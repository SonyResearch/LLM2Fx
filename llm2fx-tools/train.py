import argparse, os, functools
import wandb
import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.strategies import DDPStrategy, FSDPStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision
from pytorch_lightning.loggers import WandbLogger
from llm4mp.common_utils.config_utils import instantiate_from_config
from llm4mp.common_utils.activation_checkpointing import set_activation_checkpointing
torch.set_float32_matmul_precision('highest')
torch.backends.cuda.matmul.allow_tf32 = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(args):
    config = OmegaConf.load(args.config)
    os.makedirs(f"{config.lightning.logdir}/{config.project}/{config.lightning.version_name}", exist_ok=True)
    os.system(f"cp {args.config} {config.lightning.logdir}/{config.project}/{config.lightning.version_name}/config.yaml")

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if config.lightning.trainer.precision in ["16", "16-mixed"]:
        target_dtype = torch.float16
    elif config.lightning.trainer.precision in ["bf16", "bf16-mixed"]:
        target_dtype = torch.bfloat16
    elif config.lightning.trainer.precision in ["32", "32-true"]:
        target_dtype = torch.float32
    else:
        raise ValueError(f"Precision {config.lightning.trainer.precision} not supported")

    pl.seed_everything(42 + rank)
    data_module = instantiate_from_config(config.data)
    data_module.prepare_data()
    model = instantiate_from_config(config.model)
    model.to(target_dtype)
    trainer_config = config.lightning.trainer
    if config.model_type == "llama":
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        layer_cls = LlamaDecoderLayer
    elif config.model_type == "mistral":
        from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
        layer_cls = MistralDecoderLayer
    elif config.model_type == "qwen3":
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
        layer_cls = Qwen3DecoderLayer
    if trainer_config["strategy"] == "fsdp":
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={layer_cls},
        )
        mixed_precision_policy = MixedPrecision(
            param_dtype=target_dtype,
            reduce_dtype=target_dtype,
            buffer_dtype=target_dtype,
        )
        # bn_modules = [m for m in model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm))]
        strategy = FSDPStrategy(
            auto_wrap_policy=auto_wrap_policy,
            cpu_offload=False,
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            mixed_precision=mixed_precision_policy,
            use_orig_params=True,
            # ignored_modules=bn_modules
        )
        trainer_config["sync_batchnorm"] = True
        
    elif trainer_config["strategy"] == "ddp":
        strategy = DDPStrategy(find_unused_parameters=False)
    if config.activation_checkpointing:
        print("start activation check pointing")
        model.lm.config.use_cache = False # for activation checkpointing
        set_activation_checkpointing(model, layer_cls)
    del trainer_config["strategy"]
    trainer_kwargs = vars(argparse.Namespace(**trainer_config))

    if config.lightning.logger_type == "tensorboard":
        logger = pl.loggers.TensorBoardLogger(
            save_dir=config.lightning.logdir,
            name=config.project,
            version=config.lightning.version_name,
        )
    else:
        wandb_logger = WandbLogger(
            project=config.project,
            name=config.lightning.version_name,
            save_dir=config.lightning.logdir,
            version=config.lightning.version_name,
            settings=wandb.Settings(init_timeout=300)
        )
        wandb_logger.watch(model)
        logger = wandb_logger
    # Check model dtype
    first_param = next(model.parameters())
    print(f"Model dtype: {first_param.dtype}")

    callbacks = []
    for cb_name, cb_conf in config.lightning.callbacks.items():
        if cb_name in ["save_checkpoint", "learning_rate", "demo"]:
            callback = instantiate_from_config(cb_conf)
            callbacks.append(callback)
    
    trainer = pl.Trainer(
        callbacks=callbacks,
        strategy=strategy,
        devices="auto",
        logger=logger,
        **trainer_kwargs
    )

    trainer.fit(
        model,
        data_module,
        ckpt_path=args.resume if args.resume else None
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--gpu", default=None, type=str, required=False, help="GPU ID")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint for resuming training")
    args = parser.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["NCCL_P2P_DISABLE"] = "1"
    main(args)
