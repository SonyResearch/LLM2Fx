import os
import json
import datetime
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
from llm4mp.common_utils.config_utils import instantiate_from_config
from llm4mp.models.llm2fx2.llm.model_setup import setup_llm
from llm4mp.modules.audio_encoder import load_audio_encoder
from llm4mp.modules.adapter.al_adapter import AL_Adapter

def load_llm2fx2(ckpt_path: str, ckpt_config_path: str, inference_only=True):
    exp_config = OmegaConf.load(ckpt_config_path)
    model_params = exp_config["model"]["params"]
    model = LMTrainingWrapper.load_from_checkpoint(
        ckpt_path,
        model_config=model_params["model_config"],
        optimizer_config=model_params["optimizer_config"],
        scheduler_config=model_params["scheduler_config"],
        map_location="cpu"
    )
    if inference_only:
        import gc
        torch.cuda.empty_cache()
        gc.collect()
    return model

class LMTrainingWrapper(pl.LightningModule):
    def __init__(self, model_config=None, optimizer_config=None, scheduler_config=None):
        """Constructor setup"""
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.model_type = model_config.model_type
        self.expdir = model_config.expdir
        self.dry_audio_dropout = model_config.dry_audio_dropout
        self.max_length = model_config.max_length
        self.audio_encoder_type = model_config.audio_encoder_type
        self.projector_type = model_config.projector_type
        self.target_dtype = self.setup_precision()
        self.audio_encoder = self.setup_audio_encoder()
        self.lm, self.tokenizer = setup_llm(
            self.model_config.model_path,
            self.model_config.attn_implementation,
            self.model_config.cache_dir,
            self.target_dtype
        )      
        self.lm_hidden_size = self.lm.config.hidden_size
        if self.projector_type == "mlp":
            self.projection = nn.Sequential(
                nn.Linear(self.model_config.audio_hidden_size, self.lm_hidden_size, bias=False),
                nn.ReLU(),
                nn.Linear(self.lm_hidden_size, self.lm_hidden_size, bias=False)
            )
        elif self.projector_type == "adapter":
            self.projection = AL_Adapter(
                num_layers=6,
                num_learnable_latents=32,
                i_dim=self.model_config.audio_hidden_size,
                o_dim=self.lm_hidden_size,
                num_heads=12
            )
        if self.model_config.use_pretrained_projection:
            self.projection.load_state_dict(torch.load(os.path.join(self.model_config.projection_path, f"{self.projector_type}_weights.pth")))
        self.audio_encoder.to(self.target_dtype)
        self.projection.to(self.target_dtype)
        self.lm.to(self.target_dtype)    
        self.wet_audio_prefix = "reference audio"
        self.dry_audio_prefix = "dry audio"
        self.start_of_think = "<think>"
        self.end_of_think = "</think>"
        self.start_of_audio = "<|vision_start|>"
        self.wet_token_id = self.tokenizer.encode(self.wet_audio_prefix)
        self.dry_token_id = self.tokenizer.encode(self.dry_audio_prefix)
        self.think_token_id = self.tokenizer.encode(self.start_of_think)
        self.audio_token_id = self.tokenizer.encode(self.start_of_audio)
        self.lm.eval()
        self.audio_encoder.eval()
        self.projection.train()
        for param in self.lm.parameters():
            param.requires_grad = False
        if self.model_config.apply_lora:
            self.apply_lora()        
        if self.model_config.loss_type == "ce_ntl":
            from llm4mp.modules.ntloss import NTLoss
            activated_vocab = self.lm.lm_head.weight.shape[0]
            self.ntl_fn = NTLoss(tokenizer=self.tokenizer, activated_vocab=activated_vocab)
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def loss_fn(self, logits, labels):
        if self.model_config.loss_type == "ce":
            return nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100
            )
        elif self.model_config.loss_type == "ce_ntl":
            ntl = self.ntl_fn(logits, labels)
            ce_loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100
            )
            return ce_loss + (ntl * 0.3)

    def apply_lora(self):
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=self.model_config.lora_rank,  # Rank of LoRA update matrices
            lora_alpha=(self.model_config.lora_rank * 2),  # Alpha scaling factor
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.lm = get_peft_model(self.lm, lora_config)
        self.lm.print_trainable_parameters()

    def setup_audio_encoder(self):
        audio_encoder = load_audio_encoder("fxenc_plusplus")
        audio_encoder.eval()
        return audio_encoder
        
    def setup_precision(self):
        if self.model_config.precision in ["bf16", "bf16-mixed"]:
            target_dtype = torch.bfloat16
        elif self.model_config.precision in ["32", "32-mixed"]:
            target_dtype = torch.float32
        return target_dtype

    @property
    def device(self):
        return list(self.parameters())[0].device

    @property
    def dtype(self):
        return list(self.parameters())[0].dtype

    def get_audio_embedding(self, audio_dry, audio_wet):
        B, L, D = audio_dry.size()
        concat_audios = torch.cat([audio_dry, audio_wet], dim=0)
        with torch.no_grad():
            audio_embeds = self.audio_encoder.get_local_embedding(concat_audios)
            audio_embeds = audio_embeds.detach() # for stop gradient
        dry_embeds = audio_embeds[:B, :, :]
        wet_embeds = audio_embeds[B:, :, :]
        z_dry = self.projection(dry_embeds)
        z_wet = self.projection(wet_embeds)
        return z_dry, z_wet

    def training_multimodal_embedding(self, input_ids, output_ids, input_attention_mask, output_attention_mask, z_dry, z_wet):
        audio_token_id = torch.tensor([self.audio_token_id], device=self.device)
        wet_token_id = torch.tensor([self.wet_token_id], device=self.device)
        dry_token_id = torch.tensor([self.dry_token_id], device=self.device)
        audio_insert_index = torch.where(input_ids == audio_token_id)[1].tolist()
        if self.model_config.apply_lora:
            inputs_embeds = self.lm.model.model.embed_tokens(input_ids).to(self.device)
            output_embeds = self.lm.model.model.embed_tokens(output_ids).to(self.device)
            wet_token_emb = self.lm.model.model.embed_tokens(wet_token_id).repeat(z_dry.shape[0], 1, 1)
            dry_token_emb = self.lm.model.model.embed_tokens(dry_token_id).repeat(z_dry.shape[0], 1, 1)
        else:
            inputs_embeds = self.lm.model.embed_tokens(input_ids).to(self.device)
            output_embeds = self.lm.model.embed_tokens(output_ids).to(self.device)
            wet_token_emb = self.lm.model.embed_tokens(wet_token_id).repeat(z_dry.shape[0], 1, 1)
            dry_token_emb = self.lm.model.embed_tokens(dry_token_id).repeat(z_dry.shape[0], 1, 1)
        z_dry = torch.cat([dry_token_emb, z_dry], dim=1)
        z_wet = torch.cat([wet_token_emb, z_wet], dim=1)
        if torch.rand(1) < self.dry_audio_dropout:
            dry_audio_attention_mask = torch.zeros(z_dry.shape[:2], device=self.device)
        else:
            dry_audio_attention_mask = torch.ones(z_dry.shape[:2], device=self.device)
        wet_audio_attention_mask = torch.ones(z_wet.shape[:2], device=self.device)

        batch_embeds = []
        batch_attention_mask = []
        batch_labels = []
        for batch_idx, audio_insert_idx in enumerate(audio_insert_index):
            embeds = torch.cat([inputs_embeds[batch_idx, :audio_insert_idx], z_dry[batch_idx, :], z_wet[batch_idx, :], inputs_embeds[batch_idx, audio_insert_idx+1:], output_embeds[batch_idx, :]], dim=0)
            attention_mask = torch.cat([input_attention_mask[batch_idx, :audio_insert_idx], dry_audio_attention_mask[batch_idx, :], wet_audio_attention_mask[batch_idx, :], input_attention_mask[batch_idx, audio_insert_idx+1:], output_attention_mask[batch_idx, :]], dim=0)
            labels = torch.full(embeds.shape[:1], fill_value=-100, device=self.device, dtype=torch.long)
            labels[-output_embeds.shape[1]:] = output_ids[batch_idx, :] # alpaca-style loss function
            batch_embeds.append(embeds)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)
        batch_embeds = torch.stack(batch_embeds)
        batch_attention_mask = torch.stack(batch_attention_mask)
        batch_labels = torch.stack(batch_labels)
        pad_mask = (batch_labels == self.tokenizer.pad_token_id)  # Find padding tokens
        batch_labels[pad_mask] = -100  # Replace pad tokens with -100
        return batch_embeds[:,:self.max_length], batch_attention_mask[:,:self.max_length], batch_labels[:,:self.max_length] # for training stability

    
    def inference_multimodal_embedding(self, input_ids, input_attention_mask, z_dry, z_wet):
        audio_token_id = torch.tensor([self.audio_token_id], device=self.device)
        wet_token_id = torch.tensor([self.wet_token_id], device=self.device)
        dry_token_id = torch.tensor([self.dry_token_id], device=self.device)
        think_token_id = torch.tensor([self.think_token_id], device=self.device)
        audio_insert_index = torch.where(input_ids == audio_token_id)[1].tolist()
        if self.model_config.apply_lora:
            inputs_embeds = self.lm.model.model.embed_tokens(input_ids).to(self.device)
            wet_token_emb = self.lm.model.model.embed_tokens(wet_token_id).repeat(z_dry.shape[0], 1, 1)
            dry_token_emb = self.lm.model.model.embed_tokens(dry_token_id).repeat(z_dry.shape[0], 1, 1)
            think_token_emb = self.lm.model.model.embed_tokens(think_token_id).repeat(z_dry.shape[0], 1, 1)
        else:
            inputs_embeds = self.lm.model.embed_tokens(input_ids).to(self.device)
            wet_token_emb = self.lm.model.embed_tokens(wet_token_id).repeat(z_dry.shape[0], 1, 1)
            dry_token_emb = self.lm.model.embed_tokens(dry_token_id).repeat(z_dry.shape[0], 1, 1)
            think_token_emb = self.lm.model.embed_tokens(think_token_id).repeat(z_dry.shape[0], 1, 1)
        z_dry = torch.cat([dry_token_emb, z_dry], dim=1)
        z_wet = torch.cat([wet_token_emb, z_wet], dim=1)
        dry_audio_attention_mask = torch.ones(z_dry.shape[:2], device=self.device)
        wet_audio_attention_mask = torch.ones(z_wet.shape[:2], device=self.device)
        thunk_token_attention_mask = torch.ones(think_token_emb.shape[:2], device=self.device)
        batch_embeds = []
        batch_attention_mask = []
        for batch_idx, audio_insert_idx in enumerate(audio_insert_index):
            embeds = torch.cat([inputs_embeds[batch_idx, :audio_insert_idx], z_dry[batch_idx, :], z_wet[batch_idx, :], inputs_embeds[batch_idx, audio_insert_idx+1:], think_token_emb[batch_idx, :]], dim=0)
            attention_mask = torch.cat([input_attention_mask[batch_idx, :audio_insert_idx], dry_audio_attention_mask[batch_idx, :], wet_audio_attention_mask[batch_idx, :], input_attention_mask[batch_idx, audio_insert_idx+1:], thunk_token_attention_mask[batch_idx, :]], dim=0)
            batch_embeds.append(embeds)
            batch_attention_mask.append(attention_mask)
        batch_embeds = torch.stack(batch_embeds)
        batch_attention_mask = torch.stack(batch_attention_mask)
        return batch_embeds, batch_attention_mask

    def forward_pass(self, batch):
        input_tokens = self.tokenizer(
            batch['input_text'],
            return_tensors="pt",
            max_length=self.model_config.input_length,
            padding="max_length",
            # padding=True,
            truncation=True,
            padding_side="left"
        )
        output_tokens = self.tokenizer(
            batch['output_text'],
            return_tensors="pt",
            max_length=self.model_config.output_length,
            padding="max_length",
            # padding=True,
            truncation=True,
            padding_side="right"
        )        
        input_ids = input_tokens.input_ids.to(self.device)
        input_attention_mask = input_tokens.attention_mask.to(self.device)
        output_ids = output_tokens.input_ids.to(self.device)
        output_attention_mask = output_tokens.attention_mask.to(self.device)
        z_dry, z_wet = self.get_audio_embedding(batch['audio_dry'], batch['audio_wet'])
        inputs_embeds, attention_mask, labels = self.training_multimodal_embedding(input_ids, output_ids, input_attention_mask, output_attention_mask, z_dry, z_wet)
        outputs = self.lm.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = outputs.logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        loss = self.loss_fn(logits, labels)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.forward_pass(batch)
        self.log("train_loss", loss, sync_dist=True, prog_bar=True)
        return loss

    def generation_step(self, batch):
        input_tokens = self.tokenizer(
            batch['input_text'], # instruction
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        input_ids = input_tokens.input_ids
        input_attention_mask = input_tokens.attention_mask
        z_dry, z_wet = self.get_audio_embedding(batch['audio_dry'], batch['audio_wet'])
        inputs_embeds, attention_mask = self.inference_multimodal_embedding(input_ids, input_attention_mask, z_dry, z_wet)
        outputs = self.lm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=self.max_length,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            top_p=0.95
        )
        output_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for output_text in output_texts:
            print("\n---start generation---")
            print(output_text)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss = self.forward_pass(batch)
        self.log("val_loss", loss, sync_dist=True)
        # Store the loss in the outputs dictionary
        self.validation_step_outputs = getattr(self, "validation_step_outputs", [])
        self.validation_step_outputs.append({"val_loss": loss})
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        # Calculate average validation loss using stored outputs
        val_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean()
        # Calculate perplexity
        # Log metrics
        self.log("mean_val_loss", val_loss, sync_dist=True)
        # Clear the outputs to free memory
        self.validation_step_outputs = []
        if self.model_config.apply_lora == False:
            weight_dict = self.projection.state_dict()
            save_weight_dict = weight_dict.copy()
            for key in save_weight_dict:
                save_weight_dict[key] = save_weight_dict[key].detach()
            torch.save(save_weight_dict, os.path.join(self.expdir, f"{self.projector_type}_weights.pth"))
            
    def configure_optimizers(self):
        optimizer = instantiate_from_config(
            self.optimizer_config, params=filter(lambda p: p.requires_grad, self.parameters())
        )
        scheduler = instantiate_from_config(
            self.scheduler_config, optimizer=optimizer
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]


