import os
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import pytorch_lightning as pl  
from llm4mp.modules.audio_encoder import load_audio_encoder
from llm4mp.common_utils.config_utils import instantiate_from_config
from llm4mp.common_utils.fx_utils import load_fx_config
from llm4mp.modules.adapter.al_adapter import AL_Adapter

def load_regression(ckpt_path: str, ckpt_config_path: str, inference_only=True):
    exp_config = OmegaConf.load(ckpt_config_path)
    model_params = exp_config["model"]["params"]
    model = RegressionWrapper.load_from_checkpoint(
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

class RegressionWrapper(pl.LightningModule):
    def __init__(self, model_config=None, optimizer_config=None, scheduler_config=None):
        """Constructor setup"""
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.fx_config = load_fx_config(self.model_config.fx_config_type)
        self.audio_encoder_type = self.model_config.audio_encoder_type
        self.embedding_type = self.model_config.embedding_type
        self.loss_type = self.model_config.loss_type
        self.audio_encoder = self.setup_audio_encoder(self.audio_encoder_type)
        self.audio_hidden_size = model_config.audio_hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(self.audio_hidden_size, model_config.mlp_hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(model_config.mlp_hidden_size, model_config.mlp_hidden_size, bias=False)
        )
        if "parameter_regression" in self.loss_type:
            self.regression_heads = nn.ModuleDict({
                fx_type: nn.Linear(model_config.mlp_hidden_size * 2, len(self.fx_config["fx_param_keys"][fx_type]), bias=False)
                for fx_type in self.fx_config["fx_list"]
            })
        if "effect_classification" in self.loss_type:
            self.classification_head = nn.Linear(model_config.mlp_hidden_size * 2, len(self.fx_config["fx_list"]), bias=False)
    
        self.validation_step_outputs = []
        self.test_step_outputs = []

    @property
    def device(self):
        return next(self.mlp.parameters()).device

    @property
    def dtype(self):
        return next(self.mlp.parameters()).dtype

    def regression_loss_fn(self, predict_params, gt_params, mask):
        """
        mask: B
        predict_params: B x Number of params
        gt_params: B x Number of params
        """
        mse_loss = torch.nn.functional.mse_loss(predict_params, gt_params, reduction="none")  # B x Number of params
        masked_loss = mse_loss * mask
        denom = mask.sum()
        if denom == 0:
            return torch.tensor(0.0, device=predict_params.device, dtype=predict_params.dtype)
        loss = masked_loss.mean(axis=1).sum() / denom
        return loss
    
    def effect_classification(self, audio_embeds, fx_labels):
        """
        logits: B x Number of fx
        gt_labels: B
        """
        logits = self.classification_head(audio_embeds)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, fx_labels)
        return loss
        
    def setup_audio_encoder(self, audio_encoder_type):
        if audio_encoder_type == "fxenc_pp":
            audio_encoder = load_audio_encoder("fxenc_plusplus")
            audio_encoder.eval()
            for param in audio_encoder.parameters():
                param.requires_grad = False
        elif audio_encoder_type == "clap":
            audio_encoder = load_audio_encoder("clap")
            audio_encoder.eval()
            for param in audio_encoder.parameters():
                param.requires_grad = False
        elif audio_encoder_type == "stito_encoder":
            audio_encoder = load_audio_encoder("stito_encoder")
            audio_encoder.eval()
            for param in audio_encoder.parameters():
                param.requires_grad = False
        return audio_encoder
        
    def get_global_audio_embedding(self, audio_dry, audio_wet):
        B, L, D = audio_dry.size()
        concat_audios = torch.cat([audio_dry, audio_wet], dim=0)
        if self.audio_encoder_type == "fxenc_pp":
            audio_embeds = self.audio_encoder.get_global_embedding(concat_audios)
            audio_embeds = audio_embeds.detach() # for stop gradient
        elif self.audio_encoder_type == "clap":
            audio_embeds = self.audio_encoder.get_global_embedding(concat_audios)
            audio_embeds = audio_embeds.detach() # for stop gradient
        elif self.audio_encoder_type == "stito_encoder":
            audio_embeds = self.audio_encoder.get_global_embedding(concat_audios)
            audio_embeds = audio_embeds.detach() # for stop gradient
        dry_embeds = audio_embeds[:B, :]
        wet_embeds = audio_embeds[B:, :]
        z_dry = self.mlp(dry_embeds)
        z_wet = self.mlp(wet_embeds)
        z_dry = torch.nn.functional.normalize(z_dry, dim=-1)
        z_wet = torch.nn.functional.normalize(z_wet, dim=-1)
        if torch.rand(1) < self.model_config.dry_audio_dropout:
            z_dry = torch.zeros_like(z_dry, device=self.device, dtype=self.dtype) # dry audio dropout
        z_io = torch.cat([z_dry, z_wet], dim=-1)
        return z_io.squeeze(1)

    def parameter_regression(self, audio_embeds, batch):
        """
        effect_labels act as mask for parameter regression
        """
        param_loss = 0
        for fx_type in self.fx_config["fx_list"]:
            gt_params = batch[f"{fx_type}_target"] # B x Number of params
            mask = batch[f"{fx_type}_mask"]
            predict_prams = self.regression_heads[fx_type](audio_embeds) # B x Number of params
            predict_prams = torch.clamp(predict_prams, min=0.0, max=1.0) # clamp to [0,1] range
            param_loss += self.regression_loss_fn(predict_prams, gt_params, mask)
        return param_loss

    def forward_pass(self, batch):
        audio_dry = batch['audio_dry'].to(self.dtype)
        audio_wet = batch['audio_wet'].to(self.dtype)
        fx_labels = batch['fx_labels']
        audio_embeds = self.get_global_audio_embedding(audio_dry, audio_wet) # B x D
        total_loss, loss_info = 0, {}
        if "effect_classification" in self.loss_type:
            effect_loss = self.effect_classification(audio_embeds, fx_labels)
            total_loss += (0.1 * effect_loss)
        if "parameter_regression" in self.loss_type:
            param_loss = self.parameter_regression(audio_embeds, batch)
            total_loss += param_loss
        loss_info["cls_loss"] = effect_loss if "effect_classification" in self.loss_type else 0
        loss_info["reg_loss"] = param_loss if "parameter_regression" in self.loss_type else 0
        return total_loss, loss_info
    
    @torch.no_grad()
    def predict_effect(self, audio_dry, audio_wet, use_pooling=True):
        audio_embeds = self.get_global_audio_embedding(audio_dry, audio_wet) # B x D
        if use_pooling:
            audio_embeds = audio_embeds.mean(dim=0, keepdim=True)
        logits = self.classification_head(audio_embeds)
        logits = torch.sigmoid(logits) # clamp to [0,1] range
        return logits

    @torch.no_grad()
    def predict_parameter(self, audio_dry, audio_wet, fx_type, use_pooling=True):
        audio_embeds = self.get_global_audio_embedding(audio_dry, audio_wet) # B x D
        if use_pooling:
            audio_embeds = audio_embeds.mean(dim=0, keepdim=True)
        predict_params = self.regression_heads[fx_type](audio_embeds)
        predict_params = torch.clamp(predict_params, min=0.0, max=1.0) # clamp to [0,1] range
        return predict_params

    def training_step(self, batch, batch_idx):
        batch_size = batch['audio_dry'].size(0)
        loss, loss_info = self.forward_pass(batch)
        self.log("train_loss", loss, sync_dist=True, prog_bar=True, batch_size=batch_size)
        for k, v in loss_info.items():
            self.log(f"train_{k}", v, sync_dist=True, batch_size=batch_size)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        batch_size = batch['audio_dry'].size(0)
        loss, loss_info = self.forward_pass(batch)
        self.log("val_loss", loss, sync_dist=True, batch_size=batch_size)
        for k, v in loss_info.items():
            self.log(f"val_{k}", v, sync_dist=True, batch_size=batch_size)
        # Store the loss in the outputs dictionary
        self.validation_step_outputs = getattr(self, "validation_step_outputs", [])
        self.validation_step_outputs.append({"val_loss": loss})
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        # Calculate average validation loss using stored outputs
        val_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean()
        self.log("mean_val_loss", val_loss, sync_dist=True)
        self.validation_step_outputs = []

    def configure_optimizers(self):
        optimizer = instantiate_from_config(
            self.optimizer_config, params=filter(lambda p: p.requires_grad, self.parameters())
        )
        scheduler = instantiate_from_config(
            self.scheduler_config, optimizer=optimizer
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]