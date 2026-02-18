from typing import List, Optional, Tuple
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities.types import LRSchedulerConfig

from ..tasks import DEGClassificationTask
from ..input import CellLineAutoencoderInputTransform
from ..fusion.encoder_layer import EncoderLayer

class FusionModel(L.LightningModule):
    def __init__(
        self,
        input_transforms: List[nn.Module],
        task: nn.Module,
        d_model: int,
        n_heads: int,
        n_layers: int,
        warmup_epochs: int,
        learning_rate: float,
        use_gate: bool,
        use_gelu: bool,
        xavier_init: bool,
        reconstruction_weight: float,
        dropout: float,
        contextualize_cell_line: bool,
        final_layer_cls: bool,
        add_cell_line_to_tokens: bool = False,
        autoencoder_weight: float = 0.1,
        modal_drop_prob: float = 0,
        **kwargs, 
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['input_transforms', 'task'])

        self.input_transforms = nn.ModuleList(input_transforms)
        self.task = task
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.reconstruction_weight = reconstruction_weight
        self.d_model = d_model
        self.n_heads = n_heads
        self.use_gelu = use_gelu
        self.use_gate = use_gate
        self.xavier_init = xavier_init
        self.dropout = dropout
        self.contextualize_cell_line = contextualize_cell_line
        self.final_layer_cls = final_layer_cls
        self.add_cell_line_to_tokens = add_cell_line_to_tokens
        self.autoencoder_weight = autoencoder_weight
        self.modal_drop_prob = modal_drop_prob

        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    embed_dim=d_model,
                    num_heads=n_heads,
                    batch_first=False,
                    use_gate=use_gate,
                    dropout=dropout,
                    use_gelu=use_gelu,
                    xavier_init=xavier_init,
                )
                for _ in range(n_layers)
            ] 
        )

        self.cls_token = nn.Parameter(torch.randn(1, d_model))

        if self.final_layer_cls:
            self.final_encoder_layer = EncoderLayer(
                embed_dim=d_model,
                num_heads=n_heads,
                batch_first=False,
                use_gate=False,
                dropout=dropout,
                use_gelu=use_gelu,
                xavier_init=xavier_init,
            )

        if self.reconstruction_weight > 0:
            n_reco_modalities = len(input_transforms) - 1
            self.reconstruction_heads = nn.ModuleList()
            for i in range(n_reco_modalities):
                if hasattr(input_transforms[i], 'forward') and hasattr(input_transforms[i], 'linear'):
                    input_dim = input_transforms[i].linear.in_features
                    self.reconstruction_heads.append(nn.Linear(d_model, input_dim))
                else:
                    self.reconstruction_heads.append(nn.Linear(d_model, d_model))

    def forward(self, batch, return_inputs=False, augment=False):
        
        # Apply the input-transforms to our batch
        transformed_inputs, original_inputs, padding_inputs = [], [], []
        for tr in self.input_transforms:

            # Get transformed and original inputs
            if hasattr(tr, 'forward') and hasattr(tr, 'linear'):
                transformed, normalized = tr(batch, return_normalized_input=True)
                transformed_inputs.append(transformed)
                original_inputs.append(normalized)
            else:
                transformed_inputs.append(tr(batch))
                original_inputs.append(None)
            
            # Get padding mask
            if tr.input_key + "_attn_mask" in batch:
                padding_inputs.append(batch[tr.input_key + "_attn_mask"].ravel())
            else:
                padding_inputs.append(torch.zeros(len(batch[tr.input_key]), dtype=torch.bool))

        # Split modality tokens from the cell line token
        modality_inputs, cell_line_input = transformed_inputs[:-1], transformed_inputs[-1]
        padding_inputs = padding_inputs[:-1]

        # Apply modality dropout
        if augment and self.modal_drop_prob > 0:
            augmented_modality_inputs = []
            for x in modality_inputs:
                mask = torch.bernoulli(torch.full((x.shape[0], 1), 1.0 - self.modal_drop_prob, device=x.device))
                augmented_modality_inputs.append(x * mask)
            modality_inputs = augmented_modality_inputs
        
        # Stack inputs
        x_modalities = torch.stack(modality_inputs, dim=0)  # (n_modalities, batch_size, dim)
        key_padding_mask = torch.stack(padding_inputs, dim=1)  # (batch_size, n_modalities)
        batch_size = x_modalities.shape[1]

        # Append cell line as a separate token (if we are not adding it to every token later on)
        if self.contextualize_cell_line and not self.add_cell_line_to_tokens:
            x_modalities = torch.cat([x_modalities, cell_line_input.unsqueeze(0)], dim=0)
            key_padding_mask = torch.cat([
                key_padding_mask,
                torch.zeros((batch_size, 1), dtype=torch.bool, device=key_padding_mask.device)
            ], dim=1)
        
        # Prepend the CLS token (if we're including it in the attention)
        if not self.final_layer_cls:
            cls_tokens = self.cls_token.unsqueeze(0).expand(1, batch_size, self.d_model)
            x_modalities = torch.cat([cls_tokens, x_modalities], dim=0)
            key_padding_mask = torch.cat([
                torch.zeros((batch_size, 1), dtype=torch.bool, device=key_padding_mask.device),
                key_padding_mask
            ], dim=1)
        
        # Sum cell line with all tokens
        if self.add_cell_line_to_tokens:
            x_modalities = x_modalities + cell_line_input.unsqueeze(0)

        # Check that there is no perturbation in the batch that is masked everywhere (key_padding_mask[i, j] == True for all j)
        assert (((~key_padding_mask).sum(1) == 0).sum() == 0), "The current embeddings do not cover a perturbation in this batch"
        
        # Apply encoder layers
        for layer in self.encoder_layers:
            x_modalities = layer(x_modalities, key_padding_mask=key_padding_mask)
        
        # If CLS wasn't included in attention, do it now
        if self.final_layer_cls:

            # Get the cell line token
            cls_tokens = self.cls_token.unsqueeze(0).expand(1, batch_size, self.d_model)
            if self.add_cell_line_to_tokens:
                cls_tokens = cls_tokens + cell_line_input.unsqueeze(0)

            # Prepend it
            x_modalities = torch.cat([cls_tokens, x_modalities], dim=0)
            key_padding_mask = torch.cat([
                torch.zeros((batch_size, 1), dtype=torch.bool, device=key_padding_mask.device), 
                key_padding_mask,
            ], dim=1)

            # Go through a last layer of encoder
            x_modalities = self.final_encoder_layer(x_modalities, key_padding_mask=key_padding_mask)

        # If we haven't added the cell line to tokens yet, do it now
        if not self.contextualize_cell_line and not self.add_cell_line_to_tokens:
            x_modalities = torch.cat([x_modalities, cell_line_input.unsqueeze(0)], dim=0)

        return (x_modalities, original_inputs) if return_inputs else x_modalities

    def _calculate_recon_loss(self, encoded_x, original_inputs):
        # Given [CLS, Mod1, ..., ModN] or [CLS, Mod1, ..., ModN, CellLine],
        # get only [Mod1, ..., ModN]
        if self.add_cell_line_to_tokens:
            encoded_modality_tokens = encoded_x[1:]
        else:
            encoded_modality_tokens = encoded_x[1:-1]
        original_modality_tokens = original_inputs[:-1]
        assert encoded_modality_tokens.shape[0] > 0

        # Compute the mse reconstruction loss for all modalities
        individual_losses = []
        for i, head in enumerate(self.reconstruction_heads):
            assert original_modality_tokens[i] is not None
            reconstructed_input = head(encoded_modality_tokens[i])
            loss = F.mse_loss(reconstructed_input, original_modality_tokens[i])
            individual_losses.append(loss)

        # Return the average across modalities
        return torch.stack(individual_losses).mean()

    def _compute_autoencoder_loss(self, batch):
        autoencoder_loss, count = 0, 0
        for tr in self.input_transforms:
            if isinstance(tr, CellLineAutoencoderInputTransform):
                _, reconstruction, original = tr(batch, return_reconstruction=True)
                autoencoder_loss = F.mse_loss(reconstruction, original)
                count += 1
        assert count <= 1
        return autoencoder_loss

    def training_step(self, batch, batch_idx):

        # Get encoded and original embeddings
        encoded_x, original_inputs = self.forward(batch, return_inputs=True, augment=True)

        # Compute the task loss (regression or classification)
        y_hat = self.task(encoded_x)
        task_loss = self.task.loss_fn(y_hat, batch)

        # Compute the autoencoder loss (returns 0 if no cell-line autoencoder)
        autoencoder_loss = self._compute_autoencoder_loss(batch)

        # Compute the reconstruction loss
        if self.reconstruction_weight > 0:
            recon_loss = self._calculate_recon_loss(encoded_x, original_inputs)
        else:
            recon_loss = 0

        # Compute the total loss
        total_loss = task_loss + self.reconstruction_weight * recon_loss + self.autoencoder_weight * autoencoder_loss

        self.log_dict({
            "train_loss": task_loss,
            "train_total_loss": total_loss,
            "train_recon_loss": recon_loss,
            "train_autoencoder_loss": autoencoder_loss,
        }, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        # Get encoded and original embeddings
        encoded_x, original_inputs = self.forward(batch, return_inputs=True)

        # Compute the task loss (regression or classification)
        y_hat = self.task(encoded_x)
        task_loss = self.task.loss_fn(y_hat, batch)

        # Compute the autoencoder loss (returns 0 if no cell-line autoencoder)
        autoencoder_loss = self._compute_autoencoder_loss(batch)

        # Compute the reconstruction loss
        if self.reconstruction_weight == 0.0:
            recon_loss = 0
        else:
            recon_loss = self._calculate_recon_loss(encoded_x, original_inputs)

        # Compute the total loss
        total_loss = task_loss + self.reconstruction_weight * recon_loss + self.autoencoder_weight * autoencoder_loss

        self.log_dict({
            "val_loss": task_loss,
            "val_total_loss": total_loss,
            "val_recon_loss": recon_loss,
            "val_autoencoder_loss": autoencoder_loss,
        }, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return total_loss

    def predict_step(self, batch, batch_idx):
        x = self.forward(batch, return_inputs=False)
        return self.task(x), batch

    def configure_optimizers(self):
        self._optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return self._optimizer
    
    def on_train_start(self):

        assert self.trainer.max_epochs is not None
        assert self.trainer.estimated_stepping_batches is not None

        # Get the number of steps per epoch
        total_steps = self.trainer.estimated_stepping_batches
        steps_per_epoch = total_steps // self.trainer.max_epochs
        
        # Compute the number of warmup steps
        warmup_steps = steps_per_epoch * self.warmup_epochs
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self._optimizer, start_factor=1e-8, total_iters=warmup_steps
        )

        self.trainer.strategy.lr_scheduler_configs = [
            LRSchedulerConfig(warmup_scheduler, interval="step")
        ]
