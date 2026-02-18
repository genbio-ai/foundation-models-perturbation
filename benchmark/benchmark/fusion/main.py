import hydra
from lightning_fabric.utilities.seed import seed_everything
from omegaconf import DictConfig, OmegaConf
import wandb
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint

from benchmark.fusion.utils.compute_cell_line_controls import compute_average_control_expressions
from benchmark.fusion.model.input import CellLineAutoencoderInputTransform
from benchmark.fusion.model.fusion import FusionModel
from benchmark.paths import ESSENTIAL_CONTROLS_ADATA_PATH, ESSENTIAL_CONTROLS_CACHE_PATH

@hydra.main(version_base=None, config_path="config", config_name=None)
def main(cfg: DictConfig) -> None:

    seed = seed_everything(123)
    best_val_loss = None

    # Define the W&B logger
    wandb_logger = instantiate(cfg.wandb_logger)
    wandb_logger.experiment.name = f"fold-{cfg.datamodule.fold} ({cfg.model_name})"
    wandb_logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))

    # Define the datamodule
    datamodule = instantiate(cfg.datamodule, _convert_="partial")

    # Instantiate the fusion model (it's more complicated if we use the cell line autoencoder)
    if not OmegaConf.select(cfg, 'use_cell_line_autoencoder', default=False):
        fusion_model = instantiate(cfg.fusion_model)
    else:
        # Get control expression
        assert cfg.datamodule.dataset_type == "essential_lfc"
        le = datamodule.train_dataset.condition_encoders[0].le
        cell_line_label_mapping = {name: le.transform([name])[0] for name in le.classes_}
        cell_line_expressions = compute_average_control_expressions(
            adata_path=ESSENTIAL_CONTROLS_ADATA_PATH,
            cache_path=ESSENTIAL_CONTROLS_CACHE_PATH,
            control_gene="non-targeting",
            gene_key="gene",
            cell_line_key="cell_line",
            cell_line_label_mapping=cell_line_label_mapping,
        )
        
        # Pass control expression to the cell line autoencoder
        input_transforms = []
        for transform_cfg in cfg.fusion_model.input_transforms:
            if transform_cfg._target_ == "model.input.CellLineAutoencoderInputTransform":
                autoencoder = instantiate(transform_cfg.autoencoder)
                transform = CellLineAutoencoderInputTransform(
                    autoencoder=autoencoder,
                    cell_line_expressions=cell_line_expressions,
                    input_key=transform_cfg.input_key,
                    normalize_input=transform_cfg.get('normalize_input', True),
                    normalize_output=transform_cfg.get('normalize_output', True),
                )
                input_transforms.append(transform)
            else:
                input_transforms.append(instantiate(transform_cfg))
        
        # Instantiate de model
        task = instantiate(cfg.fusion_model.task)
        fusion_model_cfg = OmegaConf.to_container(cfg.fusion_model, resolve=True)
        fusion_model_cfg.pop('input_transforms')
        fusion_model_cfg.pop('task')
        fusion_model_cfg.pop('_target_')
        fusion_model = FusionModel(input_transforms=input_transforms, task=task, **fusion_model_cfg)

    # Train the model
    trainer = instantiate(cfg.trainer, logger=wandb_logger)
    trainer.fit(model=fusion_model, datamodule=datamodule)

    # Get the best validation loss
    for callback in trainer.callbacks:
        if isinstance(callback, ModelCheckpoint):
            best_val_loss = callback.best_model_score.item()
    wandb_logger.log_metrics({"best_val_loss": best_val_loss})
    wandb.finish()

    return best_val_loss

if __name__ == "__main__":
    main()