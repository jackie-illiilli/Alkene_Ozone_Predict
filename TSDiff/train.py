import os
import shutil
import yaml
from easydict import EasyDict
from glob import glob
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pathlib import Path
from .models.epsnet import get_model
from .utils.datasets import ConformationDataset, TSDataset
from .utils.transforms import CountNodesPerGraph
from .utils.misc import seed_all, get_new_log_dir, get_logger, get_checkpoint_path
from .utils.common import get_optimizer, get_scheduler
import torch_geometric


# ------------------- 原来的 LightningModule / DataModule（保持不变） -------------------
class DiffusionLightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(dict(config))
        self.model = get_model(config.model)
        self.config = config
        self.anneal_power = config.train.anneal_power
        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx):
        loss = self.model.get_loss(
            atom_type=batch.atom_type,
            r_feat=batch.r_feat,
            p_feat=batch.p_feat,
            pos=batch.pos,
            bond_index=batch.edge_index,
            bond_type=batch.edge_type,
            batch=batch.batch,
            num_nodes_per_graph=batch.num_nodes_per_graph,
            num_graphs=batch.num_graphs,
            anneal_power=self.anneal_power,
        )
        self.log("train/loss", loss.mean(), on_step=True, prog_bar=True)
        return loss.mean()

    def validation_step(self, batch, batch_idx):
        loss = self.model.get_loss(
            atom_type=batch.atom_type,
            r_feat=batch.r_feat,
            p_feat=batch.p_feat,
            pos=batch.pos,
            bond_index=batch.edge_index,
            bond_type=batch.edge_type,
            batch=batch.batch,
            num_nodes_per_graph=batch.num_nodes_per_graph,
            num_graphs=batch.num_graphs,
            anneal_power=self.anneal_power,
        )
        self.validation_step_outputs.append({"loss_sum": loss.sum(), "n": loss.size(0)})
        return loss

    def on_validation_epoch_end(self):
        if self.validation_step_outputs:
            total_loss = sum(out["loss_sum"] for out in self.validation_step_outputs)
            total_n = sum(out["n"] for out in self.validation_step_outputs)
            avg_loss = total_loss / total_n if total_n > 0 else 0
            self.log("val_loss", avg_loss, prog_bar=True)
            self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = get_optimizer(self.config.train.optimizer, self.model)
        scheduler = get_scheduler(self.config.train.scheduler, optimizer)

        lr_scheduler_config = {"scheduler": scheduler}

        if self.config.train.scheduler.type == "plateau":
            lr_scheduler_config.update({
                "monitor": "val_loss",
                "interval": "epoch",
            })
        else:
            lr_scheduler_config.update({
                "interval": "epoch",
                "frequency": self.config.train.val_freq,
            })

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}


class DiffusionDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transforms = CountNodesPerGraph()

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            if hasattr(self.config.model, "TS") and self.config.model.TS:
                self.train_set = TSDataset(self.config.dataset.train, transform=self.transforms)
                self.val_set   = TSDataset(self.config.dataset.val,   transform=self.transforms)
            else:
                self.train_set = ConformationDataset(self.config.dataset.train, transform=self.transforms)
                self.val_set   = ConformationDataset(self.config.dataset.val,   transform=self.transforms)

    def train_dataloader(self):
        return torch_geometric.loader.DataLoader(
            self.train_set,
            batch_size=self.config.train.batch_size,
            shuffle=True,
            num_workers=getattr(self.config.train, "num_workers", 4),
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch_geometric.loader.DataLoader(
            self.val_set,
            batch_size=self.config.train.batch_size,
            shuffle=False,
            num_workers=getattr(self.config.train, "num_workers", 4),
            pin_memory=True,
        )

def train(parameter: dict):
    """
    封装后的训练入口

    参数
    ----
    parameter : dict 或 EasyDict
        必须包含以下字段（原来的 argparse + config 文件的内容）：

        Required keys
        - config: str                     # config yaml 文件路径，或已经是一个 config dict
        - device: str = "cuda" / "cpu"
        - logdir: str = "./logs"
        - project: str = None            # wandb project（留空则不使用 wandb）
        - name: str = None               # wandb run name
        - tag: str = None
        - fn: str = None
        - resume_iter: int = None
        - pretrain: str = ""             # 预训练权重路径
        - resume: str = None             # 如果是文件夹路径，表示继续之前的实验

        其它所有超参数会从 config yaml 中读取。

    示例
    ----
    >>> train({
    ...     "config": "./configs/train_config_KwonFirst_2000_xtb.yml",
    ...     "device": "cuda",
    ...     "logdir": "./logs",
    ...     "project": "MyDiff",
    ...     "name": "exp001",
    ...     "tag": "test",
    ... })
    """
    param = EasyDict(parameter)

    # ==================== 【前面所有代码保持不变，直到 Trainer 部分】 ====================
    if isinstance(param.config, str):
        with open(param.config, "r") as f:
            config = EasyDict(yaml.safe_load(f))
        config_path = param.config
    else:
        config = EasyDict(param.config)
        config_path = "./in_memory_config.yml"

    config_name = Path(config_path).stem
    seed_all(config.train.get("seed", 42))

    resume_from = getattr(param, "resume", None)
    tag = param.tag or param.name or ""

    if resume_from:
        log_dir = get_new_log_dir(param.logdir, prefix=config_name,
                                  tag=f"{tag}_resume", fn=param.get("fn"))
        os.symlink(os.path.realpath(resume_from),
                   os.path.join(log_dir, Path(resume_from).name))
    else:
        log_dir = get_new_log_dir(param.logdir, prefix=config_name,
                                  tag=tag, fn=param.get("fn"))
        # shutil.copytree("TSDiff/models", os.path.join(log_dir, "models"), dirs_exist_ok=True)

    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    logger = get_logger("train", log_dir)
    logger.info(f"Arguments: {param}")
    logger.info(f"Config: {config}")

    wandb_logger = None
    if param.get("project") and param.get("name"):
        wandb_logger = WandbLogger(project=param.project, name=param.name, save_dir=log_dir)
        wandb_logger.experiment.config.update(config)

    if isinstance(parameter.get("config"), str):
        shutil.copyfile(parameter["config"], os.path.join(log_dir, Path(parameter["config"]).name))

    data_module = DiffusionDataModule(config)
    model = DiffusionLightningModule(config)

    if param.get("pretrain"):
        ckpt = torch.load(param.pretrain, map_location="cpu", weights_only=False)
        # state_dict = {'.'.join('model' key.split('.')[1:]):ckpt['state_dict'][key] for key in ckpt['state_dict'].keys()}
        model.load_state_dict(ckpt['state_dict'])

        # logger.info(f"Loading pretrain checkpoint: {param.pretrain}")
        # ckpt = torch.load(param.pretrain, map_location=param.device, weights_only=False)
        # model.load_state_dict(ckpt["model"], strict=False)

    resume_ckpt_path = None
    if resume_from:
        ckpt_path, _ = get_checkpoint_path(os.path.join(resume_from, "checkpoints"),
                                           it=param.get("resume_iter"))
        if ckpt_path:
            resume_ckpt_path = ckpt_path
            logger.info(f"Resume from checkpoint: {resume_ckpt_path}")

    # ==================== 关键修改：保存 top-k=3，方便后面挑选最优 ====================
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{step}-{val_loss:.6f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,           # 改成保存 top-3，防止只剩 last.ckpt
        save_last=True,
        every_n_train_steps=None,
        every_n_epochs=None,
        
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.0,
        patience=200,
        verbose=True,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=-1,
        max_steps=config.train.max_iters,
        accelerator="gpu" if param.device.startswith("cuda") else "cpu",
        devices=1 if param.device.startswith("cuda") else None,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10,
        gradient_clip_val=config.train.get("max_grad_norm", None),
        precision=config.train.get("precision", "32"),
        default_root_dir=log_dir,
    )

    logger.info("Start training...")
    trainer.fit(model, datamodule=data_module, ckpt_path=resume_ckpt_path)

    # ============================== 训练结束后：找出最佳模型 ==============================
    logger.info("Training finished! Selecting best checkpoint...")

    # 方法1：直接用 callback 保存的 best_model_path（最推荐！）
    if checkpoint_callback.best_model_path:
        best_ckpt_path = checkpoint_callback.best_model_path
        logger.info(f"Best model found (by Lightning): {best_ckpt_path}")
        logger.info(f"Best val_loss = {checkpoint_callback.best_model_score:.6f}")
    else:
        # 方法2：手动在 checkpoints 目录里找 val_loss 最小的
        ckpt_files = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
        if not ckpt_files:
            logger.warning("No checkpoint found! Return last model.")
            best_ckpt_path = os.path.join(ckpt_dir, "last.ckpt")
        else:
            # 解析文件名中的 val_loss
            def extract_loss(path):
                fname = Path(path).name
                try:
                    return float(fname.split("val_loss=")[-1].split(".ckpt")[0])
                except:
                    return float('inf')
            best_ckpt_path = min(ckpt_files, key=extract_loss)
            logger.info(f"Best model found (manual scan): {best_ckpt_path}")

    logger.info(f"All logs  → {log_dir}")
    logger.info(f"Best model → {best_ckpt_path}")

    # 可选：把 best.ckpt 复制一份方便后续直接加载
    final_best_path = os.path.join(log_dir, "best.ckpt")
    if os.path.exists(best_ckpt_path) and best_ckpt_path != final_best_path:
        shutil.copyfile(best_ckpt_path, final_best_path)
        logger.info(f"Copied to {final_best_path} for easy loading")

    return log_dir, final_best_path   # 推荐直接用这个路径进行后续采样