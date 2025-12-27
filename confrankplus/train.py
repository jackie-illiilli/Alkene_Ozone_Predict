# train_test_pipeline.py
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger
from torch_geometric.loader import DataLoader
from argparse import ArgumentParser

from .ConfRankPlus.training.lightning import LightningWrapper
from .ConfRankPlus.data.dataset import PairDataset, NewMolecularDataset
from .ConfRankPlus.inference.radius_graph import SmartRadiusGraph
from .ConfRankPlus.inference.loading import load_ConfRankPlus
from .ConfRankPlus.model import ConfRankPlus


# ==================== 公共工具函数 ====================
def get_device(gpu_id: int = 0):
    return torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')


def build_model(
    hidden_channels: int = 128,
    num_blocks: int = 2,
    int_emb_size: int = 64,
    out_emb_channels: int = 96,
    pair_basis_dim: int = 16,
    triplet_basis_dim: int = 16,
    cutoff: float = 6.5,
    cutoff_threebody: float = 4.0,
    additive_repulsion_energy: bool = True,
    dataset_encoding_dim: int = 2,
    num_dataset_embeddings: int = 5,
    pretrained_path: bool = False,          # 可选：加载老模型权重
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
    compute_forces: bool = False,
) -> ConfRankPlus:
    """
    创建（并可选地加载预训练权重）的 ConfRankPlus 模型
    """        
    model = ConfRankPlus(
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            int_emb_size=int_emb_size,
            out_emb_channels=out_emb_channels,
            pair_basis_dim=pair_basis_dim,
            triplet_basis_dim=triplet_basis_dim,
            cutoff=cutoff,
            cutoff_threebody=cutoff_threebody,
            additive_repulsion_energy=additive_repulsion_energy,
            dataset_encoding_dim=dataset_encoding_dim,
            num_dataset_embeddings=num_dataset_embeddings,
        )
    if pretrained_path:
        old_model, _ = load_ConfRankPlus(
            device=device, dtype=dtype, compute_forces=compute_forces
        )
        static_dict = old_model.state_dict()
        model.load_state_dict(static_dict)

    return model.to(device)


def build_lightning_module(
    model: ConfRankPlus,
    energy_loss_fn=None,
    forces_tradeoff: float = 0.0,
    decay_factor: float = 0.5,
    decay_patience: int = 3,
    weight_decay: float = 1e-8,
    xy_lim=None,
    pairwise: bool = True,
    ckpt_path: str = None,          # 可选：继续训练时加载 checkpoint
) -> LightningWrapper:
    if energy_loss_fn is None:
        energy_loss_fn = lambda x, y: torch.nn.functional.l1_loss(x, y)

    lightning_module = LightningWrapper(
        model=model,
        energy_key='energy',
        forces_key=None,
        forces_tradeoff=forces_tradeoff,
        atomic_numbers_key="z",
        decay_factor=decay_factor,
        decay_patience=decay_patience,
        energy_loss_fn=energy_loss_fn,
        weight_decay=weight_decay,
        xy_lim=xy_lim,
        pairwise=pairwise,
    )

    if ckpt_path:
        checkpoint = torch.load(ckpt_path, weights_only=False)
        lightning_module.load_state_dict(checkpoint["state_dict"], strict=False)

    return lightning_module


def get_dataloaders(
    project_name: str,
    data_root: str = "Data",
    batch_size: int = 100,
    num_workers: int = 1,
    cutoff: float = 6.5,
    train_shuffle: bool = True,
) -> dict:
    radius_graph_transform = SmartRadiusGraph(radius=cutoff)

    train_file = os.path.join(data_root, project_name, "train.pt")
    val_file   = os.path.join(data_root, project_name, "val.pt")
    test_file  = os.path.join(data_root, project_name, "test.pt")

    trainset = PairDataset(torch.load(train_file, weights_only=False), transform=radius_graph_transform)
    valset   = PairDataset(torch.load(val_file,   weights_only=False), transform=radius_graph_transform)
    testset  = PairDataset(torch.load(test_file,  weights_only=False), transform=radius_graph_transform)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=train_shuffle, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(valset,   batch_size=batch_size, shuffle=False,        num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(testset,  batch_size=batch_size, shuffle=False,        num_workers=num_workers, pin_memory=True)

    return {
        "train": train_loader,
        "val":   val_loader,
        "test":  test_loader,
        "raw_test_file": test_file,
    }


# ==================== 训练函数 ====================
def train(
    project_name: str,
    # ---- 模型超参数 ----
    hidden_channels: int = 128,
    num_blocks: int = 2,
    cutoff: float = 6.5,
    # ---- 训练超参数 ----
    batch_size: int = 100,
    max_epochs: int = 100,
    patience: int = 20,
    init_ckpt: str = None,                # 初始权重（比如你原来那個 epoch=0 的 ckpt）
    gpu_id: int = 0,
    data_root: str = '',
    log_every_n_steps: int = 200,
    pretrain_old_model: bool = False,
) -> str:
    """
    返回值：最优模型的 checkpoint 完整路径（可直接喂给 test()）
    """
    device = get_device(gpu_id)

    # 1. 模型
    model = build_model(
        hidden_channels=hidden_channels,
        num_blocks=num_blocks,
        cutoff=cutoff,
        pretrained_path=pretrain_old_model,           
        device=device,
    )

    # 2. Lightning wrapper（可加载初始权重）
    lightning_module = build_lightning_module(
        model=model,
        ckpt_path=init_ckpt,            # 你原来的 'epoch=0-step=396.ckpt'
        pairwise=True,
    )

    # 3. 数据
    loaders = get_dataloaders(
        project_name=project_name,
        data_root=data_root,
        batch_size=batch_size,
        cutoff=cutoff,
    )

    # 4. Callbacks
    ckpt_dir = os.path.join(data_root, project_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    monitor_metric = "ptl/val_loss_pairwise"
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,
        save_top_k=1,
        mode="min",
        dirpath=ckpt_dir,
        filename="best-{epoch:02d}",
    )
    early_stop_callback = EarlyStopping(
        monitor=monitor_metric,
        min_delta=0.0,
        patience=patience,
        verbose=True,
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # 5. Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[gpu_id] if torch.cuda.is_available() else None,
        precision=32,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        log_every_n_steps=log_every_n_steps,
        enable_progress_bar=True,
        deterministic=False,
        benchmark=True,

    )

    # 6. Fit
    trainer.fit(
        lightning_module,
        train_dataloaders=loaders["train"],
        val_dataloaders=loaders["val"],
    )

    best_ckpt_path = checkpoint_callback.best_model_path
    print(f"\nTraining finished! Best model: {best_ckpt_path}\n")
    torch.cuda.empty_cache()
    return best_ckpt_path


# ==================== 测试/推理函数 ====================
def test(
    project_name: str,
    test_set,
    best_ckpt_path: str,          # 直接用 train() 返回的路径
    pretrained_path = True,
    batch_size: int = 50,
    cutoff: float = 6.5,
    data_root: str = "",
    gpu_id: int = 0,
    save_file: bool = True,
):
    """
    在 test set 上做推理，保存 Pred.pt
    返回保存的文件路径
    """
    if torch.torch.cuda.is_available():
        device = get_device(gpu_id)
    else:
        device = torch.device("cpu")

    # 1. 加载最优模型（通过 LightningWrapper 自动恢复）
    # 先重建一个空模型结构，再让 LightningWrapper 加载 checkpoint
    dummy_model = build_model(cutoff=cutoff, device=device, pretrained_path=pretrained_path)   # 参数随意，只要结构一致
    lightning_module = build_lightning_module(model=dummy_model)
    if best_ckpt_path:
        checkpoint = torch.load(best_ckpt_path, weights_only=False)
        lightning_module.load_state_dict(checkpoint["state_dict"])
    lightning_module.eval()
    lightning_module.to(device)

    # 2. 测试数据（使用 NewMolecularDataset，跟你原来一样）
    if isinstance(test_set, str):
        test_file = os.path.join(test_set)
        test_file = torch.load(test_file, weights_only=False)
    else:
        test_file = test_set
    radius_graph_transform = SmartRadiusGraph(radius=cutoff)
    testset = NewMolecularDataset('./temp', test_file,
                                  transform=radius_graph_transform)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # 3. 推理
    final_energies = {}
    with torch.no_grad(), torch.jit.optimized_execution(False):
        for batch in test_loader:
            batch = batch.to(device)
            energy = lightning_module.model.forward(batch)['energy'].detach().cpu().numpy().tolist()
            for ensbid, confid, e in zip(batch.ensbid, batch.confid, energy):
                final_energies[f"{ensbid}_{confid}"] = e

    # 4. 保存
    if save_file:
        save_path = os.path.join(data_root, project_name, "Pred.pt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(final_energies, save_path)
        print(f"Prediction saved to {save_path}")
        return save_path
    else:
        return final_energies


# ==================== 示例调用（可直接当脚本跑） ====================
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--project", type=str, default="Kwon_2000/FFTsGuess")
    parser.add_argument("--init_ckpt", type=str, default="Data/Kwon_2000/FFTsGuess/FFTsGuess/epoch=0-step=396.ckpt")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    # 1. 训练 → 得到最优 checkpoint
    best_ckpt = train(
        project_name=args.project,
        init_ckpt=args.init_ckpt,      # 你原来加载的那个初始权重
        gpu_id=args.gpu,
        max_epochs=100,
        patience=20,
    )

    # 2. 直接用最优模型做测试
    test(
        project_name=args.project,
        best_ckpt_path=best_ckpt,
        gpu_id=args.gpu,
    )