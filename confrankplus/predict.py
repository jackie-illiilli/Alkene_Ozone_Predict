import sys
import os

sys.path.append("../")

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger
from torch_geometric.loader import DataLoader
from argparse import ArgumentParser

from ConfRankPlus.training.lightning import LightningWrapper
from ConfRankPlus.data.dataset import PairDataset, NewMolecularDataset
from ConfRankPlus.inference.radius_graph import SmartRadiusGraph
from ConfRankPlus.inference.loading import load_ConfRankPlus
from ConfRankPlus.model import ConfRankPlus

device = torch.device('cpu')
dtype = torch.float32
compute_forces = False

old_model, fidelity_mapping = load_ConfRankPlus(device=device,
                                                dtype=dtype,
                                                compute_forces=compute_forces)
# new
model = ConfRankPlus(
    hidden_channels=128, 
    num_blocks=2, 
    int_emb_size=64,
    out_emb_channels=96,
    pair_basis_dim=16, 
    triplet_basis_dim=16, 
    cutoff=6.5, 
    cutoff_threebody=4.0, 
    additive_repulsion_energy=True,
    dataset_encoding_dim = 2,
    num_dataset_embeddings = 5,
)
# static_dict = old_model.state_dict()
# model.load_state_dict(static_dict)

energy_loss_fn = lambda x, y: torch.nn.functional.l1_loss(x, y)
lightning_module = LightningWrapper(
    model=model,
    energy_key='energy',
    forces_key=None,
    forces_tradeoff=0.0,
    atomic_numbers_key="z",
    decay_factor=0.5,
    decay_patience=3,
    energy_loss_fn=energy_loss_fn,
    weight_decay=1E-8,
    xy_lim=None,
    pairwise=True,
)

checkpoint = torch.load('Data/KwonRingConf_ts1&2/epoch=61-step=8556.ckpt', weights_only=False)
lightning_module.load_state_dict(checkpoint["state_dict"], strict=False)

batch_size = 100
num_workers = 1
radius_graph_transform = SmartRadiusGraph(radius=model.cutoff)

test_file = 'Data/Kwon_Chiral2/Kwon_Chiral2.pt'
temp_testset = NewMolecularDataset('./temp', torch.load(test_file, weights_only=False), 
                                    transform=radius_graph_transform)
temp_test_loader = DataLoader(
    temp_testset,
    batch_size=batch_size,
    shuffle=False,
)

final_energies = {}
with torch.jit.optimized_execution(False):
    for each_batch in temp_test_loader:
        predict_result = lightning_module.model.forward(each_batch)['energy'].detach().numpy().tolist()
        for ensbid, confid, result in zip(each_batch.ensbid, each_batch.confid, predict_result):
            final_energies[f"{ensbid}_{confid}"] = result
        # break
torch.save(final_energies, 'Data/Kwon_Chiral2/Kwon_Chiral2_pred.pt')