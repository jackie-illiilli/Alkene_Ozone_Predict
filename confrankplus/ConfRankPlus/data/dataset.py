from __future__ import annotations

from typing import Dict, List, Optional

import h5py
import json
import random
import numpy as np
import torch
from copy import copy
from itertools import combinations
from pathlib import Path
from typing import Callable, Optional
from torch import Tensor
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data.data import BaseData
from torch_geometric.transforms import BaseTransform
from torch.utils.data import Dataset


class HDF5Dataset(InMemoryDataset):

    def __init__(
            self,
            data: BaseData,
            slices: Dict[str, Tensor],
            transform: Optional[BaseTransform] = None,
    ):
        super().__init__("./", transform)
        self.data, self.slices = data, slices

    @staticmethod
    def from_hdf5(
            filepath: str,
            precision: int = 32,
            transform: Optional[BaseTransform] = None,
            exclude_keys: Optional[List[str]] = None,
    ) -> HDF5Dataset:
        _exclude_keys = exclude_keys if exclude_keys is not None else []
        data = {}
        slices = {}
        with h5py.File(filepath, "r") as f:
            if "additional" in f.keys() and "utf-8-encoded" in f["additional"].keys():
                decode_utf8 = [
                    key.decode("utf-8") for key in f["additional"]["utf-8-encoded"][:]
                ]
            else:
                decode_utf8 = []
            for key in f["data"].keys():
                np_arrays = {"data": f["data"][key][:], "slices": f["slices"][key][:]}
                for prop, val in np_arrays.items():
                    if val.dtype == np.uint64:
                        np_arrays[prop] = val.astype(np.int64)
                    if val.dtype == np.float64 and precision == 32:
                        np_arrays[prop] = val.astype(np.float32)
                    elif val.dtype == np.float32 and precision == 64:
                        np_arrays[prop] = val.astype(np.float64)
                if key in decode_utf8:
                    data[key] = np_arrays["data"].tolist()
                    data[key] = [bs.decode("utf-8") for bs in data[key]]
                    slices[key] = torch.from_numpy(np_arrays["slices"])
                else:
                    data[key] = torch.from_numpy(np_arrays["data"])
                    slices[key] = torch.from_numpy(np_arrays["slices"])
        dataset = HDF5Dataset(
            data=Data.from_dict(data),
            slices=slices,
            transform=transform,
        )
        return dataset

    @staticmethod
    def from_ase(
            filepaths: List[str],
            file_format: Optional[str] = None,
            precision: int = 32,
            index: Optional[slice] = ':',
            transform: Optional[BaseTransform] = None,
    ) -> HDF5Dataset:

        from ase.io import read
        assert precision in [32, 64], 'Precision must be either 32 or 64!'

        data = {}
        slices = {}
        dtype = torch.double if precision == 64 else torch.float32

        for filepath in filepaths:
            atoms_list = read(filepath, index=index, format=file_format)
            for i, atoms in enumerate(atoms_list):
                atom_data = {
                    "pos": torch.tensor(atoms.positions, dtype=dtype),
                    "z": torch.tensor(atoms.numbers, dtype=torch.long).view(-1),
                }

                for key, value in atom_data.items():
                    if key not in data:
                        data[key] = []
                        slices[key] = [0]
                    data[key].append(value)
                    increment_slice = value.shape[0]
                    slice_index = slices[key][-1] + increment_slice
                    slices[key].append(slice_index)

        for key in data:
            data[key] = torch.cat(data[key], dim=0)
            slices[key] = torch.tensor(slices[key], dtype=torch.long)

        dataset = HDF5Dataset(
            data=Data.from_dict(data),
            slices=slices,
            transform=transform,
        )
        return dataset


class NewMolecularDataset(InMemoryDataset):
    def __init__(self, root, data_list, transform=None, pre_transform=None):
        """
        :param root: Directory to store processed data (e.g., './my_dataset').
        :param data_list: Your large list of dicts.
        :param transform: Optional on-the-fly transform (not used here for edge_index).
        :param pre_transform: Your radius_graph_transform to pre-compute edge_index.
        """
        self.raw_data_list = data_list  # Store the original list
        super().__init__("./", transform)
        self.transfer_data()

    @property
    def raw_file_names(self):
        return []  # No raw files needed since data is in memory

    @property
    def processed_file_names(self):
        return 'data.pt'  # The file where processed Data objects are saved

    def download(self):
        pass  # No download required

    def transfer_data(self):
        processed_list = []
        for idx, raw_dict in enumerate(self.raw_data_list):
            # Convert dict to Data object
            data = Data()
            # Assume 'z' is atomic numbers: convert to tensor (num_atoms, 1) for node features
            data.z = torch.tensor(raw_dict['z'], dtype=torch.long)  # Or keep as data.z if not using as x
            data.pos = torch.tensor(raw_dict['pos'], dtype=torch.float)  # (num_atoms, 3)
            
            # Custom attributes or labels
            data.confid = raw_dict['confid']  # Can be string/int
            data.ensbid = raw_dict['ensbid']
            if 'energy' in raw_dict.keys():
                data.energy = torch.tensor(raw_dict['energy'], dtype=torch.float)  # e.g., as y for regression
            data.total_charge = torch.tensor(raw_dict['total_charge'], dtype=torch.float)
            data.batch_index = torch.tensor(idx)  # Optional: track original index
            data.dataset_idx=torch.full_like(data["z"], dtype=torch.long, fill_value=0)
            # Apply pre_transform to compute edge_index (e.g., radius graph)
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            
            processed_list.append(data)
        
        # Collate and save the list of Data objects
        self.data, self.slices = self.collate(processed_list)


class PairDataset(Dataset):
    """Collect pairs of samples from dataset. Per default only take pairs from same ensemble."""

    def __init__(
        self,
        dataset,
        sample_pairs_randomly: bool = False,
        lowest_k: Optional[int] = None,
        additional_k: Optional[int] = None,
        transform: Callable | BaseTransform | None = None,
        # dtype=torch.float64,
    ):
        """
        :param path_to_hdf5: path of hdf5 file storing the data
        :param sample_pairs_randomly: If False, all pairs (i,j) with i<j are computed for each ensemble.
        If True, random sampling is used and lowest_k and additional_k have to be specified.
        :param lowest_k: Number of conformers with the lowest energy in an ensemble,
        only has an effect if sample_pairs_randomly is True; default: None
        :param additional_k: Number of conformers that are samples randomly,
        only has an effect if sample_pairs_randomly is True; default: None
        :param transform: Transformation for on-the-fly post-processing of data points; default: None
        :param dtype: precision that used; default: torch.float64
        """
        # self.dataset = dataset
        # self.dataset = [Data.from_dict(transform(each)) for each in dataset]
        self.dataset = NewMolecularDataset(root='./temp', data_list=dataset, transform=transform)
        self.sample_pairs_randomly = sample_pairs_randomly
        self.lowest_k = lowest_k
        self.additional_k = additional_k
        if self.sample_pairs_randomly:
            assert self.lowest_k is not None
            assert self.additional_k is not None

        self.ensembles = self.get_ensembles(self.dataset)
        self.pairs = self._setup_pairs()

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            index1, index2 = self.pairs[idx]
            return self.dataset[index1], self.dataset[index2]
        elif isinstance(idx, slice):
            new_pairs = self.pairs[idx]
            new_dataset = copy(self)
            new_dataset.pairs = new_pairs
            return new_dataset
        elif isinstance(idx, (list, np.ndarray, torch.Tensor)):
            new_pairs = [self.pairs[i] for i in idx]
            new_dataset = copy(self)
            new_dataset.pairs = new_pairs
            return new_dataset
        else:
            raise IndexError("Invalid index type")

    def __str__(self):
        return f"PairDataset({len(self)})"

    def split_by_ensemble(self, train_size, val_size, test_size, seed=42):
        assert train_size + val_size + test_size == 1, "train+val+test sizes must be 1"
        random.seed(seed)
        n_train = int(train_size * len(self.ensembles))
        n_val = int(val_size * len(self.ensembles))
        ensemble_uids = list(self.ensembles.keys())
        random.shuffle(ensemble_uids)
        ensemble_uids_train = ensemble_uids[:n_train]
        ensemble_uids_val = ensemble_uids[n_train : n_train + n_val]
        train_idx = []
        val_idx = []
        test_idx = []
        for idx, sample in enumerate(self):
            ensbid = sample[0].ensbid
            assert (
                sample[1].ensbid == ensbid
            ), f"sample[0] has ensemble id {ensbid} and sample[1] has ensemble id {sample[1].ensbid}"
            if ensbid in ensemble_uids_train:
                train_idx.append(idx)
            elif ensbid in ensemble_uids_val:
                val_idx.append(idx)
            else:
                test_idx.append(idx)
        return self[train_idx], self[val_idx], self[test_idx]

    def get_ensembles(self, dataset) -> dict[str, int] | dict[str, list[int]]:
        """Obtain mapping of samples and ensembles."""
        print("Calculating ensembles ...")
        ensembles = {}
        for idx, d in enumerate(dataset):
            euid = d.ensbid
            ensembles.setdefault(euid, [])
            ensembles[euid].append(idx)
        return ensembles

    def save_ensembles(self, path: str | Path):
        """Save ensemble info to file for further usage."""
        with open(path, "w") as json_file:
            json.dump(self.ensembles, json_file)

    def load_ensembles(self, path: str | Path) -> dict[str, int]:
        """Load ensemble info from file."""
        with open(path, "r") as json_file:
            ensembles = json.load(json_file)
        return ensembles

    def _setup_pairs(self) -> list[tuple[int]]:
        """Initialize pairs to draw samples from."""
        if self.sample_pairs_randomly:
            pairs = self.pair_generation_ensemble_random_sampled()
        else:
            pairs = self.pair_generation_ensemble()
        return pairs

    def pair_generation_ensemble(self):
        """Add pairs all pairs up to permutation"""
        pairs = []
        # get tuples (i,j) with i, j stemming from same ensemble and i < j
        for ensbid, idcs in self.ensembles.items():
            combs = combinations(idcs, 2)
            pairs.extend(combs)  # no permutations
        return pairs

    def pair_generation_ensemble_random_sampled(self):
        """Generate randomly sampled pairs up to permutation"""
        pairs = []
        # get tuples (i,j) with i, j stemming from same ensemble
        for ensbid, idcs in self.ensembles.items():
            # get energies
            energies = np.array(
                [self.dataset[i]["total_energy_ref"].item() for i in idcs]
            )
            # sort by energy
            sort_idx = np.argsort(energies).reshape(-1)
            idxs_i = np.array(idcs)[sort_idx]
            idxs_i = np.concatenate(
                [
                    idxs_i[: self.lowest_k],
                    np.random.choice(
                        idxs_i[self.lowest_k :],
                        replace=False,
                        size=(min(self.additional_k, len(idxs_i[self.lowest_k :])),),
                    ),
                ]
            )
            nn_combs = []
            for p1 in idxs_i:
                for p2 in idxs_i:
                    if p2 < p1:
                        nn_combs.append((p1, p2))
            pairs.extend(nn_combs)
        return pairs