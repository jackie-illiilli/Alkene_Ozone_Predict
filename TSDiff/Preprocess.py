import pandas as pd
import pickle
import os
import numpy as np
import torch
import random
from tqdm import tqdm
import argparse
from TSDiff.utils.datasets import generate_ts_data2, read_xyz_block, generate_ts_data
from TSDiff.utils.parse_xyz import parse_xyz_corpus
from typing import List


def random_split(data_list: List, train: float = 0.8, valid: float = 0.1, seed: int = 1234):
    """
    Randomly split a dataset into non-overlapping train/valid/test set.
    args :
        data_list (list): a list of data
        train (float): ratio of train data
        valid (float): ratio of valid data
        seed (int): random seed
    return :
        train_data (list): a list of train data
        valid_data (list): a list of valid data
        test_data (list): a list of test data
    """
    assert train + valid < 1
    random.seed(seed)
    random.shuffle(data_list)
    N = len(data_list)
    n_train = int(N * train)
    n_valid = int(N * valid)
    train_data = data_list[:n_train]
    valid_data = data_list[n_train: n_train + n_valid]
    test_data = data_list[n_train + n_valid:]

    return train_data, valid_data, test_data

def check_dir(dir_name):
    """
    Check the directory exists or not
    If not, make the directory
    Check wheather train_data.pkl, valid_data.pkl, test_data.pkl are exist or not.
    If exist, raise error.

    args :
       dir_name (str): directory name
    return :
         None
    """
    os.makedirs(dir_name, exist_ok=True)
    # if os.path.isfile(os.path.join(dir_name, "train_data.pkl")):
    #     raise ValueError("train_data.pkl is already exist.")
    # if os.path.isfile(os.path.join(dir_name, "valid_data.pkl")):
    #     raise ValueError("valid_data.pkl is already exist.")
    # if os.path.isfile(os.path.join(dir_name, "test_data.pkl")):
    #     raise ValueError("test_data.pkl is already exist.")


def preprocess(save_dir, rxn_smarts_file, ts_data, ts_guess, feat_dict, train_indices, val_indices, test_indices):
    df = pd.read_csv(rxn_smarts_file)
    rxn_smarts = df.AAM
    if ts_data == None:
        xyz_blocks = [None] * len(df)
    else:
        xyz_blocks = parse_xyz_corpus(ts_data)
    if ts_guess == None:
        all_ts_guess = [None] * len(df)
    else:
        all_ts_guess = parse_xyz_corpus(ts_guess)
    
    feat_dict = pickle.load(open(feat_dict, "rb"))
    data_list = []
    for idx, (a_smarts, xyz_block, ts_guess) in tqdm(enumerate(zip(rxn_smarts, xyz_blocks, all_ts_guess))):
        r, p = a_smarts.split(">>")
        data, feat_dict = generate_ts_data2(r, p, xyz_block, feat_dict=feat_dict)
        data_list.append(data)
        data.rxn_index = idx
        data.augmented = False 
        if ts_guess != None:
            data.ts_guess = torch.tensor(read_xyz_block(ts_guess)[-1], dtype=torch.float32)

    # convert features to one-hot encoding
    num_cls = [len(v) for k, v in feat_dict.items()]
    for data in data_list:
        feat_onehot = []
        feats = data.r_feat.T
        for feat, n_cls in zip(feats, num_cls):
            feat_onehot.append(torch.nn.functional.one_hot(feat, num_classes=n_cls))
        data.r_feat = torch.cat(feat_onehot, dim=-1)

        feat_onehot = []
        feats = data.p_feat.T
        for feat, n_cls in zip(feats, num_cls):
            feat_onehot.append(torch.nn.functional.one_hot(feat, num_classes=n_cls))
        data.p_feat = torch.cat(feat_onehot, dim=-1)

    train_data, test_data, valid_data = [], [], []
    
    
    test_data = [data_list[each] for each in test_indices]

    check_dir(save_dir)
    # save the data, feat_dict, index_dict at the save_dir with pickle format. (.pkl)
    if ts_data != None:
        train_data = [data_list[each] for each in train_indices]
        with open(os.path.join(save_dir, "train_data.pkl"), "wb") as f:
            pickle.dump(train_data, f)
        valid_data = [data_list[each] for each in val_indices]
        with open(os.path.join(save_dir, "valid_data.pkl"), "wb") as f:
            pickle.dump(valid_data, f)
    pickle.dump(data_list, open(os.path.join(save_dir, "all_data.pkl"), 'wb'))

    with open(os.path.join(save_dir, "test_data.pkl"), "wb") as f:
        pickle.dump(test_data, f)
    with open(os.path.join(save_dir, "feat_dict.pkl"), "wb") as f:
        pickle.dump(feat_dict, f)

    print("Preprocessing done. Train: {}, Valid: {}, Test: {}".format(
        len(train_data), len(valid_data), len(test_data)
    ))  

