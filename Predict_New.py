import os, glob, shutil, pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
import torch
from DFTStructureGenerator import mol_manipulation, logfile_process, xtb_process, FormatConverter, Kwon
from TSDiff.Preprocess import preprocess as tsdiff_preprocess
from TSDiff.train import train as tsdiff_train
from TSDiff.sampling import sample as tsdiff_sample
from confrankplus.train import train as confrankplus_train
from confrankplus.train import test as confrankplus_test

smiles_list = ["C[C@@H]1CC[C@@H](C(C)C)C=C1", 'C[C@@H]1CC[C@H](C(C)C)C=C1']
project_name = "p-menth-2-ene"
ene_dict = {"Index": {index: index for index, ene in enumerate(smiles_list)},"Ene": {index: ene for index, ene in enumerate(smiles_list)},}
df = pd.DataFrame(ene_dict, columns=['Index', 'Ene'])
if not os.path.exists(f'/results/{project_name}'):
    os.makedirs(f'/results/{project_name}')
df.to_csv(f'/results/{project_name}/Final.csv', index=False)

all_ts_guess = []
All_AAM = []
All_Name = []
for row_id, row in tqdm(new_df.iterrows()):
    Index, smiles, Site_A, Site_B, Z_Pos, Rot = int(row['Index']), row['Ene'], int(row['Site_A']), int(row['Site_B']), int(row['Z_Pos']), int(row['Rot'])
    AAMs = Kwon.smiles_to_AAM(smiles, return_mediate=True)
    for AAM in AAMs:
        if AAM[1][0] == Site_A and AAM[1][1] == Site_B:
            target_AAM = AAM[0]
            break
    reactant_mol = Chem.MolFromSmarts(target_AAM.split(">>")[0])
    reactant_atom_map_list = [atom.GetAtomMapNum() for atom in reactant_mol.GetAtoms()]
    reactant_atom_list = np.array([atom.GetSymbol() for atom in reactant_mol.GetAtoms()])[np.argsort(reactant_atom_map_list)]
    target_AAMs = [">>".join(target_AAM.split(">>")[:2]), ">>".join(target_AAM.split(">>")[1:])]
    for idx in range(2):
        mol_name = f'{['ts1', 'ts2'][idx]}_{Index:05}_{Site_A:03}_{Site_B:03}_{Z_Pos}_{Rot}'
        if idx == 0 and mol_name not in ts_mol_dict:
            continue
        ts_guess_mol = ts_mol_dict[mol_name] if idx == 0 else ts2_mol_dict[mol_name]
        ts_guess_position = np.array(ts_guess_mol.GetConformer().GetPositions())
        all_ts_guess.append({
            'name': mol_name,
            'atomlist': reactant_atom_list,
            'positions': ts_guess_position
        })
        All_AAM.append(target_AAMs[idx])
        All_Name.append(mol_name)

if not os.path.exists(f'/results/{project_name}/row'):
    os.makedirs(f'/results/{project_name}/row')
FormatConverter.write_xyz_file(f'/results/{project_name}/row/ts_guess.xyz', all_ts_guess)
new_df.to_csv(f'/results/{project_name}/Detail.csv', index=False)
new_df_ = pd.DataFrame({'Index': np.arange(len(All_AAM)), 'AAM': All_AAM, 'name': All_Name})
new_df_.to_csv(f'/results/{project_name}/row/Input.csv', index=False)

tsdiff_preprocess(os.path.join("/results", project_name, 'processed'), 
                  os.path.join("/results", project_name, 'row', 'Input.csv'), 
                  None, 
                  os.path.join("/results", project_name, 'row', 'ts_guess.xyz'), 
                  'Trained_model/feat_dict.pkl', 
                  None, None, np.arange(len(new_df_)))

sample_path = tsdiff_sample({
    "ckpt": 'Trained_model/tsdiff.ckpt', 
    "test_set": os.path.join("/results", project_name, 'processed', 'test_data.pkl'),
    "feat_dict": os.path.join("/results", project_name, 'processed', 'feat_dict.pkl'),
    "save_dir": os.path.join("/results", project_name, 'processed'),
})

train_data, val_data, test_data = [], [], []
root_dir = os.path.join("/results", project_name)
with open(sample_path, "rb") as f:
    results = pickle.load(f)
df = pd.read_csv(os.path.join("/results", project_name, 'row', 'Input.csv'), index_col='Index')
for row_id, row in df.iterrows():
    each_result = results[row_id]
    rxn_index = each_result.rxn_index
    smiles = each_result.smiles
    AAM = row['AAM']
    assert AAM == smiles and rxn_index == row_id

    log_name = row['name'].split("_")
    confid = '_'.join(log_name[2:])
    ensbid = '_'.join(log_name[:2])
    charge = np.sum([each.GetFormalCharge() for each in each_result.rdmol[0].GetAtoms()])
    symbol_list = each_result.atom_type
    data = {
    'confid':confid,
    'ensbid':ensbid,
    'total_charge':torch.tensor(charge, dtype=torch.float32),
    'z':torch.tensor(symbol_list, dtype=torch.long),
    'pos':torch.tensor(each_result.ts_guess, dtype=torch.float32),
    }
    test_data.append(data)
    # raise NameError

if not os.path.exists(os.path.join(root_dir, 'cfrk')):
    os.makedirs(os.path.join(root_dir, 'cfrk'))
torch.save(test_data, os.path.join(root_dir, 'cfrk', 'test.pt'))

cfrk_pred = confrankplus_test(
        project_name=os.path.join(root_dir, 'cfrk'),
        best_ckpt_path='/results/KwonFirst_2000/cfrk/best-epoch=40.ckpt',
        gpu_id=0,
    )
target_df_test = pd.read_csv(f'/results/{project_name}/Detail.csv')
pred_energies_dict = torch.load(cfrk_pred)
pred_TS = []
pred_TS2 = []
for row_id, row in target_df_test.iterrows():
    Index, Site_A, Site_B, Z_Pos, Rot = row['Index'], row['Site_A'], row['Site_B'], row['Z_Pos'], row['Rot']
    for idx in range(2):
        ts_guess_name = f'{['ts1', 'ts2'][idx]}_{Index:05}_{Site_A:03}_{Site_B:03}_{Z_Pos}_{Rot}'
        if idx == 0 and ts_guess_name not in pred_energies_dict.keys():
            ts_guess_name = f'{['ts1', 'ts2'][idx]}_{Index:05}_{Site_B:03}_{Site_A:03}_{Z_Pos}_{Rot}'
            if ts_guess_name not in pred_energies_dict.keys():
                if idx == 0:
                    pred_TS.append(np.nan)
                else:
                    pred_TS2.append(np.nan)
                continue
        if idx == 0:
            pred_TS.append(pred_energies_dict[ts_guess_name])
        else:
            pred_TS2.append(pred_energies_dict[ts_guess_name])
target_df_test['pred_TS'] = pred_TS
target_df_test['pred_TS2'] = pred_TS2
target_df_test.to_csv(f'/results/{project_name}/Detail.csv', index=False)
detail_df = pd.read_csv(f'/results/{project_name}/Detail.csv')

# column_TS1_TS2 = [['TS_G(kcal/mol)', 'TS2_G(kcal/mol)', 'B-A Energy'], ['pred_TS', 'pred_TS2', 'B-A Energy_pred']]
column_TS1_TS2 = [['pred_TS', 'pred_TS2', 'B-A Energy_pred']]
result_df = Kwon.calc_DDG(result_df=pd.read_csv(f'/results/{project_name}/Final.csv'), 
                          detail_df=detail_df, 
                          column_TS1_TS2= [['pred_TS', 'pred_TS2', 'B-A Energy_pred']], 
                          banned_ene = [])
result_df.to_csv(f'/results/{project_name}/Final.csv')
                          

