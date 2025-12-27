import glob, os, shutil, itertools, copy
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
import numpy as np
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt
from . import logfile_process, FormatConverter, xtb_process, mol_manipulation, Tool

def smiles_DFT_calc(root_dir='first_xtb', 
                    mol_dir='mol', 
                    dft_dir='mol_dft', 
                    method="opt freq b3lyp/6-31g* em=gd3bj",
                    conf_limit=3,
                    rmsd_limit=1.5,
                    SpinMultiplicity = None,
                    restrict = [],
                    pre_DFT_dir = None,
                    useConfRank = False,
                    ):

    """通过Xtb结果，优化得到Gaussian优化输入文件

    Args:
        root_dir (str, optional): Xtb的根目录. Defaults to 'first_xtb'.
        mol_dir (str, optional): Mol分子的目录. Defaults to 'mol'.
        dft_dir (str, optional): 要存储Gaussian输入文件的目录. Defaults to 'mol_dft'.
        method (str, optional): Gaussian方法. Defaults to "opt freq b3lyp/6-31g* em=gd3bj".
        conf_limit (int, optional): Xtb读取结构的构象数量限制. Defaults to 3.
        rmsd_limit (float, optional): Xtb读取结构的RMSD限制. Defaults to 1.5.
        SpinMultiplicity (int, optional): 设定的自旋多重度. Defaults to None.
    """                    
    # all_files = glob.glob(root_dir + "/*/*/*")
    # for xtb_file in all_files:
    #     if ("crest.out" in xtb_file) or ("best" in xtb_file) or ("crest_conf" in xtb_file) or ("crest_conf" in xtb_file):
    #         pass
    #     else:
    #         if os.path.isdir(xtb_file):
    #             shutil.rmtree(xtb_file)
    #         else:
    #             os.remove((xtb_file))
    xtb_dirs = glob.glob(root_dir + "/*/*")
    for i, xtb_dir in enumerate(xtb_dirs):
        mol_name = os.path.split(xtb_dir)[-1][:-2]
        mol_file = mol_dir + f"/{mol_name}.mol" 
        mol = Chem.MolFromMolFile(mol_file, removeHs=False, sanitize=False)
        title = "Singlemol"
        try:
            if len(restrict):
                title = " ".join([str(each) for each in restrict[mol_name][0] + restrict[mol_name][1]])
                xtb_process.after_xtb(mol,xtb_dir=xtb_dir, save_dir=dft_dir, xtb_title=title, method=method, conf_limit=conf_limit, rmsd_limit=rmsd_limit, SpinMultiplicity=SpinMultiplicity, freeze=restrict[mol_name], pre_DFT_dir=pre_DFT_dir, useConfRank=useConfRank)
            else:
                xtb_process.after_xtb(mol,xtb_dir=xtb_dir, save_dir=dft_dir, xtb_title=title, method=method, conf_limit=conf_limit, rmsd_limit=rmsd_limit, SpinMultiplicity=SpinMultiplicity, pre_DFT_dir=pre_DFT_dir, useConfRank=useConfRank)
        except:
            continue

def SPE_DFT_calc(target_dir, opt_name="Reactants", eng_name='Reactants_eng', save_chk=None, method="b3lyp/6-311+g(d,p) em=gd3bj"):
    opt_file_dir = os.path.join(target_dir, opt_name)
    eng_dir = os.path.join(target_dir, eng_name)
    log_files = glob.glob(opt_file_dir + "/" + "*.log")
    for log_file in log_files:
        try:
            new_log_name = eng_dir + "/" + os.path.split(log_file)[-1].split('.')[0] + ".gjf" 
            opt_log = logfile_process.Logfile(log_file)
            assert len(opt_log.running_positions) != 0
            title, charge, symbol_list, position,= opt_log.title, opt_log.charge, opt_log.symbol_list, opt_log.running_positions[-1]
            title = " ".join(str(each) for each in title)
            if save_chk:
                savechk = os.path.split(new_log_name.strip(".gjf"))[-1]
            else:
                savechk = None
            FormatConverter.block_to_gjf(symbol_list, position, new_log_name, charge, opt_log.multiplicity, title,
                        method=method, savechk=savechk)
        except:
            continue

def collection_dft_single(result_path, mol_dir, dft_dir, spe_dir=None, save_path=None):
    error_reason, E_energy, G_energy, conf_idxs = [],[],[], []
    result_file = pd.read_csv(result_path)
    for line_id, line in tqdm(result_file.iterrows()):
        dft_dir_ = dft_dir
        spe_dir_ = spe_dir
        smiles = line['Ene']
        Index = line['Index']
        Site_A = line['Site_A']
        Site_B = line['Site_B']
        rot = line['Rot']
        mol_file = glob.glob(os.path.join(mol_dir, f'ts2_{Index:05}_{Site_A:03}_{Site_B:03}_0_{rot:01}.mol'))
        if len(mol_file) == 0:
            print(smiles, Index, "Is Error")
            error_reason.append("DFT Error")
            E_energy.append(np.nan)
            G_energy.append(np.nan)
            conf_idxs.append(np.nan)
            continue
        mol_file = mol_file[0]
        opt_files = glob.glob(os.path.join(dft_dir_, f'ts2_{Index:05}_{Site_A:03}_{Site_B:03}_0_{rot:01}_*.log'))
        temp_idx, temp_E, temp_G = [], [], []
        for opt_file in opt_files:
            opt = mol_manipulation.logfile_process.Logfile(opt_file)
            if spe_dir_ is not None:
                spe_files = glob.glob(os.path.join(spe_dir_, os.path.split(opt_file)[-1]))
                if len(spe_files) == 0:
                    print(smiles, Index, "OPT Error")
                    continue
                spe_file = spe_files[0]
                spe = mol_manipulation.logfile_process.Logfile(spe_file)
                electric_energy = spe.all_engs[0]
            else:
                electric_energy = opt.all_engs[0]
            conf_id = int(opt_file.split('.')[0].split("_")[-1])
            G_cor = opt.all_engs[-1]
            temp_idx.append(conf_id)
            temp_E.append(electric_energy)
            temp_G.append(G_cor + electric_energy)
        if len(temp_G) == 0:
            error_reason.append("DFT Error")
            E_energy.append(np.nan)
            G_energy.append(np.nan)
            conf_idxs.append(np.nan)
        else:
            min_index = np.argmin(temp_G)
            E_energy.append(temp_E[min_index])
            G_energy.append(temp_G[min_index])
            conf_idxs.append(temp_idx[min_index])
            error_reason.append(np.nan)
    result_file[f"E_energy"] = E_energy
    result_file[f"G_energy"] = G_energy
    result_file[f"conf_idxs"] = conf_idxs
    result_file[f"error_reason"] = error_reason
    if save_path == None:
        save_path = result_path
    result_file.to_csv(save_path, index=False)

def collection_dft_ts(ts_csv_path, root_dir, calc_SPE=False):
    all_results = []
    if ts_csv_path.endswith(".csv"):
        result_file = pd.read_csv(ts_csv_path)
    else:
        result_file = pd.read_excel(ts_csv_path)
    mol_dir = os.path.join(root_dir, "Mols")
    for line_id, line in tqdm(result_file.iterrows()):
        smiles = line['Ene']
        Index = line['Index']
        Site_A = line['Site_A']
        Site_B = line['Site_B']
        Z_Pos = line['Z_Pos']
        rot = line['Rot']
        temp_mol_name_gs = f'gs2_{Index:05}_{Site_A:03}_{Site_B:03}_{Z_Pos:01}_{rot:01}'
        temp_mol_name_ts = f'ts1_{Index:05}_{Site_A:03}_{Site_B:03}_{Z_Pos:01}_{rot:01}'
        if not os.path.isfile(os.path.join(mol_dir, temp_mol_name_gs + '.mol')):
            temp_mol_name_gs = f'gs2_{Index:05}_{Site_B:03}_{Site_A:03}_{Z_Pos:01}_{rot:01}'
            temp_mol_name_ts = f'ts1_{Index:05}_{Site_B:03}_{Site_A:03}_{Z_Pos:01}_{rot:01}'

        mol_names = [
            f"gs1_{Index:05}",
            temp_mol_name_ts,
            temp_mol_name_gs, 
            f"ts2_{Index:05}_{Site_A:03}_{Site_B:03}_{Z_Pos:01}_{rot:01}"
        ]
        if not os.path.isfile(os.path.join(mol_dir, mol_names[0] + '.mol')):
            print(smiles, Index, "Mol Error")
            all_results.append([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
            continue
        file_paths = ['OPT', "TS", 'OPT2', "TS2"]
        temp_result = []
        for id_, [mol_name, file_path] in enumerate(zip(mol_names, file_paths)):
            if not os.path.isfile(os.path.join(mol_dir, mol_name + '.mol')):
                print(os.path.join(mol_dir, mol_name + '.mol'))
            assert os.path.isfile(os.path.join(mol_dir, mol_name + '.mol'))
            all_logs = glob.glob(os.path.join(root_dir, file_path, mol_name+ "*.log"))
            temp_idx, temp_G = [], []
            for log in all_logs:
                opt = mol_manipulation.logfile_process.Logfile(log)
                if calc_SPE:
                    spe_files = glob.glob(os.path.join(root_dir, file_path + "_eng", os.path.split(log)[-1]))
                    if len(spe_files) == 0:
                        print(smiles, Index, "OPT Error")
                        continue
                    spe_file = spe_files[0]
                    spe = mol_manipulation.logfile_process.Logfile(spe_file)
                    electric_energy = spe.all_engs[0]
                else:
                    electric_energy = opt.all_engs[0]
                G_cor = opt.all_engs[-1]
                conf_id = int(log.split('.')[0].split("_")[-1])
                temp_idx.append(conf_id)
                temp_G.append(G_cor + electric_energy)
            if len(temp_G) == 0:
                temp_result += [np.nan, np.nan]
            else:
                min_index = np.argmin(temp_G)
                temp_result += [temp_G[min_index], temp_idx[min_index]]
        all_results.append(temp_result)
    all_results = np.array(all_results)
    result_file_ = pd.DataFrame(all_results, columns=["GS_G", "GS_conf", "TS_G", "TS_conf", "GS2_G", "GS2_conf", "TS2_G", "TS2_conf"])
    result_file = pd.concat([result_file, result_file_], axis=1)
    result_file = result_file.sort_values(by=['Index', "Site_A", "Z_Pos", 'Rot']).reset_index()
    result_file['GS_dG(kcal/mol)'] = 0
    result_file['TS_dG(kcal/mol)'] = (result_file['TS_G'] - result_file['GS_G'] + 225.262318) * 627.5
    result_file['GS2_dG(kcal/mol)'] = (result_file['GS2_G'] - result_file['GS_G'] + 225.262318) * 627.5
    result_file['TS2_dG(kcal/mol)'] = (result_file['TS2_G'] - result_file.groupby(['Index', "Site_A", "Z_Pos"])['GS2_G'].transform('min')) * 627.5

    return result_file
            

def reaction_calc_ts(target_dir, method, om_name="OM", ts_name="TS"):
    om_file_dir = target_dir + "/" + om_name
    ts_dir = target_dir + "/" + ts_name
    if not os.path.isdir(ts_dir):
        os.mkdir(ts_dir)
    om_log_files = glob.glob(om_file_dir + "/*.log")
    for om_log_file in om_log_files:
        try:
            om_log = logfile_process.Logfile(om_log_file)
            # assert om_log.bond_attach
            mol_manipulation.om_to_ts(om_log, 1, new_dir=ts_dir, method=method)     
        except:
            continue     

def reaction_calc_irc(target_dir, ts_name='ts', irc_name='irc'):
    # 仅针对频率较低或者振动方向错误的
    ts_file_dir = target_dir + "/" + ts_name
    irc_dir = target_dir + "/" + irc_name
    if not os.path.isdir(irc_dir):
        os.mkdir(irc_dir)
    ts_log_files = glob.glob(ts_file_dir + "/*.log")
    for ts_log_file in ts_log_files:
        print(ts_log_file, end='\r')
        ts_log = logfile_process.Logfile(ts_log_file)
        assert ts_log.bond_attach
        title = [each - 1 for each in ts_log.title]
        position = ts_log.running_positions[-1]
        bond = mol_manipulation.Tool.get_atoms_distance(position[title[0]], position[title[1]])
        if bond > 3.0 or bond < 2.0:
            print("%s May have wrong B_Cl bond!!! " % ts_log.file_dir)
        else:
            # continue
            if ts_log.is_right_ts:
                if float(ts_log.first_unreal_freq) <= -100:
                    continue
                else:
                    print("%s May have wrong vibration freq: %.4f!!! " % (ts_log.file_dir, float(ts_log.first_unreal_freq)))
            else:
                print("%s May have wrong vibration direction!!! " % ts_log.file_dir)
        mol_manipulation.ts_to_irc(ts_log, new_dir=irc_dir)


