from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom
import numpy as np
import pandas as pd
import copy, os, itertools
from skfp.preprocessing import MolFromSmilesTransformer, ConformerGenerator

from . import mol_manipulation, Tool

G_to_K = lambda x: np.exp(-4180 * x / 8.314 / 195.15)
G_to_rate = lambda x: 100 / (1 + np.exp(-4180 * x / 8.314 / 195.15))
K_to_G = lambda x: -8.314 * 195.15 * np.log(x) / 4180

def hand_write_addition(mol, FFOpt = True, z_dist = 4.0):
    pattern = Chem.MolFromSmarts("C=C")
    matches = mol.GetSubstructMatches(pattern)
    mol_batches = []
    ozone_mol = mol_manipulation.smiles2mol("[O+]O[O-]", 1)
    for match in matches:
        atom_a, atom_b = match[0], match[1]
        neighbor_atoms_a, neighbor_atoms_b = mol.GetAtomWithIdx(atom_a).GetNeighbors(), mol.GetAtomWithIdx(atom_b).GetNeighbors()
        neighbor_atoms_a = [each for each in neighbor_atoms_a if each.GetIdx() != atom_b]
        neighbor_atoms_b = [each for each in neighbor_atoms_b if each.GetIdx() != atom_a]
        position = mol.GetConformer(0).GetPositions()
        position = mol_manipulation.trfm_rot(position[match[0]], position[match[1]], position[neighbor_atoms_a[0].GetIdx()], position, center_point=np.array([0, 0, 0]))[:, :3]
        ozone_position = ozone_mol.GetConformer(0).GetPositions()

        if len(Chem.FindMolChiralCenters(mol)) > 0:
            all_z_pos = [-z_dist, z_dist]
        else:
            all_z_pos = [z_dist]

        final_mols = Chem.CombineMols(mol, ozone_mol)
        rwmol = Chem.RWMol(final_mols)
        rwmol.GetBondBetweenAtoms(match[0], match[1]).SetBondType(Chem.BondType.SINGLE)
        rwmol.AddBond(match[0], rwmol.GetNumAtoms() - 3, Chem.BondType.SINGLE)
        rwmol.AddBond(match[1], rwmol.GetNumAtoms() - 1, Chem.BondType.SINGLE)
        rwmol.RemoveBond(rwmol.GetNumAtoms() - 1, rwmol.GetNumAtoms() - 2)
        rwmol.RemoveBond(rwmol.GetNumAtoms() - 2, rwmol.GetNumAtoms() - 3)
        rwmol.AddBond(rwmol.GetNumAtoms() - 1, rwmol.GetNumAtoms() - 2, Chem.BondType.SINGLE)
        rwmol.AddBond(rwmol.GetNumAtoms() - 2, rwmol.GetNumAtoms() - 3, Chem.BondType.SINGLE)
        rwmol.GetAtomWithIdx(rwmol.GetNumAtoms() - 3).SetFormalCharge(0)
        rwmol.GetAtomWithIdx(rwmol.GetNumAtoms() - 1).SetFormalCharge(0)
        final_mols = rwmol.GetMol()
        Chem.SanitizeMol(final_mols)
        

        for z_id, z_pos in enumerate(all_z_pos):
            new_ozone_position = mol_manipulation.trfm_rot(ozone_position[0], ozone_position[2], ozone_position[1], ozone_position, center_point=np.array([0, 0, z_pos]))[:, :3]
            final_position = np.concatenate([position, new_ozone_position], axis=0)
            new_final_mols = copy.deepcopy(final_mols)
            for i in range(new_final_mols.GetNumConformers()): 
                new_final_mols.RemoveConformer(i)
            conformer = Chem.rdchem.Conformer(new_final_mols.GetNumAtoms())
            conformer.SetId(0)
            for i, c in enumerate(final_position):
                conformer.SetAtomPosition(i, c[:3])
            new_final_mols.AddConformer(conformer)
            Chem.AssignStereochemistryFrom3D(new_final_mols)
            if FFOpt:
                AllChem.UFFOptimizeMolecule(new_final_mols)
            NumAtoms = new_final_mols.GetNumAtoms()
            mol_batches.append([new_final_mols, [match[0], NumAtoms - 3, NumAtoms - 2, NumAtoms - 1, match[1]], neighbor_atoms_a, neighbor_atoms_b, z_id])
    return mol_batches

def EmbedConfs(mol, num_confs=20, random_seed=42, restrict = []):
    mol2 = copy.deepcopy(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.pruneRmsThresh = 0.5
    D = np.array(AllChem.Get3DDistanceMatrix(mol2))
    bm = np.array(rdDistGeom.GetMoleculeBoundsMatrix(mol2))
    UL = bm - bm.T
    UL * 1.5
    for atom_a, atom_b in restrict:
        max_a, min_a = max(atom_a, atom_b), min(atom_a, atom_b)
        UL[max_a, min_a] = -0.001
        UL[min_a, max_a] = 0.001
    new_bm = D + UL
    params.SetBoundsMat(new_bm)
    # mol2.RemoveAllConformers()
    AllChem.EmbedMultipleConfs(mol2, num_confs, params=params)
    return mol2

def write_noV3000_mol(mol, mol_file, confId=0):
    temp_mol = copy.deepcopy(mol)
    temp_mol = Chem.RWMol(temp_mol)
    temp_mol.SetStereoGroups([])
    mol_block = Chem.MolToMolBlock(temp_mol, forceV3000=False)
    # with open(mol_file, 'wt') as f:
    #     f.write(mol_block)
    Chem.MolToMolFile(temp_mol, mol_file, forceV3000=False, confId=confId)

def generate_mol_input(target_df, mol_dir=None, save_mol=True, banned_ene = [], seed=3):
    
    new_df = pd.DataFrame({"Index":{}, "Ene": {}, "Site_A": {}, "Site_B": {}, 'Z_Pos': {}, "Rot": {}})
    new_df_id = 0
    gs_mols, gs_names = [], []
    ts_mols, ts_names, ts_restrict = [], [], {}
    gs_2_mols, gs_2_names, gs_2_restrict = [], [], {}
    ts_2_mols, ts_2_names, ts_2_restrict = [], [], {}
    restrict_dict = {}
    restrict_dict_2 = {}
    restrict_angle_1 = [-133, 133]
    restrict_angle_2 = [80, 160]
    for row_id, row in target_df.iterrows():
        target_ene = row['Ene']
        Index = row['Index']
        if Index in banned_ene:
            print(f'Banned SMILES: {Index} {target_ene}')
            continue
        if not Chem.MolFromSmiles(target_ene).HasSubstructMatch(Chem.MolFromSmarts("C=C")):
            print(f"Not Found Ene in SMILES {target_ene}")
            continue
        # 获得烯烃前体结构
        gs_mol_name = f'gs1_{Index:05}'
        if mol_dir == None or not os.path.isfile(os.path.join(mol_dir, f'{gs_mol_name}.mol')):
            # mol = mol_manipulation.smiles2mol(target_ene, 1)
            try:
                mfs = MolFromSmilesTransformer()
                mols = mfs.transform([row['Ene']])
                conf_gen = ConformerGenerator(random_state=seed)
                [mol] = conf_gen.transform(mols) 
            except:
                print(f"New Banned SMILES : {Index} {target_ene}")
            if save_mol:
                write_noV3000_mol(mol, os.path.join(mol_dir, f'{gs_mol_name}.mol'))
        else:
            mol = Chem.MolFromMolFile(os.path.join(mol_dir, f'{gs_mol_name}.mol'), removeHs=False)

        gs_position = mol.GetConformer(0).GetPositions()
        gs_mols.append(mol)
        gs_names.append(gs_mol_name)

        all_mol_batchs = hand_write_addition(mol)
        for [reactant, match, neighbor_atoms_a, neighbor_atoms_b, z_id] in all_mol_batchs:
            position = reactant.GetConformer(0).GetPositions()


            # 识别两端的烯烃
            pairs = []
            for neighbor_a, neighbor_b in itertools.product(neighbor_atoms_a, neighbor_atoms_b):
                dihedral_angle = Tool.get_dihedral_angle(position[neighbor_a.GetIdx()], position[match[0]], position[match[-1]], position[neighbor_b.GetIdx()])
                if np.abs(dihedral_angle) > 80:
                    if dihedral_angle < 0:
                        pairs.append([neighbor_a.GetIdx(), match[0], match[-1], neighbor_b.GetIdx(), -1])
                    else:
                        pairs.append([neighbor_a.GetIdx(), match[0], match[-1], neighbor_b.GetIdx(), 1])
                    dihedral_angle_2 = Tool.get_dihedral_angle(gs_position[neighbor_a.GetIdx()], gs_position[match[0]], gs_position[match[-1]], gs_position[neighbor_b.GetIdx()])
                    if np.abs(dihedral_angle_2) < 80:
                        print(row_id, "Error")
            require_restrict = len(pairs) >1

            for id, angle in enumerate(restrict_angle_1):
                # 获得[3+2]产物结构
                temp_react = copy.deepcopy(reactant)
                ff = Chem.AllChem.UFFGetMoleculeForceField(temp_react)
                ff.UFFAddTorsionConstraint(match[2], match[1], match[3], match[0], False, angle - 1, angle + 1, 1)
                ff.Initialize()
                ff.Minimize()
                gs_2_mol_name = f'gs2_{Index:05}_{match[0]:03}_{match[4]:03}_{z_id:01}_{id:01}'
                if save_mol:
                    write_noV3000_mol(temp_react, os.path.join(mol_dir, f'{gs_2_mol_name}.mol'))
                gs_2_mols.append(temp_react)
                gs_2_names.append(gs_2_mol_name)
                gs_2_restrict[gs_2_mol_name] = [[match[2] + 1, match[1] + 1, match[3] + 1, match[0] + 1]]

                # 搭建3+2过渡态
                temp_react = copy.deepcopy(reactant)
                ts_mol_name = f'ts1_{Index:05}_{match[0]:03}_{match[4]:03}_{z_id:01}_{id:01}'
                restrict_1 = [[match[0] + 1, match[1] + 1], [match[3] + 1, match[4] + 1]]
                ff = Chem.AllChem.UFFGetMoleculeForceField(temp_react)
                ff.UFFAddDistanceConstraint(match[0], match[1], False, 2.2, 2.2 + 0.05, 10000)
                ff.UFFAddDistanceConstraint(match[3], match[4], False, 2.2, 2.2 + 0.05, 10000)
                ff.UFFAddTorsionConstraint(match[2], match[1], match[3], match[0], False, angle - 1, angle + 1, 1)
                ff.Initialize()
                ff.Minimize()
                if save_mol:
                    write_noV3000_mol(temp_react, os.path.join(mol_dir, f'{ts_mol_name}.mol'))
                ts_mols.append(temp_react)
                ts_names.append(ts_mol_name)
                # ts_restrict.append(restrict_1)
                restrict_dict[ts_mol_name] = restrict_1
                if not require_restrict:
                    ts_restrict[ts_mol_name] = restrict_1
                    break
                else:
                    ts_restrict[ts_mol_name] = restrict_1 + [[match[2] + 1, match[1] + 1, match[3] + 1, match[0] + 1]]


            for temp_match in [match, match[::-1]]:
                ids = np.arange(len(restrict_angle_2))
                for id, reverse_id in zip(ids, ids[::-1]):
                    ts_mol_name2 = f'ts2_{Index:05}_{temp_match[0]:03}_{temp_match[4]:03}_{z_id:01}_{id:01}'
                    restrict_2 = [[temp_match[0] + 1, temp_match[4] + 1], [temp_match[2] + 1, temp_match[3] + 1]]
                    new_mol = copy.deepcopy(reactant)

                    distance_a1 = 1.8
                    distance_a2 = 2.0
                    

                    ff = Chem.AllChem.UFFGetMoleculeForceField(new_mol)
                    ff.UFFAddDistanceConstraint(temp_match[0], temp_match[4], False, distance_a1 + 0.05, distance_a1 + 0.15, 10000)
                    ff.UFFAddDistanceConstraint(temp_match[2], temp_match[3], False, distance_a2 + 0.05, distance_a2 + 0.15, 10000)
                    if require_restrict:
                        ff.UFFAddTorsionConstraint(pairs[0][0], pairs[0][1], pairs[0][2], pairs[0][3], False, restrict_angle_2[id] * pairs[0][4] - 1, restrict_angle_2[id] * pairs[0][4] + 1, 10)
                        ff.UFFAddTorsionConstraint(pairs[1][0], pairs[1][1], pairs[1][2], pairs[1][3], False, restrict_angle_2[reverse_id] * pairs[1][4] - 1, restrict_angle_2[reverse_id] * pairs[1][4] + 1, 10)
                    ff.Initialize()
                    ff.Minimize()

                    if save_mol:
                        write_noV3000_mol(new_mol, os.path.join(mol_dir, f'{ts_mol_name2}.mol'))
                    ts_2_mols.append(new_mol)
                    ts_2_names.append(ts_mol_name2)
                    restrict_dict_2[ts_mol_name2] = restrict_2
                    if not require_restrict:
                        # ts_2_restrict.append(restrict_2)
                        ts_2_restrict[ts_mol_name2] = restrict_2
                    else:
                        # ts_2_restrict.append(restrict_2 + [[pairs[0][0] + 1, pairs[0][1] + 1, pairs[0][2] + 1, pairs[0][3] + 1]] + [[pairs[1][0] + 1, pairs[1][1] + 1, pairs[1][2] + 1, pairs[1][3] + 1]])
                        ts_2_restrict[ts_mol_name2] = restrict_2 + [[pairs[0][0] + 1, pairs[0][1] + 1, pairs[0][2] + 1, pairs[0][3] + 1]] + [[pairs[1][0] + 1, pairs[1][1] + 1, pairs[1][2] + 1, pairs[1][3] + 1]]
                    new_df.loc[new_df_id] = [Index, target_ene, temp_match[0], temp_match[4], z_id, id]
                    new_df_id += 1
                    if not require_restrict:
                        break
    gs_pack = [gs_mols, gs_names]
    gs2_pack = [gs_2_mols, gs_2_names, gs_2_restrict]
    ts_pack = [ts_mols, ts_names, ts_restrict]
    ts2_pack = [ts_2_mols, ts_2_names, ts_2_restrict]
    return gs_pack, gs2_pack, ts_pack, ts2_pack, restrict_dict, restrict_dict_2, new_df

def smiles_to_AAM(smiles, return_mediate=False):
    result = []
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    pattern = Chem.MolFromSmarts("C=C")
    matches = mol.GetSubstructMatches(pattern)
    ozone_mol = Chem.MolFromSmiles("[O+]O[O-]")
    reaction_mols = Chem.CombineMols(mol, ozone_mol)
    [atom.SetAtomMapNum(atom.GetIdx() + 1) for atom in reaction_mols.GetAtoms()]
    reactant_smiles = Chem.MolToSmiles(reaction_mols)
    for match_ in matches:
        for match in [match_, match_[::-1]]:
            rwmol = Chem.RWMol(reaction_mols)
            rwmol.RemoveBond(match[0], match[1])
            rwmol.AddBond(match[0], rwmol.GetNumAtoms() - 3, Chem.BondType.DOUBLE)
            rwmol.AddBond(match[1], rwmol.GetNumAtoms() - 1, Chem.BondType.DOUBLE)
            rwmol.RemoveBond(rwmol.GetNumAtoms() - 1, rwmol.GetNumAtoms() - 2)
            rwmol.GetAtomWithIdx(rwmol.GetNumAtoms() - 3).SetFormalCharge(1)
            rwmol.GetAtomWithIdx(rwmol.GetNumAtoms() - 2).SetFormalCharge(-1)
            rwmol.GetAtomWithIdx(rwmol.GetNumAtoms() - 1).SetFormalCharge(0)
            final_mols = rwmol.GetMol()
            Chem.SanitizeMol(final_mols)
            if return_mediate:
                rwmol_mediate = Chem.RWMol(reaction_mols)
                rwmol_mediate.RemoveBond(match[0], match[1])
                rwmol_mediate.AddBond(match[0], rwmol.GetNumAtoms() - 3, Chem.BondType.SINGLE)
                rwmol_mediate.AddBond(match[1], rwmol.GetNumAtoms() - 1, Chem.BondType.SINGLE)
                rwmol_mediate.AddBond(match[0], match[1], Chem.BondType.SINGLE)
                rwmol_mediate.GetAtomWithIdx(rwmol.GetNumAtoms() - 3).SetFormalCharge(0)
                rwmol_mediate.GetAtomWithIdx(rwmol.GetNumAtoms() - 2).SetFormalCharge(0)
                rwmol_mediate.GetAtomWithIdx(rwmol.GetNumAtoms() - 1).SetFormalCharge(0)
                mediate = rwmol_mediate.GetMol()
                Chem.SanitizeMol(mediate)
                result.append([reactant_smiles + ">>" + Chem.MolToSmiles(mediate) + ">>" + Chem.MolToSmiles(final_mols), match])
            else:
                result.append([reactant_smiles + ">>" + Chem.MolToSmiles(final_mols), match])
    return result

def FF_TS1(temp_ts, match, angle):
    for conf_id in range(temp_ts.GetNumConformers()):
        ff = Chem.AllChem.UFFGetMoleculeForceField(temp_ts, confId = conf_id)
        ff.UFFAddDistanceConstraint(match[0], match[1], False, 2.3, 2.3 + 0.05, 10000)
        ff.UFFAddDistanceConstraint(match[3], match[4], False, 2.3, 2.3 + 0.05, 10000)
        ff.UFFAddTorsionConstraint(match[0], match[2], match[3], match[0], False, angle - 1, angle + 1, 1)
        ff.Initialize()
        ff.Minimize()
    return temp_ts

def FF_TS2(new_mol, temp_match, pairs, require_restrict, restrict_angle_2, id, reverse_id, distance_a1 = 1.9, distance_a2 = 2.0, distance_a3 = 2.4):
    for conf_id in range(new_mol.GetNumConformers()):
        ff = Chem.AllChem.UFFGetMoleculeForceField(new_mol, confId = conf_id)
        ff.UFFAddDistanceConstraint(temp_match[0], temp_match[4], False, distance_a1 - 0.05, distance_a1 + 0.05, 10000)
        ff.UFFAddDistanceConstraint(temp_match[2], temp_match[3], False, distance_a2 - 0.05, distance_a2 + 0.05, 10000)
        ff.UFFAddDistanceConstraint(temp_match[2], temp_match[4], False, distance_a3 - 0.05, distance_a3 + 0.05, 10000)
        if require_restrict:
            ff.UFFAddTorsionConstraint(pairs[0][0], pairs[0][1], pairs[0][2], pairs[0][3], False, restrict_angle_2[id] * pairs[0][4] - 1, restrict_angle_2[id] * pairs[0][4] + 1, 1)
            ff.UFFAddTorsionConstraint(pairs[1][0], pairs[1][1], pairs[1][2], pairs[1][3], False, restrict_angle_2[reverse_id] * pairs[1][4] - 1, restrict_angle_2[reverse_id] * pairs[1][4] + 1, 1)
        ff.Initialize()
        ff.Minimize()
    return new_mol

def generate_mol_input_MultiConf(target_df, mol_dir, conf_num = 20, load_mol = False, save_mol=True, banned_ene = []):
    
    new_df = pd.DataFrame({"Index":{}, "Ene": {}, "Site_A": {}, "Site_B": {}, 'Z_Pos': {}, "Rot": {}})
    new_df_id = 0
    gs_mols, gs_names = [], []
    ts_mols, ts_names, ts_restrict = [], [], {}
    gs_2_mols, gs_2_names, gs_2_restrict = [], [], {}
    ts_2_mols, ts_2_names, ts_2_restrict = [], [], {}
    restrict_dict = {}
    restrict_dict_2 = {}
    restrict_angle_1 = [-133, 133]
    restrict_angle_2 = [80, 160]
    for row_id, row in target_df.iterrows():
        # try:
        target_ene = row['Ene']
        Index = row['Index']
        print(f"Process SMILES ID: {Index}", end='\r')
        # if not Chem.MolFromSmiles(target_ene).HasSubstructMatch(Chem.MolFromSmarts("C1=C****1")):
        #     continue
        # 获得烯烃前体结构
        if Index in banned_ene:
            print(f'Banned SMILES: {Index} {target_ene}')
            continue
        if not Chem.MolFromSmiles(target_ene).HasSubstructMatch(Chem.MolFromSmarts("C=C")):
            print(f"Not Found Ene in SMILES {target_ene}")
            continue
        gs_mol_name = f'gs1_{Index:05}'
        if mol_dir == None or not os.path.isfile(os.path.join(mol_dir, f'{gs_mol_name}.mol')):
            # mol = mol_manipulation.smiles2mol(target_ene, 1)
            try:
                mfs = MolFromSmilesTransformer()
                mols = mfs.transform([row['Ene']])
                conf_gen = ConformerGenerator()
                [mol] = conf_gen.transform(mols) 
            except:
                print(f"New Banned SMILES : {Index} {target_ene}")
            if save_mol:
                write_noV3000_mol(mol, os.path.join(mol_dir, f'{gs_mol_name}.mol'))
        else:
            mol = Chem.MolFromMolFile(os.path.join(mol_dir, f'{gs_mol_name}.mol'), removeHs=False)
        gs_position = mol.GetConformer(0).GetPositions()
        # Chem.MolToMolFile(mol, os.path.join(mol_dir, f'{gs_mol_name}.mol'), includeStereo=False)
        gs_mols.append(mol)
        gs_names.append(gs_mol_name)

        all_mol_batchs = hand_write_addition(mol)
        for [reactant, match, neighbor_atoms_a, neighbor_atoms_b, z_id] in all_mol_batchs:
            position = reactant.GetConformer(0).GetPositions()


            # 识别两端的烯烃
            pairs = []
            for neighbor_a, neighbor_b in itertools.product(neighbor_atoms_a, neighbor_atoms_b):
                dihedral_angle = Tool.get_dihedral_angle(position[neighbor_a.GetIdx()], position[match[0]], position[match[-1]], position[neighbor_b.GetIdx()])
                if np.abs(dihedral_angle) > 70:
                    if dihedral_angle < 0:
                        pairs.append([neighbor_a.GetIdx(), match[0], match[-1], neighbor_b.GetIdx(), -1])
                    else:
                        pairs.append([neighbor_a.GetIdx(), match[0], match[-1], neighbor_b.GetIdx(), 1])
                    dihedral_angle_2 = Tool.get_dihedral_angle(gs_position[neighbor_a.GetIdx()], gs_position[match[0]], gs_position[match[-1]], gs_position[neighbor_b.GetIdx()])
                    if np.abs(dihedral_angle_2) < 70:
                        print(row_id, "Error")
            require_restrict = len(pairs) >1

            for id, angle in enumerate(restrict_angle_1):
                # 获得[3+2]产物结构
                temp_react = copy.deepcopy(reactant)
                ff = Chem.AllChem.UFFGetMoleculeForceField(temp_react)
                ff.UFFAddTorsionConstraint(match[2], match[1], match[3], match[0], False, angle - 1, angle + 1, 1)
                ff.Initialize()
                ff.Minimize()
                gs_2_mol_name = f'gs2_{Index:05}_{match[0]:03}_{match[4]:03}_{z_id:01}_{id:01}'
                if save_mol:
                    write_noV3000_mol(temp_react, os.path.join(mol_dir, gs_2_mol_name + ".mol"))
                gs_2_mols.append(temp_react)
                gs_2_names.append(gs_2_mol_name)
                gs_2_restrict[gs_2_mol_name] = [[match[2] + 1, match[1] + 1, match[3] + 1, match[0] + 1]]

                # 搭建3+2过渡态
                temp_ts = copy.deepcopy(temp_react)
                ts_mol_name = f'ts1_{Index:05}_{match[0]:03}_{match[4]:03}_{z_id:01}_{id:01}'
                restrict_1 = [[match[0] + 1, match[1] + 1], [match[3] + 1, match[4] + 1]]
                
                if load_mol:
                    if isinstance(load_mol, bool):
                        temp_ts = Chem.MolFromMolFile(os.path.join(mol_dir, f'{ts_mol_name}.mol'), removeHs=False)
                    elif ts_mol_name in load_mol.keys():
                        temp_ts = load_mol[ts_mol_name]
                    else:
                        print(ts_mol_name, "no found in load_mol")
                        continue
                else:
                    temp_ts = FF_TS1(temp_ts, match, angle)
                embed_restrict = [[match[0], match[1]], [match[3], match[4]], [match[0], match[3]], [match[1], match[4]]]
                if len(pairs) >=1:
                    embed_restrict += [[match[2], pairs[0][0]]]
                if conf_num > 0:
                    temp_ts_ = EmbedConfs(temp_ts, conf_num, restrict = embed_restrict)
                    if len(temp_ts_.GetConformers()) > 0:
                        temp_ts_ = FF_TS1(temp_ts_, match, angle)
                        temp_ts_.AddConformer(temp_ts.GetConformer(0), assignId=True)
                        temp_ts = temp_ts_

                if save_mol:
                    write_noV3000_mol(temp_ts, os.path.join(mol_dir, ts_mol_name + '.mol'))
                ts_mols.append(temp_ts)
                ts_names.append(ts_mol_name)
                # ts_restrict.append(restrict_1)
                restrict_dict[ts_mol_name] = restrict_1
                if not require_restrict:
                    ts_restrict[ts_mol_name] = restrict_1
                    break
                else:
                    ts_restrict[ts_mol_name] = restrict_1 + [[match[2] + 1, match[1] + 1, match[3] + 1, match[0] + 1]]


            for temp_match in [match, match[::-1]]:
                ids = np.arange(len(restrict_angle_2))
                for id, reverse_id in zip(ids, ids[::-1]):
                    ts_mol_name2 = f'ts2_{Index:05}_{temp_match[0]:03}_{temp_match[4]:03}_{z_id:01}_{id:01}'
                    restrict_2 = [[temp_match[0] + 1, temp_match[4] + 1], [temp_match[2] + 1, temp_match[3] + 1]]
                    new_mol = copy.deepcopy(reactant)

                    if load_mol:
                        if isinstance(load_mol, bool):
                            new_mol = Chem.MolFromMolFile(os.path.join(mol_dir, f'{ts_mol_name2}.mol'), removeHs=False)
                        elif ts_mol_name2 in load_mol.keys():
                            new_mol = load_mol[ts_mol_name2]
                        else:
                            print(ts_mol_name2, "no found in load_mol")
                            continue
                    else:
                        new_mol = FF_TS2(new_mol, temp_match, pairs, require_restrict, restrict_angle_2, id, reverse_id)
                    embed_restrict = [[temp_match[0], temp_match[4]], [temp_match[2], temp_match[3]], 
                                                                [temp_match[0], temp_match[3]], [temp_match[2], temp_match[4]], 
                                                                [temp_match[1], temp_match[4]], [temp_match[1], temp_match[3]]]
                    if require_restrict:
                        embed_restrict += [[pairs[0][0], temp_match[2]], [pairs[0][3], temp_match[2]],
                                                                [pairs[1][0], temp_match[2]], [pairs[1][3], temp_match[2]]]
                    if conf_num > 0:
                        new_mol_ = EmbedConfs(new_mol, conf_num, restrict=embed_restrict, )
                        if len(new_mol_.GetConformers()) > 0:
                            new_mol_ = FF_TS2(new_mol_, temp_match, pairs, require_restrict, restrict_angle_2, id, reverse_id)
                            new_mol_.AddConformer(new_mol.GetConformer(0), assignId=True)
                            new_mol = new_mol_


                    if save_mol:
                        write_noV3000_mol(new_mol, os.path.join(mol_dir, ts_mol_name2 + ".mol"))
                    ts_2_mols.append(new_mol)
                    ts_2_names.append(ts_mol_name2)
                    restrict_dict_2[ts_mol_name2] = restrict_2
                    if not require_restrict:
                        ts_2_restrict[ts_mol_name2] = restrict_2
                    else:
                        ts_2_restrict[ts_mol_name2] = restrict_2 + [[pairs[0][0] + 1, pairs[0][1] + 1, pairs[0][2] + 1, pairs[0][3] + 1]] + [[pairs[1][0] + 1, pairs[1][1] + 1, pairs[1][2] + 1, pairs[1][3] + 1]]
                    new_df.loc[new_df_id] = [Index, target_ene, temp_match[0], temp_match[4], z_id, id]
                    new_df_id += 1
                    if not require_restrict:
                        break
        # except:
        #     continue
    gs_pack = [gs_mols, gs_names]
    gs2_pack = [gs_2_mols, gs_2_names, gs_2_restrict]
    ts_pack = [ts_mols, ts_names, ts_restrict]
    ts2_pack = [ts_2_mols, ts_2_names, ts_2_restrict]
    return gs_pack, gs2_pack, ts_pack, ts2_pack, restrict_dict, restrict_dict_2, new_df

def calc_DDG(result_df, detail_df, column_TS1_TS2, banned_ene=[]):

    A_site = []
    B_site = []
    B_A_Energies = [[] for _ in range(len(column_TS1_TS2))]
    for row_id, row in result_df.iterrows():
        fail = 0
        ene_index = row['Index']
        if ene_index in banned_ene:
            fail = 1
        temp_df = detail_df.loc[detail_df['Index'] == ene_index]
        # if len(temp_df) < 2 or len(temp_df) != len(temp_df.dropna(subset=['TS_G', 'TS2_G'])):
        #     fail = 1
        # if 1 in temp_df['Z_Pos'].to_numpy():
            # temp_df = temp_df.dropna(subset=['TS_G(kcal/mol)']).reset_index(drop=True)
            # temp_df = temp_df.loc[temp_df['TS_dG(kcal/mol)'] > 0].reset_index(drop=True)
        sites = temp_df['Site_A'].unique()
        if len(sites) != 2:
            fail = 1
        else:
            temp_site_A = temp_df.loc[temp_df['Site_A'] == sites[0]]
            temp_site_B = temp_df.loc[temp_df['Site_A'] == sites[1]]

        if fail != 1 and 1 in temp_df['Z_Pos'].to_numpy():
            temp_pos_A0 = temp_site_A.loc[temp_site_A['Z_Pos'] == 0]
            temp_pos_A1 = temp_site_A.loc[temp_site_A['Z_Pos'] == 1]
            temp_pos_B0 = temp_site_B.loc[temp_site_B['Z_Pos'] == 0]
            temp_pos_B1 = temp_site_B.loc[temp_site_B['Z_Pos'] == 1]
            if len(temp_pos_A0) == 0 or len(temp_pos_A1) == 0 or len(temp_pos_B1) == 0 or len(temp_pos_B0) == 0:
                fail = 1
            else:
                for col_id, [TS1_column_name_pred, TS2_column_name_pred, B_A_Energy_name] in enumerate(column_TS1_TS2):
                    K0_pred = G_to_K(np.min(temp_pos_A1[TS1_column_name_pred].to_numpy()) - np.min(temp_pos_A0[TS1_column_name_pred].to_numpy()))
                    K1_pred = G_to_K(np.min(temp_pos_B0[TS2_column_name_pred].to_numpy()) - np.min(temp_pos_A0[TS2_column_name_pred].to_numpy()))
                    K2_pred = G_to_K(np.min(temp_pos_B1[TS2_column_name_pred].to_numpy()) - np.min(temp_pos_A1[TS2_column_name_pred].to_numpy()))
                    KFinal_pred = (K1_pred * (K2_pred + 1) + K0_pred * K2_pred * (K1_pred + 1)) / (K2_pred + 1 + K0_pred * (K1_pred + 1))
                    B_A_Energies[col_id].append(K_to_G(KFinal_pred))
                
                A_site.append(temp_site_A['Site_A'].iloc[0])
                B_site.append(temp_site_B['Site_A'].iloc[0])
        elif fail != 1:
            for col_id, [TS1_column_name_pred, TS2_column_name_pred, B_A_Energy_name] in enumerate(column_TS1_TS2):
                B_A_Energies[col_id].append(np.min(temp_site_B[TS2_column_name_pred].to_numpy()) - np.min(temp_site_A[TS2_column_name_pred].to_numpy()))
            A_site.append(temp_site_A['Site_A'].iloc[0])
            B_site.append(temp_site_B['Site_A'].iloc[0])
        if fail == 1:
            A_site.append(np.nan)
            B_site.append(np.nan)
            for col_id, [TS1_column_name_pred, TS2_column_name_pred, B_A_Energy_name] in enumerate(column_TS1_TS2):
                B_A_Energies[col_id].append(np.nan)

    result_df['A_site'] = A_site
    # result_df['A_Rot'] = A_Rot
    # result_df['A_Zpos'] = A_Zpos
    result_df['B_site'] = B_site
    # result_df['B_Rot'] = B_Rot
    for col_id, [TS1_column_name_pred, TS2_column_name_pred, B_A_Energy_name] in enumerate(column_TS1_TS2):
        result_df[B_A_Energy_name] = B_A_Energies[col_id]
    return result_df
