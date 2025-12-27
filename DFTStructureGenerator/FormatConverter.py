# 一些文件转换的脚本写在这里
import numpy as np
import pandas as pd
import shutil, os

from rdkit import Chem
from rdkit.Chem import AllChem

from . import Tool
# def MolFormatConversion(input_file: str, output_file: str, input_format="xyz", output_format="sdf"):
#     """简单的完成xyz和sdf转换，

#     Args:
#         input_file (str): file dir
#         output_file (str): file dir
#         input_format (str, optional): _description_. Defaults to "xyz".
#         output_format (str, optional): _description_. Defaults to "sdf".
#     """    
#     molecules = readfile(input_format, input_file)
#     output_file_writer = Outputfile(output_format, output_file, overwrite=True)
#     i = 0
#     for molecule in molecules:
#         output_file_writer.write(molecule)
#         i += 1
#     output_file_writer.close()
#     print('%d molecules converted' % (i))

def read_gjf(gjf_file):
    with open(gjf_file, "rt") as f:
        lines = f.readlines()
    for id_, line in enumerate(lines):
        if line.startswith("0 1") or line.startswith("0 2") or line.startswith("1 2"):
            break
    atoms = []
    positions = []
    for line in lines[id_ + 1:]:
        gjf_list = line.strip('\n').split()
        if len(gjf_list) != 4:
            break
        atoms.append(gjf_list[0])
        positions.append([float(each) for each in gjf_list[1:]])
    return atoms, positions

def xyz_to_gjf(choose="opt"):
    """write xyzfile to gjf
    Useless!!!

    Args:
        choose (str, optional): _description_. Defaults to "opt".
    """    
    files = os.listdir("Data/xyz")
    for file in files:
        xyz_dir = "Data/xyz/" + file
        gjf_dir = "Data/gjf/" + file[:-4] + ".gjf"
        with open(gjf_dir, "wt") as af:
            if file in ["Tri-25.xyz", "Tri-26.xyz", "Tri-27.xyz"]:
                af.write(
                    "%%nprocshared=28\n%mem=56GB\n#p opt freq b3lyp/6-31g(d)\n\nNone\n\n1 1\n")
            else:
                af.write(
                    "%%nprocshared=28\n%mem=56GB\n#p opt freq b3lyp/6-31g(d)\n\nNone\n\n0 1\n")
            with open(xyz_dir, "rt") as bf:
                a = bf.readlines()
                for each in a[2:]:
                    af.write(each)
            af.write("\n\n")

def mol_to_xyz(mol, atom_list=None, position_list=None, file_dir="test.xyz",title=None, confid=None):
    """最基础的mol转化为xyz文件的过程

    Args:
        mol (Chem.Mol): 
        file_dir (str, optional): _description_. Defaults to "test.xyz".
        title (str, optional): _description_. Defaults to None.
        

    Returns:
        filename: _description_
    """   
    if mol is None and (atom_list is None or position_list is None):
        print("mol, atom_list/position_list must contain one")
        return None 
    
    
    if mol is not None:
        atom_num = mol.GetNumAtoms()
        atom_list = [atom.GetSymbol() for atom in mol.GetAtoms()]
        position_list = [conf.GetPositions() for conf in mol.GetConformers()]
    else:
        atom_num = len(atom_list)
    if confid != None:
        position_list = [position_list[confid]]
    else:
        file_dir = file_dir.split(".")[0]
    file_names = []
    for i, position in enumerate(position_list):
        if confid != None:
            file_name = file_dir
        else:
            file_name = "%s_%s.xyz" % (file_dir, str(i))
        with open(file_name, "wt") as f:
            f.write(" %d\n" % atom_num)
            f.write("%s\n" % title)
            for j, atom in enumerate(atom_list):
                f.write("%s %.8f %.8f %.8f\n" % (atom, position[j][0],position[j][1],position[j][2]))
        file_names.append(file_name)
    return file_names
            
def mol_to_gjf(mol, file_="test_data/mol2gjf.gjf", charge=None, SpinMultiplicity=None, title="Title", method="opt freq b3lyp/6-311g(d,p)", confid=0, ignore_warning=False, freeze=[], difreeze=[], savechk=None, readchk=None, final_line=None):
    """cover mol to gjf, title line with charge

    Args:
        mol (Chem.Mol): _description_
        file_dir (str, optional): file_dir. Defaults to "test_data/mol2gjf.gjf".
        charge (int, optional): 0 or None. Defaults to None.
        title (str, optional): title. Defaults to "Title".
        method (str, optional): method use to calculation. Defaults to "opt freq b3lyp/6-311g(d,p)".
        confid (int, optional): use special conformers with mol. Defaults to 0.
    """    
    if charge == None:
        charge = sum([atom.GetFormalCharge() for atom in mol.GetAtoms()])
    if SpinMultiplicity == None:
        SpinMultiplicity = Tool.GetSpinMultiplicity(mol)
    file_dir, filename = os.path.split(file_)
    if file_dir != "":
        if not os.path.isdir(file_dir):
            os.mkdir(file_dir)
    with open(file_, "wt") as f:
        if savechk != None:
            f.write("%%chk=%s.chk\n" % savechk)
        if readchk != None:
            f.write("%%oldchk=%s.chk\n" % readchk)
        f.write("%nprocshared=28\n%mem=56GB\n#p")
        f.write(" %s\n\n" % method)
        f.write("$$$$%s####%d????\n\n" % (title, charge))
        f.write("%d %d\n" % (int(charge), SpinMultiplicity))
        if SpinMultiplicity != 1 and not ignore_warning:
            print("%s 's SpinMultiplicity != 1, check it" % file_)
        if len(mol.GetConformers()) == 0:
            mol = AllChem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            AllChem.EmbedMultipleConfs(mol, numConfs=1)
        positions = mol.GetConformer(confid).GetPositions()
        for i, atom in enumerate(mol.GetAtoms()):
            f.write(" %s                 " % atom.GetSymbol())
            f.write("%f %f %f\n" % tuple(positions[i]))
        f.write("\n")
        for each in freeze:
            f.write("B %d %d F\n" % tuple(each))
        for each in difreeze:
            f.write("D %d %d %d %d F\n" % tuple(each))
        if final_line != None:
            f.write(final_line)
            f.write("\n")
        f.write("\n\n")

def block_to_gjf(symbol_list, positions, file="test_data/mol2gjf.gjf", charge=0, multiplicity=1, title="Title", method="opt freq b3lyp/6-311g(d,p)", freeze=[], difreeze=[], savechk=None, readchk=None, final_line=None):
    """cover symbol_list and position into gjf, can write link1, as om&ts

    Args:
        symbol_list (_type_): _description_
        positions (_type_): _description_
        file (str, optional): _description_. Defaults to "test_data/mol2gjf.gjf".
        charge (int, optional): _description_. Defaults to 0.
        title (str, optional): _description_. Defaults to "Title".
        method (str, optional): _description_. Defaults to "opt freq b3lyp/6-311g(d,p)".
        method2 (_type_, optional): _description_. Defaults to None.
        freeze (_type_, optional): Freeze atoms, as TS calculation. Defaults to [].
        savechk (str, optional): Name of Chkfile, 省略.chk. Defaults to None.
    """    
    file_dir, filename = os.path.split(file)
    if not os.path.isdir(file_dir):
        os.mkdir(file_dir)
    assert len(symbol_list) == len(positions)
    with open(file, "wt") as f:
        if savechk != None:
            f.write("%%chk=%s.chk\n" % savechk)
        if readchk != None:
            f.write("%%oldchk=%s.chk\n" % readchk)
        f.write("%nprocshared=28\n%mem=56GB\n#p")
        f.write(" %s\n\n" % method)
        f.write("$$$$%s####%d????\n\n" % (title, int(charge)))
        f.write("%d %d\n" % (int(charge), int(multiplicity)))
        # positions = mol.GetConformer(0).GetPositions()
        for i, atom in enumerate(symbol_list):
            f.write(" %s                 " % atom)
            f.write("%f %f %f\n" % tuple(positions[i][:3]))
        f.write("\n")
        for each in freeze:
            f.write("B %d %d F\n" % tuple(each))
        for each in difreeze:
            f.write("D %d %d %d %d F\n" % tuple(each))
        if final_line != None:
            f.write(final_line)
            f.write("\n")
        f.write("\n\n")

def _split_files(gjf_files, split_dir, split_num):
    for file_id, each_file in enumerate(gjf_files):
        target_dir = split_dir + str(file_id % split_num)
        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)
        target_file = target_dir + "/" + os.path.split(each_file)[1]
        shutil.copy(each_file, target_file)
        # print(target_file)


def conc_files(conc_dir, gjf_files):
    for file_id, each_file in enumerate(gjf_files):
        target_dir = conc_dir 
        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)
        target_file = target_dir + "/" + os.path.split(each_file)[1]
        shutil.copy(each_file, target_file)

def read_smiles_file(files):
    all_smiles = []
    for file in files:
        with open(file, "r") as f:
            lines = f.readlines()
        for eachline in lines:
            smiles = eachline.strip("\n")
            all_smiles.append(smiles)
    return all_smiles
    
def write_smi_file(smiles_list, file_name):
    file_dir = os.path.split(file_name)[0]
    if not os.path.isdir(file_dir):
        os.mkdir(file_dir)
    with open(file_name, "wt") as f:
        for each in smiles_list:
            f.write(each)
            f.write('\n')
            
def write_smi_csv(smiles_list, file_name):
    a = {"Index":{}, "Smiles":{}}
    for idx, smiles in enumerate(smiles_list):
        a["Index"][idx] = idx
        a["Smiles"][idx] = smiles
    file_dir = os.path.split(file_name)[0]
    if not os.path.isdir(file_dir):
        os.mkdir(file_dir)
    a = pd.DataFrame(a)
    a.to_csv(file_name)

def read_chg_file(file):
    return_dict = {"index":{}, "symbol":{}, "charge":{}}
    with open(file) as f:
        lines = f.readlines()
    for idx, line in enumerate(lines):
        line_list = line.split()
        assert len(line_list) == 5
        return_dict["index"][idx] = idx
        return_dict["symbol"][idx] = line_list[0]
        return_dict["charge"][idx] = float(line_list[-1])
    return pd.DataFrame(return_dict)

def read_chg_from_sdf(file):
    return_dict = {"index":{}, "symbol":{}, "charge":{}}
    mol = Chem.SDMolSupplier(file, removeHs=False)[0]
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        return_dict["index"][idx] = idx
        return_dict["symbol"][idx] = atom.GetSymbol()
        return_dict["charge"][idx] = atom.GetPropsAsDict()["molFileAlias"]
    return pd.DataFrame(return_dict)

def write_xyz_file(filename, molecules):
    """
    将多个分子的原子列表和位置写入一个 XYZ 文件。

    :param filename: 输出的 XYZ 文件名
    :param molecules: 包含多个分子的列表，每个分子是一个字典，格式如下：
                      {
                          'name': '分子名称',
                          'atomlist': ['H', 'O', 'C'],  # 原子类型列表
                          'positions': [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]  # 原子坐标列表
                      }
    """
    with open(filename, 'w') as f:
        # 写入每个分子的原子信息
        for molecule in molecules:
            atomlist = molecule['atomlist']
            positions = molecule['positions']
            num_atoms = len(atomlist)

            # 写入分子的标题行
            f.write(f"{num_atoms}\n")
            f.write(f"{molecule['name']}\n")

            # 写入原子信息
            for atom, position in zip(atomlist, positions):
                f.write(f"{atom} {' '.join(map(str, position))}\n")