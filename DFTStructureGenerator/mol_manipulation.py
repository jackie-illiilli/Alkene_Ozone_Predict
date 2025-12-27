import os, glob, math, shutil, pickle, random, copy
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
# from rdkit.Chem.Draw import MolDrawing, DrawingOptions
from rdkit.Geometry import Point3D
from . import FormatConverter, xtb_process
from . import Tool, logfile_process
op_dir = "../files/"

# DrawingOptions.bondLineWidth = 1.8
# DrawingOptions.atomLabelFontSize = 14
# Draw.DrawingOptions.includeAtomNumbers = True
atom_num = [None, "H", "He", "Li", "Be", "B", "C", "N", "O", "F",
            "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
            "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr"]


def mol_add_Hs(molfile):
    """给molfile文件补氢

    Args:
        molfile (_type_): _description_
    """    
    mol = Chem.MolFromMolFile(molfile)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    Chem.MolToMolFile(mol, molfile)


def group_atoms(coordinates, threshold):
    """
    只给出原子坐标的情况下，把原子按照设定的坐标间距阈值划分成若干组，适用于识别过渡态的两部分。
    
    参数：
    - coordinates: N x 3 的 NumPy 数组，表示 N 个原子的坐标。
    - threshold: 阈值，表示两个原子之间的距离不超过该值时，这两个原子应该被划分到同一组中。
    
    返回值：
    - 一个包含若干组的列表，每一组由若干原子的序号构成。
    """
    # 构建 KD 树
    from scipy.spatial import KDTree
    tree = KDTree(coordinates)

    # 查找近邻点
    groups = []
    visited = set()
    for i in range(len(coordinates)):
        if i in visited:
            continue
        group = []
        queue = [i]
        while queue:
            j = queue.pop(0)
            if j in visited:
                continue
            visited.add(j)
            group.append(j)
            neighbors = tree.query_ball_point(coordinates[j], threshold)
            queue.extend([k for k in neighbors if k not in visited])
        groups.append(group)

    return groups

def move(a):
    """将三位旋转平移矩阵a转化为四维旋转平移矩阵

    Args:
        a (iterable): 

    Returns:
        array: size = 4*4
    """    
    x, y, z = a[:3]
    return np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])

def rotation(a, sin, cos):
    """生成4维矩阵，与之相乘能沿a轴旋转sin，cos角度

    Args:
        a (array): 3维
        sin (float): 
        cos (float): 

    Returns:
        array: size 4*4
    """    
    a = np.array(a)
    a = a / np.sqrt(a @ a.T)
    u, v, w = a[:3]
    return np.array([[u * u + (1 - u * u) * cos, u * v * (1 - cos) - w * sin, u * w * (1 - cos) + v * sin, 0],
                     [u * v * (1 - cos) + w * sin, v * v + (1 - v * v)
                      * cos, v * w * (1 - cos) - u * sin, 0],
                     [u * w * (1 - cos) - v * sin, v * w * (1 - cos) +
                      u * sin, w * w + (1 - w * w) * cos, 0],
                     [0, 0, 0, 1]])

def trfm_rot(a, b, c, position=[], center_point=np.array([0, 0, 0])):
    """针对a,b,c三点，平移旋转使ab矢量平行于x轴，a,b,c在xoy平面上，a,b关于center_point对称

    Args:
        a (_type_): 3维坐标，最后会位于x轴负方向
        b (_type_): 3维坐标，最后会位于x轴正方向
        c (_type_): 3维坐标，调整后其y轴坐标为正，z轴坐标为0
        position (list, optional): 如果给定了position就会输出转化后的position，没有则输出旋转平移矩阵. Defaults to [].
        center_point (_type_, optional): _description_. Defaults to np.array([0, 0, 0]).

    Returns:
        array: new array (4*4)
        or tr_array, rot1_array, rot2_array, tr_array2
    """    
    zero_point = np.array([0, 0, 0])
    x, y, z = np.array(a), np.array(b), np.array(c)
    x_axis = np.array([1, 0, 0])
    mid_point = (x + y)/2
    tr_array_3d = zero_point - mid_point
    tr_array = move(tr_array_3d)
    x += tr_array_3d
    y += tr_array_3d
    z += tr_array_3d
    tr_array2 = move(center_point - zero_point)
    xy = y - x
    xy = xy / np.sqrt((xy @ xy))
    law_axis = np.cross(xy, x_axis)
    if (law_axis == np.array([0, 0, 0])).all() == False:
        law_axis /= np.sqrt((law_axis @ law_axis))
    sin = np.sqrt(xy[1] * xy[1] + xy[2] * xy[2])
    cos = xy[0]
    rot1_array = rotation(law_axis, sin, cos)
    oz = z - zero_point
    oz = rot1_array[:3, :3] @ oz
    oz[0] = 0
    oz = oz / np.sqrt((oz @ oz))
    sin = oz[2]
    cos = oz[1]
    rot2_array = rotation(-1 * x_axis, sin, cos)
    if len(position) == 0:
        return tr_array, rot1_array, rot2_array, tr_array2
    else:
        if len(position[0]) == 3:
            up_position = np.insert(position, 3, np.ones(len(position)), 1).T
        else:
            up_position = position.T
        up_position = tr_array @ up_position
        up_position = rot1_array @ up_position
        up_position = rot2_array @ up_position
        up_position = tr_array2 @ up_position
        up_position = up_position.T
    return up_position

def rot_mol(mol, axis=np.array([0, 1, 0]), sin=0, cos=-1):
    """旋转分子的所有构象，默认围绕y轴旋转180°

    Args:
        mol (_type_): _description_
        axis (array, optional): 转轴. Defaults to np.array([0, 1, 0]).
        sin (int, optional): _description_. Defaults to 0.
        cos (int, optional): _description_. Defaults to -1.

    Returns:
        mol: _description_
    """    
    # Define rotation corner is 180 degree
    react1 = copy.deepcopy(mol)
    for conformer in react1.GetConformers():
        position = conformer.GetPositions()
        up_position = np.insert(position, 3, np.ones(len(position)), 1).T
        matrix = rotation(axis, sin, cos)
        up_position = (matrix @ up_position).T
        for i, c in enumerate(up_position):
            conformer.SetAtomPosition(i, Point3D(c[0], c[1], c[2]))
    return react1


def move_mol(mol, array=np.array([0, 0, 1.5])):
    """移动分子的所有构象，默认向z轴移动1.5个单位（ene）

    Args:
        mol (_type_): _description_
        array (array, optional): _description_. Defaults to np.array([0, 0, 1.5]).

    Returns:
        _type_: _description_
    """    
    react1 = copy.deepcopy(mol)
    for conformer in react1.GetConformers():
        position = conformer.GetPositions()
        up_position = np.insert(position, 3, np.ones(len(position)), 1).T
        matrix = move(array)
        up_position = (matrix @ up_position).T
        for i, c in enumerate(up_position):
            conformer.SetAtomPosition(i, Point3D(c[0], c[1], c[2]))
    return react1

def check_double_bond_ZE(mol, old_position, new_position, ignore=[]):
    """检查前后双键的ZE是否发生了变化

    Args:
        mol (_type_): mol分子
        old_position (_type_): 前面的坐标
        new_position (_type_): 后面的坐标
        ignore (list, optional): 需要忽略的原子编号列表. Defaults to [].

    Returns:
        _type_: _description_
    """    
    is_error = []
    for bond in mol.GetBonds():
        # if bond.GetStereo() == Chem.BondStereo.STEREONONE:
        #     continue
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        continue_num = 0
        for each_ignore in ignore:
            if atom1.GetIdx() in each_ignore and atom2.GetIdx() in each_ignore:
                continue_num = 1
        if continue_num:
            continue
        if atom1.GetSymbol() not in ["C", "N"] or atom2.GetSymbol() not in ["C", "N"]:
            continue
        a_neighbor = [each.GetIdx() for each in mol.GetAtomWithIdx(atom1.GetIdx()).GetNeighbors() if each.GetIdx() != atom2.GetIdx() and each.GetAtomicNum() != 1]
        b_neighbor = [each.GetIdx() for each in mol.GetAtomWithIdx(atom2.GetIdx()).GetNeighbors() if each.GetIdx() != atom1.GetIdx() and each.GetAtomicNum() != 1]
        if len(a_neighbor) <= 0 or len(b_neighbor) <= 0:
            continue
        a_nei_id = a_neighbor[0]
        b_nei_id = b_neighbor[0]
        old_cos = Tool.get_torsion(old_position[a_nei_id], old_position[atom1.GetIdx()], old_position[atom2.GetIdx()], old_position[b_nei_id])
        if abs(old_cos) < 0.3:
            continue
        new_cos = Tool.get_torsion(new_position[a_nei_id], new_position[atom1.GetIdx()], new_position[atom2.GetIdx()], new_position[b_nei_id])
        if old_cos * new_cos < 0:
            is_error = [a_nei_id, atom1.GetIdx(), atom2.GetIdx(), b_nei_id]
            break
    return is_error

def eng_to_om(log_file:logfile_process.Logfile, diene_mol, new_dir="om", assert_title=None, write_gjf=True, distance=2.1, 
method='opt=modredundant freq b3lyp/6-31g(d) em=gd3bj', target_cos = -0.9396926, difreeze=True):
    """将后体优化log结果读取，拉长反应键，限制性优化

    Args:
        file_name (str, optional): 设定的. Defaults to "../file/test6/TSa_opt_0.log".
    """    
    new_name = os.path.split(log_file.file_dir)[-1].split(".")[0] + ".gjf"
    newfile = new_dir + "/" + new_name 
    title = log_file.title
    charge = log_file.charge
    symbol_list = log_file.symbol_list
    position = log_file.running_positions[-1]
    # title, charge, symbol_list, position = read_log(file_name, allow_unreal_freq=1)
    if assert_title:
        title = assert_title
    diene_num1, diene_num2, dieno_num1, dieno_num2, start = [
        int(each) for each in title][:5]
    diene_point1 = position[diene_num1]
    diene_point2 = position[diene_num2]
    dieno_point1 = position[dieno_num1 + start]
    dieno_point2 = position[dieno_num2 + start]

    # 平面定位
    new_position = trfm_rot(diene_point1, diene_point2, (dieno_point1 + dieno_point2)/2, position)
    diene_center_point = (
        new_position[diene_num1] + new_position[diene_num2])/2
    dieno_center_point = (
        new_position[dieno_num1 + start] + new_position[dieno_num2 + start])/2
    y_distance = distance - np.sqrt((diene_center_point - dieno_center_point)
                                @ (diene_center_point - dieno_center_point).T)# 所需拉伸的距离
    diene_position = new_position[:start]
    dieno_position = new_position[start:]
    # 拉伸
    move_matrix = move(np.array([0, y_distance, 0]))
    dieno_position = (move_matrix @ dieno_position.T).T
    # 旋转至同一平面
    diene_array = diene_position[diene_num1] - diene_position[diene_num2]
    dieno_array = dieno_position[dieno_num1] - dieno_position[dieno_num2]
    law_array = np.cross(diene_array[:3], dieno_array[:3])
    cos = (diene_array @ dieno_array)/(np.sqrt(diene_array @ diene_array) * np.sqrt(dieno_array @ dieno_array))
    sin = np.sqrt(1 - cos ** 2)
    rot_matrix = rotation(law_array, sin, cos)
    diene_position = (rot_matrix @ diene_position.T).T
    new_position = np.append(diene_position, dieno_position, axis=0)
    new_position = np.array([each[:3] for each in new_position])
    # 重新定位
    diene_point1 = new_position[diene_num1]
    diene_point2 = new_position[diene_num2]
    dieno_point1 = new_position[dieno_num1 + start]
    dieno_point2 = new_position[dieno_num2 + start]
    diene_atom_lists = diene_atom_Idx(diene_mol, select_diene=0)
    for each in diene_atom_lists:
        if each[0] == diene_num1 and each[-1] == diene_num2:
            atomb_id, atomc_id = each[1:3]
            break
    function_groups = set()
    function_groups.update(find_sustation_group(diene_mol, diene_num1, [atomb_id, atomc_id]))
    function_groups.update(find_sustation_group(diene_mol, diene_num2, [atomb_id, atomc_id]))

    diene_point3 = new_position[atomb_id]
    # print(diene_point3)
    new2_position = trfm_rot(diene_point1, diene_point2, diene_point3, new_position)
    # new2_position = np.array([each[:3] for each in new2_position])
    # 求角度
    diene_point1 = new2_position[diene_num1][:3]
    diene_point2 = new2_position[diene_num2][:3]
    dieno_point1 = new2_position[dieno_num1 + start][:3]
    dieno_point2 = new2_position[dieno_num2 + start][:3]
    diene_point3 = new2_position[atomb_id][:3]
    
    now_cos = Tool.get_torsion(diene_point3, diene_point1, diene_point2, dieno_point2)
    if now_cos < target_cos or dieno_point1[2] < 0:
        diene_position = new2_position[:start]
        dieno_position = new2_position[start:]
        if dieno_point1[2] > 0:
            cos = (target_cos * now_cos + np.sqrt(1-target_cos**2) * np.sqrt(1-now_cos ** 2))
        else:
            cos = (target_cos * now_cos - np.sqrt(1-target_cos**2) * np.sqrt(1-now_cos ** 2))
        sin = -np.sqrt(1-cos ** 2)
        rot_matrix = rotation(np.array([1, 0, 0]), sin, cos)
        dieno_position = (rot_matrix @ dieno_position.T).T
        new2_position = np.append(diene_position, dieno_position, axis=0)

        # 移动取代基
        if cos < 0.5:
            cos = 0.5
            sin = -np.sqrt(1-cos ** 2)
            rot_matrix = rotation(np.array([1, 0, 0]), sin, cos)
        new2_position = np.array(new2_position)
        sustation_position = new2_position[list(function_groups)]
        sustation_position = (rot_matrix @ sustation_position.T).T
        for sustation_id, position_id in enumerate(list(function_groups)):
            new2_position[position_id] = sustation_position[sustation_id]


    title = " ".join(str(each) for each in title)
    # savechk = new_name.split(".")[0]
    if write_gjf:
        if difreeze:
            FormatConverter.block_to_gjf(symbol_list, new2_position, newfile, charge, title,
                    method=method,
                    freeze=[[diene_num1 + 1, dieno_num1 + start + 1], [diene_num2 + 1, dieno_num2 + start + 1]], 
                    difreeze=[[atomb_id + 1, diene_num1 + 1, diene_num2 + 1, dieno_num1 + start + 1], [atomc_id + 1, diene_num2 + 1, diene_num1 + 1, dieno_num2 + start + 1]])
        else:
            FormatConverter.block_to_gjf(symbol_list, new2_position, newfile, charge, title,
                    method=method,
                    freeze=[[diene_num1 + 1, dieno_num1 + start + 1], [diene_num2 + 1, dieno_num2 + start + 1]])
    return title, symbol_list, new2_position, charge

def om_to_om2(log_file:logfile_process.Logfile, diene_mol, new_dir="om", assert_title=None, write_gjf=True, distance=1.9, 
method='opt=modredundant freq b3lyp/6-31g(d) em=gd3bj', target_cos = -np.sqrt(3)/2, difreeze = True):
    """将后体优化log结果读取，拉长反应键，限制性优化

    Args:
        file_name (str, optional): 设定的. Defaults to "../file/test6/TSa_opt_0.log".
    """    
    new_name = os.path.split(log_file.file_dir)[-1].split(".")[0] + ".gjf"
    newfile = new_dir + "/" + new_name 
    title = log_file.title
    charge = log_file.charge
    multiplicity = log_file.multiplicity
    symbol_list = log_file.symbol_list
    position = log_file.running_positions[-1]
    # title, charge, symbol_list, position = read_log(file_name, allow_unreal_freq=1)
    if assert_title:
        title = assert_title
    diene_num1, diene_num2, dieno_num1, dieno_num2, start = [
        int(each) for each in title][:5]

    # 重新定位
    diene_point1 = position[diene_num1]
    diene_point2 = position[diene_num2]
    dieno_point1 = position[dieno_num1 + start]
    dieno_point2 = position[dieno_num2 + start]
    diene_atom_lists = diene_atom_Idx(diene_mol, select_diene=0)
    for each in diene_atom_lists:
        if each[0] == diene_num1 and each[-1] == diene_num2:
            atomb_id, atomc_id = each[1:3]
            break
    function_groups = set()
    function_groups.update(find_sustation_group(diene_mol, diene_num1, [atomb_id, atomc_id]))
    function_groups.update(find_sustation_group(diene_mol, diene_num2, [atomb_id, atomc_id]))


    diene_point3 = position[atomb_id]
    # print(diene_point3)
    new2_position = trfm_rot(diene_point1, diene_point2, diene_point3, position)
    # new2_position = np.array([each[:3] for each in new2_position])
    # 求角度
    diene_point1 = new2_position[diene_num1][:3]
    diene_point2 = new2_position[diene_num2][:3]
    dieno_point1 = new2_position[dieno_num1 + start][:3]
    dieno_point2 = new2_position[dieno_num2 + start][:3]
    diene_point3 = new2_position[atomb_id][:3]
    
    now_cos = Tool.get_torsion(diene_point3, diene_point1, diene_point2, dieno_point2)
    # target_cos = -np.sqrt(2)/2
    if now_cos < target_cos:
        diene_position = new2_position[:start]
        dieno_position = new2_position[start:]
        cos = (target_cos * now_cos + np.sqrt(1-target_cos**2) * np.sqrt(1-now_cos ** 2))
        sin = -np.sqrt(1-cos ** 2)
        rot_matrix = rotation(np.array([1, 0, 0]), sin, cos)
        dieno_position = (rot_matrix @ dieno_position.T).T
        new2_position = np.append(diene_position, dieno_position, axis=0)




    title = " ".join(str(each) for each in title)
    # savechk = new_name.split(".")[0]
    if write_gjf:
        if difreeze:
            FormatConverter.block_to_gjf(symbol_list, new2_position, newfile, charge, multiplicity, title,
                    method=method,
                    freeze=[[diene_num1 + 1, dieno_num1 + start + 1], [diene_num2 + 1, dieno_num2 + start + 1]], 
                    difreeze=[[atomb_id + 1, diene_num1 + 1, diene_num2 + 1, dieno_num1 + start + 1], [atomc_id + 1, diene_num2 + 1, diene_num1 + 1, dieno_num2 + start + 1]])
        else:
            FormatConverter.block_to_gjf(symbol_list, new2_position, newfile, charge, multiplicity, title,
                    method=method,
                    freeze=[[diene_num1 + 1, dieno_num1 + start + 1], [diene_num2 + 1, dieno_num2 + start + 1]])
    return title, symbol_list, new2_position, charge



def om_to_ts(log_file:logfile_process.Logfile, write_gjf=True, new_dir="ts",
    method="opt=(calcfc,ts,noeigen) freq b3lyp/6-31g* em=gd3bj"):
    """将Gaussian限制性优化的输出文件转化为过渡态计算的输入文件或者输入信息

    Args:
        log_file (logfile_process.Logfile): log文件
        write_gjf (bool, optional): 是否直接写输入文件. Defaults to True.
        new_dir (str, optional): 过渡态输入文件的存放地址. Defaults to "ts".
        method (str, optional): #p行. Defaults to "opt=(calcfc,ts,noeigen) freq b3lyp/6-31g* em=gd3bj".

    Returns:
        _type_: _description_
    """    
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)
    new_name = os.path.split(log_file.file_dir)[-1].split(".")[0] + ".gjf"
    newfile = new_dir + "/" + new_name 
    title = log_file.title
    charge = log_file.charge
    symbol_list = log_file.symbol_list
    multiplicity = log_file.multiplicity
    position = log_file.running_positions[-1]
    title = " ".join(str(each) for each in title)
    multiplicity = log_file.multiplicity
    if write_gjf:
        FormatConverter.block_to_gjf(symbol_list, position, newfile, charge, multiplicity, title,
                    method=method,
                    freeze=[])
    
    return title, symbol_list, position, charge, multiplicity


def ts_to_irc(log_file:logfile_process.Logfile, new_dir):
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)
    new_names = [os.path.split(log_file.file_dir)[-1].split(".")[0] + "forward.gjf",
    os.path.split(log_file.file_dir)[-1].split(".")[0] + "reverse.gjf"]
    methods = ['irc=(calcfc,stepsize=30,maxpoints=30,forward, lqa) b3lyp/6-31g(d) em=gd3bj scrf=(smd,solvent=toluene) nosymm', 
    'irc=(calcfc,stepsize=30,maxpoints=30,reverse, lqa) b3lyp/6-31g(d) em=gd3bj scrf=(smd,solvent=toluene) nosymm']

    title = log_file.title
    title = " ".join(str(each) for each in title)
    charge = log_file.charge
    multiplicity = log_file.multiplicity
    symbol_list = log_file.symbol_list
    position = log_file.running_positions[-1]
    for new_name, method in zip(new_names, methods):
        newfile = new_dir + "/" + new_name 
        FormatConverter.block_to_gjf(symbol_list, position, newfile, charge, multiplicity, title,
                    method=method,
                    freeze=[])
    
    return title, symbol_list, position, charge

# def log_to_eng(file_name, new_file_name=None,method="b3lyp/6-311+g(d,p) em=gd3bj"):
#     if new_file_name == None:
#         file_dir, name = os.path.split(file_name)
#         name = name.split(".")[0]
#         newname = name + "_eng"
#         newfile = file_dir + "/" + newname + ".gjf"
#     else:
#         newfile = new_file_name
#     # file_name = "om_test2_0"
#     title, charge, symbol_list, position= read_log(file_name, allow_unreal_freq=1)
#     title = " ".join(str(each) for each in title)
#     block_to_gjf(symbol_list, position, newfile, charge, title,
#                  method=method)

def smiles2mol(smiles, conf_num=20, randomSeed = 0):
    """smiles 转化为mol，包含AddHs和Embed的3D结构生成

    Args:
        smiles (str): _description_

    Returns:
        mol: _description_
    """    
    mol = Chem.MolFromSmiles(smiles)
    if mol == None:
        print(smiles, "can't be read")
        return None
    Hmol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(Hmol, numConfs=conf_num, maxAttempts=100, randomSeed=randomSeed)
    try:
        AllChem.MMFFOptimizeMolecule(Hmol)
    except:
        pass
    
    return Hmol

def add_conformer(mol, conf_num=50):
    for _ in range(conf_num):
        b = copy.deepcopy(mol)
        Chem.AllChem.MMFFOptimizeMolecule(b)
        mol.AddConformer(b.GetConformer(0), assignId=True)
    return mol

def find_sustation_group(mol, mother_atom:int, ignore_atoms = []):
    """给定取代基原子mother_atom， 和母体原子ignore_atoms，找到取代基上面的所有原子

    Args:
        mol (_type_): _description_
        mother_atom (int): _description_
        ignore_atoms (list, optional): _description_. Defaults to [].

    Returns:
        _type_: _description_
    """    
    all_atoms = set()
    neighbor = [each.GetIdx() for each in mol.GetAtomWithIdx(mother_atom).GetNeighbors() if each.GetIdx() not in ignore_atoms]
    all_atoms.update(neighbor)
    for atom in neighbor:
        new_ignore_atoms = ignore_atoms + [atom]
        all_atoms.update(find_sustation_group(mol, atom, new_ignore_atoms))
    return all_atoms

def read_reactant(csvfile, index_lists=None):
    """读取指定的.csv,存储了diene/ene的编号（Index），smiles和能量

    Args:
        csvfile (_type_): _description_
        index_lists (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """    
    file = pd.read_csv(csvfile, index_col="Index").to_numpy()
    if index_lists ==None:
        smiles, energy = file[:, 0], file[:, -1]
    else:
        smiles, energy = file[index_lists, 0], file[index_lists, -1]
    return smiles, energy

def GetAtomIdxBetweenBonds(mol, bond1, bond2):
    """判断连着指定两根键的原子

    Args:
        mol (Chem.Mol): molecule
        bond1 (Chem.Bond): bond1
        bond2 (Chem.Bond): bond2

    Returns:
        int: atom_id
    """    
    atomlist1 = set([bond1.GetBeginAtomIdx(), bond1.GetEndAtomIdx()])
    atomlist2 = set([bond2.GetBeginAtomIdx(), bond2.GetEndAtomIdx()])
    atom = atomlist1 & atomlist2
    if len(atom) == 0:
        return None
    else:
        return list(atom)[-1]

def stretch_bond(coords, a, b, move_ids, x):
    """
    将分子中move_ids原子 沿着原子 a-b 的方向拉伸至当前键长的 x 倍。

    参数：
    coords: 一个 N×3 的二维数组，表示所有原子的坐标。
    a: 要拉伸的原子 a 的编号。
    b: 要拉伸的原子 b 的编号。
    x: 拉伸倍数。

    返回值：
    一个新的 N×3 的二维数组，表示拉伸后的所有原子的坐标。
    """
    # 计算 a 和 b 之间的距离和方向向量
    vec_ab = coords[b] - coords[a]
    dist_ab = np.linalg.norm(vec_ab)

    # 计算拉伸后的距离和拉伸比例，并计算新位置
    dist_new = dist_ab * x
    scale = dist_new / dist_ab
    new_pos_b = coords[a] + vec_ab * scale
    move_vec = new_pos_b - coords[b]

    # 构造新的坐标矩阵并返回
    new_coords = np.array(coords)
    for id, each in enumerate(new_coords):
        if id in move_ids:
            new_coords[id] = each + move_vec
    return new_coords

def shortest_distance(coords):
    """
    计算给定分子中任意两个原子之间的最短距离。

    参数：
    coords: N×3 的 numpy 数组，表示所有原子的坐标。

    返回值：
    分子中任意两个原子之间的最短距离。
    """
    # 计算原子间距离矩阵
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_mat = np.linalg.norm(diff, axis=-1)

    # 忽略对角线元素并取上三角部分
    upper_dist_mat = np.triu(dist_mat, k=1)

    # 找到最小的非零距离值作为最短距离
    min_dist = np.amin(upper_dist_mat[upper_dist_mat > 0])

    return min_dist

def error_improve(target_dir, mol_dir, file_name, dust_bin='dust_bin', improve_dir='improve', bond_attach_std='mol', maxcycles=0, method=None, bond_addition_function=None, bond_ignore_list=None, Inv_dir = 'Inv3', yqc_dir = 'yqc'):
    """与log_process模块配合，对Gaussian的输出文件进行错误识别和修改，处理虚频问题。
    会将修改后的输入文件和算错/有虚频的输出文件挪到指定文件夹。

    Args:
        target_dir (str): 根目录
        mol_dir (_type_): Mol分子对应的目录
        file_name (_type_): 根目录下存放有Gaussian输出文件的文件夹
        dust_bin (str, optional): 无药可救的输出文件的垃圾桶. Defaults to 'dust_bin'.
        improve_dir (str, optional): 修改后的Gaussian输入文件存放地址. Defaults to 'improve'.
    """    
    opt_file_dir = target_dir + "/" + file_name
    dust_bin_dir = target_dir + "/" + dust_bin
    improve_dir_ = target_dir + "/" + improve_dir
    Inv_dir_ = target_dir + "/" + Inv_dir
    yqc_dir_ = target_dir + "/" + yqc_dir
    mol_files = glob.glob(mol_dir + "/*.mol")
    
    for mol_file in mol_files:
        try:
            print("process %s" % mol_file, end='\r')
            log_files = glob.glob(opt_file_dir + "/" + os.path.split(mol_file)[-1].split(".")[0] + "*.log")
            if len(log_files) == 0:
                continue 
            for log_file in log_files:
                fail = 0
                with open(log_file, "r") as f:
                    lines = f.readlines()
                if len(lines) <= 15:
                    fail = 2
                else:
                    # try:
                    opt_log= logfile_process.Logfile(log_file, mol_file_dir=mol_file, bond_attach_std=bond_attach_std, bond_addition_function=bond_addition_function, bond_ignore_list=bond_ignore_list)
                    if opt_log.multiplicity <= 0:# or opt_log.S_2 < 0:
                        fail = 1
                    elif not opt_log.bond_attach:
                        fail = 1
                    elif opt_log.file_type == "OM" and opt_log.unreal_freq == 0:
                        print("%s may not be a right OM for unreal freq num of %d" % (opt_log.file_dir, opt_log.unreal_freq))
                        # fail = 1
                    elif opt_log.file_type == "TS":
                        if opt_log.unreal_freq >= 0 and opt_log.unreal_freq != 1: # or not opt_log.is_right_ts:
                            print("%s is not a right TS for unreal freq num of %d" % (opt_log.file_dir, opt_log.unreal_freq))
                            # if opt_log.unreal_freq == 0:
                            fail = 1
                            # else:
                            #     opt_log.is_normal_end = 0
                            #     opt_log.error_reason = ""
                        if opt_log.is_right_ts == False:
                            fail = 1
                    if opt_log.file_type == "IRC":
                        if opt_log.irc_result == False:
                            fail = 1
                        else:
                            continue
                    # except:
                    #     fail = 2 
                    
                if fail == 1:
                    new_log_name = dust_bin_dir + "/" + os.path.split(log_file)[-1] 
                    # new_log_name =  new_log_name.split(".")[0] + "%s.log" % opt_log.file_type
                    if not os.path.isdir(dust_bin_dir):
                        os.mkdir(dust_bin_dir) 
                    shutil.move(log_file, new_log_name)
                    continue
                if fail == 2:
                    new_log_name = improve_dir_ + "/" + os.path.split(log_file)[-1] 
                    if not os.path.isdir(improve_dir_):
                        os.mkdir(improve_dir_) 
                    shutil.move(log_file, new_log_name)
                    gjf_file = log_file.split(".")[0] + ".gjf"
                    new_gjf_name = improve_dir_ + "/" + os.path.split(gjf_file)[-1] 
                    shutil.move(gjf_file, new_gjf_name)
                    continue
                new_log_name = target_dir + '/' + improve_dir
                savechk = None
                readchk = None
                # if opt_log.file_type == "OM":
                #     savechk = os.path.split(log_file)[-1].split(".")[0]
                # if opt_log.file_type == "TS":
                #     readchk = os.path.split(log_file)[-1].split(".")[0]
                if not opt_log.normal_end:
                    opt_log.solve_error_logfile(new_log_name, Inv_dir=Inv_dir_, yqc_dir=yqc_dir_, savechk=savechk, readchk=readchk, maxcycles=maxcycles, method=method)
                elif opt_log.unreal_freq and opt_log.file_type not in ["OM", "TS"]:
                    opt_log.unreal_freq_improve(new_log_name, savechk=savechk, readchk=readchk, method=method)
        except:
            continue

