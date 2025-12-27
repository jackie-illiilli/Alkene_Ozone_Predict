# 一些常用工具的集合文档
import numpy as np
import pandas as pd
from rdkit import Chem
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def find_first_line(fileline, find_str, find_type="start"):
    assert find_type in ['start', 'end', 'all', 'in']
    if find_type == "start":
        line_function = lambda x, y: x.startswith(y)
    elif find_type == 'end':
        line_function = lambda x, y: x.endswith(y)
    elif find_type == 'all':
        line_function = lambda x, y: x == y
    else:
        line_function = lambda x, y: y in x

    for idx, line in enumerate(fileline):
        if line_function(line, find_str):
            return idx, line
    return None, None

def remove_same(lists):
    """Remove duplicates from a list of lists.

    Args:
        lists (list): A list of lists.

    Returns:
        list: A list of lists without duplicates.
    """
    seen = set()
    return_list = []
    for each in lists:
        # Convert each list to a tuple so it can be hashed
        t = tuple(each)
        # Check if the tuple is already in the set
        if t not in seen:
            # Add it to the set and the return list
            seen.add(t)
            return_list.append(each)
    return return_list

def save_load(file_name, smiles_lists = None):
    if smiles_lists != None and len(smiles_lists) != 0:
        with open(file_name, "wt") as f:
            for each in smiles_lists:
                f.write(each + "\n")
        return None
    else:
        smiles_lists = []
        with open(file_name, "rt") as f:
            for eachline in f.readlines():
                smiles_lists.append(eachline.strip("\n"))
        print(len(smiles_lists))
        smiles_lists_ = smiles_lists
        smiles_lists = []
        return smiles_lists_

def clean_nan(input_list):
    return np.nan_to_num(input_list, nan=0)

def get_array_cos(array1, array2):
    return array1 @ array2.T / (np.sqrt(array1 @ array1.T)
                              * np.sqrt(array2 @ array2.T))

def get_atoms_distance(atom_positionA, atom_positionB):
    """[summary]

    Args:
        atom_positionA (array): [description]
        atom_positionA (array): [description]

    Returns:
        array: distance
    """
    return np.sqrt(sum((atom_positionA - atom_positionB) ** 2))

def get_bond_angle(atom_positionA, atom_positionB, atom_positionC):
    conf = Chem.rdchem.Conformer(3)
    all_positions = [atom_positionA, atom_positionB, atom_positionC]
    for i in range(3):
        conf.SetAtomPosition(i, all_positions[i][:3])
    bond_angle = Chem.rdMolTransforms.GetAngleDeg(conf, 0, 1, 2)
    return bond_angle

def get_dihedral_angle(atom_positionA, atom_positionB, atom_positionC, atom_positionD):
    conf = Chem.rdchem.Conformer(4)
    all_positions = [atom_positionA, atom_positionB, atom_positionC, atom_positionD]
    for i in range(4):
        conf.SetAtomPosition(i, all_positions[i][:3])
    dihedral_angle = Chem.rdMolTransforms.GetDihedralDeg(conf, 0, 1, 2, 3)
    return dihedral_angle

def get_torsion(A, B, C, D):
    """计算A-B-C-D二面角的cos值

    Args:
        A (array): points
        B (array): 
        C (array): 
        D (array): 

    Returns:
        cos: _description_
    """    
    AB_AC = np.cross((B - A), (C - A))
    DB_DC = np.cross((B - D), (C - D))
    cos0 = get_array_cos(AB_AC, DB_DC)

    return cos0


def GetSpinMultiplicity(Mol, CheckMolProp = True):
    """From RDKitUtil.py
    Get spin multiplicity of a molecule. The spin multiplicity is either
    retrieved from 'SpinMultiplicity' molecule property or calculated from
    from the number of free radical electrons using Hund's rule of maximum
    multiplicity defined as 2S + 1 where S is the total electron spin. The
    total spin is 1/2 the number of free radical electrons in a molecule.

    Arguments:
        Mol (object): RDKit molecule object.
        CheckMolProp (bool): Check 'SpinMultiplicity' molecule property to
            retrieve spin multiplicity.

    Returns:
        int : Spin multiplicity.

    """
    
    Name = 'SpinMultiplicity'
    if (CheckMolProp and Mol.HasProp(Name)):
        return int(float(Mol.GetProp(Name)))

    # Calculate spin multiplicity using Hund's rule of maximum multiplicity...
    NumRadicalElectrons = 0
    for Atom in Mol.GetAtoms():
        NumRadicalElectrons += Atom.GetNumRadicalElectrons()

    TotalElectronicSpin = NumRadicalElectrons/2
    SpinMultiplicity = 2 * TotalElectronicSpin + 1
    
    return int(SpinMultiplicity)


def stablize_smileses(smiles_list):
    return [Chem.MolToSmiles(Chem.MolFromSmiles(each)) for each in smiles_list]

def plot_scatter_with_metrics(x, y, title=None, min_=-10, max_=10, figsize = (5,5)):
    """
    绘制散点图并显示回归性能指标
    
    参数：
    x: 一维数组类型，表示x轴数据。
    y: 一维数组类型，表示y轴数据。
    title: 字符串类型，表示图的标题。
    
    返回值：
    None
    
    """

    # 计算回归性能指标
    r2 = r2_score(x, y)
    mae = mean_absolute_error(x, y)
    mse = mean_squared_error(x, y)

    # 绘制散点图
    plt.figure(figsize=figsize, facecolor='white')
    plt.xlim(min_, max_)
    plt.ylim(min_, max_)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    # plt.xlabel("Real", fontsize=18)
    # plt.ylabel("Prediction", fontsize=18)
    if title != None:
        plt.title("%s\nR2:%.3f, MAE:%.3f, MSE:%.3f" % (title, r2, mae, mse), fontsize=24)
        # plt.title("%s"%title,fontsize=24)
    z = np.linspace(min_, max_, 10000)
    plt.plot(z, z)

    plt.scatter(x, y, marker="*", c="g")
    # 添加回归性能指标到图像的第二行
    
    # 显示图像
    plt.savefig("test.png", format="png", dpi=300, bbox_inches='tight')
    plt.show()

def calc_distribution2(y, eachsize=0.01, title=None, xlab=None, ylab="Count", y_max=None, y_min=None, figure_size = (4,3)):
    if y_max == None:    y_max = np.max(y)
    if y_min == None:    y_min = np.min(y)
    X = np.arange(y_min, y_max + eachsize, eachsize)
    des = [0 for each in X]
    z = (y - y_min)/eachsize
    for each in z:
        try:
            des[int(each)] += 1
        except:
            continue
    des = np.array(des)
    # des = des / len(y)
    
    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(111)
    ax.patch.set_alpha(0.0)
    plt.bar(X, des, width=eachsize/2, color="green")
    plt.xlim(y_min - eachsize, y_max + eachsize)
    plt.ylim(0, np.max(des) * 1.2)
    plt.xlabel(xlab, fontsize=30)
    plt.ylabel(ylab, fontsize=30)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    if title != None:
        plt.title = title
    plt.tight_layout()
    plt.savefig('test.svg', format='svg')
    plt.show()
    return des 


# Model 
def normalize_axis(arr, axis=0, mean=[], std=[]):
    """
    对数组中的某一维进行标准化（z-score normalization）
    
    参数：
    arr: ndarray，输入的数组
    axis: int，标准化的维度
    
    返回值：
    normalized_arr: ndarray，标准化后的数组
    """
    if len(mean) == 0 or len(mean) == 0:
        mean = np.mean(arr, axis=axis, keepdims=True)  # 计算均值
        std = np.std(arr, axis=axis, keepdims=True)  # 计算标准差
    normalized_arr = (arr - mean) / std  # 标准化
    normalized_arr = np.nan_to_num(normalized_arr, 0)
    return normalized_arr, mean, std

def half_ood_folds(react_data, n_folds=3, seed=0, ignore_B=0, ignore_N = 0, ignore_Cl = 0):
    np.random.seed(seed)
    all_B_ids = np.unique(react_data["B_Index"].to_numpy())
    all_N_ids = np.unique(react_data["N_Index"].to_numpy())
    all_Cl_ids = np.unique(react_data["Cl_Index"].to_numpy())
    Blist, Nlist, Cllist = np.arange(len(all_B_ids)), np.arange(len(all_N_ids)), np.arange(len(all_Cl_ids))
    np.random.shuffle(Blist)
    np.random.shuffle(Nlist)
    np.random.shuffle(Cllist)
    B_random_list ={id: int(each) % n_folds for id, each in zip(all_B_ids, Blist)}
    N_random_list ={id: int(each) % n_folds for id, each in zip(all_N_ids, Nlist)}
    Cl_random_list ={id: int(each) % n_folds for id, each in zip(all_Cl_ids, Cllist)}
    folds = [[[], []] for _ in range(n_folds)]
    for id, B_Index in enumerate(react_data["B_Index"]):
        N_Index = react_data['N_Index'][id]
        Cl_Index = react_data['Cl_Index'][id]
        B_idx, N_idx, Cl_idx = B_random_list[B_Index], N_random_list[N_Index], Cl_random_list[Cl_Index]
        for idx in range(n_folds):
            if (ignore_B or idx == B_idx) and (ignore_N or idx == N_idx) and (ignore_Cl or idx == Cl_idx):
                folds[idx][1].append(id)
                continue
            if (ignore_B or idx != B_idx) and (ignore_N or idx != N_idx) and (ignore_Cl or idx != Cl_idx):
                folds[idx][0].append(id)
    return folds

def draw_heatmap(x_labels, y_labels, values, title="None", figure_size=(40, 6), min_value = 0.0, max_value = 1):
    import seaborn as sns
    sns.set()
    # desc_labels = ["rdkit_mf", "morgan_mf", "rdkit_des", "modred_des"]
    # model_labels = ["GB", "XGB", "RF", "ET", "AdaB", "Line", "MLP"]
    # model_labels = ["no_product", "with_product", "with_structure"]

    plt.rcParams['font.sans-serif']='Arial'#设置中文显示，必须放在sns.set之后

    uniform_data = values #设置二维矩阵
    f, ax = plt.subplots(figsize=figure_size)
    annot_kws = {"fontsize": 30}
    #heatmap后第一个参数是显示值,vmin和vmax可设置右侧刻度条的范围,
    #参数annot=True表示在对应模块中注释值
    # 参数linewidths是控制网格间间隔
    #参数cbar是否显示右侧颜色条，默认显示，设置为None时不显示
    #参数cmap可调控热图颜色，具体颜色种类参考：https://blog.csdn.net/ztf312/article/details/102474190
    sns.heatmap(uniform_data, ax=ax,vmin=min_value,vmax=max_value,cmap='Blues',linewidths=2,cbar=True, annot=True,annot_kws=annot_kws, fmt='.3f')

    ax.set_title(title, fontsize=40) #plt.title('热图'),均可设置图片标题
    # ax.set_ylabel('descriptor', fontsize=10)  #设置纵轴标签
    # ax.set_xlabel('model', fontsize=10)  #设置横轴标签
    ax.set_xticklabels(x_labels, fontsize=30)
    ax.set_yticklabels(y_labels, fontsize=30)
    # #设置坐标字体方向，通过rotation参数可以调节旋转角度
    label_y =  ax.get_yticklabels()
    plt.setp(label_y, rotation=0, horizontalalignment='right')
    label_x =  ax.get_xticklabels()
    plt.setp(label_x, rotation=0, horizontalalignment='center')
    plt.savefig('test.svg', format='svg')
    plt.show()
    return plt
