# -*- coding: utf-8 -*-
"""


@author: Li-Cheng Xu
"""
import numpy as np
import glob, math
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from mordred import Calculator, descriptors
from dscribe.descriptors import ACSF, SOAP, LMBTR, MBTR
from ase import Atoms as ASE_Atoms
from traitlets import All
from tqdm import tqdm
from .Tool import clean_nan
period_table = Chem.GetPeriodicTable()


def getmorganfp(mol,radius=2,nBits=2048,useChirality=True):
    '''
    
    Parameters
    ----------
    mol : mol
        RDKit mol object.

    Returns
    -------
    mf_desc_map : ndarray
        ndarray of molecular fingerprint descriptors.

    '''
    fp = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol,radius=radius,nBits=nBits,useChirality=useChirality)
    return np.array(list(map(eval,list(fp.ToBitString()))))



def Mol2Atoms(mol):
    positions = mol.GetConformer().GetPositions()
    atom_types = [period_table.GetElementSymbol(atom.GetAtomicNum()) for atom in mol.GetAtoms()]
    atoms = ASE_Atoms(symbols=atom_types,positions=positions)
    return atoms



def getusidx(desc):
    usidx = []
    desc_scale = desc.max(axis=0) - desc.min(axis=0)
    for i in range(len(desc_scale)):
        if np.isnan(desc_scale[i]) or desc_scale[i] == 0:
            continue
        usidx.append(i)
    return usidx
class generate2Ddesc():

    def __init__(self, df, smiles_columns, mol_dir=None, has_product=True):
        
        '''
        
        Parameters
        ----------
        df : DataFrame(pandas)
            dataframe which needed process.
        smiles_columns : [columns_name] which contain SMILES

        Returns
        -------
        gen2d: obeject
        '''
        if smiles_columns != None:
            self.smi_set = np.concatenate([df[each].to_numpy() for each in smiles_columns])
            if mol_dir:
                mol_files = glob.glob(mol_dir + "/*.mol")
                mols = [Chem.MolFromMolFile(mol_file) for mol_file in mol_files]
                mol_set = []
                for smiles in self.smi_set:
                    mol_id = list(df[smiles_columns[0]].to_dict().values()).index(smiles)
                    mol_set.append(mols[mol_id])
                self.mol_set = mol_set

            else:
                self.mol_set = [Chem.MolFromSmiles(tmp_smi) for tmp_smi in self.smi_set]
            
    def getmorganfp(self,mol,radius=2,nBits=2048,useChirality=True):
        '''
        
        Parameters
        ----------
        mol : mol
            RDKit mol object.

        Returns
        -------
        mf_desc_map : ndarray
            ndarray of molecular fingerprint descriptors.

        '''
        fp = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol,radius=radius,nBits=nBits,useChirality=useChirality)
        return np.array(list(map(eval,list(fp.ToBitString()))))
    def calc_rdkit_desc(self):
        '''
        
        Parameters
        ----------
        

        Returns
        -------
        rdkit_desc_map : dict
            map of RDKit descriptors.

        '''
        descs = [desc_name[0] for desc_name in Descriptors._descList]
        desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(descs)
        rdkit_desc_map = {self.smi_set[i] : np.array(desc_calc.CalcDescriptors(self.mol_set[i])) for i in range(len(self.smi_set))}
        return rdkit_desc_map
    def calc_modred_desc(self):
        '''
        

        Returns
        -------
        modred_desc_map : dict
            map of modred descriptors.

        '''
        calc = Calculator(descriptors, ignore_3D=True)
        modred_df = calc.pandas(self.mol_set)
        modred_desc_map = {self.smi_set[i]: np.array(clean_nan([float(each) for each in modred_df.iloc[i]])) for i in range(len(self.smi_set))}
        return modred_desc_map
    def calc_morgan_mf(self):
        '''
    
        Returns
        -------
        mf_desc_map : dict
            map of molecular fingerprint descriptors.

        '''
        
        morgan_mf_map = {self.smi_set[i]:self.getmorganfp(self.mol_set[i]) for i in range(len(self.smi_set))}
        return morgan_mf_map
    
    def calc_rdkit_mf(self, size=2048):
        """return rdkit molecular fingerprint

        Returns:
            rdkit_mf_mp: {smiles: descriptor}
        """        
        rdkit_desc_map = {eachsmiles : np.array(clean_nan(Chem.RDKFingerprint(self.mol_set[i], fpSize=size))) for i, eachsmiles in enumerate(self.smi_set)}
        return rdkit_desc_map

class generate3Ddesc():   
    def __init__(self,df, smiles_columns=None, mol_dir=None):
        self.EmbedMol = True
        if smiles_columns != None:
            self.smi_set = np.concatenate([df[each].to_numpy() for each in smiles_columns])
            if mol_dir:
                mol_files = glob.glob(mol_dir + "/*.mol")
                mols = [Chem.MolFromMolFile(mol_file) for mol_file in mol_files]
                mol_set = []
                for smiles in self.smi_set:
                    mol_id = list(df[smiles_columns[0]].to_dict().values()).index(smiles)
                    mol_set.append(mols[mol_id])
                self.mol_set = mol_set
                self.EmbedMol = False
            else:
                self.mol_set = []
                for smi in self.smi_set:
                    mol = Chem.MolFromSmiles(smi)
                    mol = Chem.AddHs(mol)
                    Chem.AllChem.EmbedMolecule(mol)
                    Chem.AllChem.UFFOptimizeMolecule(mol)
                    self.mol_set.append()

    def getkeyatom(self,tmp_mol):
        pos = tmp_mol.GetConformer().GetPositions()
        atom_weights = np.array([tmp_atom.GetMass() for tmp_atom in tmp_mol.GetAtoms()]).reshape(-1,1)
        atom_weights = np.concatenate([atom_weights,atom_weights,atom_weights],axis=1)
        weight_cent = np.sum(pos*atom_weights,axis=0)/atom_weights.sum()
        key_atom = np.argmin(np.sum((pos - weight_cent)**2,axis=1))
        return key_atom
    def Mol2Atoms(self,mol):
        positions = mol.GetConformer().GetPositions()
        atom_types = [period_table.GetElementSymbol(atom.GetAtomicNum()) for atom in mol.GetAtoms()]
        atoms = ASE_Atoms(symbols=atom_types,positions=positions)
        return atoms
    def getkeyatomspecies(self):
        
        key_atoms = []
        atom_species = []
        for id, tmp_mol in enumerate(self.mol_set):
            if tmp_mol == None:
                print(self.smi_set[id])
            if self.EmbedMol:
                tmp_mol = Chem.AddHs(tmp_mol)
                AllChem.EmbedMolecule(tmp_mol)
                AllChem.MMFFOptimizeMolecule(tmp_mol)
            tmp_sym = [tmp_at.GetSymbol() for tmp_at in tmp_mol.GetAtoms()]
            atom_species += tmp_sym
            key_atoms.append(self.getkeyatom(tmp_mol))
        atom_species = list(set(atom_species)) 
        self.key_atoms = key_atoms
        self.atom_species = atom_species
        return key_atoms,atom_species
    
    def calc_acsf_desc(self):
        key_atoms,atom_species = self.getkeyatomspecies()
        rcut=6.0
        g2_params=[[1, 1], [1, 2], [1, 3]]
        g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]]
        calc = ACSF(species=atom_species,rcut=rcut,g2_params=g2_params,g4_params=g4_params)
        acsf_desc_map = {}
        for i, tmp_mol in tqdm(enumerate(self.mol_set), desc="Generate ACSF"):
            tmp_smi = self.smi_set[i]
            if self.EmbedMol:
                tmp_mol = Chem.AddHs(tmp_mol)
                AllChem.EmbedMolecule(tmp_mol)
                AllChem.MMFFOptimizeMolecule(tmp_mol)
            tmp_atom = self.Mol2Atoms(tmp_mol)
            tmp_desc = calc.create(tmp_atom, positions=[key_atoms[i]])
            tmp_desc = np.concatenate(tmp_desc)
            acsf_desc_map[tmp_smi] = tmp_desc

        return acsf_desc_map
    
    def calc_soap_desc(self):
        key_atoms,atom_species = self.getkeyatomspecies()
        rcut = 6.0
        nmax = 4
        lmax = 3
        calc = SOAP(species=atom_species,rcut=rcut,nmax=nmax,lmax=lmax)
        soap_desc_map = {}
        for i, tmp_mol in tqdm(enumerate(self.mol_set), desc="Generate SOAP"):
            tmp_smi = self.smi_set[i]
            if self.EmbedMol:
                tmp_mol = Chem.AddHs(tmp_mol)
                AllChem.EmbedMolecule(tmp_mol)
                AllChem.MMFFOptimizeMolecule(tmp_mol)
            tmp_atom = self.Mol2Atoms(tmp_mol)
            tmp_desc = calc.create(tmp_atom, positions=[key_atoms[i]])
            tmp_desc = np.concatenate(tmp_desc)
            soap_desc_map[tmp_smi] = tmp_desc

        return soap_desc_map
    def calc_lmbtr_desc(self):
        key_atoms,atom_species = self.getkeyatomspecies()
        k2={
                "geometry": {"function": "inverse_distance"},
                "grid": {"min": 0, "max": 1, "n": 10, "sigma": 0.1},
                "weighting": {"function": "exp", "scale": 0.5, "cutoff": 1e-3, 'threshold': 1e-3},
            }
        k3={
                "geometry": {"function": "cosine"},
                "grid": {"min": -1, "max": 1, "n": 10, "sigma": 0.1},
                "weighting": {"function": "exp", "scale": 0.5, "cutoff": 1e-3, 'threshold': 1e-3},
            }
        calc = LMBTR(species=atom_species,k2=k2,k3=k3,periodic=False)
        lmbtr_desc_map = {}
        for i, tmp_mol in tqdm(enumerate(self.mol_set), desc="Generate LMBTR"):
            tmp_smi = self.smi_set[i]
            if self.EmbedMol:
                tmp_mol = Chem.AddHs(tmp_mol)
                AllChem.EmbedMolecule(tmp_mol)
                AllChem.MMFFOptimizeMolecule(tmp_mol)
            tmp_atom = self.Mol2Atoms(tmp_mol)
            tmp_desc = calc.create(tmp_atom, positions=[key_atoms[i]])
            tmp_desc = np.concatenate(tmp_desc)
            lmbtr_desc_map[tmp_smi] = tmp_desc
        return lmbtr_desc_map

    def calc_mbtr_desc(self):
        key_atoms,atom_species = self.getkeyatomspecies()
        k1={
                "geometry": {"function": "atomic_number"},
                "grid": {"min": 0, "max": 8, "n": 10, "sigma": 0.1},
            }
        k2={
                "geometry": {"function": "inverse_distance"},
                "grid": {"min": 0, "max": 1, "n": 10, "sigma": 0.1},
                "weighting": {"function": "exp", "scale": 0.5, "cutoff": 1e-3, 'threshold': 1e-3},
            }
        k3={
                "geometry": {"function": "cosine"},
                "grid": {"min": -1, "max": 1, "n": 10, "sigma": 0.1},
                "weighting": {"function": "exp", "scale": 0.5, "cutoff": 1e-3, 'threshold': 1e-3},
            }
        calc = MBTR(species=atom_species,k1=k1,k2=k2,k3=k3,periodic=False,normalization="l2_each")
        mbtr_desc_map = {}
        for i, tmp_mol in tqdm(enumerate(self.mol_set), desc="Generate MBTR"):
            tmp_smi = self.smi_set[i]
            if self.EmbedMol:
                tmp_mol = Chem.AddHs(tmp_mol)
                AllChem.EmbedMolecule(tmp_mol)
                AllChem.MMFFOptimizeMolecule(tmp_mol)
            tmp_atom = self.Mol2Atoms(tmp_mol)
            tmp_desc = calc.create(tmp_atom)
            # tmp_desc = np.concatenate(tmp_desc)
            mbtr_desc_map[tmp_smi] = tmp_desc
        return mbtr_desc_map

    