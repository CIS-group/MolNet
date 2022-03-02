from keras.utils import to_categorical, Sequence
from rdkit import Chem
from rdkit.Chem import rdmolops, AllChem
import numpy as np
import pandas as pd
import pickle
import random
import math
import os

def one_hot(x, allowable_set):
    # If x is not in allowed set, use last index
    if x not in allowable_set:
        x = allowable_set[-1]

    return list(map(lambda s: x == s, allowable_set))


def find_distance(a1, num_atoms, bond_adj_list, max_distance=7):
    """Computes distances from provided atom.
      Parameters
      ----------
      a1: RDKit atom
        The source atom to compute distances from.
      num_atoms: int
        The total number of atoms.
      bond_adj_list: list of lists
        `bond_adj_list[i]` is a list of the atom indices that atom `i` shares a
        bond with. This list is symmetrical so if `j in bond_adj_list[i]` then `i in
        bond_adj_list[j]`.
      max_distance: int, optional (default 7)
        The max distance to search.
      Returns
      -------
      distances: np.ndarray
        Of shape `(num_atoms, max_distance)`. Provides a one-hot encoding of the
        distances. That is, `distances[i]` is a one-hot encoding of the distance
        from `a1` to atom `i`.
      """
    distance = np.zeros((num_atoms, max_distance))
    radial = 0
    # atoms `radial` bonds away from `a1`
    adj_list = set(bond_adj_list[a1])
    # atoms less than `radial` bonds away
    all_list = set([a1])
    while radial < max_distance:
        distance[list(adj_list), radial] = 1
        all_list.update(adj_list)
        # find atoms `radial`+1 bonds away
        next_adj = set()
        for adj in adj_list:
            next_adj.update(bond_adj_list[adj])
        adj_list = next_adj - all_list
        radial = radial + 1
    return np.array(distance)


def graph_distance(num_atoms, adj, max_distance=7):
    # Get canonical adjacency list
    bond_adj_list = [[] for mol_id in range(num_atoms)]
    for i in range(num_atoms):
        for j in range(num_atoms):
            if adj[i, j] == 1:
                bond_adj_list[i].append(j)
                bond_adj_list[j].append(i)
    
    distance_matrix = np.zeros([num_atoms, num_atoms, max_distance])
    
    for a1 in range(num_atoms):
        # distance is a matrix of 1-hot encoded distances for all atoms
        distance = find_distance(a1, num_atoms, bond_adj_list, max_distance=max_distance)
        distance_matrix[a1, :, :] = distance
        
    return distance_matrix
    

def degree_rotation_matrix(axis, degree):
    theta = degree / 180 * np.pi
    if axis == "x":
        r = np.array([1, 0, 0, 0, np.cos(theta), -np.sin(theta), 0, np.sin(theta), np.cos(theta)]).reshape([3, 3])
    elif axis == "y":
        r = np.array([np.cos(theta), 0, np.sin(theta), 0, 1, 0, -np.sin(theta), 0, np.cos(theta)]).reshape([3, 3])
    elif axis == "z":
        r = np.array([np.cos(theta), -np.sin(theta), 0, np.sin(theta), np.cos(theta), 0, 0, 0, 1]).reshape([3, 3])
    else:
        raise ValueError("Unsupported axis for rotation: {}".format(axis))

    return r


def optimize_conformer(m, algo="MMFF"):
    # print("Calculating {}: {} ...".format(Chem.MolToSmiles(m)))
    
    mol = Chem.AddHs(m)
    mol2 = Chem.Mol(mol)
    
    if algo == "ETKDG":
        # Landrum et al. DOI: 10.1021/acs.jcim.5b00654
        k = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        
        if k != 0:
            return None, None
    
    elif algo == "UFF":
        # Universal Force Field
        AllChem.EmbedMultipleConfs(mol, 50, pruneRmsThresh=0.5)
        try:
            arr = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=2000)
        except ValueError:
            return None, None
        
        if not arr:
            return None, None
        
        else:
            arr = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=20000)
            idx = np.argmin(arr, axis=0)[1]
            conf = mol.GetConformers()[idx]
            mol.RemoveAllConformers()
            mol.AddConformer(conf)
    
    elif algo == "MMFF":
        # Merck Molecular Force Field
        AllChem.EmbedMultipleConfs(mol, 50, pruneRmsThresh=0.5)
        try:
            arr = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000)
        except ValueError:
            return None, None
        
        if not arr:
            return None, None
        
        else:
            arr = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=20000)
            idx = np.argmin(arr, axis=0)[1]
            conf = mol.GetConformer(id=int(idx))
            # conf = mol.GetConformers()[idx]
            # mol.RemoveAllConformers()
            # mol.AddConformer(conf)
            mol2.AddConformer(conf)
            return Chem.RemoveHs(mol2)
    
    mol = Chem.RemoveHs(mol)
    
    return mol


class Dataset(object):
    def __init__(self, dataset, fold, iter, split, adj_cutoff=None, batch=128,
                 add_bf=False, model_type="3dgcn"):
        self.dataset = dataset
        self.path = "./dataset/{}.sdf".format(dataset)
        
        self.task = None  # "binary"
        self.target_name = None  # "active"
        self.max_atoms = 0

        self.split = split
        self.fold = fold
        self.iter = iter
        self.adj_cutoff = adj_cutoff
        
        self.add_bf = add_bf
        self.model_type = model_type

        self.batch = batch
        self.outputs = 1

        self.mols = []
        self.coords = []
        self.target = []
        self.x, self.c, self.y = {}, {}, {}

        self.use_overlap_area = False
        
        self.use_full_ec = False
        self.use_atom_symbol = True
        self.use_degree = True
        self.use_hybridization = True
        self.use_implicit_valence = True
        self.use_partial_charge = False
        self.use_formal_charge = True
        self.use_ring_size = True
        self.use_hydrogen_bonding = True
        self.use_acid_base = True
        self.use_aromaticity = True
        self.use_chirality = True
        self.use_num_hydrogen = True

        # Load data
        self.load_dataset(iter)

        # Calculate number of features
        if self.add_bf:
            print("Add bond features")
            mp = MPGenerator_bond([], [], [], 1,
                                  model_type=self.model_type,
                                  use_overlap_area=self.use_overlap_area,
                                  use_full_ec=self.use_full_ec,
                                  use_atom_symbol=self.use_atom_symbol,
                                  use_degree=self.use_degree,
                                  use_hybridization=self.use_hybridization,
                                  use_implicit_valence=self.use_implicit_valence,
                                  use_partial_charge=self.use_partial_charge,
                                  use_formal_charge=self.use_formal_charge,
                                  use_ring_size=self.use_ring_size,
                                  use_hydrogen_bonding=self.use_hydrogen_bonding,
                                  use_acid_base=self.use_acid_base,
                                  use_aromaticity=self.use_aromaticity,
                                  use_chirality=self.use_chirality,
                                  use_num_hydrogen=self.use_num_hydrogen)
        else:
            mp = MPGenerator([], [], [], 1,
                             use_full_ec=self.use_full_ec,
                             use_atom_symbol=self.use_atom_symbol,
                             use_degree=self.use_degree,
                             use_hybridization=self.use_hybridization,
                             use_implicit_valence=self.use_implicit_valence,
                             use_partial_charge=self.use_partial_charge,
                             use_formal_charge=self.use_formal_charge,
                             use_ring_size=self.use_ring_size,
                             use_hydrogen_bonding=self.use_hydrogen_bonding,
                             use_acid_base=self.use_acid_base,
                             use_aromaticity=self.use_aromaticity,
                             use_chirality=self.use_chirality,
                             use_num_hydrogen=self.use_num_hydrogen)

        self.num_features = mp.get_num_features()

        # Normalize
        if self.task == "regression":
            self.mean = np.mean(self.y["train"])
            self.std = np.std(self.y["train"])
    
            self.y["train"] = (self.y["train"] - self.mean) / self.std
            self.y["test"] = (self.y["test"] - self.mean) / self.std
            try:
                self.y["valid"] = (self.y["valid"] - self.mean) / self.std
            except:
                print("Cannot normalize")
        else:
            self.mean = 0
            self.std = 1

    def load_dataset(self, iter):
        # Dataset parameters
        if self.dataset == "bace_reg" or self.dataset == "delaney" or self.dataset == "freesolv":
            self.task = "regression"
            self.target_name = "target"
            self.loss = "mse"

        elif self.dataset == "bace_cla" or self.dataset == "hiv":
            self.task = "binary"
            self.target_name = "active"
            self.loss = "binary_crossentropy"

        elif self.dataset == "tox21":
            self.target_name = "NR-ER"

        # elif self.dataset == "tox21":  # Multitask tox21
        #     self.target_name = ["NR-Aromatase", "NR-AR", "NR-AR-LBD", "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma", "NR-AhR",
        #                    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]

        else:
            pass

        # Load file
        x, c, y = [], [], []
        try:
            mols = Chem.SDMolSupplier(self.path)
        except:
            mols = Chem.SDMolSupplier("../dataset/{}.sdf".format(self.dataset))

        for mol in mols:
            if mol is not None:
                if mol.GetNumAtoms() > 200:
                    continue
                # Multitask
                if type(self.target_name) is list:
                    y.append([float(mol.GetProp(t)) if t in mol.GetPropNames() else -1 for t in self.target_name])
                    self.outputs = len(self.target_name)

                # Single task
                elif self.target_name in mol.GetPropNames():
                    _y = float(mol.GetProp(self.target_name))
                    if _y == -1:
                        continue
                    else:
                        y.append(_y)

                else:
                    continue

                x.append(mol)
                c.append(mol.GetConformer().GetPositions())
        assert len(x) == len(y)

        # Filter and update maximum number of atoms
        new_x, new_c, new_y = [], [], []
        if self.max_atoms > 0:
            for mol, coo, tar in zip(x, c, y):
                if mol.GetNumAtoms() <= self.max_atoms:
                    new_x.append(mol)
                    new_c.append(coo)
                    new_y.append(tar)

            x = new_x
            c = new_c
            y = new_y

        else:
            for mol, tar in zip(x, y):
                self.max_atoms = max(self.max_atoms, mol.GetNumAtoms())

        if self.task != "regression":
            self.mols, self.coords, self.target = np.array(x), np.array(c), np.array(y, dtype=int)
        else:
            self.mols, self.coords, self.target = np.array(x), np.array(c), np.array(y)

        # Shuffle data
        np.random.seed(25)
        # np.random.seed(100) -> before 1012
        idx = np.random.permutation(len(self.mols))
        self.mols, self.coords, self.target = self.mols[idx], self.coords[idx], self.target[idx]

        # Split data
        if self.split == "random" or self.fold == 1:
            print('random split')
            # Split data
            spl1 = int(len(self.mols) * 0.2)
            spl2 = int(len(self.mols) * 0.1)
    
            self.x = {"train": self.mols[spl1:],
                      "valid": self.mols[spl2:spl1],
                      "test": self.mols[:spl2]}
            self.c = {"train": self.coords[spl1:],
                      "valid": self.coords[spl2:spl1],
                      "test": self.coords[:spl2]}
            self.y = {"train": self.target[spl1:],
                      "valid": self.target[spl2:spl1],
                      "test": self.target[:spl2]}

        elif self.split == "":
            print('cross-validation split')
            # Split data
            len_test = int(len(self.mols) / self.fold)
    
            if iter == (self.fold - 1):
                self.x = {"train": self.mols[:len_test * iter],
                          "test": self.mols[len_test * iter:]}
                self.c = {"train": self.coords[:len_test * iter],
                          "test": self.coords[len_test * iter:]}
                self.y = {"train": self.target[:len_test * iter],
                          "test": self.target[len_test * iter:]}
            else:
                self.x = {"train": np.concatenate((self.mols[:len_test * iter], self.mols[len_test * (iter + 1):])),
                          "test": self.mols[len_test * iter:len_test * (iter + 1)]}
                self.c = {"train": np.concatenate((self.coords[:len_test * iter], self.coords[len_test * (iter + 1):])),
                          "test": self.coords[len_test * iter:len_test * (iter + 1)]}
                self.y = {"train": np.concatenate((self.target[:len_test * iter], self.target[len_test * (iter + 1):])),
                          "test": self.target[len_test * iter:len_test * (iter + 1)]}

        elif self.split == "stratified":
            # Dataset parameters
            if self.dataset == "bace_cla" or self.dataset == "hiv":
                # Shuffle data
                idx_inactive = np.squeeze(np.argwhere(self.target == 0))  # 825
                idx_active = np.squeeze(np.argwhere(self.target == 1))  # 653

                #np.random.seed(100)  # 37
                #np.random.shuffle(idx_inactive)
                #np.random.seed(100)  # 37
                #np.random.shuffle(idx_active)
        
                # Split data
                len_inactive = int(len(idx_inactive) / self.fold)
                len_active = int(len(idx_active) / self.fold)
        
                if iter == (self.fold - 1):
                    test_idx = np.append(idx_inactive[len_inactive * iter:],
                                         idx_active[len_active * iter:])
                    train_idx = np.append(idx_inactive[:len_inactive * iter],
                                          idx_active[:len_active * iter])
        
                else:
                    test_idx = np.append(idx_inactive[len_inactive * iter:len_inactive * (iter + 1)],
                                         idx_active[len_active * iter:len_active * (iter + 1)])
                    train_idx = np.append(np.append(idx_inactive[:len_inactive * iter], idx_inactive[len_inactive * (iter + 1):]),
                                          np.append(idx_active[:len_active * iter], idx_active[len_active * (iter + 1):]))
        
                test_idx = np.random.permutation(test_idx)
                train_idx = np.random.permutation(train_idx)
                
                #print('permut')
                assert len(test_idx) > 0 and len(train_idx) > 0
        
                self.x = {"train": self.mols[train_idx], "test": self.mols[test_idx]}
                self.c = {"train": self.coords[train_idx], "test": self.coords[test_idx]}
                self.y = {"train": self.target[train_idx], "test": self.target[test_idx]}
                
                print('Finish stratified splitting')
    
            else:
                print("Cannot stratified splitting")

        else:
            print('Select proper dataset splitting type.')

        assert len(self.x["train"]) == len(self.c["train"]) == len(self.y["train"])
        assert len(self.x["test"]) == len(self.c["test"]) == len(self.y["test"])
        try:
            assert len(self.x["valid"]) == len(self.c["valid"]) == len(self.y["valid"])
            print('Train/valid/test dataset size: ', len(self.y["train"]), len(self.y["valid"]), len(self.y["test"]))

        except:
            # print("No valid dataset")
            print('Train/test dataset size: ', len(self.y["train"]), len(self.y["test"]))

    def save_dataset(self, path, pred=None, target="test", filename=None):
        mols = []
        for idx, (x, c, y) in enumerate(zip(self.x[target], self.c[target], self.y[target])):
            x.SetProp("true", str(y * self.std + self.mean))
            if pred is not None:
                x.SetProp("pred", str(pred[idx][0] * self.std + self.mean))
            mols.append(x)

        if filename is not None:
            w = Chem.SDWriter(path + filename + ".sdf")
        else:
            w = Chem.SDWriter(path + target + ".sdf")
        for mol in mols:
            if mol is not None:
                w.write(mol)

    def replace_dataset(self, path, subset="test", target_name="target"):
        x, c, y = [], [], []
        mols = Chem.SDMolSupplier(path)

        for mol in mols:
            if mol is not None:
                # Multitask
                if type(target_name) is list:
                    y.append([float(mol.GetProp(t)) if t in mol.GetPropNames() else -1 for t in target_name])
                    self.outputs = len(self.target_name)

                # Singletask
                elif target_name in mol.GetPropNames():
                    _y = float(mol.GetProp(target_name))

                    if _y == -1:
                        continue
                    else:
                        y.append(_y)

                else:
                    y.append(float(0))
                x.append(mol)
                c.append(mol.GetConformer().GetPositions())

        # Normalize
        x = np.array(x)
        c = np.array(c)
        y = (np.array(y) - self.mean) / self.std

        self.x[subset] = x
        self.c[subset] = c
        self.y[subset] = y.astype(int) if self.task != "regression" else y

    def set_features(self, use_overlap_area=False, use_full_ec=True,
                     use_atom_symbol=True, use_degree=True, use_hybridization=True, use_implicit_valence=True,
                     use_partial_charge=False, use_formal_charge=True, use_ring_size=True, use_hydrogen_bonding=True,
                     use_acid_base=True, use_aromaticity=True, use_chirality=True, use_num_hydrogen=True):
        
        self.use_overlap_area = use_overlap_area
    
        self.use_full_ec = use_full_ec
        self.use_atom_symbol = use_atom_symbol
        self.use_degree = use_degree
        self.use_hybridization = use_hybridization
        self.use_implicit_valence = use_implicit_valence
        self.use_partial_charge = use_partial_charge
        self.use_formal_charge = use_formal_charge
        self.use_ring_size = use_ring_size
        self.use_hydrogen_bonding = use_hydrogen_bonding
        self.use_acid_base = use_acid_base
        self.use_aromaticity = use_aromaticity
        self.use_chirality = use_chirality
        self.use_num_hydrogen = use_num_hydrogen

        # Calculate number of features
        if self.add_bf:
            mp = MPGenerator_bond([], [], [], 1,
                                  model_type=self.model_type,
                                  use_overlap_area=self.use_overlap_area,
                                  use_full_ec=self.use_full_ec,
                                  use_atom_symbol=self.use_atom_symbol,
                                  use_degree=self.use_degree,
                                  use_hybridization=self.use_hybridization,
                                  use_implicit_valence=self.use_implicit_valence,
                                  use_partial_charge=self.use_partial_charge,
                                  use_formal_charge=self.use_formal_charge,
                                  use_ring_size=self.use_ring_size,
                                  use_hydrogen_bonding=self.use_hydrogen_bonding,
                                  use_acid_base=self.use_acid_base,
                                  use_aromaticity=self.use_aromaticity,
                                  use_chirality=self.use_chirality,
                                  use_num_hydrogen=self.use_num_hydrogen)
            self.num_bond_features = mp.num_bond_features
            
        else:
            mp = MPGenerator([], [], [], 1,
                             use_full_ec=self.use_full_ec,
                             use_atom_symbol=self.use_atom_symbol,
                             use_degree=self.use_degree,
                             use_hybridization=self.use_hybridization,
                             use_implicit_valence=self.use_implicit_valence,
                             use_partial_charge=self.use_partial_charge,
                             use_formal_charge=self.use_formal_charge,
                             use_ring_size=self.use_ring_size,
                             use_hydrogen_bonding=self.use_hydrogen_bonding,
                             use_acid_base=self.use_acid_base,
                             use_aromaticity=self.use_aromaticity,
                             use_chirality=self.use_chirality,
                             use_num_hydrogen=self.use_num_hydrogen)

        self.num_features = mp.get_num_features()

    def generator(self, target, task=None):
        if self.add_bf:
            return MPGenerator_bond(self.x[target], self.c[target], self.y[target], self.batch,
                                    task=task if task is not None else self.task,
                                    model_type=self.model_type,
                                    num_atoms=self.max_atoms,
                                    use_full_ec=self.use_full_ec,
                                    use_overlap_area=self.use_overlap_area,
                                    use_atom_symbol=self.use_atom_symbol,
                                    use_degree=self.use_degree,
                                    use_hybridization=self.use_hybridization,
                                    use_implicit_valence=self.use_implicit_valence,
                                    use_partial_charge=self.use_partial_charge,
                                    use_formal_charge=self.use_formal_charge,
                                    use_ring_size=self.use_ring_size,
                                    use_hydrogen_bonding=self.use_hydrogen_bonding,
                                    use_acid_base=self.use_acid_base,
                                    use_aromaticity=self.use_aromaticity,
                                    use_chirality=self.use_chirality,
                                    use_num_hydrogen=self.use_num_hydrogen)
        else:
            return MPGenerator(self.x[target], self.c[target], self.y[target], self.batch,
                               adj_cutoff=self.adj_cutoff,
                               task=task if task is not None else self.task,
                               num_atoms=self.max_atoms,
                               use_full_ec=self.use_full_ec,
                               use_atom_symbol=self.use_atom_symbol,
                               use_degree=self.use_degree,
                               use_hybridization=self.use_hybridization,
                               use_implicit_valence=self.use_implicit_valence,
                               use_partial_charge=self.use_partial_charge,
                               use_formal_charge=self.use_formal_charge,
                               use_ring_size=self.use_ring_size,
                               use_hydrogen_bonding=self.use_hydrogen_bonding,
                               use_acid_base=self.use_acid_base,
                               use_aromaticity=self.use_aromaticity,
                               use_chirality=self.use_chirality,
                               use_num_hydrogen=self.use_num_hydrogen)


class MPGenerator(Sequence):
    def __init__(self, x_set, c_set, y_set, batch, task="binary", num_atoms=0, adj_cutoff=None,
                 use_full_ec=False,
                 use_degree=True, use_hybridization=True, use_implicit_valence=True, use_partial_charge=False,
                 use_formal_charge=True, use_ring_size=True, use_hydrogen_bonding=True, use_acid_base=True,
                 use_aromaticity=True, use_chirality=True, use_num_hydrogen=True, use_atom_symbol=True):
        self.x, self.c, self.y = x_set, c_set, y_set

        self.batch = batch
        self.task = task
        self.num_atoms = num_atoms
        self.adj_cutoff = adj_cutoff

        self.use_full_ec = use_full_ec
        self.use_atom_symbol = use_atom_symbol
        self.use_degree = use_degree
        self.use_hybridization = use_hybridization
        self.use_implicit_valence = use_implicit_valence
        self.use_partial_charge = use_partial_charge
        self.use_formal_charge = use_formal_charge
        self.use_ring_size = use_ring_size
        self.use_hydrogen_bonding = use_hydrogen_bonding
        self.use_acid_base = use_acid_base
        self.use_aromaticity = use_aromaticity
        self.use_chirality = use_chirality
        self.use_num_hydrogen = use_num_hydrogen

        self.hydrogen_donor = Chem.MolFromSmarts("[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]")
        self.hydrogen_acceptor = Chem.MolFromSmarts(
            "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]")
        self.acidic = Chem.MolFromSmarts("[$([C,S](=[O,S,P])-[O;H1,-1])]")
        self.basic = Chem.MolFromSmarts(
            "[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))])]")

        self.use_bond_adj = True

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch:(idx + 1) * self.batch]
        batch_c = self.c[idx * self.batch:(idx + 1) * self.batch]
        batch_y = self.y[idx * self.batch:(idx + 1) * self.batch]

        if self.task == "category":
            return self.tensorize(batch_x, batch_c), to_categorical(batch_y)
        elif self.task == "binary":
            return self.tensorize(batch_x, batch_c), np.array(batch_y, dtype=int)
        elif self.task == "regression":
            return self.tensorize(batch_x, batch_c), np.array(batch_y, dtype=float)
        elif self.task == "input_only":
            return self.tensorize(batch_x, batch_c)

    def tensorize(self, batch_x, batch_c):
        atom_tensor = np.zeros((len(batch_x), self.num_atoms, self.get_num_features()))
        adjm_tensor = np.zeros((len(batch_x), self.num_atoms, self.num_atoms))
        posn_tensor = np.zeros((len(batch_x), self.num_atoms, self.num_atoms, 3))

        for mol_idx, mol in enumerate(batch_x):
            Chem.RemoveHs(mol)
            mol_atoms = mol.GetNumAtoms()

            # Atom features
            atom_tensor[mol_idx, :mol_atoms, :] = self.get_atom_features(mol)

            # Adjacency matrix
            if self.adj_cutoff is not None:
                # Crop atoms
                distance = np.array(rdmolops.Get3DDistanceMatrix(mol))
                adjms = np.where(distance <= self.adj_cutoff, np.ones_like(distance), np.zeros_like(distance))
            else:
                adjms = np.array(rdmolops.GetAdjacencyMatrix(mol), dtype="float")

            ###
            if self.use_bond_adj:
                # adj based degree of bond
                m = np.zeros([mol_atoms, mol_atoms, 4], dtype=int)
                for i in range(mol.GetNumAtoms()):
                    for j in range(mol.GetNumAtoms()):
                        bond = mol.GetBondBetweenAtoms(i, j)
                        if bond is not None:
                            bond_type = bond.GetBondType()
                            # bond.GetBeginAtom(), bond.GetEndAtom()
                            
                            m[i, j] = np.squeeze(np.array([
                                bond_type == Chem.rdchem.BondType.SINGLE, bond_type == Chem.rdchem.BondType.DOUBLE,
                                bond_type == Chem.rdchem.BondType.TRIPLE, bond_type == Chem.rdchem.BondType.AROMATIC]))

                # adj based degree of bond
                bond_single = np.where(m[:, :, 0] == 1,
                                       np.full_like(adjms, 1), np.zeros_like(adjms))
                bond_double = np.where(m[:, :, 1] == 1,
                                       np.full_like(adjms, 2), np.zeros_like(adjms))
                bond_triple = np.where(m[:, :, 2] == 1,
                                       np.full_like(adjms, 3), np.zeros_like(adjms))
                bond_aromatic = np.where(m[:, :, 3] == 1,
                                         np.full_like(adjms, 1.5), np.zeros_like(adjms))

                adjms = bond_single + bond_double + bond_triple + bond_aromatic
            
            # Normalize adjacency matrix by D^(-1/2) * A_hat * D^(-1/2), Kipf et al. 2016
            adjms += np.eye(mol_atoms)
            degree = np.array(adjms.sum(1))
            deg_inv_sqrt = np.power(degree, -0.5)
            deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.
            deg_inv_sqrt = np.diag(deg_inv_sqrt)

            adjms = np.matmul(np.matmul(deg_inv_sqrt, adjms), deg_inv_sqrt)

            adjm_tensor[mol_idx, : mol_atoms, : mol_atoms] = adjms

            # Relative position matrix
            for atom_idx in range(mol_atoms):
                pos_c = batch_c[mol_idx][atom_idx]

                for neighbor_idx in range(mol_atoms):
                    pos_n = batch_c[mol_idx][neighbor_idx]

                    # Direction should be Neighbor -> Center
                    n_to_c = [pos_c[0] - pos_n[0], pos_c[1] - pos_n[1], pos_c[2] - pos_n[2]]
                    posn_tensor[mol_idx, atom_idx, neighbor_idx, :] = n_to_c

        return [atom_tensor, adjm_tensor, posn_tensor]

    def get_num_features(self):
        mol = Chem.MolFromSmiles("CC")
        return len(self.get_atom_features(mol)[0])
    
    def get_atom_features(self, mol):
        AllChem.ComputeGasteigerCharges(mol)
        Chem.AssignStereochemistry(mol)

        hydrogen_donor_match = sum(mol.GetSubstructMatches(self.hydrogen_donor), ())
        hydrogen_acceptor_match = sum(mol.GetSubstructMatches(self.hydrogen_acceptor), ())
        acidic_match = sum(mol.GetSubstructMatches(self.acidic), ())
        basic_match = sum(mol.GetSubstructMatches(self.basic), ())

        ring = mol.GetRingInfo()

        m = []
        for atom_idx in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(atom_idx)

            o = []
            if self.use_full_ec:
                possible_atoms = ['Al', 'Ge', 'Cl', 'N', 'S', 'P', 'Br', 'I', 'F', 'O', 'Si', 'As', 'C', 'B', 'Se',
                                  'Na']  # 16
                possible_group = [1, 13, 14, 15, 16, 17]
                possible_period = [2, 3, 4, 5]
                possible_block = ['s', 'p']
                block_dict = {'Na': 's1',
                              'B': 'p1', 'Al': 'p1',
                              'C': 'p2', 'Si': 'p2', 'Ge': 'p2',
                              'N': 'p3', 'P': 'p3', 'As': 'p3',
                              'O': 'p4', 'S': 'p4', 'Se': 'p4',
                              'F': 'p5', 'Cl': 'p5', 'Br': 'p5', 'I': 'p5'}

                ptable = pd.read_pickle('./dataset/ptable.pickle')
                symbol, group, period, _ = (ptable.loc[ptable['symbol'] == atom.GetSymbol()]).iloc[0]

                # ['s1', 's2', 'p1', 'p2', 'p3', 'p4', 'p5']
                fulleconf = [0 for _ in range(8 * 4)]
                for p in range(period - 2):
                    fulleconf[8 * p:8 * (p + 1)] = [0, 1, 0, 0, 0, 0, 0, 1]
    
                if symbol in ['Na', 'K']:
                    fulleconf[8 * (period - 2):8 * (period - 1)] = [1, 0, 0, 0, 0, 0, 0, 0]
                elif symbol in ['B', 'Al', 'Ga']:
                    fulleconf[8 * (period - 2):8 * (period - 1)] = [0, 1, 1, 0, 0, 0, 0, 0]
                elif symbol in ['C', 'Si', 'Ge', 'Sn']:
                    fulleconf[8 * (period - 2):8 * (period - 1)] = [0, 1, 0, 1, 0, 0, 0, 0]
                elif symbol in ['N', 'P', 'As']:
                    fulleconf[8 * (period - 2):8 * (period - 1)] = [0, 1, 0, 0, 1, 0, 0, 0]
                elif symbol in ['O', 'S', 'Se']:
                    fulleconf[8 * (period - 2):8 * (period - 1)] = [0, 1, 0, 0, 0, 1, 0, 0]
                elif symbol in ['F', 'Cl', 'Br', 'I']:
                    fulleconf[8 * (period - 2):8 * (period - 1)] = [0, 1, 0, 0, 0, 0, 1, 0]
    
                o += fulleconf
            
            o += one_hot(atom.GetSymbol(), ['C', 'O', 'N', 'S', 'Cl', 'F', 'Br', 'P',
                                            'I', 'Si', 'B', 'Na', 'Sn', 'Se', 'other']) if self.use_atom_symbol else []
            o += one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) if self.use_degree else []
            o += one_hot(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP,
                                                   Chem.rdchem.HybridizationType.SP2,
                                                   Chem.rdchem.HybridizationType.SP3,
                                                   Chem.rdchem.HybridizationType.SP3D,
                                                   Chem.rdchem.HybridizationType.SP3D2]) if self.use_hybridization else []
            o += one_hot(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) if self.use_implicit_valence else []
            o += one_hot(atom.GetFormalCharge(), [-3, -2, -1, 0, 1, 2, 3]) if self.use_degree else []
            # o += [atom.GetProp("_GasteigerCharge")] if self.use_partial_charge else [] # some molecules return NaN
            o += [atom.GetIsAromatic()] if self.use_aromaticity else []
            o += [ring.IsAtomInRingOfSize(atom_idx, 3),
                  ring.IsAtomInRingOfSize(atom_idx, 4),
                  ring.IsAtomInRingOfSize(atom_idx, 5),
                  ring.IsAtomInRingOfSize(atom_idx, 6),
                  ring.IsAtomInRingOfSize(atom_idx, 7),
                  ring.IsAtomInRingOfSize(atom_idx, 8)] if self.use_ring_size else []
            o += one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) if self.use_num_hydrogen else []

            if self.use_chirality:
                try:
                    o += one_hot(atom.GetProp('_CIPCode'), ["R", "S"]) + [atom.HasProp("_ChiralityPossible")]
                except:
                    o += [False, False] + [atom.HasProp("_ChiralityPossible")]
            if self.use_hydrogen_bonding:
                o += [atom_idx in hydrogen_donor_match]
                o += [atom_idx in hydrogen_acceptor_match]
            if self.use_acid_base:
                o += [atom_idx in acidic_match]
                o += [atom_idx in basic_match]

            m.append(o)

        return np.array(m, dtype=float)


class MPGenerator_bond(Sequence):
    def __init__(self, x_set, c_set, y_set, batch, task="binary", num_atoms=0,
                 model_type="3dgcn",
                 use_overlap_area=False, use_full_ec=False,
                 use_degree=True, use_hybridization=True, use_implicit_valence=True, use_partial_charge=False,
                 use_formal_charge=True, use_ring_size=True, use_hydrogen_bonding=True, use_acid_base=True,
                 use_aromaticity=True, use_chirality=True, use_num_hydrogen=True, use_atom_symbol=True):
        
        self.x, self.c, self.y = x_set, c_set, y_set
        
        self.batch = batch
        self.task = task
        self.num_atoms = num_atoms

        self.model_type = model_type
        
        self.use_overlap_area = use_overlap_area
        self.use_full_ec = use_full_ec
        self.use_atom_symbol = use_atom_symbol
        self.use_degree = use_degree
        self.use_hybridization = use_hybridization
        self.use_implicit_valence = use_implicit_valence
        self.use_partial_charge = use_partial_charge
        self.use_formal_charge = use_formal_charge
        self.use_ring_size = use_ring_size
        self.use_hydrogen_bonding = use_hydrogen_bonding
        self.use_acid_base = use_acid_base
        self.use_aromaticity = use_aromaticity
        self.use_chirality = use_chirality
        self.use_num_hydrogen = use_num_hydrogen
        
        self.hydrogen_donor = Chem.MolFromSmarts("[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]")
        self.hydrogen_acceptor = Chem.MolFromSmarts(
            "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]")
        self.acidic = Chem.MolFromSmarts("[$([C,S](=[O,S,P])-[O;H1,-1])]")
        self.basic = Chem.MolFromSmarts(
            "[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))])]")

        #self.num_bond_features = self.get_num_bond_features()

        self.use_bond_adj = True
        self.use_overlap_area = False
        self.only_bond_adj_oa = False
        
        self.num_bond_features = 10
        if self.use_overlap_area:
            self.num_bond_features += 4
        if self.use_bond_adj:
            self.num_bond_features -= 4

        if self.only_bond_adj_oa:
            self.use_bond_adj = True
            self.use_overlap_area = True
            self.num_bond_features = 4

        if self.model_type == "gcn":
            self.num_bond_features = 0
            self.only_bond_adj_oa = False
            self.use_bond_adj = False
            self.use_overlap_area = False
        elif self.model_type == "mpnn":
            self.num_bond_features = 4  # basic=4, dist_bin=14, dist_raw=5
            self.only_bond_adj_oa = False
            self.use_bond_adj = False
            self.use_overlap_area = False
        elif self.model_type == "weave":
            self.num_bond_features = 7
            self.only_bond_adj_oa = False
            self.use_bond_adj = False
            self.use_overlap_area = False
            
    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch)))
    
    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch:(idx + 1) * self.batch]
        batch_c = self.c[idx * self.batch:(idx + 1) * self.batch]
        batch_y = self.y[idx * self.batch:(idx + 1) * self.batch]
        
        if self.task == "category":
            return self.tensorize(batch_x, batch_c), to_categorical(batch_y)
        elif self.task == "binary":
            return self.tensorize(batch_x, batch_c), np.array(batch_y, dtype=int)
        elif self.task == "regression":
            return self.tensorize(batch_x, batch_c), np.array(batch_y, dtype=float)
        elif self.task == "input_only":
            return self.tensorize(batch_x, batch_c)
    
    def tensorize(self, batch_x, batch_c):
        atom_tensor = np.zeros((len(batch_x), self.num_atoms, self.get_num_features()))
        bond_tensor = np.zeros((len(batch_x), self.num_atoms, self.num_atoms, self.num_bond_features))
        adjm_tensor = np.zeros((len(batch_x), self.num_atoms, self.num_atoms))
        posn_tensor = np.zeros((len(batch_x), self.num_atoms, self.num_atoms, 3))
        
        #print('tensor:: ', len(batch_x), self.num_atoms, self.get_num_features(), self.num_bond_features)
        
        for mol_idx, mol in enumerate(batch_x):
            Chem.RemoveHs(mol)
            mol_atoms = mol.GetNumAtoms()
            
            # Atom features
            atom_tensor[mol_idx, :mol_atoms, :] = self.get_atom_features(mol)

            # Bond features
            #print('bond: ', bond_tensor.shape, bond_tensor[mol_idx, :mol_atoms, :mol_atoms, :].shape, self.get_bond_features(mol).shape)
            #bond_tensor[mol_idx, :mol_atoms, :mol_atoms, :] = self.get_bond_features(mol)

            if self.only_bond_adj_oa:
                bond_temp = self.get_bond_features(mol)
    
                # Adjacency matrix
                adjms = np.array(rdmolops.GetAdjacencyMatrix(mol), dtype="float")
    
                # adj based degree of bond
                bond_single = np.where(bond_temp[:, :, 0] == 1,
                                       np.full_like(adjms, 1), np.zeros_like(adjms))
                bond_double = np.where(bond_temp[:, :, 1] == 1,
                                       np.full_like(adjms, 2), np.zeros_like(adjms))
                bond_triple = np.where(bond_temp[:, :, 2] == 1,
                                       np.full_like(adjms, 3), np.zeros_like(adjms))
                bond_aromatic = np.where(bond_temp[:, :, 3] == 1,
                                       np.full_like(adjms, 1.5), np.zeros_like(adjms))
                
                adjms = bond_single + bond_double + bond_triple + bond_aromatic
                bond_tensor[mol_idx, :mol_atoms, :mol_atoms, :] = bond_temp[:, :, 4:]

            elif self.use_bond_adj:
                bond_temp = self.get_bond_features(mol)
    
                # Adjacency matrix
                adjms = np.array(rdmolops.GetAdjacencyMatrix(mol), dtype="float")
    
                # adj based degree of bond
                bond_single = np.where(bond_temp[:, :, 0] == 1,
                                       np.full_like(adjms, 1), np.zeros_like(adjms))
                bond_double = np.where(bond_temp[:, :, 1] == 1,
                                       np.full_like(adjms, 2), np.zeros_like(adjms))
                bond_triple = np.where(bond_temp[:, :, 2] == 1,
                                       np.full_like(adjms, 3), np.zeros_like(adjms))
                bond_aromatic = np.where(bond_temp[:, :, 3] == 1,
                                       np.full_like(adjms, 1.5), np.zeros_like(adjms))
                
                adjms = bond_single + bond_double + bond_triple + bond_aromatic
                bond_tensor[mol_idx, :mol_atoms, :mol_atoms, :] = bond_temp[:, :, 4:]

            else:
                bond_tensor[mol_idx, :mol_atoms, :mol_atoms, :] = self.get_bond_features(mol)
    
                # Adjacency matrix
                adjms = np.array(rdmolops.GetAdjacencyMatrix(mol), dtype="float")

            # Normalize adjacency matrix by D^(-1/2) * A_hat * D^(-1/2), Kipf et al. 2016
            if self.model_type == "mpnn":
                pass
            else:
                adjms += np.eye(mol_atoms)
            degree = np.array(adjms.sum(1))
            deg_inv_sqrt = np.power(degree, -0.5)
            deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.
            deg_inv_sqrt = np.diag(deg_inv_sqrt)
            
            adjms = np.matmul(np.matmul(deg_inv_sqrt, adjms), deg_inv_sqrt)
            
            adjm_tensor[mol_idx, : mol_atoms, : mol_atoms] = adjms
            
            # Relative position matrix
            for atom_idx in range(mol_atoms):
                pos_c = batch_c[mol_idx][atom_idx]
                
                for neighbor_idx in range(mol_atoms):
                    pos_n = batch_c[mol_idx][neighbor_idx]
                    
                    # Direction should be Neighbor -> Center
                    n_to_c = [pos_c[0] - pos_n[0], pos_c[1] - pos_n[1], pos_c[2] - pos_n[2]]
                    posn_tensor[mol_idx, atom_idx, neighbor_idx, :] = n_to_c

        return [atom_tensor, bond_tensor, adjm_tensor, posn_tensor]
    
    def get_num_features(self):
        mol = Chem.MolFromSmiles("CC")
        return len(self.get_atom_features(mol)[0])
    
    def get_atom_features(self, mol):
        AllChem.ComputeGasteigerCharges(mol)
        Chem.AssignStereochemistry(mol)
        
        hydrogen_donor_match = sum(mol.GetSubstructMatches(self.hydrogen_donor), ())
        hydrogen_acceptor_match = sum(mol.GetSubstructMatches(self.hydrogen_acceptor), ())
        acidic_match = sum(mol.GetSubstructMatches(self.acidic), ())
        basic_match = sum(mol.GetSubstructMatches(self.basic), ())
        ring = mol.GetRingInfo()

        m = []

        if self.model_type == "mpnn":
            for atom_idx in range(mol.GetNumAtoms()):
                atom = mol.GetAtomWithIdx(atom_idx)
        
                o = []
                o += one_hot(atom.GetSymbol(), ['C', 'O', 'N', 'S', 'Cl', 'F', 'Br', 'P',
                                                'I', 'Si', 'B', 'Na', 'Sn', 'Se', 'other'])
                o += [atom.GetAtomicNum()]
                # o += one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
                o += one_hot(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP,
                                                       Chem.rdchem.HybridizationType.SP2,
                                                       Chem.rdchem.HybridizationType.SP3])
                # o += one_hot(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) if self.use_implicit_valence else []
                o += [atom.GetFormalCharge()]
                # o += one_hot(atom.GetFormalCharge(), [-3, -2, -1, 0, 1, 2, 3]) if self.use_degree else []
                o += [atom.GetIsAromatic()]
                o += [atom.GetTotalNumHs()]
                # o += one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) if self.use_num_hydrogen else []
        
                # hydrogen_bonding:
                #o += [atom_idx in hydrogen_donor_match]
                #o += [atom_idx in hydrogen_acceptor_match]
        
                # acid_base:
                o += [atom_idx in acidic_match]
                o += [atom_idx in basic_match]
        
                m.append(o)
    
            return np.array(m, dtype=float)

        elif self.model_type == "gcn":
            AllChem.ComputeGasteigerCharges(mol)
    
            for atom_idx in range(mol.GetNumAtoms()):
                atom = mol.GetAtomWithIdx(atom_idx)
        
                o = []
                o += one_hot(atom.GetSymbol(), ['C', 'O', 'N', 'S', 'Cl', 'F', 'Br', 'P',
                                                'I', 'Si', 'B', 'Na', 'Sn', 'Se',
                                                'other'])
                # o += one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
                o += one_hot(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP,
                                                       Chem.rdchem.HybridizationType.SP2,
                                                       Chem.rdchem.HybridizationType.SP3])
                # o += one_hot(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) if self.use_implicit_valence else []
                o += [atom.GetFormalCharge()]
                # o += one_hot(atom.GetFormalCharge(), [-3, -2, -1, 0, 1, 2, 3]) if self.use_degree else []
                o += [atom.GetDoubleProp('_GasteigerCharge')]
                # o += [atom.GetProp("_GasteigerCharge")] if self.use_partial_charge else [] # some molecules return NaN
                o += [atom.GetIsAromatic()]
                o += [ring.IsAtomInRingOfSize(atom_idx, 3),
                      ring.IsAtomInRingOfSize(atom_idx, 4),
                      ring.IsAtomInRingOfSize(atom_idx, 5),
                      ring.IsAtomInRingOfSize(atom_idx, 6),
                      ring.IsAtomInRingOfSize(atom_idx, 7),
                      ring.IsAtomInRingOfSize(atom_idx, 8)]
                # o += one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) if self.use_num_hydrogen else []
        
                try:
                    o += one_hot(atom.GetProp('_CIPCode'), ["R", "S"])
                except:
                    o += [False, False]
                # hydrogen_bonding:
                o += [atom_idx in hydrogen_donor_match]
                o += [atom_idx in hydrogen_acceptor_match]
                
                # acid_base:
                #o += [atom_idx in acidic_match]
                #o += [atom_idx in basic_match]
        
                m.append(o)
    
            return np.array(m, dtype=float)
        
        elif self.model_type == "weave":
            AllChem.ComputeGasteigerCharges(mol)
    
            for atom_idx in range(mol.GetNumAtoms()):
                atom = mol.GetAtomWithIdx(atom_idx)
        
                o = []
                atom_symbol = atom.GetSymbol()
                if atom_symbol in ['Na', 'Al', 'Sn']:
                    atom_symbol = "metal"
                o += one_hot(atom_symbol, ['C', 'O', 'N', 'S', 'Cl', 'F', 'Br', 'P', 'I', 'Si', 'B', 'Se', 'metal'])
                # o += one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
                o += one_hot(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP,
                                                       Chem.rdchem.HybridizationType.SP2,
                                                       Chem.rdchem.HybridizationType.SP3])
                # o += one_hot(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) if self.use_implicit_valence else []
                o += [atom.GetFormalCharge()]
                # o += one_hot(atom.GetFormalCharge(), [-3, -2, -1, 0, 1, 2, 3]) if self.use_degree else []
                o += [atom.GetDoubleProp('_GasteigerCharge')]
                # o += [atom.GetProp("_GasteigerCharge")] if self.use_partial_charge else [] # some molecules return NaN
                o += [atom.GetIsAromatic()]
                o += [ring.IsAtomInRingOfSize(atom_idx, 3),
                      ring.IsAtomInRingOfSize(atom_idx, 4),
                      ring.IsAtomInRingOfSize(atom_idx, 5),
                      ring.IsAtomInRingOfSize(atom_idx, 6),
                      ring.IsAtomInRingOfSize(atom_idx, 7),
                      ring.IsAtomInRingOfSize(atom_idx, 8)]
                # o += one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) if self.use_num_hydrogen else []
        
                try:
                    o += one_hot(atom.GetProp('_CIPCode'), ["R", "S"])
                except:
                    o += [False, False]
                # hydrogen_bonding:
                o += [atom_idx in hydrogen_donor_match]
                o += [atom_idx in hydrogen_acceptor_match]
                m.append(o)
    
            return np.array(m, dtype=float)
    
        # normal features
        for atom_idx in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(atom_idx)
            
            o = []

            if self.use_full_ec:
                possible_atoms = ['Al', 'Ge', 'Cl', 'N', 'S', 'P', 'Br', 'I', 'F', 'O', 'Si', 'As', 'C', 'B', 'Se',
                                  'Na']  # 16
                possible_group = [1, 13, 14, 15, 16, 17]
                possible_period = [2, 3, 4, 5]
                possible_block = ['s', 'p']
                block_dict = {'Na': 's1',
                              'B': 'p1', 'Al': 'p1',
                              'C': 'p2', 'Si': 'p2', 'Ge': 'p2',
                              'N': 'p3', 'P': 'p3', 'As': 'p3',
                              'O': 'p4', 'S': 'p4', 'Se': 'p4',
                              'F': 'p5', 'Cl': 'p5', 'Br': 'p5', 'I': 'p5'}
    
                ptable = pd.read_pickle('./dataset/ptable.pickle')
                symbol, group, period, _ = (ptable.loc[ptable['symbol'] == atom.GetSymbol()]).iloc[0]

                # valence electron: e in s orbital, e in p orbital
                if symbol in ['Na', 'K']:
                    fulleconf = [1, 0]
                elif symbol in ['B', 'Al', 'Ga']:
                    fulleconf = [2, 1]
                elif symbol in ['C', 'Si', 'Ge', 'Sn']:
                    fulleconf = [2, 2]
                elif symbol in ['N', 'P', 'As']:
                    fulleconf = [2, 3]
                elif symbol in ['O', 'S', 'Se']:
                    fulleconf = [2, 4]
                elif symbol in ['F', 'Cl', 'Br', 'I']:
                    fulleconf = [2, 5]
                else:
                    print('Not preprossed: ', symbol)
                    return None
                o += fulleconf
                o += one_hot(period, possible_period)

            o += one_hot(atom.GetSymbol(), ['C', 'O', 'N', 'S', 'Cl', 'F', 'Br', 'P',
                                            'I', 'Si', 'B', 'Na', 'Sn', 'Se', 'other']) if self.use_atom_symbol else []
            o += one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) if self.use_degree else []
            o += one_hot(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP,
                                                   Chem.rdchem.HybridizationType.SP2,
                                                   Chem.rdchem.HybridizationType.SP3,
                                                   Chem.rdchem.HybridizationType.SP3D,
                                                   Chem.rdchem.HybridizationType.SP3D2]) if self.use_hybridization else []
            o += one_hot(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) if self.use_implicit_valence else []
            o += one_hot(atom.GetFormalCharge(), [-3, -2, -1, 0, 1, 2, 3]) if self.use_degree else []
            # o += [atom.GetProp("_GasteigerCharge")] if self.use_partial_charge else [] # some molecules return NaN
            o += [atom.GetIsAromatic()] if self.use_aromaticity else []
            o += [ring.IsAtomInRingOfSize(atom_idx, 3),
                  ring.IsAtomInRingOfSize(atom_idx, 4),
                  ring.IsAtomInRingOfSize(atom_idx, 5),
                  ring.IsAtomInRingOfSize(atom_idx, 6),
                  ring.IsAtomInRingOfSize(atom_idx, 7),
                  ring.IsAtomInRingOfSize(atom_idx, 8)] if self.use_ring_size else []
            o += one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) if self.use_num_hydrogen else []

            if self.use_chirality:
                try:
                    o += one_hot(atom.GetProp('_CIPCode'), ["R", "S"]) + [atom.HasProp("_ChiralityPossible")]
                except:
                    o += [False, False] + [atom.HasProp("_ChiralityPossible")]
            if self.use_hydrogen_bonding:
                o += [atom_idx in hydrogen_donor_match]
                o += [atom_idx in hydrogen_acceptor_match]
            if self.use_acid_base:
                o += [atom_idx in acidic_match]
                o += [atom_idx in basic_match]
            
            m.append(o)
        
        return np.array(m, dtype=float)
    
    def get_bond_features(self, mol):
        AllChem.ComputeGasteigerCharges(mol)
        Chem.AssignStereochemistry(mol)
        
        _num_atom = mol.GetNumAtoms()

        if self.use_overlap_area:
            distance = np.array(rdmolops.Get3DDistanceMatrix(mol))
            # coords = conf.GetConformer().GetPositions()
            # coords = np.around(coords, decimals=2)

            overlap_area = np.zeros_like(distance)
            if _num_atom > 1:
                for i in range(_num_atom):
                    for j in range(_num_atom):

                        if not i == j:
                            # Get the coordinates of the central site
                            r_i = Chem.GetPeriodicTable().GetRvdw(mol.GetAtomWithIdx(i).GetAtomicNum())
                            r_j = Chem.GetPeriodicTable().GetRvdw(mol.GetAtomWithIdx(j).GetAtomicNum())
                            # print('atom ', mol.GetAtomWithIdx(i), mol.GetAtomWithIdx(j), ' radius: ', r_i, r_j)
    
                            d = distance[i][j]
                            if d < (r_i + r_j):
                                s = (r_i + r_j + d) / 2.
                                h = 2. * np.sqrt(s * (s - r_i) * (s - r_j) * (s - d)) / d
                                #h = np.round(2. * np.sqrt(s * (s - r_i) * (s - r_j) * (s - d)) / d, 3)
                                # print('len: ', r_i, r_j, d, h)
                                h = np.round(np.power(h, 2), 3)
        
                                overlap_area[i, j] = h
                                overlap_area[j, i] = h

                overlap_area = overlap_area / max(overlap_area.flatten())
                # print('overlap norm: ', overlap_area)
        
        # m = []
        if self.use_bond_adj:
            m = np.zeros([mol.GetNumAtoms(), mol.GetNumAtoms(), self.num_bond_features+4], dtype=float)
        else:
            m = np.zeros([mol.GetNumAtoms(), mol.GetNumAtoms(), self.num_bond_features], dtype=float)
            
        if self.only_bond_adj_oa:
            for i in range(mol.GetNumAtoms()):
                for j in range(mol.GetNumAtoms()):
                    bond = mol.GetBondBetweenAtoms(i, j)

                    if bond is not None:
                        bond_type = bond.GetBondType()
                        # bond.GetBeginAtom(), bond.GetEndAtom()
    
                        bond_feats = [
                            bond_type == Chem.rdchem.BondType.SINGLE, bond_type == Chem.rdchem.BondType.DOUBLE,
                            bond_type == Chem.rdchem.BondType.TRIPLE, bond_type == Chem.rdchem.BondType.AROMATIC
                        ]
                        
                        h = overlap_area[i, j]

                        if h < 0.6:
                            oa_bin = None
                        elif 0.6 < h <= 0.7:
                            oa_bin = 0
                        elif 0.7 < h <= 0.8:
                            oa_bin = 1
                        elif 0.8 < h <= 0.9:
                            oa_bin = 2
                        elif 0.9 < h <= 1.:
                            oa_bin = 3
    
                        bond_feats += one_hot(oa_bin, [0, 1, 2, 3])

                        m[i, j] = np.squeeze(np.array(bond_feats))

            return np.array(m, dtype=float)

        if self.model_type == "gcn":
            return np.array(m, dtype=float)

        elif self.model_type == "mpnn":
            distance = np.array(rdmolops.Get3DDistanceMatrix(mol))
            edge_rep = None  # "dist_bin", "dist_raw"
    
            for i in range(mol.GetNumAtoms()):
                for j in range(mol.GetNumAtoms()):
                    bond = mol.GetBondBetweenAtoms(i, j)
                    
                    if bond is not None:
                        bond_type = bond.GetBondType()
    
                        bond_feats = [
                            bond_type == Chem.rdchem.BondType.SINGLE, bond_type == Chem.rdchem.BondType.DOUBLE,
                            bond_type == Chem.rdchem.BondType.TRIPLE, bond_type == Chem.rdchem.BondType.AROMATIC
                        ]
                        
                        if edge_rep == "dist_bin":
                            bond_feats += [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        elif edge_rep == "dist_raw":
                            bond_feats += [distance[i, j]]
                
                        m[i, j] = np.squeeze(np.array(bond_feats))
                        
                    else:
                        if edge_rep == "dist_bin":
                            if distance[i, j] <= 2.:
                                distance_bin = 0
                            elif 2. <= distance[i, j] <= 2.5:
                                distance_bin = 1
                            elif 2.5 <= distance[i, j] <= 3.:
                                distance_bin = 2
                            elif 3. <= distance[i, j] <= 3.5:
                                distance_bin = 3
                            elif 3.5 <= distance[i, j] <= 4.:
                                distance_bin = 4
                            elif 4. <= distance[i, j] <= 4.5:
                                distance_bin = 5
                            elif 4.5 <= distance[i, j] <= 5.:
                                distance_bin = 6
                            elif 5. <= distance[i, j] <= 5.5:
                                distance_bin = 7
                            elif 5.5 <= distance[i, j] <= 6.:
                                distance_bin = 8
                            elif distance[i, j] >= 6.:
                                distance_bin = 9
                                
                            bond_feats = [0, 0, 0, 0]
                            bond_feats += one_hot(distance_bin, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
                            
                            m[i, j] = np.squeeze(np.array(bond_feats))

            return np.array(m, dtype=float)

        elif self.model_type == "weave":
            ring_info = mol.GetRingInfo()
            bond_rings = ring_info.BondRings()
            
            # Graph distance matrix
            adj = np.array(rdmolops.GetAdjacencyMatrix(mol), dtype="float")
            graph_dist = graph_distance(mol.GetNumAtoms(), adj, max_distance=2)
            
            for i in range(mol.GetNumAtoms()):
                for j in range(mol.GetNumAtoms()):
                    bond = mol.GetBondBetweenAtoms(i, j)
                    if bond is not None:
                        bond_type = bond.GetBondType()
    
                        bond_feats = [
                            bond_type == Chem.rdchem.BondType.SINGLE, bond_type == Chem.rdchem.BondType.DOUBLE,
                            bond_type == Chem.rdchem.BondType.TRIPLE, bond_type == Chem.rdchem.BondType.AROMATIC,
                            bond.IsInRing(),
                            1, 0
                        ]

                    else:
                        in_same_ring = False
                        for bond_ring in bond_rings:
                            if i in bond_ring and j in bond_ring:
                                in_same_ring = True
    
                        # bond type, in same ring, graph distance
                        bond_feats = [0, 0, 0, 0, in_same_ring, 0, graph_dist[i, j, 1]]

                    m[i, j] = np.squeeze(np.array(bond_feats))

            return np.array(m, dtype=float)

        # normal features
        for i in range(mol.GetNumAtoms()):
            for j in range(mol.GetNumAtoms()):
                bond = mol.GetBondBetweenAtoms(i, j)
                if bond is not None:
                    bond_type = bond.GetBondType()
                    
                    bond_feats = [
                        bond_type == Chem.rdchem.BondType.SINGLE, bond_type == Chem.rdchem.BondType.DOUBLE,
                        bond_type == Chem.rdchem.BondType.TRIPLE, bond_type == Chem.rdchem.BondType.AROMATIC,
                        bond.GetIsConjugated(),
                        bond.IsInRing()
                    ]
                    bond_feats += one_hot(str(bond.GetStereo()),
                                          ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])

                    if self.use_overlap_area:
                        h = overlap_area[i, j]
                        if h == 0.:
                            bond_feats += [0, 0, 0, 0]
                        elif h <= 0.7:
                            bond_feats += one_hot(0, [0, 1, 2, 3])
                        elif 0.7 < h <= 0.8:
                            bond_feats += one_hot(1, [0, 1, 2, 3])
                        elif 0.8 < h <= 0.9:
                            bond_feats += one_hot(2, [0, 1, 2, 3])
                        elif 0.9 < h <= 1.:
                            bond_feats += one_hot(3, [0, 1, 2, 3])
                            
                    m[i, j] = np.squeeze(np.array(bond_feats))

        return np.array(m, dtype=float)
