import os
import os.path as osp
import re
from typing import Callable, List, Optional

import numpy as np
import random
import torch
import torch.nn.functional as F
from torch_geometric.data.makedirs import makedirs
from tqdm import tqdm
import copy
from functools import lru_cache
from rdkit import Chem
import selfies as sf
from data_provider.mol_mapping_utils import build_rdkit2cano_smiles_withoutH_mapping, get_smiles2selfies_mapping, invalid_int, build_rdkit2rand_smiles_withoutH_mapping
from data_provider.conf_gen_cal_metrics import get_best_rmsd, set_rdmol_positions

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from data_provider.conf_gen_cal_metrics import my_single_process_data, generate_conformers
from evaluation.jodo.bond_analyze import allowed_bonds
from rdkit.Chem.rdchem import BondType as BT
from torch_geometric.datasets import QM9
from openbabel import openbabel, pybel
from evaluation.eval_functions import check_3D_stability


def my_add_hs(mol):
    mol = copy.deepcopy(mol)
    mol = Chem.RemoveHs(mol)
    if Chem.GetFormalCharge(mol) == 0:
        mol = Chem.AddHs(mol)
    else:
        add_hs = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "H":
                continue
            if atom.GetFormalCharge() == 0:
                add_hs.append(atom.GetIdx())
            elif atom.GetFormalCharge() > 0 and atom.GetTotalValence() == atom.GetExplicitValence() and atom.GetTotalValence() < allowed_bonds[atom.GetSymbol()]:
                add_hs.append(atom.GetIdx())
        if add_hs:
            mol = Chem.AddHs(mol, onlyOnAtoms=add_hs, explicitOnly=False)
    Chem.SanitizeMol(mol)
    return mol


def my_add_hs_old(mol, debug=False, additional_fix=False):
    mol = copy.deepcopy(mol)
    mol = Chem.RemoveHs(mol)
    add_hs = []
    for atom in mol.GetAtoms():
        if Chem.GetFormalCharge(mol) == 0:
            if atom.GetImplicitValence() != 0 and False:
                add_hs.append(atom.GetIdx())
                print(atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalValence(), atom.GetExplicitValence(), atom.GetImplicitValence(), atom.GetNoImplicit(), atom.IsInRing(), atom.GetChiralTag(), atom.GetIsAromatic(), atom.GetHybridization(), atom.GetDegree(), 'ring')
                # if atom.IsInRing():
                #     add_hs.append(atom.GetIdx())
                #     print(atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalValence(), atom.GetExplicitValence(), atom.GetImplicitValence(), atom.GetNoImplicit(), atom.IsInRing(), atom.GetChiralTag(), atom.GetIsAromatic(), atom.GetHybridization(), atom.GetDegree(), 'ring')
                # else:
                #     print(atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalValence(), atom.GetExplicitValence(), atom.GetImplicitValence(), atom.GetNoImplicit(), atom.IsInRing(), atom.GetChiralTag(), atom.GetIsAromatic(), atom.GetHybridization(), atom.GetDegree(), 'not ring')
            else:
                if atom.GetSymbol() == 'C' and atom.GetDegree() == 3 and atom.GetImplicitValence() > 0:
                    if debug:
                        print(atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalValence(), atom.GetExplicitValence(), atom.GetImplicitValence(), atom.GetNoImplicit(), atom.IsInRing(), atom.GetChiralTag(), atom.GetIsAromatic(), atom.GetHybridization(), atom.GetDegree(), 'stable C')
                    neis = {n.GetSymbol() for n in atom.GetNeighbors()}
                    if neis == {'C'}:
                        add_hs.append(atom.GetIdx())
                        # print('add CC')
                    continue
                elif atom.GetSymbol() == 'N' and atom.GetDegree() == 2 and atom.GetImplicitValence() > 0:
                    if debug:
                        print(atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalValence(), atom.GetExplicitValence(), atom.GetImplicitValence(), atom.GetNoImplicit(), atom.IsInRing(), atom.GetChiralTag(), atom.GetIsAromatic(), atom.GetHybridization(), atom.GetDegree(), 'stable N')
                    continue
                add_hs.append(atom.GetIdx())
                if debug:
                    print(atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalValence(), atom.GetExplicitValence(), atom.GetImplicitValence(), atom.GetNoImplicit(), atom.IsInRing(), atom.GetChiralTag(), atom.GetIsAromatic(), atom.GetHybridization(), atom.GetDegree(), 'stable 0')
                neis = {n.GetSymbol() for n in atom.GetNeighbors()}
                if debug:
                    print(neis)

        elif (atom.GetTotalValence() - atom.GetExplicitValence() == 0) and atom.GetFormalCharge() == 0:
            add_hs.append(atom.GetIdx())
            if debug:
                print(atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalValence(), atom.GetExplicitValence(), atom.GetImplicitValence(), atom.GetNoImplicit(), atom.IsInRing(), atom.GetChiralTag(), atom.GetIsAromatic(), atom.GetHybridization(), atom.GetDegree(), 'implicit')
            neis = {n.GetSymbol() for n in atom.GetNeighbors()}
            # print(neis)
        elif atom.GetFormalCharge() < 0:
            add_hs.append(atom.GetIdx())
            if debug:
                print(atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalValence(), atom.GetExplicitValence(), atom.GetImplicitValence(), atom.GetNoImplicit(), atom.IsInRing(), atom.GetChiralTag(), atom.GetIsAromatic(), atom.GetDegree(), 'charge')
        elif atom.GetFormalCharge() > 0 and atom.GetTotalValence() == atom.GetExplicitValence() and atom.GetTotalValence() < allowed_bonds[atom.GetSymbol()]:
            add_hs.append(atom.GetIdx())
            if debug:
                print(atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalValence(), atom.GetExplicitValence(), atom.GetImplicitValence(), atom.GetNoImplicit(), atom.IsInRing(), atom.GetChiralTag(), atom.GetIsAromatic(), atom.GetHybridization(), atom.GetDegree(), 'valency')
        else:
            # pass
            if debug:
                print(atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalValence(), atom.GetExplicitValence(), atom.GetImplicitValence(), atom.GetNoImplicit(), atom.IsInRing(), atom.GetChiralTag(), atom.GetIsAromatic(), atom.GetHybridization(), atom.GetDegree(), 'nothing')

    if debug:
        print('add_hs', add_hs)
    if len(add_hs) > 0:
        mol = Chem.AddHs(mol, onlyOnAtoms=add_hs, explicitOnly=False)

    if additional_fix:
        if not check_3d_stable(mol)[0]:
            Chem.SanitizeMol(mol)
            # mol = Chem.AddHs(mol)
            add_hs = []
            for atom in mol.GetAtoms():
                if atom.GetFormalCharge() != 0 and atom.GetTotalValence() == atom.GetExplicitValence() and atom.GetTotalValence() < allowed_bonds[atom.GetSymbol()]:
                    add_hs.append(atom.GetIdx())
            if len(add_hs) > 0:
                mol = Chem.AddHs(mol, onlyOnAtoms=add_hs, explicitOnly=False)

    return mol

HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])

atomrefs = {
    6: [0., 0., 0., 0., 0.],
    7: [
        -13.61312172, -1029.86312267, -1485.30251237, -2042.61123593,
        -2713.48485589
    ],
    8: [
        -13.5745904, -1029.82456413, -1485.26398105, -2042.5727046,
        -2713.44632457
    ],
    9: [
        -13.54887564, -1029.79887659, -1485.2382935, -2042.54701705,
        -2713.42063702
    ],
    10: [
        -13.90303183, -1030.25891228, -1485.71166277, -2043.01812778,
        -2713.88796536
    ],
    11: [0., 0., 0., 0., 0.],
}

Cv_atomref = [2.981, 2.981, 2.981, 2.981, 2.981]


def files_exist(files: List[str]) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])


def check_3d_stable(mol):
    bonds = {BT.SINGLE: 1, BT.DOUBLE: 2, BT.TRIPLE: 3, BT.AROMATIC: 1.5}  # 0 -> without edge
    nr_bonds = [0 for _ in range(mol.GetNumAtoms())]
    for bond in mol.GetBonds():
        nr_bonds[bond.GetBeginAtomIdx()] += bonds[bond.GetBondType()]
        nr_bonds[bond.GetEndAtomIdx()] += bonds[bond.GetBondType()]
    correct = 0
    for a in mol.GetAtoms():
        if nr_bonds[a.GetIdx()] == (allowed_bonds[a.GetSymbol()] + a.GetFormalCharge()):
            correct += 1
        else:
            pass
    return correct == mol.GetNumAtoms(), nr_bonds


def construct_mol(atoms, coordinates, title=None):
    mol = openbabel.OBMol()
    for atom, (x, y, z) in zip(atoms, coordinates):
        ob_atom = mol.NewAtom()
        ob_atom.SetAtomicNum(atom)
        ob_atom.SetVector(x, y, z)
    mol.ConnectTheDots()
    mol.PerceiveBondOrders()
    if title:
        mol.SetTitle(title)
    return mol


class QM9(InMemoryDataset):
    r"""The QM9 dataset from the `"MoleculeNet: A Benchmark for Molecular
    Machine Learning" <https://arxiv.org/abs/1703.00564>`_ paper, consisting of
    about 130,000 molecules with 19 regression targets.
    Each molecule includes complete spatial information for the single low
    energy conformation of the atoms in the molecule.
    In addition, we provide the atom features from the `"Neural Message
    Passing for Quantum Chemistry" <https://arxiv.org/abs/1704.01212>`_ paper.

    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | Target | Property                         | Description                                                                       | Unit                                        |
    +========+==================================+===================================================================================+=============================================+
    | 0      | :math:`\mu`                      | Dipole moment                                                                     | :math:`\textrm{D}`                          |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 1      | :math:`\alpha`                   | Isotropic polarizability                                                          | :math:`{a_0}^3`                             |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 2      | :math:`\epsilon_{\textrm{HOMO}}` | Highest occupied molecular orbital energy                                         | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 3      | :math:`\epsilon_{\textrm{LUMO}}` | Lowest unoccupied molecular orbital energy                                        | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 4      | :math:`\Delta \epsilon`          | Gap between :math:`\epsilon_{\textrm{HOMO}}` and :math:`\epsilon_{\textrm{LUMO}}` | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 5      | :math:`\langle R^2 \rangle`      | Electronic spatial extent                                                         | :math:`{a_0}^2`                             |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 6      | :math:`\textrm{ZPVE}`            | Zero point vibrational energy                                                     | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 7      | :math:`U_0`                      | Internal energy at 0K                                                             | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 8      | :math:`U`                        | Internal energy at 298.15K                                                        | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 9      | :math:`H`                        | Enthalpy at 298.15K                                                               | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 10     | :math:`G`                        | Free energy at 298.15K                                                            | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 11     | :math:`c_{\textrm{v}}`           | Heat capacity at 298.15K                                                          | :math:`\frac{\textrm{cal}}{\textrm{mol K}}` |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 12     | :math:`U_0^{\textrm{ATOM}}`      | Atomization energy at 0K                                                          | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 13     | :math:`U^{\textrm{ATOM}}`        | Atomization energy at 298.15K                                                     | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 14     | :math:`H^{\textrm{ATOM}}`        | Atomization enthalpy at 298.15K                                                   | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 15     | :math:`G^{\textrm{ATOM}}`        | Atomization free energy at 298.15K                                                | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 16     | :math:`A`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 17     | :math:`B`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 18     | :math:`C`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)

    Stats:
        .. list-table::
            :widths: 10 10 10 10 10
            :header-rows: 1

            * - #graphs
              - #nodes
              - #edges
              - #features
              - #tasks
            * - 130,831
              - ~18.0
              - ~37.3
              - 11
              - 19
    """  # noqa: E501

    raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
               'molnet_publish/qm9.zip')
    raw_url2 = 'https://ndownloader.figshare.com/files/3195404'

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None, selfies_tokenizer: Optional[List[str]] = None, rand_smiles=False, addHs=False):
        self.selfies_tokenizer = selfies_tokenizer
        self.rand_smiles = rand_smiles
        self.addHs = addHs # note, this parameter on applies on selfies but not rdmols
        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])
        self.add_unseen_selfies_tokens(self.selfies_tokenizer)

    def add_unseen_selfies_tokens(self, tokenizer):
        with open(self.processed_paths[1], 'r') as f:
            unseen_tokens = f.read().splitlines()
        vocab = tokenizer.get_vocab()
        for token in unseen_tokens:
            if token not in vocab:
                tokenizer.add_tokens(token)

    def atomref(self, target) -> Optional[torch.Tensor]:
        if target in atomrefs:
            out = torch.zeros(100)
            out[torch.tensor([1, 6, 7, 8, 9])] = torch.tensor(atomrefs[target])
            return out.view(-1, 1)
        return None

    @property
    def raw_file_names(self) -> List[str]:
        return ['gdb9.sdf', 'gdb9.sdf.csv', 'uncharacterized.txt']

    @property
    def processed_file_names(self) -> str:
        return ['data_qm9.pt', 'unseen_selfies_tokens.txt', 'babel_mol.sdf', 'id2id.txt']

    def _download(self):
        if files_exist(self.processed_paths):
            return

        if files_exist(self.raw_paths):  # pragma: no cover
            return

        makedirs(self.raw_dir)
        self.download()

    def download(self):
        file_path = download_url(self.raw_url, self.raw_dir)
        extract_zip(file_path, self.raw_dir)
        os.unlink(file_path)

        file_path = download_url(self.raw_url2, self.raw_dir)
        os.rename(osp.join(self.raw_dir, '3195404'),
                  osp.join(self.raw_dir, 'uncharacterized.txt'))


    def process(self):
        try:
            import rdkit
            from rdkit import Chem, RDLogger
            from rdkit.Chem.rdchem import BondType as BT
            from rdkit.Chem.rdchem import HybridizationType
            RDLogger.DisableLog('rdApp.*')

        except ImportError:
            raise ImportError("Please install 'rdkit' to alternatively process the raw data.")

        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        bonds = {BT.SINGLE: 1, BT.DOUBLE: 2, BT.TRIPLE: 3, BT.AROMATIC: 4}  # 0 -> without edge
        charge_dict = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}

        with open(self.raw_paths[1], 'r') as f:
            target = f.read().split('\n')[1:-1]
            target = [[float(x) for x in line.split(',')[1:20]]
                      for line in target]
            target = torch.tensor(target, dtype=torch.float)
            target = torch.cat([target[:, 3:], target[:, :3]], dim=-1)
            target = target * conversion.view(1, -1)

        with open(self.raw_paths[2], 'r') as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

        if not osp.exists(self.processed_paths[2]) or not osp.exists(self.processed_paths[3]):
            babel_mol_list = []
            name_list = []
            suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False, sanitize=False)

            id2id = {}
            new_id = 0
            split_id = 0
            for qm9_id, mol in enumerate(tqdm(suppl)):
                if qm9_id in skip:
                    continue
                try:
                    Chem.SanitizeMol(mol)
                except:
                    split_id += 1
                    continue
                id2id[new_id] = (qm9_id, split_id)
                name = mol.GetProp('_Name')
                babel_mol = construct_mol([atom.GetAtomicNum() for atom in mol.GetAtoms()], mol.GetConformer().GetPositions(), name)
                babel_mol_list.append(babel_mol)
                name_list.append(name)

                new_id += 1
                split_id += 1

            output = pybel.Outputfile("sdf", self.processed_paths[2], overwrite=True)
            for mol in babel_mol_list:
                mol = pybel.Molecule(mol)
                # Write each molecule to the SDF file
                output.write(mol)
            output.close()
            with open(self.processed_paths[3], 'w') as f:
                for k, (v1, v2) in id2id.items():
                    f.write(f'{k} {v1} {v2}\n')

        with open(self.processed_paths[3], 'r') as f:
            id2id = {}
            for line in f:
                k, v1, v2 = line.split()
                id2id[int(k)] = (int(v1), int(v2))

        suppl = Chem.SDMolSupplier(self.processed_paths[2], removeHs=False, sanitize=False)
        data_list = []

        for new_id, mol in enumerate(tqdm(suppl)):
            try:
                Chem.SanitizeMol(mol)
            except:
                continue

            rdmol2smiles, cano_smiles_woh = build_rdkit2cano_smiles_withoutH_mapping(mol)
            selfies = sf.encoder(cano_smiles_woh)

            name = mol.GetProp('_Name')
            N = mol.GetNumAtoms()
            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)

            type_idx = []
            charges = []
            formal_charges = []

            for atom in mol.GetAtoms():
                atom_str = atom.GetSymbol()
                type_idx.append(types[atom_str])
                charges.append(charge_dict[atom_str])
                formal_charges.append(atom.GetFormalCharge())

            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                if bond.GetBondType() == BT.AROMATIC:
                    print('meet aromatic bond!')

                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()]]

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]

            qm9_id, split_id = id2id[new_id]
            x = torch.tensor(type_idx)
            y = target[qm9_id].unsqueeze(0)
            data = Data(atom_type=x, pos=pos, charge=torch.tensor(charges), fc=torch.tensor(formal_charges),
                        edge_index=edge_index, edge_type=edge_type, y=y, num_atom=N, idx=qm9_id, split_id=split_id, rdmol=copy.deepcopy(mol), selfies=selfies, rdmol2smiles=rdmol2smiles, cano_smiles_woh=cano_smiles_woh, gdb_id=name)

            if self.pre_filter is not None and not self.pre_filter(data):
                assert False
                continue
            if self.pre_transform is not None:
                assert False
                data = self.pre_transform(data)
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

        ## check and save the unseen selfies tokens
        unseen_selfies_tokens = set()
        vocab = self.selfies_tokenizer.get_vocab()
        for data in data_list:
            selfies_tokens = sf.split_selfies(data.selfies)
            for t in selfies_tokens:
                if t not in vocab:
                    unseen_selfies_tokens.add(t)
        unseen_selfies_tokens = list(unseen_selfies_tokens)
        unseen_selfies_tokens.sort()
        with open(self.processed_paths[1], 'w') as f:
            for t in unseen_selfies_tokens:
                f.write(t + '\n')



    def get_idx_split(self):
        # load split idx for train, val, test
        split_path = osp.join(self.processed_dir, 'split_dict_qm9.pt')
        if osp.exists(split_path):
            print('Loading existing split data.')
            return torch.load(split_path)

        data_num = len(self.indices())
        # assert data_num == 130831
        # data_num = 127924
        train_num = 100000
        test_num = int(0.1 * data_num)
        valid_num = data_num - (train_num + test_num)

        # Generate random permutation
        np.random.seed(0)
        data_perm = np.random.permutation(data_num)
        train, valid, test, extra = np.split(
            data_perm, [train_num, train_num + valid_num, train_num + valid_num + test_num])

        split_id = self._data.split_id.tolist()
        split_id2new_id = {}
        for i, idx in enumerate(split_id):
            split_id2new_id[idx] = i

        train = train.tolist()
        valid = valid.tolist()
        test = test.tolist()
        train = [split_id2new_id[x] for x in train if x in split_id2new_id]
        valid = [split_id2new_id[x] for x in valid if x in split_id2new_id]
        test = [split_id2new_id[x] for x in test if x in split_id2new_id]
        train = np.array(train)
        valid = np.array(valid)
        test = np.array(test)

        train = np.array(self.indices())[train]
        valid = np.array(self.indices())[valid]
        test = np.array(self.indices())[test]

        splits = {'train': train, 'valid': valid, 'test': test}
        torch.save(splits, split_path)
        return splits

    def get_cond_idx_split(self):
        # load conditional generation split idx for first train, second train, val, test
        split_path = osp.join(self.processed_dir, 'split_dict_cond_qm9.pt')
        return torch.load(split_path)

    def compute_property_mean_mad(self, prop2idx):
        prop_values = []

        prop_ids = torch.tensor(list(prop2idx.values()))
        for idx in range(len(self.indices())):
            data = self.get(self.indices()[idx])
            tars = []
            for prop_id in prop_ids:
                if prop_id == 11:
                    tars.append(self.sub_Cv_thermo(data).reshape(1))
                else:
                    tars.append(data.y[0][prop_id].reshape(1))
            tars = torch.cat(tars)
            prop_values.append(tars)
        prop_values = torch.stack(prop_values, dim=0)
        mean = torch.mean(prop_values, dim=0, keepdim=True)
        ma = torch.abs(prop_values - mean)
        mad = torch.mean(ma, dim=0)

        prop_norm = {}
        for tmp_i, key in enumerate(prop2idx.keys()):
            prop_norm[key] = {
                'mean': mean[0, tmp_i].item(),
                'mad': mad[tmp_i].item()
            }
        return prop_norm

    # add the property of Cv thermo
    @staticmethod
    def sub_Cv_thermo(data):
        atom_types = data.atom_type
        atom_counts = torch.bincount(atom_types, minlength=len(Cv_atomref))
        property = data.y[0, 11] - torch.sum((atom_counts * torch.tensor(Cv_atomref)))
        return property


class QM9Dataset(QM9):
    def download(self):
        super(QM9Dataset, self).download()

    def process(self):
        super(QM9Dataset, self).process()

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        data = self.get(self.indices()[idx])
        data = data if self.transform is None else self.transform(data)

        rdmol2smiles, output_smiles = build_rdkit2rand_smiles_withoutH_mapping(data.rdmol, self.rand_smiles)
        rdmol2smiles = rdmol2smiles.tolist()
        smiles2selfies, selfies_tokens, selfies = get_smiles2selfies_mapping(output_smiles) # smiles2selfies is a dict

        ## update the data object with new information
        data['smiles'] = output_smiles
        data['selfies'] = selfies
        data['rdmol2smiles'] = rdmol2smiles
        data.pop('cano_smiles_woh')

        ## add the mapping from rdmol to selfies
        rdmol2selfies = torch.zeros((data.rdmol.GetNumAtoms(), len(selfies_tokens)), dtype=torch.float) # shape = [num_atoms, num_selfies_tokens]
        rdmol2selfies_mask = torch.zeros((data.rdmol.GetNumAtoms(),), dtype=torch.bool)
        for i, v in enumerate(rdmol2smiles):
            if v in smiles2selfies:
                for j in smiles2selfies[v]:
                    rdmol2selfies[i, j] = 1
                rdmol2selfies_mask[i] = True
        data['rdmol2selfies'] = rdmol2selfies
        data['rdmol2selfies_mask'] = rdmol2selfies_mask

        return data


class QM9LMDataset(QM9):
    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None, selfies_tokenizer: Optional[List[str]] = None, rand_smiles=False, addHs=False, aug_inv=False):
        super(QM9LMDataset, self).__init__(root, transform, pre_transform, pre_filter, selfies_tokenizer, rand_smiles, addHs)
        self.aug_inv = aug_inv

    def download(self):
        super(QM9LMDataset, self).download()

    def process(self):
        super(QM9LMDataset, self).process()

    def __getitem__(self, idx):
        data = self.get(self.indices()[idx])
        data = data if self.transform is None else self.transform(data)

        if self.rand_smiles and self.rand_smiles not in {'None', 'none', 'False', 'false'}:
            selfies = data['selfies']
            smiles = sf.decoder(selfies)
            if self.rand_smiles == 'restricted':
                rand_smiles = restricted_random_smiles(smiles, self.addHs)
            elif self.rand_smiles == 'unrestricted':
                rand_smiles = unrestricted_random_smiles(smiles, self.addHs)
            else:
                raise NotImplementedError()
            rand_selfies = sf.encoder(rand_smiles)
            data['selfies'] = rand_selfies
            if self.aug_inv:
                if self.rand_smiles == 'restricted':
                    rand_smiles = restricted_random_smiles(smiles, self.addHs)
                elif self.rand_smiles == 'unrestricted':
                    rand_smiles = unrestricted_random_smiles(smiles, self.addHs)
                else:
                    raise NotImplementedError()
                rand_selfies2 = sf.encoder(rand_smiles)
                data['selfies2'] = rand_selfies2
        return data


def randomSmiles(smiles):
    m1 = Chem.MolFromSmiles(smiles)
    m1.SetProp("_canonicalRankingNumbers", "True")
    idxs = list(range(0,m1.GetNumAtoms()))
    random.shuffle(idxs)
    for i,v in enumerate(idxs):
        m1.GetAtomWithIdx(i).SetProp("_canonicalRankingNumber", str(v))
    return Chem.MolToSmiles(m1)


def restricted_random_smiles(smiles, addHs=False):
    '''following https://github.com/undeadpixel/reinvent-randomized/blob/master/utils/chem.py'''
    mol = Chem.MolFromSmiles(smiles)
    if addHs:
        mol = Chem.AddHs(mol)
    new_atom_order = list(range(mol.GetNumAtoms()))
    random.shuffle(new_atom_order)
    random_mol = Chem.RenumberAtoms(mol, newOrder=new_atom_order)
    return Chem.MolToSmiles(random_mol, canonical=False, isomericSmiles=False)

def unrestricted_random_smiles(smiles, addHs=False):
    '''following https://github.com/undeadpixel/reinvent-randomized/blob/master/utils/chem.py'''
    mol = Chem.MolFromSmiles(smiles)
    if addHs:
        mol = Chem.AddHs(mol)

    while True:
        try:
            rand_smiles = Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False)
            break
        except:
            continue
    return rand_smiles

def remove_hs_and_build_map(mol):
    for atom in mol.GetAtoms():
        atom.SetProp("atom_index", str(atom.GetIdx()))
    new_mol = Chem.RemoveHs(mol)
    mapping = []
    for i, atom in enumerate(new_mol.GetAtoms()):
        assert atom.GetIdx() == i
        mapping.append(int(atom.GetProp("atom_index")))
    mapping = np.asarray(mapping)
    return new_mol, mapping

def build_smiles2selfies_mapping(molecule):
    canonical_smiles = Chem.MolToSmiles(molecule, canonical=True)

    smiles_atom_finder = re.compile(r"""
    (
    Cl? |             # Cl and Br are part of the organic subset
    Br? |
    [NOSPFIbcnosp*] | # as are these single-letter elements
    \[[^]]+\]         # everything else must be in []s
    )
    """, re.X)
    atom_positions = [x.start() for x in smiles_atom_finder.finditer(canonical_smiles)]

    selfies, attribution = sf.encoder(canonical_smiles, attribute=True)
    # selfies, attribution = modified_encoder(canonical_smiles, attribute=True) # TODO: modify selfies encoder

    return_type = "list" # TODO: determine return type
    if return_type == "bool":
        mapping = np.full((len(canonical_smiles), len(list(sf.split_selfies(selfies)))), False)
    elif return_type == "onehot":
        mapping = np.full((len(canonical_smiles), len(list(sf.split_selfies(selfies)))), 0)
    elif return_type == "list":
        mapping = [[] for _ in range(len(canonical_smiles))]

    for selfies_token in attribution:
        if selfies_token.attribution is None:
            continue

        for smiles_token in selfies_token.attribution:
                if return_type == "bool":
                    mapping[smiles_token.index, selfies_token.index] = True
                elif return_type == "onehot":
                    mapping[smiles_token.index, selfies_token.index] += 1
                elif return_type == "list":
                    mapping[smiles_token.index].append(selfies_token.index)

    smiles2selfies = mapping # TODO: atom assert?

    return smiles2selfies, selfies


if __name__ == '__main__':
    if False:
        with open('train_idx1.txt', 'r') as f:
            train_idx1 = f.read().splitlines()
        with open('valid_idx1.txt', 'r') as f:
            valid_idx1 = f.read().splitlines()
        with open('test_idx1.txt', 'r') as f:
            test_idx1 = f.read().splitlines()
        with open('train_idx2.txt', 'r') as f:
            train_idx2 = f.read().splitlines()
        with open('valid_idx2.txt', 'r') as f:
            valid_idx2 = f.read().splitlines()
        with open('test_idx2.txt', 'r') as f:
            test_idx2 = f.read().splitlines()

        print(len(set(train_idx1) - set(train_idx2)))
        print(len(set(valid_idx1) - set(valid_idx2)))
        print(len(set(test_idx1) - set(test_idx2)))

        exit()
    from transformers import AutoTokenizer
    from data_provider.qm9_dataset_v5 import QM9Dataset as QM9DatasetV5
    tokenizer = AutoTokenizer.from_pretrained('acharkq/MoLlama')

    dataset = QM9Dataset(root='./data/qm9v7', selfies_tokenizer=tokenizer, rand_smiles='restricted')


    exit()
    if True:
        splits = dataset.get_idx_split()
        train_idx = splits['train']
        valid_idx = splits['valid']
        test_idx = splits['test']

        ## filter the ones with no selfies
        selfies = np.array(dataset.data.selfies)
        print(len(train_idx), len(valid_idx), len(test_idx))
        train_idx = train_idx[selfies[train_idx] != np.array('')]
        valid_idx = valid_idx[selfies[valid_idx] != np.array('')]
        test_idx = test_idx[selfies[test_idx] != np.array('')]
        print(len(train_idx), len(valid_idx), len(test_idx))

        train_dataset = dataset.index_select(train_idx)
        val_dataset = dataset.index_select(valid_idx)
        test_dataset = dataset.index_select(test_idx)

    if True:
        for i in range(len(train_dataset)):
            data = train_dataset[i]
            assert data.selfies != '', print([data.selfies])
            # print(data)
            # input()
        for i in range(len(val_dataset)):
            assert val_dataset[i].selfies != ''
        for i in range(len(test_dataset)):
            assert test_dataset[i].selfies != ''

    dataset2 = QM9DatasetV5(root='./data/archive/qm9v6', selfies_tokenizer=tokenizer, rand_smiles='restricted')
    idx1 = dataset._data.idx.tolist()
    idx2 = dataset2._data.idx.tolist()
    idx_set2 = set(idx2)
    for i in idx1:
        assert i in idx_set2

    exit()
    if True:
        splits = dataset.get_idx_split()
        train_idx = splits['train']
        valid_idx = splits['valid']
        test_idx = splits['test']

        ## filter the ones with no selfies
        selfies = np.array(dataset.data.selfies)
        print(len(train_idx), len(valid_idx), len(test_idx))
        train_idx = train_idx[selfies[train_idx] != np.array('')]
        valid_idx = valid_idx[selfies[valid_idx] != np.array('')]
        test_idx = test_idx[selfies[test_idx] != np.array('')]
        print(len(train_idx), len(valid_idx), len(test_idx))


        print('running here 1')
        train_dataset2 = dataset.index_select(train_idx)
        val_dataset2 = dataset.index_select(valid_idx)
        test_dataset2 = dataset.index_select(test_idx)

        print('running here 2')
        train_idx2 = [int(train_dataset2[i].idx) for i in range(len(train_dataset2))]
        valid_idx2 = [int(val_dataset2[i].idx) for i in range(len(val_dataset2))]
        test_idx2 = [int(test_dataset2[i].idx) for i in range(len(test_dataset2))]
        with open('train_idx2.txt', 'w') as f:
            for i in train_idx2:
                f.write(str(i) + '\n')
        with open('valid_idx2.txt', 'w') as f:
            for i in valid_idx2:
                f.write(str(i) + '\n')
        with open('test_idx2.txt', 'w') as f:
            for i in test_idx2:
                f.write(str(i) + '\n')
        print('running here 3')
        train_idx1 = [int(train_dataset[i].idx) for i in range(len(train_dataset))]
        valid_idx1 = [int(val_dataset[i].idx) for i in range(len(val_dataset))]
        test_idx1 = [int(test_dataset[i].idx) for i in range(len(test_dataset))]
        with open('train_idx1.txt', 'w') as f:
            for i in train_idx1:
                f.write(str(i) + '\n')
        with open('valid_idx1.txt', 'w') as f:
            for i in valid_idx1:
                f.write(str(i) + '\n')
        with open('test_idx1.txt', 'w') as f:
            for i in test_idx1:
                f.write(str(i) + '\n')

        print('running here 4')
        # train_idx_set2 = set(train_idx2)
        # valid_idx_set2 = set(valid_idx2)
        test_idx_set2 = set(test_idx2)
        # for i in train_idx1:
        #     assert i in train_idx_set2
        # for i in valid_idx1:
        #     assert i in valid_idx_set2
        for i in test_idx1:
            assert i in test_idx_set2

