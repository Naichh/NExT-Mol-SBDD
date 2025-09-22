import os
import pickle
from torch.utils.data import Dataset
from rdkit import Chem
import selfies as sf
import torch


import os
import pickle
from torch.utils.data import Dataset
from rdkit import Chem
import selfies as sf
import torch
import random


def restricted_random_smiles(smiles, addHs=False):
    mol = Chem.MolFromSmiles(smiles)
    if addHs:
        mol = Chem.AddHs(mol)
    new_order = list(range(mol.GetNumAtoms()))
    random.shuffle(new_order)
    new_mol = Chem.RenumberAtoms(mol, newOrder=new_order)
    return Chem.MolToSmiles(new_mol, canonical=False, isomericSmiles=False)


def unrestricted_random_smiles(smiles, addHs=False):
    mol = Chem.MolFromSmiles(smiles)
    if addHs:
        mol = Chem.AddHs(mol)
    while True:
        try:
            return Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False)
        except:
            continue


class PocketLigandPairDataset(Dataset):
    # <<< 修改点 #1: 在 __init__ 方法中增加一个参数 `data_cache` >>>
    def __init__(self, raw_path, rand_smiles='restricted', addHs=False, data_cache=None):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        self.rand_smiles = rand_smiles
        self.addHs = addHs
        
        # <<< 新增：将缓存保存为类属性 >>>
        self.data_cache = data_cache

        # 如果没有提供缓存，才加载索引文件（用于预加载阶段）
        if self.data_cache is None:
            print(f"--> [Disk Mode] 正在从 '{self.index_path}' 加载主索引文件...")
            with open(self.index_path, 'rb') as f:
                self.index = pickle.load(f)
            print(f"--> [Disk Mode] 主索引加载完成，共找到 {len(self.index)} 个样本。")
        else:
            print("--> [RAM Mode] Dataset is running in RAM-cached mode.")
            # 在缓存模式下，长度由缓存的键（也就是原始索引）决定
            # 为了能通过 Subset 访问，我们仍然需要一个索引列表
            self.index = list(self.data_cache.keys())


    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        # <<< 修改点 #2: __getitem__ 的核心逻辑 >>>
        
        # 如果缓存存在，直接从缓存中返回数据
        if self.data_cache is not None:
            # Subset 会传入索引列表的下标，我们需要通过它找到原始的idx
            original_idx = self.index[idx]
            return self.data_cache[original_idx]

        # --- 如果缓存不存在（仅在setup预加载时执行），则执行原始的磁盘读取逻辑 ---
        pocket_fn, ligand_fn, protein_fn, rmsd = self.index[idx]

        pocket_path = os.path.join(self.raw_path, pocket_fn)
        ligand_path = os.path.join(self.raw_path, ligand_fn)

        pt_embed_path = pocket_path.replace('.pdb', '.pt')
        embedding = torch.load(pt_embed_path, map_location='cpu') # 加上 map_location 是个好习惯

        if embedding is None:
            raise FileNotFoundError(f"Embedding could not be loaded for sample: {pt_embed_path}")

        mol = Chem.SDMolSupplier(ligand_path, removeHs=False, sanitize=True)[0]
        if mol is None:
            raise ValueError(f'RDKit failed to read ligand SDF: {ligand_path}')
        
        smiles = Chem.MolToSmiles(mol)

        if self.rand_smiles == 'restricted':
            smiles1 = restricted_random_smiles(smiles, self.addHs)
            smiles2 = restricted_random_smiles(smiles, self.addHs)
        elif self.rand_smiles == 'unrestricted':
            smiles1 = unrestricted_random_smiles(smiles, self.addHs)
            smiles2 = unrestricted_random_smiles(smiles, self.addHs)
        else:
            raise NotImplementedError(f"rand_smiles mode {self.rand_smiles} not supported.")

        selfies1 = sf.encoder(smiles1)
        selfies2 = sf.encoder(smiles2)

        return {
            'pdb_path': pocket_fn,
            'pdb_embedding': embedding.clone().detach(),
            'sdf_path': ligand_path,
            'selfies': selfies1,
            'selfies2': selfies2,
            'rmsd': torch.tensor(rmsd, dtype=torch.float),
            'rdmol': mol
        }








class PocketLigandPairDataset_Diffusion(Dataset):
    def __init__(self, raw_path, rand_smiles='restricted', addHs=False):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        self.rand_smiles = rand_smiles
        self.addHs = addHs

        with open(self.index_path, 'rb') as f:
            self.index = pickle.load(f)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):

        pocket_fn, ligand_fn, protein_fn, rmsd = self.index[idx]

        pocket_path = os.path.join(self.raw_path, pocket_fn)
        ligand_path = os.path.join(self.raw_path, ligand_fn)

        # embed_path = pocket_path.replace('.pdb', '.pt')
        # if not os.path.exists(embed_path):
        #     raise FileNotFoundError(f'Embedding file not found: {embed_path}')
        # embedding=torch.load(embed_path)
        embed_path = pocket_path.replace('.pdb', '.pkl')
        if not os.path.exists(embed_path):
            raise FileNotFoundError(f'Embedding file not found: {embed_path}')
        with open(embed_path, 'rb') as f:
            embedding_numpy = pickle.load(f)

        embedding = torch.from_numpy(embedding_numpy)





        mol = Chem.SDMolSupplier(ligand_path, removeHs=False, sanitize=True)[0]
        if mol is None:
            raise ValueError(f'RDKit failed to read ligand SDF: {ligand_path}')

        smiles = Chem.MolToSmiles(mol)

        # Get two random SMILES, convert to selfies
        if self.rand_smiles == 'restricted':
            smiles1 = restricted_random_smiles(smiles, self.addHs)
            smiles2 = restricted_random_smiles(smiles, self.addHs)
        elif self.rand_smiles == 'unrestricted':
            smiles1 = unrestricted_random_smiles(smiles, self.addHs)
            smiles2 = unrestricted_random_smiles(smiles, self.addHs)
        else:
            raise NotImplementedError(f"rand_smiles mode {self.rand_smiles} not supported.")

        selfies1 = sf.encoder(smiles1)
        selfies2 = sf.encoder(smiles2)

        return {
            'pdb_path': pocket_path,
            'pdb_embedding': embedding.clone().detach(),
            'sdf_path': ligand_path,
            'selfies': selfies1,
            'selfies2': selfies2,
            'rmsd': torch.tensor(rmsd, dtype=torch.float)
        }

    @staticmethod
    def collate_fn(batch):
        return {
            'pdb_path': [item['pdb_path'] for item in batch],
            'pdb_embedding': [item['pdb_embedding'] for item in batch],
            'sdf_path': [item['sdf_path'] for item in batch],
            'selfies': [item['selfies'] for item in batch],
            'selfies2': [item['selfies2'] for item in batch],
            'rmsd': torch.stack([item['rmsd'] for item in batch]),
        }


























if __name__ == "__main__":

    from torch.utils.data import DataLoader

    dataset = PocketLigandPairDataset('/home/mydata/nextmol/crossdocked_pocket')
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=PocketLigandPairDataset.collate_fn)

    for batch in loader:
        print(batch['pdb_path'])
        print(batch['pdb_embedding'][0].shape)
        print(batch['sdf_path'])
        print(batch['selfies'])
        print(batch['selfies2'])
        print(batch['rmsd'])
        break
    # raw_path = "/home/mydata/nextmol/crossdocked_pocket"
    # index_path = os.path.join(raw_path, 'index.pkl')

    # with open(index_path, 'rb') as f:
    #     index = pickle.load(f)
    # import pdb;pdb.set_trace()
