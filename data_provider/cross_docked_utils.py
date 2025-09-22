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
        # --- 如果缓存存在，逻辑不变 ---
        if self.data_cache is not None:
            original_idx = self.index[idx]
            return self.data_cache[original_idx]

        # --- 对磁盘读取逻辑进行修改 ---
        try:
            pocket_fn, ligand_fn, protein_fn, rmsd = self.index[idx]

            pocket_path = os.path.join(self.raw_path, pocket_fn)
            ligand_path = os.path.join(self.raw_path, ligand_fn)

            pt_embed_path = pocket_path.replace('.pdb', '.pt')
            
            # 核心改动：如果 embedding 文件不存在，torch.load 会抛出 FileNotFoundError
            embedding = torch.load(pt_embed_path, map_location='cpu')

            # 后续的SDF处理等逻辑保持不变
            mol = Chem.SDMolSupplier(ligand_path, removeHs=False, sanitize=True)[0]
            if mol is None:
                return None # 如果SDF有问题，也跳过这个样本
            
            smiles = Chem.MolToSmiles(mol)

            # 假设您的随机化和selfies转换函数在这里
            smiles1 = restricted_random_smiles(smiles, self.addHs)
            smiles2 = restricted_random_smiles(smiles, self.addHs)
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
        
        except FileNotFoundError:
            # 当 torch.load 找不到 .pt 文件时，捕获错误并返回None
            return None
        except Exception as e:
            # 捕获其他可能的错误，同样跳过样本
            # print(f"\n[!] 在 __getitem__ 中发生错误，跳过样本 {self.index[idx][0]}。错误: {e}")
            return None







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
