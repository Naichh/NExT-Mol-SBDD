import torch
import lightning.pytorch as pl
import os
import pickle
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataset import random_split
from .cross_docked_utils import PocketLigandPairDataset
from .cross_docked_collater import LMCollater
from pathlib import Path
from evaluation.eval_functions import get_moses_metrics

class PocketLigandDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_root: str,
        split_file: str,
        tokenizer,
        batch_size: int = 64,
        eval_batch_size: int=32,
        num_workers: int = 4,
        max_sf_tokens: int = 128,
        max_pocket_tokens: int = 128,
        **kwargs
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.split_file = split_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = tokenizer
        self.max_sf_tokens = max_sf_tokens
        self.max_pocket_tokens = max_pocket_tokens
        self.eval_batch_size=eval_batch_size
        self.cache_dir = "/data/share/liuzhiyuan/nai/NExT-Mol/datasets/cache/"
        self.cached_data_path = os.path.join(self.cache_dir, "all_data_stage1_cache.pt")

    def prepare_data(self):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
            print(f"INFO: Created temporary cache directory: {self.cache_dir}")

        if os.path.exists(self.cached_data_path):
            print(f"INFO: 找到了预计算的缓存文件 '{self.cached_data_path}'，跳过数据准备。")
            return


        print("INFO: 未找到缓存文件，开始在Rank 0上进行一次性数据预加载...")

        full_dataset_on_disk = PocketLigandPairDataset(self.dataset_root)

        print(f"INFO: Pre-loading all {len(full_dataset_on_disk)} samples into RAM. This may take a while...")
        data_cache = {}
        with open(full_dataset_on_disk.index_path, 'rb') as f:
            index_list = pickle.load(f)

        for i in tqdm(range(len(index_list)), desc="Caching data into RAM on Rank 0"):
            sample = full_dataset_on_disk[i]
            if sample is not None:
                original_idx_tuple = index_list[i]
                data_cache[original_idx_tuple] = sample

        print(f"INFO: 数据缓存完成，正在保存到 '{self.cached_data_path}'...")
        torch.save(data_cache, self.cached_data_path)
        print("INFO: 缓存文件保存成功。")

    def setup(self, stage=None):
        print(f"INFO: Rank {self.trainer.global_rank} 正在设置 DataModule...")

        print(f"INFO: Rank {self.trainer.global_rank} is loading data from '{self.cached_data_path}'")
        full_data_cache = torch.load(self.cached_data_path, map_location='cpu')

        cached_dataset = PocketLigandPairDataset(self.dataset_root, data_cache=full_data_cache)
        index_to_original_key_map = cached_dataset.index
        split = torch.load(self.split_file)

        print("INFO: Building fast lookup map...")
        short_key_to_int_idx_map = {
            (key[0], key[1]): i
            for i, key in enumerate(cached_dataset.index)
        }

        def convert_split_to_indices(file_pairs):
            indices = []
            for pocket_fn, ligand_fn in tqdm(file_pairs, desc="Converting split files to indices"):
                short_key = (pocket_fn, ligand_fn)
                idx = short_key_to_int_idx_map.get(short_key)
                if idx is not None:
                    indices.append(idx)
            return indices

        train_indices = convert_split_to_indices(split['train'])
        test_indices = convert_split_to_indices(split['test'])

        self.train_dataset = Subset(cached_dataset, train_indices)
        self.test_dataset = Subset(cached_dataset, test_indices)
        print("INFO: Efficiently extracting train molecules by index...")
        self.train_rdmols = []
        for idx in tqdm(train_indices, desc="Extracting train rdmol"):
            original_key = index_to_original_key_map[idx]
            sample = full_data_cache[original_key]
            if sample is not None and 'rdmol' in sample:
                self.train_rdmols.append(sample['rdmol'])

        print("INFO: Efficiently extracting test molecules DIRECTLY from cache...")
        self.test_rdmols = []
        for idx in tqdm(test_indices, desc="Extracting test rdmol"):
            original_key = index_to_original_key_map[idx]
            sample = full_data_cache[original_key]
            if sample is not None and 'rdmol' in sample:
                self.test_rdmols.append(sample['rdmol'])
        self.get_moses_metrics = get_moses_metrics(self.test_rdmols, 1)
        print("INFO: DataModule setup is complete. Training will now run entirely from RAM.")


    def train_dataloader(self):
        dataloader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': self.num_workers,
            'collate_fn': LMCollater(
                tokenizer=self.tokenizer,
                max_sf_tokens=self.max_sf_tokens,
                max_pocket_tokens=self.max_pocket_tokens
            ),
            'drop_last': True,
            'pin_memory': True,
        }

        if self.num_workers > 0:
            dataloader_kwargs['multiprocessing_context'] = 'spawn'
            dataloader_kwargs['persistent_workers'] = True # Recommended for performance

        return DataLoader(self.train_dataset, **dataloader_kwargs)


    def val_dataloader(self):
        dataloader_kwargs = {
            'batch_size': self.eval_batch_size,
            'shuffle': False,
            'num_workers': 0,
            'collate_fn': LMCollater(
                tokenizer=self.tokenizer,
                max_sf_tokens=self.max_sf_tokens,
                max_pocket_tokens=self.max_pocket_tokens
            ),
            'drop_last': False,
            'pin_memory': True,
        }

        # if self.num_workers > 0:
        #     dataloader_kwargs['multiprocessing_context'] = 'spawn'
        #     # persistent_workers is often set to True here as well for speed
        #     dataloader_kwargs['persistent_workers'] = True

        return DataLoader(self.test_dataset, **dataloader_kwargs)

























class PocketLigandDataModule_DF(pl.LightningDataModule):
    def __init__(
        self,
        split_file: str,
        root: str = '/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/data/sbdd/crossdocked_pocket',
        num_workers: int = 0,
        batch_size: int = 256,
        selfies_tokenizer = None,
        args=None,
        eval_batch_size: int=32,
    ):
        super().__init__()
        root = Path(root)
        self.args = args
        self.root = root
        self.split_file = split_file

        self.discrete_schedule = args.discrete_schedule if hasattr(args, 'discrete_schedule') else False
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.eval_batch_size=eval_batch_size,
        self.selfies_tokenizer = selfies_tokenizer
        self.use_eigvec = args.use_eigvec if hasattr(args, 'use_eigvec') else False
        self.infer_batch_size = args.infer_batch_size if hasattr(args, 'infer_batch_size') else 64
        self.flatten_dataset = args.flatten_dataset if hasattr(args, 'flatten_dataset') else False
        self.disable_com = args.disable_com if hasattr(args, 'disable_com') else False
        self.add_unseen_selfies_tokens(self.selfies_tokenizer, root / 'processed')

        if not hasattr(args, 'condition_property') or args.condition_property == None:
            self.transform = None
        elif args.condition_property in ['mu', 'alpha', 'homo', 'lumo', 'gap', 'Cv']:
            dataset_info = get_dataset_info('qm9_second_half')
            prop2idx = dataset_info['prop2idx']
            include_aromatic = False
            self.transform = EdgeComCondTransform(dataset_info['atom_encoder'].values(), include_aromatic, prop2idx[args.condition_property])
        else:
            raise NotImplementedError(f"{args.conditon} is not supported")

        self.rand_smiles = args.rand_smiles if hasattr(args, 'rand_smiles') else 'restricted'
        self.addHs = args.addHs if hasattr(args, 'addHs') else False
        self.infer_noise = args.infer_noise if hasattr(args, 'infer_noise') else 0.9999

        assert not self.use_eigvec, 'old version of QM9 dataset does not have eigenvectors'

        dataset = QM9Dataset(root=root, selfies_tokenizer=selfies_tokenizer, rand_smiles=self.rand_smiles)
        self.dataset = dataset
        max_atoms = int(dataset._data.num_atom.max()) + 2 # +2 because of the bos and eos token;
        self.max_atoms = max_atoms
        print('QM9 max num atoms', max_atoms)

        ## obtain max selfies token length
        selfies_list = dataset._data['selfies']
        selfies_lens = [len(list(sf.split_selfies(selfies))) for selfies in selfies_list]
        self.max_sf_tokens = max(max(selfies_lens) + 2 + 5, 96) # +2 because of the bos and eos token; +5 to enlarge the space of molecule sampling
        print('max selfies tokens', self.max_sf_tokens)

        # self.max_atoms = 31
        # self.max_sf_tokens = 28


        # print('QM9 max num atoms', self.max_atoms)
        # print('max selfies tokens', self.max_sf_tokens)

        if args.condition_property == None:
            splits = dataset.get_idx_split()
            train_idx = splits['train']
            valid_idx = splits['valid']
            test_idx = splits['test']
        elif args.condition_property in ['mu', 'alpha', 'homo', 'lumo', 'gap', 'Cv']:
            splits = dataset.get_cond_idx_split()
            first_train_idx = splits['first_train']
            second_train_idx = splits['second_train']
            valid_idx = splits['valid']
            test_idx = splits['test']

            train_idx = second_train_idx

        ## filter the ones without selfies
        selfies = np.array(dataset._data.selfies)

        print('before filtering', len(train_idx), len(valid_idx), len(test_idx))
        train_idx = train_idx[train_idx < len(dataset)]
        valid_idx = valid_idx[valid_idx < len(dataset)]
        test_idx = test_idx[test_idx < len(dataset)]
        train_idx = train_idx[selfies[train_idx] != np.array('')]
        valid_idx = valid_idx[selfies[valid_idx] != np.array('')]
        test_idx = test_idx[selfies[test_idx] != np.array('')]
        print('after filtering', len(train_idx), len(valid_idx), len(test_idx))
        self.train_dataset = tordf_version(dataset.index_select(train_idx), max_atoms, self.rand_smiles, self.addHs, self.transform, 'train')
        self.valid_dataset = tordf_version(dataset.index_select(valid_idx), max_atoms, self.rand_smiles, self.addHs, self.transform, 'valid')
        self.test_dataset = tordf_version(dataset.index_select(test_idx), max_atoms, self.rand_smiles, self.addHs, self.transform, 'infer', self.infer_noise)
        self.predict_dataset = None

        if args.condition_property == None:
            pass
        elif args.condition_property in ['mu', 'alpha', 'homo', 'lumo', 'gap', 'Cv']:
            prop2idx_sub = {
                args.condition_property: prop2idx[args.condition_property]
            }
            self.prop_norms = dataset.index_select(valid_idx).compute_property_mean_mad(prop2idx_sub)
            self.prop_dist = DistributionProperty(dataset.index_select(train_idx), prop2idx_sub, normalizer=self.prop_norms)
        else:
            raise NotImplementedError(f"{args.conditon} is not supported")

        ## load rdmols of subsets
        rdmols = dataset._data.rdmol
        train_idx = train_idx.tolist()
        valid_idx = valid_idx.tolist()
        test_idx = test_idx.tolist()
        self.train_rdmols = [rdmols[i] for i in train_idx]
        self.valid_rdmols = [rdmols[i] for i in valid_idx]
        self.test_rdmols = [rdmols[i] for i in test_idx]

        self.get_moses_metrics = get_moses_metrics(self.test_rdmols, 1)
        self.get_sub_geometry_metric = get_sub_geometry_metric(self.test_rdmols, get_dataset_info('qm9_with_h'), os.path.join(root, 'processed'))

        self.aug_rotation = (not args.not_aug_rotation) if hasattr(args, 'not_aug_rotation') else True
        self.aug_translation = args.aug_translation if hasattr(args, 'aug_translation') else False
        self.t_cond = args.t_cond if hasattr(args, 't_cond') else 't'
        # self.pos_std = self.test_dataset.pos_std
        self.pos_std = 1.7226

        noise_scheduler = args.noise_scheduler if hasattr(args, 'noise_scheduler') else 'cosine'
        continuous_beta_0 = args.continuous_beta_0 if hasattr(args, 'continuous_beta_0') else 0.1
        continuous_beta_1 = args.continuous_beta_1 if hasattr(args, 'continuous_beta_1') else 20
        self.noise_scheduler = NoiseScheduleVPV2(noise_scheduler, continuous_beta_0=continuous_beta_0, continuous_beta_1=continuous_beta_1, discrete_mode=self.discrete_schedule)

        if args.eval_smiles_path is not None:
            with open(args.eval_smiles_path, 'r') as f:
                lines = f.readlines()
                sampled_sequences = [line.strip().split() for line in lines]
                _, _, smiles_smiles = zip(*sampled_sequences)
                self.predict_dataset = PredictDataset(smiles_smiles, self.max_atoms, self.rand_smiles, args.dataset)


    def add_unseen_selfies_tokens(self, tokenizer, root_path):
        with open(root_path / 'unseen_selfies_tokens.txt', 'r') as f:
            unseen_tokens = f.read().splitlines()
        vocab = tokenizer.get_vocab()
        for token in unseen_tokens:
            if token not in vocab:
                tokenizer.add_tokens(token)

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=QM9Collater(
                max_atoms=self.max_atoms,
                max_sf_tokens=self.max_sf_tokens,
                selfies_tokenizer=self.selfies_tokenizer,
                noise_scheduler=self.noise_scheduler,
                aug_rotation=self.aug_rotation,
                t_cond=self.t_cond,
                use_eigvec=self.use_eigvec,
                disable_com=self.disable_com,
                aug_translation=self.aug_translation,
                condition=(self.transform is not None),
                prop_norm=self.prop_norms if self.transform is not None else None,
                mode='train',
            )
        )
        return loader

    def val_dataloader(self):
        print('validating')
        val_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
            collate_fn=QM9Collater(
                max_atoms=self.max_atoms,
                max_sf_tokens=self.max_sf_tokens,
                selfies_tokenizer=self.selfies_tokenizer,
                noise_scheduler=self.noise_scheduler,
                aug_rotation=self.aug_rotation,
                t_cond=self.t_cond,
                use_eigvec=self.use_eigvec,
                disable_com=self.disable_com,
                aug_translation=self.aug_translation,
                condition=(self.transform is not None),
                prop_norm=self.prop_norms if self.transform is not None else None,
                mode='infer',
            )
        )
        if hasattr(self, 'predict_dataset'):
            predict_loader = DataLoader(
                self.predict_dataset,
                batch_size=self.infer_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=False,
                persistent_workers=False,
                collate_fn=QM9InferCollater(self.max_atoms, self.max_sf_tokens, self.selfies_tokenizer, load_mapping=False)
                ),
            return [val_loader, predict_loader]
        else:
            return val_loader

    # def setup_predict_dataset(self, smiles_list):
    #     if self.predict_dataset is not None:
    #         return

    #     print('Setup predict dataset...', end='')
    #     self.predict_dataset = predict_dataset(smiles_list, self.max_atoms, self.rand_smiles, self.args.dataset)
    #     print('done')

    # def predict_dataloader(self):
    #     assert self.predict_dataset is not None
    #     loader = DataLoader(
    #             self.predict_dataset,
    #             batch_size=self.args.infer_batch_size,
    #             shuffle=False,
    #             num_workers=self.num_workers * 2,
    #             pin_memory=False,
    #             drop_last=False,
    #             persistent_workers=False,
    #             collate_fn=QM9InferCollater(self.max_atoms, self.max_sf_tokens, self.selfies_tokenizer, self.noise_scheduler, self.use_eigvec, self.disable_com),
    #         )
    #     return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--infer_batch_size', type=int, default=64)
        parser.add_argument('--rand_smiles', type=str, default='restricted')
        parser.add_argument('--root', type=str, default='data/tordf_qm9')
        parser.add_argument('--addHs', action='store_true', default=False)
        parser.add_argument('--infer_time', type=float, default=0.9946)
        parser.add_argument('--infer_noise', type=float, default=0.9999)
        parser.add_argument('--use_eigvec', action='store_true', default=False)
        parser.add_argument('--t_cond', type=str, default='t')
        parser.add_argument('--discrete_schedule', action='store_true', default=False)
        parser.add_argument('--not_aug_rotation', action='store_true', default=False)
        parser.add_argument('--aug_translation', action='store_true', default=False)
        parser.add_argument('--flatten_dataset', action='store_true', default=False)
        parser.add_argument('--condition_property', type=str, default=None)
        parser.add_argument('--eval_smiles_path', type=str, default=None)
        return parent_parser
