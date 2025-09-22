import torch
import lightning.pytorch as pl
import os
import pickle
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



    # Inside your PocketLigandDataModule class:
    def setup(self, stage=None):
        cache_dir = os.path.join(self.dataset_root, 'cache')
        file_to_idx_cache_path = os.path.join(cache_dir, 'file_to_idx.pkl')
        train_cache_path = os.path.join(cache_dir, 'train_rdmols.pkl')
        test_cache_path = os.path.join(cache_dir, 'test_rdmols.pkl')

        try:
            # --- FAST PATH: Try to load from pre-computed cache ---
            print("INFO: Attempting to load data from pre-computed cache...")
            
            # Check if all required files exist
            for path in [file_to_idx_cache_path, train_cache_path, test_cache_path]:
                if not os.path.exists(path):
                    # If any file is missing, jump to the except block
                    raise FileNotFoundError(f"Cache file not found: {path}")

            # Load all caches
            with open(file_to_idx_cache_path, 'rb') as f:
                file_to_idx = pickle.load(f)
            
            split = torch.load(self.split_file)
            full_dataset = PocketLigandPairDataset(self.dataset_root)

            # Define the conversion function locally
            def convert_split_to_indices(file_pairs):
                indices = []
                for pocket_fn, ligand_fn in file_pairs:
                    key = (pocket_fn, ligand_fn)
                    if key in file_to_idx:
                        indices.append(file_to_idx[key])
                return indices

            train_indices = convert_split_to_indices(split['train'])
            test_indices = convert_split_to_indices(split['test'])
            
            self.train_dataset = Subset(full_dataset, train_indices)
            self.test_dataset = Subset(full_dataset, test_indices)
            
            with open(train_cache_path, 'rb') as f: self.train_rdmols = pickle.load(f)
            with open(test_cache_path, 'rb') as f: self.test_rdmols = pickle.load(f)

            print("INFO: Successfully loaded all data from cache.")

        except FileNotFoundError:
            # --- SLOW PATH: Fallback to on-the-fly loading if cache is missing ---
            print("\n" + "="*80)
            print("WARNING: Cache not found or incomplete. Falling back to slow, on-the-fly data loading.")
            print("         For a significant speed-up, please run preprocess_dataset.py first.")
            print("="*80 + "\n")
            
            # Re-implement the on-the-fly logic without saving any caches
            full_dataset = PocketLigandPairDataset(self.dataset_root)
            split = torch.load(self.split_file)

            # Build the map in memory for this run only
            file_to_idx_map = {
                (p, l): i
                for i, (p, l, _, _) in enumerate(full_dataset.index)
                if p is not None and l is not None
            }
            
            def convert_split_to_indices_runtime(file_pairs):
                indices = []
                for pocket_fn, ligand_fn in file_pairs:
                    key = (pocket_fn, ligand_fn)
                    if key in file_to_idx_map:
                        indices.append(file_to_idx_map[key])
                return indices

            train_indices = convert_split_to_indices_runtime(split['train'])
            test_indices = convert_split_to_indices_runtime(split['test'])
            
            self.train_dataset = Subset(full_dataset, train_indices)
            self.test_dataset = Subset(full_dataset, test_indices)
            
            print("WARNING: Parsing RDKit molecules on-the-fly. This will be slow...")
            self.train_rdmols = [self.train_dataset[i]['rdmol'] for i in range(len(self.train_dataset)) if self.train_dataset[i]['rdmol'] is not None]
            self.test_rdmols = [self.test_dataset[i]['rdmol'] for i in range(len(self.test_dataset)) if self.test_dataset[i]['rdmol'] is not None]

        # This part is common to both paths
        self.get_moses_metrics = get_moses_metrics(self.test_rdmols, 1)
        print("INFO: DataModule setup is complete.")


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
            'num_workers': self.num_workers,
            'collate_fn': LMCollater(
                tokenizer=self.tokenizer,
                max_sf_tokens=self.max_sf_tokens,
                max_pocket_tokens=self.max_pocket_tokens
            ),
            'drop_last': False,
            'pin_memory': True,
        }

        if self.num_workers > 0:
            dataloader_kwargs['multiprocessing_context'] = 'spawn'
            # persistent_workers is often set to True here as well for speed
            dataloader_kwargs['persistent_workers'] = True 

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
