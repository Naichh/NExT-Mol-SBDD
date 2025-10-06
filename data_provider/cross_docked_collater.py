import torch
from torch_geometric.data import Batch

class LMCollater:
    def __init__(self, tokenizer, max_sf_tokens, max_pocket_tokens):
        self.tokenizer = tokenizer
        self.max_sf_tokens = max_sf_tokens
        self.max_pocket_tokens = max_pocket_tokens

    def prepare_inputs_for_causal_lm_left_pad(self, tensor_list, L):
        B = len(tensor_list)
        d = tensor_list[0].shape[1]
        input_embeddings = torch.zeros((B, L, d), dtype=tensor_list[0].dtype)
        attention_mask = torch.zeros((B, L), dtype=torch.long)
        for i, t in enumerate(tensor_list):
            n = min(t.shape[0], L)
            input_embeddings[i, L - n:] = t[:n]
            attention_mask[i, L - n:] = 1
        return input_embeddings, attention_mask


    def __call__(self, batch):
        batch = [item for item in batch if item is not None]
        if not batch:
            return None, None, None, None, None

        pockets = [item['pdb_embedding'] for item in batch]
        ground_truth_mols = [item['rdmol'] for item in batch]
        pocket_paths = [item['pdb_path'] for item in batch]

        num_augmentations = len(batch[0]['selfies_list'])

        list_of_selfies_batches = []
        eos_token = self.tokenizer.eos_token

        for i in range(num_augmentations):
            current_aug_selfies = [item['selfies_list'][i] for item in batch]

            selfies_with_eos = [s + eos_token for s in current_aug_selfies]
            selfies_batch_i = self.tokenizer(selfies_with_eos, padding='max_length', return_tensors='pt',
                                           max_length=self.max_sf_tokens, truncation=True, add_special_tokens=False)
            list_of_selfies_batches.append(selfies_batch_i)

        pocket_embs, pocket_mask = self.prepare_inputs_for_causal_lm_left_pad(pockets, self.max_pocket_tokens)

        return list_of_selfies_batches, pocket_embs, pocket_mask, ground_truth_mols, pocket_paths
class LMInferCollater:
    def __init__(self, tokenizer, max_pocket_tokens):
        self.tokenizer = tokenizer
        self.max_pocket_tokens = max_pocket_tokens

    def prepare_inputs_for_causal_lm_left_pad(self, tensor_list, L):
        B = len(tensor_list)
        d = tensor_list[0].shape[1]
        input_embeddings = torch.zeros((B, L, d), dtype=tensor_list[0].dtype)
        attention_mask = torch.zeros((B, L), dtype=torch.long)
        for i, t in enumerate(tensor_list):
            n = min(t.shape[0], L)
            input_embeddings[i, L - n:] = t[:n]
            attention_mask[i, L - n:] = 1
        return input_embeddings, attention_mask

    def __call__(self, batch):
        pockets = [item['pdb_embedding'] for item in batch]
        pocket_embs, pocket_mask = self.prepare_inputs_for_causal_lm_left_pad(pockets, self.max_pocket_tokens)
        return pocket_embs, pocket_mask


from torch_geometric.data import Batch
import torch

class InferCollater:
    def __init__(self, max_atoms, max_sf_tokens, selfies_tokenizer, disable_com=False, use_eigvec=False, load_mapping=True):
        self.max_atoms = max_atoms
        self.max_sf_tokens = max_sf_tokens
        self.selfies_tokenizer = selfies_tokenizer
        self.disable_com = disable_com
        self.use_eigvec = use_eigvec
        self.load_mapping = load_mapping

    def __call__(self, batch):
        selfies = [item['selfies'] for item in batch]
        rdmol2selfies = [item['rdmol2selfies'] for item in batch]
        rdmol2selfies_mask = [item['rdmol2selfies_mask'] for item in batch]

        # Tokenize selfies
        self.selfies_tokenizer.padding_side = 'right'
        sf_batch = self.selfies_tokenizer(
            selfies, padding='max_length', return_tensors='pt',
            max_length=self.max_sf_tokens, truncation=True, add_special_tokens=True
        )

        # Graph batching
        data_batch = Batch.from_data_list(batch)

        # Mapping padding
        B = len(batch)
        sf_len = sf_batch.input_ids.shape[1]
        atom_len = self.max_atoms

        padded_rdmol2selfies = torch.zeros((B, atom_len, sf_len), dtype=torch.long)
        padded_rdmol2selfies_mask = torch.zeros((B, atom_len), dtype=torch.bool)

        for i in range(B):
            m = rdmol2selfies[i]
            m_mask = rdmol2selfies_mask[i]
            padded_rdmol2selfies[i, :m.shape[0], :m.shape[1]] = m
            padded_rdmol2selfies_mask[i, :m_mask.shape[0]] = m_mask

        data_batch['rdmol2selfies'] = padded_rdmol2selfies
        data_batch['rdmol2selfies_mask'] = padded_rdmol2selfies_mask

        return data_batch, sf_batch
