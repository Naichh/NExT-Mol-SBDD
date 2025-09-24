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
        # 关键修改：在处理之前，过滤掉所有值为 None 的无效样本
        batch = [item for item in batch if item is not None]
        
        # 如果过滤后整个批次都空了，返回None，让训练循环跳过
        if not batch:
            # 构造一条详细的日志信息
            log_message = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] --- CRITICAL DIAGNOSIS (Worker PID: {self.worker_id}) ---\n" \
                          f"The 'if not batch:' check was successfully triggered.\n" \
                          f"This proves the LATEST version of LMCollater is running and has correctly detected an empty batch.\n" \
                          f"Returning (None, ...) to skip this batch.\n\n"
            
            # 1. 在终端打印，以便实时查看（如果可能）
            print("\n" + "*"*80, flush=True)
            print(log_message, flush=True)
            print("*"*80 + "\n", flush=True)

            # 2. 将证据写入一个独立的文件，这是我们明天检查的关键
            #    使用 'a' 模式（追加），以便多个进程都能写入
            with open("DIAGNOSIS_PROOF.log", "a") as f:
                f.write(log_message)

            # 正常返回None元组，不中断训练
            return None, None, None, None, None, None

        # --- 后续的所有代码都和原来完全一样 ---
        selfies = [item['selfies'] for item in batch]
        selfies2 = [item['selfies2'] for item in batch]
        pockets = [item['pdb_embedding'] for item in batch]
        ground_truth_mols = [item['rdmol'] for item in batch]
        pocket_paths = [item['pdb_path'] for item in batch]

        selfies_batch = self.tokenizer(selfies, padding='max_length', return_tensors='pt',
                                    max_length=self.max_sf_tokens, truncation=True, add_special_tokens=True)
        selfies2_batch = self.tokenizer(selfies2, padding='max_length', return_tensors='pt',
                                        max_length=self.max_sf_tokens, truncation=True, add_special_tokens=True)

        pocket_embs, pocket_mask = self.prepare_inputs_for_causal_lm_left_pad(pockets, self.max_pocket_tokens)

        return selfies_batch, selfies2_batch, pocket_embs, pocket_mask, ground_truth_mols, pocket_paths


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
