"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import math
import torch
from torch import optim
import lightning as L
import os
import subprocess
from transformers import AutoTokenizer
# from model.modeling_llama import LlamaForCausalLM, LlamaConfig
from transformers import LlamaForCausalLM, LlamaConfig
from peft import get_peft_model, LoraConfig
from model.help_funcs import AttrDict
from pathlib import Path
import selfies as sf
from rdkit import Chem
from tqdm import tqdm
from evaluation.eval_functions import get_2D_edm_metric, SBDD_validation
from torch.nn import CrossEntropyLoss
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import defaultdict # Make sure this is imported

from evaluation.eval_functions import run_full_evaluation 


# class ResidualAdapter(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
#         super().__init__()
#         self.block = torch.nn.Sequential(
#             torch.nn.Linear(input_dim, hidden_dim),
#             torch.nn.LayerNorm(hidden_dim),
#             torch.nn.GELU(),
#             torch.nn.Dropout(p=dropout),
#             torch.nn.Linear(hidden_dim, output_dim)
#         )
#         # 这个捷径层用于匹配输入和输出的维度
#         self.shortcut = torch.nn.Linear(input_dim, output_dim) if input_dim != output_dim else torch.nn.Identity()

#     def forward(self, x):
#         # H(x) = F(x) + x
#         return self.block(x) + self.shortcut(x)
    
    
def worker_generate_3d(data_pair):

    selfies_string, pocket_path = data_pair
    try:
        smiles = sf.decoder(selfies_string)
        if not smiles:
            return None, None, selfies_string

        # 调用静态方法来执行核心的3D生成逻辑
        mol_3d = LLMPL.generate_3d_mol(smiles,'fast')

        if mol_3d is None:
            return None, None, selfies_string
        
        return mol_3d, pocket_path, selfies_string
        
    except Exception:
        return None, None, selfies_string
class LinearWarmupCosineLRSchedulerV2:
    def __init__(
        self,
        optimizer,
        max_iters,
        min_lr,
        init_lr,
        warmup_iters=0,
        warmup_start_lr=-1,
        **kwargs
    ):
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.min_lr = min_lr
        self.init_lr = init_lr
        self.warmup_iters = warmup_iters
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr
        self.lr_decay_iters = max_iters

    def get_lr(self, it):
        # 1) linear warmup for warmup_steps steps
        if it < self.warmup_iters:
            return self.init_lr * it / self.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.lr_decay_iters:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.min_lr + coeff * (self.init_lr - self.min_lr)

    def step(self, cur_step):
        lr = self.get_lr(cur_step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr


def get_half_precision_dtype():
    if not torch.cuda.is_available():
        return torch.float16
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    else:
        return torch.float16


def set_embed_tokens_trainable(model):
    for name, param in model.named_parameters():
        if name.find('embed_tokens') >= 0:
            param.requires_grad = True
            print(name, 'requires_grad = True')


def obtain_loss_and_ppl(logits, labels, attn_mask, return_nll=False, context_length=0):
    if context_length > 0:
        logits = logits[:, context_length:, :]
        labels = labels[:, context_length:]
        attn_mask = attn_mask[:, context_length:]

    shift_attention_mask_batch = attn_mask[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(reduction="none")

    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    nll = loss_fct(shift_logits.transpose(1,2), shift_labels) * shift_attention_mask_batch
    loss = nll.sum() / shift_attention_mask_batch.sum()
    if return_nll:
        avg_nll = nll.sum(dim=1) / shift_attention_mask_batch.sum(dim=1)
        return loss, avg_nll
    else:
        ppl = torch.exp(nll.sum(dim=1) / shift_attention_mask_batch.sum(dim=1))
        return loss, ppl


class LLMPL(L.LightningModule):
    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)    
    def configure_optimizers(self):
        self.trainer.fit_loop.setup_data()
        warmup_steps = min(len(self.trainer.train_dataloader), self.args.warmup_steps)
        
        adapter_params = [p for n, p in self.named_parameters() if 'projection' in n or 'embedding_norm' in n]
        llm_params = [p for n, p in self.named_parameters() if 'projection' not in n and 'embedding_norm' not in n]
        
        # --- 核心修正: 使用“零学习率”策略以兼容DeepSpeed ---
        initial_llm_lr = 0.0
        if self.unfreeze_epoch == 0:
            initial_llm_lr = self.args.init_lr / 10.0
        
        print(f"INFO: Initializing optimizer. Adapter LR: {self.args.init_lr}, LLM LR: {initial_llm_lr}")

        optimizer_grouped_parameters = [
            {"params": adapter_params, "lr": self.args.init_lr},
            {"params": llm_params, "lr": initial_llm_lr} # 初始时LLM学习率为0
        ]
        # --- 修正结束 ---

        optimizer = optim.AdamW(optimizer_grouped_parameters, weight_decay=self.args.weight_decay)
        
        max_iters = self.args.max_epochs * len(self.trainer.train_dataloader)
        if self.args.scheduler == 'linear_warmup_cosine_lr':
            self.scheduler = LinearWarmupCosineLRSchedulerV2(optimizer, max_iters, self.args.min_lr, self.args.init_lr, warmup_steps, self.args.warmup_lr)
        else:
            self.scheduler = None
            
        return optimizer
    @classmethod
    def init_tokenizer(cls, args):
        tokenizer = AutoTokenizer.from_pretrained(
            args.llm_model,
            padding_side='left'  # <-- 核心修改：明确指定在左侧填充
        )
        
        # 关键：如果模型本身没有pad_token，必须在这里提前设置好
        # 因为AutoTokenizer在初始化时如果找不到pad_token，可能会报错或行为不一致
        if tokenizer.pad_token is None:
            # LLaMA等模型通常没有默认的pad_token，一个安全常见的做法是将其指向eos_token
            tokenizer.pad_token = tokenizer.eos_token
            print("Warning: pad_token is not set. Setting it to eos_token.")

        tokenizer.add_bos_token = True
        tokenizer.add_eos_token = True
        return tokenizer

    def on_train_epoch_start(self):
        if self.current_epoch == self.unfreeze_epoch and self.unfreeze_epoch > 0:
            target_llm_lr = self.args.init_lr / 10.0
            print(f"\nEpoch {self.current_epoch}: Unfreezing LLM parameters by setting LR to {target_llm_lr}...")
            self.trainer.optimizers[0].param_groups[1]['lr'] = target_llm_lr
            print("INFO: LLM parameter group LR updated.")
    # def on_train_epoch_start(self):
    #     if self.current_epoch == self.unfreeze_epoch and self.delta_train:
    #         print(f"Epoch {self.current_epoch}: Unfreezing LoRA parameters for fine-tuning...")
    #         # 只解冻 LoRA 参数
    #         for name, param in self.llm_model.named_parameters():
    #             if "lora" in name:
    #                 param.requires_grad = True
    #         self.llm_model.print_trainable_parameters()    
    
    @classmethod
    def init_llm(cls, args):
        config = LlamaConfig.from_pretrained(args.llm_model)
        config.attention_dropout = args.attention_dropout

        if args.load_random_llm:
            if args.use_flash_attention:
                config._attn_implementation = 'flash_attention_2'
            llm_model = LlamaForCausalLM(config).to(get_half_precision_dtype())
        else:
            if args.use_flash_attention:
                llm_model = LlamaForCausalLM.from_pretrained(args.llm_model, config=config, torch_dtype=get_half_precision_dtype(), attn_implementation='flash_attention_2')
            else:
                llm_model = LlamaForCausalLM.from_pretrained(args.llm_model, config=config, torch_dtype=get_half_precision_dtype())

        if args.llm_tune == 'freeze':
            for param in llm_model.parameters():
                param.requires_grad = False
        elif args.llm_tune == 'full':
            for param in llm_model.parameters():
                param.requires_grad = True
        elif args.llm_tune == 'lora':
            lora_config = LoraConfig(r=args.lora_r,
                                     lora_alpha=args.lora_alpha,
                                     lora_dropout=args.lora_dropout,
                                     target_modules=["q_proj", "v_proj"])
            llm_model = get_peft_model(llm_model, lora_config)
            if args.tune_embedding:
                set_embed_tokens_trainable(llm_model)
            llm_model.print_trainable_parameters()
        elif args.llm_tune == 'mid_lora':
            lora_config = LoraConfig(r=args.lora_r,
                                     lora_alpha=args.lora_alpha,
                                     lora_dropout=args.lora_dropout,
                                     target_modules=["q_proj", "v_proj", 'k_proj', 'o_proj', "gate_proj", "up_proj", "down_proj"])
            llm_model = get_peft_model(llm_model, lora_config)
            if args.tune_embedding:
                set_embed_tokens_trainable(llm_model)
            llm_model.print_trainable_parameters()
        else:
            raise NotImplementedError()

        # freeze on initial epochs
        # for param in llm_model.parameters():
        #     param.requires_grad = False
        return llm_model


    def resize_token_embeddings(self, tokenizer=None):
        if tokenizer is None:
            tokenizer = self.tokenizer
        self.llm_model.resize_token_embeddings(len(tokenizer))

    def __init__(self, args,tokenizer=None,max_pocket_tokens = 100, max_sf_tokens=100, property_distribution=None):
        super().__init__()
        if isinstance(args, dict):
            args = AttrDict(**args)
        self.args = args
        self.aug_inv = args.aug_inv
        self.max_sf_tokens = max_sf_tokens
        self.max_pocket_tokens = max_pocket_tokens
        self.num_beams=args.num_beams

        ## init llm
        self.llm_model = self.init_llm(args)
        if tokenizer is None:
            self.tokenizer = self.init_tokenizer(args)
        else:
            self.tokenizer = tokenizer
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("Warning: pad_token is not set. Setting it to eos_token.")



        self.delta_train = True if args.llm_tune == 'lora' else False #TODO: 如果用lora这里要打开
        self.unfreeze_epoch = args.unfreeze_epoch
        self.resize_token_embeddings(self.tokenizer)
        self.hidden_size = self.llm_model.config.hidden_size

        self.embedding_norm = torch.nn.LayerNorm(1536,elementwise_affine=False)
        self.projection = torch.nn.Sequential(
        torch.nn.Linear(1536, 512),
        torch.nn.LayerNorm(512), 
        torch.nn.GELU(),
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(512, self.hidden_size)
        )
        self.post_projection_norm = torch.nn.LayerNorm(self.llm_model.config.hidden_size,elementwise_affine=False)

               # --- 应用权重初始化 ---
        print("INFO: Applying Xavier uniform initialization to the projection adapter.")
        self.projection.apply(self._init_weights)
        # TODO: add pockets embeddings related code
        # self.projection = torch.nn.Sequential(
        #         torch.nn.Linear(1536, self.hidden_size * 2),
        #         torch.nn.GELU(), 
        #         torch.nn.Dropout(0.3), 
        #         torch.nn.Linear(self.hidden_size * 2, self.llm_model.config.hidden_size)
        #     )


        self.property_distribution = property_distribution
        if self.delta_train:
            print("INFO: Staged training enabled. Freezing LoRA parameters initially.")
            for name, param in self.llm_model.named_parameters():
                if 'lora' in name:
                    param.requires_grad = False


        self.save_hyperparameters(args)

    # def training_step(self, batch, batch_idx):
    #     if self.scheduler:
    #         self.scheduler.step(self.trainer.global_step)
        
    #     selfies_batch,selfies2_batch, pockets_emb, pock_attn_mask,_,_ = batch
    #     lm_loss0 = self.forward(selfies_batch, pockets_emb, pock_attn_mask)
    #     lm_loss1 = self.forward(selfies2_batch, pockets_emb, pock_attn_mask)
    #     lm_loss = (lm_loss0 + lm_loss1)/2
    #     loss = lm_loss
    #     batch_size = selfies_batch.input_ids.shape[0]

        # self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], sync_dist=True, batch_size=batch_size)
        # self.log('train_loss', loss, sync_dist=True, batch_size=batch_size)

        # return loss

    def training_step(self, batch, batch_idx):
        if self.scheduler:
            self.scheduler.step(self.trainer.global_step)
        
        selfies_batch, selfies2_batch, pockets_emb, pock_attn_mask, _, _ = batch
        batch_size = selfies_batch.input_ids.shape[0]

        # --- Standard Generative Loss (Default Behavior) ---

        lm_loss0 = self.forward(selfies_batch, pockets_emb, pock_attn_mask)
        lm_loss1 = self.forward(selfies2_batch, pockets_emb, pock_attn_mask)
        loss = (lm_loss0 + lm_loss1) / 2
        
        self.log('train_loss', loss, sync_dist=True, batch_size=batch_size)


        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], sync_dist=True, batch_size=batch_size)
    
        return loss


    @torch.no_grad()
    def on_validation_epoch_start(self):
        self.validation_outputs=[]
        self.sampled_seq = []

    # In LLMPL class
    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        train_epoch_condition = (self.current_epoch + 1) % self.args.generate_eval_epoch  == 0 and self.args.mode == 'train'
        eval_condition = self.args.mode in {'eval', 'eval_conf'}

        selfies_batch, selfies2_batch, pockets_emb, pock_attn_mask, ground_truth_mols, pocket_paths = batch

        # --- 1. 深入的诊断逻辑 (仅在第一个batch执行) ---
        if self.trainer.is_global_zero and batch_idx == 0:
            print("\n" + "="*80)
            print(f"|  <<<<<<<<<<<<<<<<< DIAGNOSTIC REPORT (Epoch: {self.current_epoch}) >>>>>>>>>>>>>>>>>  |")
            print("="*80)

            # --- 诊断 Part A: Projection 之前的表征相似度 ---
            normalized_pockets_emb_before = self.embedding_norm(pockets_emb)
            
            # [--- 代码修正 ---]
            # (B, L, D) -> (B, D) by averaging
            avg_emb_before = torch.mean(normalized_pockets_emb_before, dim=1)
            # Normalize for cosine similarity calculation
            avg_emb_before_norm = torch.nn.functional.normalize(avg_emb_before, p=2, dim=1)
            similarity_matrix_before = torch.matmul(avg_emb_before_norm, avg_emb_before_norm.T)
            # [--- 代码修正结束 ---]
            
            mask = torch.triu(torch.ones_like(similarity_matrix_before), diagonal=1).bool()
            
            if mask.sum() > 0:
                avg_sim_before = similarity_matrix_before[mask].mean().item()
                max_sim_before = similarity_matrix_before[mask].max().item()
                print(f"|  [DIAGNOSIS - BEFORE Projection]")
                print(f"|  - Avg Cosine Similarity of Input Embeds:  {avg_sim_before:.4f}")
                print(f"|  - Max Cosine Similarity of Input Embeds:  {max_sim_before:.4f}")
                print("."*80)

            # --- 诊断 Part B: Projection 之后的表征相似度 ---
            projected_emb = self.projection(normalized_pockets_emb_before)
            prompt_embeds_after = self.post_projection_norm(projected_emb)

            # [--- 代码修正 ---]
            avg_emb_after = torch.mean(prompt_embeds_after, dim=1)
            avg_emb_after_norm = torch.nn.functional.normalize(avg_emb_after, p=2, dim=1)
            similarity_matrix_after = torch.matmul(avg_emb_after_norm, avg_emb_after_norm.T)
            # [--- 代码修正结束 ---]
            
            if mask.sum() > 0:
                avg_sim_after = similarity_matrix_after[mask].mean().item()
                max_sim_after = similarity_matrix_after[mask].max().item()
                print(f"|  [DIAGNOSIS - AFTER Projection]")
                print(f"|  - Avg Cosine Similarity of Output Embeds: {avg_sim_after:.4f}")
                print(f"|  - Max Cosine Similarity of Output Embeds: {max_sim_after:.4f}")
                print("."*80)

            # --- 诊断 Part C: Logits 检查 ---
            prompt_embeds_for_logits = prompt_embeds_after.to(torch.bfloat16)
            outputs = self.llm_model(inputs_embeds=prompt_embeds_for_logits, attention_mask=pock_attn_mask, return_dict=True)
            self.debug_inspect_logits(logits=outputs.logits, prompt_length=prompt_embeds_for_logits.shape[1],
                                      epoch=self.current_epoch, step=self.trainer.global_step, batch_idx=batch_idx)
            
            print(f"|  <<<<<<<<<<<<<<<<<<<<<<<< END OF REPORT >>>>>>>>>>>>>>>>>>>>>>>  |")
            print("="*80 + "\n")


 
        # --- 1. 计算并记录 loss ---
        lm_loss0 = self.forward(selfies_batch, pockets_emb, pock_attn_mask)
        lm_loss1 = self.forward(selfies2_batch, pockets_emb, pock_attn_mask)
        loss = (lm_loss0 + lm_loss1) / 2
        self.log('val_loss', loss, sync_dist=True, batch_size=pockets_emb.shape[0])

        if not train_epoch_condition and not eval_condition:
            return
        current_epoch = self.current_epoch
        num_output_for_this_epoch = 0 # <-- ADD THIS INITIALIZATION LINE

        is_2d_eval_epoch = (self.current_epoch + 1) % self.args.eval_2d_every_n_epochs == 0
        is_3d_eval_epoch = (self.current_epoch + 1) % self.args.eval_3d_every_n_epochs == 0
        if is_3d_eval_epoch:
            num_output_for_this_epoch = self.args.num_output_3d
        elif is_2d_eval_epoch:
            num_output_for_this_epoch = self.args.num_output_2d

        # --- 2. 直接调用重构后的函数来生成分子 ---
        output_text, _ = self.sample_selfies_for_pocket(
            pockets_emb=pockets_emb,       # <-- 传递原始的 pockets_emb
            attention_mask=pock_attn_mask,
            num_beams=self.args.num_beams,
            do_sample=self.args.do_sample,
            temperature=self.args.temperature,
            num_output=num_output_for_this_epoch,
            max_length=self.max_sf_tokens - 1,
            batch_idx=batch_idx
        )

        # --- 后续处理逻辑保持不变 ---
        num_outputs_per_prompt = len(output_text) // len(pocket_paths)
        expanded_gt_mols = [mol for mol in ground_truth_mols for _ in range(num_outputs_per_prompt)]
        expanded_pocket_paths = [path for path in pocket_paths for _ in range(num_outputs_per_prompt)]


        self.sampled_seq.extend(output_text)
        save_dir = "/data/share/liuzhiyuan/nai/NExT-Mol/sample/2D"
        epoch_save_dir = os.path.join(save_dir, f"epoch_{self.current_epoch}")
        for i, (gen_selfies, gt_mol, pdb_path) in enumerate(zip(output_text, expanded_gt_mols, expanded_pocket_paths)):
            self.validation_outputs.append({
                'generated': gen_selfies,
                'ground_truth': gt_mol,
                'pocket_path': pdb_path
            })

            # --- MOVE THE SAVING LOGIC INSIDE THE LOOP HERE ---
            # Now 'i' and 'gen_selfies' are correctly defined and available
            if batch_idx == 0 and i < 2:
                try:
                    os.makedirs(epoch_save_dir, exist_ok=True)

                    smiles = sf.decoder(gen_selfies)
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        AllChem.Compute2DCoords(mol)
                        mol.SetProp("_Name", f"epoch_{self.current_epoch}_batch_{batch_idx}_sample_{i}")
                        save_path = os.path.join(epoch_save_dir, f"epoch_{self.current_epoch}_batch_{batch_idx}_sample_{i}.sdf")

                        with Chem.SDWriter(save_path) as writer:
                            writer.write(mol)

                        print(f"\n--- [INFO] Saved sample molecule to: {save_path} ---")
                except Exception as e:
                    print(f"\n--- [WARNING] Failed to save sample molecule due to: {e} ---")
                
        


    @torch.no_grad()
    @torch.cuda.amp.autocast(dtype=torch.bfloat16)
    def on_validation_epoch_end(self):

        if self.trainer.is_global_zero: # 只在主进程执行
            print("\n--- [DEBUG] Sanity Check: Unconditioned Generation ---")
            try:
                bos_token_id = self.tokenizer.bos_token_id
                input_ids = torch.tensor([[bos_token_id]], device=self.device)
                # 让模型自由生成，不加任何条件
                outputs = self.llm_model.generate(input_ids, max_new_tokens=50, num_return_sequences=1, do_sample=True, temperature=0.7)
                decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"Unconditioned Output: '{decoded}'")
            except Exception as e:
                print(f"Unconditioned generation failed: {e}")
            print("---------------------------------------------------\n")



                # ======================= [ 新增的调试代码块 ] =======================
        print("\n--- [DEBUG] Raw Generation Analysis (First 5 Samples) ---")
        for i, raw_selfies in enumerate(self.sampled_seq[:50]):
            print(f"\n--- Sample {i} ---")
            print(f"Raw Output String: '{raw_selfies}'")
            try:
                # 尝试解码
                decoded_smiles = sf.decoder(raw_selfies)
                print(f"Decoded SMILES: {decoded_smiles if decoded_smiles else 'DECODING_FAILED'}")

                # 尝试拆分token并检查词汇表
                split_tokens = sf.split_selfies(raw_selfies)
                print(f"Split Tokens: {split_tokens}")
                
                invalid_tokens = [tok for tok in split_tokens if tok not in self.tokenizer.vocab]
                if invalid_tokens:
                    print(f"!!! INVALID TOKENS FOUND: {invalid_tokens}")
                else:
                    print("All tokens are valid in tokenizer vocab.")

            except Exception as e:
                print(f"!!! ERROR during processing: {e}")
        print("------------------------------------------------------\n")
        # ======================= [ 调试代码块结束 ] =======================
        print(f"\n--- DEBUG: Entering on_validation_epoch_end for Epoch {self.current_epoch} ---")
        train_epoch_condition = (self.current_epoch + 1) % self.args.generate_eval_epoch == 0 and self.args.mode == 'train'
        eval_condition = self.args.mode in {'eval'}
        if  not eval_condition and self.current_epoch< self.args.epoch_without_eval:
            print(f"--- DEBUG: Skipping evaluation because current_epoch ({self.current_epoch}) < epoch_without_eval ({self.args.epoch_without_eval}). ---\n")

            return

        if not train_epoch_condition and not eval_condition:
            print(f"--- DEBUG: Skipping evaluation due to generate_eval_epoch condition. ---\n")
            
            return
        print("--- DEBUG: Passed all initial checks. Starting generation and evaluation. ---")

        if not self.trainer.is_global_zero:
            print("--- DEBUG: Not on global rank zero, returning. ---")

            return
        run_2d_eval = (self.current_epoch + 1) % self.args.eval_2d_every_n_epochs == 0
        if run_2d_eval:
            # sampled_sequences = self.sample_molecules()
            print("--- DEBUG: Starting molecule generation via self.sampled_seq processing... ---")

            tuple_list = [reencode_selfies(item) for item in self.sampled_seq]
            tuple_list_valid = []
            for index, (selfies, smiles_with_chirality, smiles_without_chirality) in enumerate(tuple_list):
                if not selfies:
                    continue
                selfies_tokens = sf.split_selfies(selfies)
                skip = False
                for token in selfies_tokens:
                    if token not in self.tokenizer.vocab:
                        skip = True
                        break
                if skip:
                    continue

                tuple_list_valid.append((selfies, smiles_with_chirality, smiles_without_chirality))


            smiles_without_chirality_list = [results_tuple[2] for results_tuple in tuple_list_valid]
            sampled_rdmols = [Chem.MolFromSmiles(smiles_without_chirality) for smiles_without_chirality in smiles_without_chirality_list]
            sampled_rdmols = [Chem.AddHs(mol) for mol in sampled_rdmols if mol is not None]
            sampled_rdmols=[mol for mol in sampled_rdmols if mol is not None]
            if len(sampled_rdmols) == 0:
                print(f"\n[WARNING] Epoch {self.current_epoch}: No valid molecules were generated. Skipping detailed evaluation.")
                return 
            

            print(f"--- DEBUG: Prepared {len(sampled_rdmols)} valid RDKit molecules for evaluation. ---")

            print("--- DEBUG: Starting get_2D_edm_metric... ---")
            eval_results_2d = get_2D_edm_metric(sampled_rdmols, self.trainer.datamodule.train_rdmols)
            print("--- DEBUG: Finished get_2D_edm_metric. ---")

            print("--- DEBUG: Starting get_moses_metrics... ---")
            moses_metrics = self.trainer.datamodule.get_moses_metrics(sampled_rdmols)
            print("--- DEBUG: Finished get_moses_metrics. ---")
            print("--- DEBUG: Starting to log all metrics... ---")
            self.log('QED', moses_metrics['QED'], rank_zero_only=True)#
            self.log('SA', moses_metrics['SA'], rank_zero_only=True)#
            self.log('logP', moses_metrics['logP'], rank_zero_only=True)#
            print("--- DEBUG: Finished metrics calculation and logging. ---\n")
            print("\n" + "="*60)
            print(f"--- Evaluation Metrics for Epoch {self.current_epoch} ---")        

            # --- SBDD 相似度评估 ---
            print(f"--- DEBUG: Preparing pairs for SBDD similarity evaluation... ---")
            generated_mols = []
            ground_truth_mols = []
            pocket_paths=[]
            for pair in self.validation_outputs:
                # 解码生成的 SELFIES 为 RDKit Mol 对象
                try:
                    gen_smiles = sf.decoder(pair['generated'])
                    mol_gen = Chem.MolFromSmiles(gen_smiles)
                    if mol_gen:
                        generated_mols.append(mol_gen)
                        ground_truth_mols.append(pair['ground_truth'])
                        pocket_paths.append(pair['pocket_path'])
                except Exception:
                    continue # 如果解码失败则跳过

            print(f"--- DEBUG: Starting SBDD_validation for {len(generated_mols)} valid pairs. ---")
            
            # 调用我们新封装的函数
            sbdd_metrics = SBDD_validation(generated_mols, ground_truth_mols)
            
            # --- 打印和记录新的SBDD指标 ---
            print("\n[--- SBDD Similarity Metrics vs. Ground Truth ---]")
            for key, value in sbdd_metrics.items():
                print(f"  - {key:<20}: {value:.4f}")
                self.log(key, value, rank_zero_only=True) # 记录到日志系统
                
            print("="*60 + "\n")

            print("\n[--- Basic 2D Metrics ---]")
            for key, value in eval_results_2d.items():
                print(f"  - {key:<12}: {value:.4f}") 
            print("\n[--- Advanced MOSES & Drug-likeness Metrics ---]")
            for key, value in moses_metrics.items():
                print(f"  - {key:<12}: {value:.4f}")
            print("="*60 + "\n")

            print(f"--- DEBUG: Starting 3D conversion and docking for {len(self.validation_outputs)} generated sequences. ---")

        run_3d_eval = (self.current_epoch + 1) % self.args.eval_3d_every_n_epochs == 0
        if run_3d_eval:
            # --- 2. 准备3D分子并进行全面评估 (现在使用多进程并行) ---
            print(f"\n--- DEBUG: Starting PARALLEL 3D conversion for {len(self.validation_outputs)} generated sequences. ---")

            tasks = [(pair['generated'], pair['pocket_path']) for pair in self.validation_outputs]
            
            generated_mols_3d = []
            pocket_paths_for_vina = []
            failed_molecules_info = [] 
            successful_molecules_info = []

            # 获取CPU核心数，可以留一两个核心给系统
            import multiprocessing as mp
            import os

            slurm_cpus_str = os.getenv('SLURM_CPUS_PER_TASK')
            if slurm_cpus_str:
                cpu_cores = int(slurm_cpus_str)
                print(f"--- INFO: Detected Slurm allocation. Using {cpu_cores} allocated cores for 3D generation. ---")
            else:
                # 如果不在Slurm环境中，回退到使用mp.cpu_count()
                cpu_cores = mp.cpu_count()
                print(f"--- INFO: Not in a Slurm environment. Using total system cores: {cpu_cores} for 3D generation. ---")
            
            # 根据获取的核心数计算进程数，为操作系统和主进程留出余地
            num_processes = max(1, cpu_cores - 2)
            print(f"--- INFO: Using {num_processes} CPU cores for parallel processing. ---")

            with mp.Pool(processes=num_processes) as pool:
                results_iterator = pool.imap_unordered(worker_generate_3d, tasks)
                
                for mol_3d, pocket_path, original_selfies in tqdm(results_iterator, total=len(tasks), desc="Generating 3D conformers"):
                    if mol_3d is not None and pocket_path is not None:
                        generated_mols_3d.append(mol_3d)
                        pocket_paths_for_vina.append(pocket_path)
                        try:
                            smiles = sf.decoder(original_selfies)
                        except:
                            smiles = "DECODING_ERROR"
                        successful_molecules_info.append({'selfies': original_selfies, 'smiles': smiles})
                    else:
                        try:
                            smiles = sf.decoder(original_selfies) if original_selfies else "EMPTY_SELFIES"
                        except:
                            smiles = "DECODING_ERROR"
                        failed_molecules_info.append({'selfies': original_selfies, 'smiles': smiles})
            
            # --- 打印成功和失败的样本 ---
            if failed_molecules_info:
                print("\n--- [DEBUG] Molecules that failed 3D generation (sample) ---")
                for i, failure in enumerate(failed_molecules_info[:5]): 
                    print(f"  - Failed Mol {i+1}:")
                    print(f"    - Original SELFIES: {failure['selfies']}")
                    print(f"    - Decoded SMILES  : {failure['smiles']}")
                print("----------------------------------------------------------\n")
            
            if successful_molecules_info:
                print("\n--- [DEBUG] Sample of SUCCESSFULLY generated 3D molecules ---")
                for i, success in enumerate(successful_molecules_info[:5]): 
                    print(f"  - Success Mol {i+1}:")
                    print(f"    - Original SELFIES: {success['selfies']}")
                    print(f"    - Decoded SMILES  : {success['smiles']}")
                print("----------------------------------------------------------\n")

                num_generated = len(generated_mols_3d)
                num_failed = len(failed_molecules_info)
                total_attempted = num_generated + num_failed


                grouped_mols_for_eval = defaultdict(list)
                
                
                
                for mol, path in zip(generated_mols_3d, pocket_paths_for_vina):
                    grouped_mols_for_eval[path].append(mol)
                # --- Final check and call to the evaluation function ---
                
                if not grouped_mols_for_eval:
                    print("\n[WARNING] No valid 3D molecules to evaluate. Skipping comprehensive evaluation.")
                else:
                    print("\n--- [INFO] 3D Molecule Generation Summary ---")
                    print(f"  - Total sequences attempted  : {total_attempted}")
                    print(f"  - Successfully generated     : {num_generated} ({num_generated/total_attempted*100:.2f}%)")
                    print(f"  - Failed to generate         : {num_failed} ({num_failed/total_attempted*100:.2f}%)")
                    print(f"  - Number of keys:", len(grouped_mols_for_eval))
                    print("----------------------------------------------------------\n")
                    
                    # Call the single, powerful evaluation function
                    eval_metrics = run_full_evaluation(
                    grouped_mols=grouped_mols_for_eval 
                    )





                    # Log the results
                    print("\n" + "="*60)
                    print(f"--- Logging Final Metrics for Epoch {self.current_epoch} ---")
                    if eval_metrics:
                        for key, value in eval_metrics.items():
                            log_key = key.replace(' ', '_').replace('(%)', 'percent').replace('(', '').replace(')', '')
                            self.log(f'eval/{log_key}', value, sync_dist=True)
                            print(f"  - Logged eval/{log_key}: {value:.4f}")
                    print("="*60 + "\n")


    @torch.no_grad()
    def sample_molecules(self):
        ## sample selfies from the molecule language model
        sample_num = self.args.sample_num
        print('sample_num:', sample_num)
        loop_count = 0
        sampled_sequences = [] # we use smiles as the intermediate data structure for its easy conversion to rdkit mol
        pbar = tqdm(total=sample_num, desc='sample molecules sequences')

        while True:
            sf_list, context = self.sample_selfies(
                batch_size=200,
                num_beams=self.args.num_beams,
                temperature=self.args.temperature,
                num_output=1,
                max_length=self.max_sf_tokens - 1) # -1 for the bos token, which is already included

            tuple_list = [reencode_selfies(item) for item in sf_list]
            tuple_list_valid = []
            for index, (selfies, smiles_with_chirality, smiles_without_chirality) in enumerate(tuple_list):
                if not selfies:
                    continue
                selfies_tokens = sf.split_selfies(selfies)
                skip = False
                for token in selfies_tokens:
                    if token not in self.tokenizer.vocab:
                        skip = True
                        break
                if skip:
                    continue
                if context is not None:
                    tuple_list_valid.append((selfies, smiles_with_chirality, smiles_without_chirality, context[index]))
                else:
                    tuple_list_valid.append((selfies, smiles_with_chirality, smiles_without_chirality))

            sampled_sequences.extend(tuple_list_valid)
            loop_count += 1
            pbar.update(len(tuple_list_valid))
            pbar.set_postfix(loop_count=loop_count)
            if len(sampled_sequences) >= sample_num:
                pbar.close()
                break

        sampled_sequences = list(sampled_sequences)[:sample_num]
        sampled_sequences.sort()

        log_dir = Path(self.logger.log_dir)
        ## save the sampled sequences
        if self.args.condition_property is None:
            save_path = log_dir / f'sequences_epoch{self.current_epoch}.txt'
            with save_path.open('w', encoding='utf8') as f:
                for selfies, smiles_with_chirality, smiles_without_chirality in sampled_sequences:
                    f.write(f'{selfies}\t{smiles_with_chirality}\t{smiles_without_chirality}' + '\n')
        else:
            save_path = log_dir / f'sequences_epoch{self.current_epoch}_{self.args.condition_property}.txt'
            with save_path.open('w', encoding='utf8') as f:
                for selfies, smiles_with_chirality, smiles_without_chirality, context in sampled_sequences:
                    f.write(f'{selfies}\t{smiles_with_chirality}\t{smiles_without_chirality}\t{context}' + '\n')

        return sampled_sequences
    
    # In LLMPL class
    @torch.no_grad()
    def sample_selfies_for_pocket(
        self,
        pockets_emb,  
        attention_mask=None,
        do_sample=True,
        num_beams=5,
        max_length=30,
        temperature=1,
        num_output=2,
        batch_idx=0
        ):
        batch_size = pockets_emb.shape[0]
        device = pockets_emb.device

        # --- 1. 在函数内部处理 pocket embedding ---

        normalized_pockets_emb = self.embedding_norm(pockets_emb)
        pocket_emb_lm = self.projection(normalized_pockets_emb)
        pocket_emb_lm = self.post_projection_norm(pocket_emb_lm)

        # --- 2. 准备 BOS token embedding ---
        bos_token_id = self.tokenizer.bos_token_id
        bos_token_tensor = torch.tensor([[bos_token_id]] * batch_size, device=device)
        bos_embedding = self.llm_model.get_input_embeddings()(bos_token_tensor)
                
        # --- 3. 强制确保所有部分都是 bfloat16 再拼接 ---
        pocket_emb_lm = pocket_emb_lm.to(torch.bfloat16)
        bos_embedding = bos_embedding.to(torch.bfloat16)
        
        inputs_embeds = torch.cat([bos_embedding, pocket_emb_lm], dim=1)
        # --- [ 新增调试代码 ] ---
        if batch_idx == 0: # 只在第一个batch打印，避免刷屏
            print(f"\n--- [DEBUG] Prompt Embedding Stats: Mean={pocket_emb_lm.mean():.4f}, Std={pocket_emb_lm.std():.4f}, IsNaN={torch.isnan(pocket_emb_lm).any()} ---")
        # --- [ 调试代码结束 ] ---


        if attention_mask is not None:
            bos_attention_mask = torch.ones_like(bos_token_tensor)
            attention_mask = torch.cat([bos_attention_mask, attention_mask], dim=1)

        # --- 4. 最终调用 generate 函数 ---
        outputs = self.llm_model.generate(
            inputs_embeds=inputs_embeds,  
            attention_mask=attention_mask,
            do_sample=do_sample,
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=max_length,
            min_length=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            num_return_sequences=num_output
        )

        output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return output_text, None



    def forward(self, selfies_batch, pockets_emb, pock_attn_mask):
        # Check for NaNs in the original input
        if torch.isnan(pockets_emb).any():
            print("!!! WARNING: NaN found in pockets_emb !!!")


        normalized_pockets_emb = self.embedding_norm(pockets_emb)
        pocket_emb_lm = self.projection(normalized_pockets_emb)
        pocket_emb_lm = self.post_projection_norm(pocket_emb_lm)
        input_emb_selfies = self.llm_model.get_input_embeddings()(selfies_batch.input_ids)

        input_emb = torch.cat([pocket_emb_lm, input_emb_selfies], dim=1)


        input_emb = input_emb.to(torch.bfloat16)

        target_selfies = selfies_batch.input_ids.masked_fill(~selfies_batch.attention_mask.bool(), -100)
        batch_size, seq_len = pocket_emb_lm.shape[:2]
        empty_target = torch.full((batch_size, seq_len), -100, dtype=torch.long).to(pocket_emb_lm.device)
        targets = torch.cat([empty_target, target_selfies], dim=1)

        attention_mask = torch.cat([pock_attn_mask, selfies_batch.attention_mask], dim=1)

        outputs = self.llm_model(inputs_embeds=input_emb,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
                output_hidden_states=True)

        if torch.isnan(outputs.logits).any():
            print("!!! WARNING: NaN found in model logits !!!")
        if torch.isnan(outputs.loss):
            print("!!! WARNING: Loss is NaN. Logits min/max:", outputs.logits.min(), outputs.max())

        lm_loss = outputs.loss
        return lm_loss
    # def forward(self, selfies_batch, pockets_emb, pock_attn_mask):
    #     if torch.isnan(pockets_emb).any():
    #         print("!!! WARNING: NaN found in pockets_emb !!!")
    # # --- END DEBUG CHECKS ---

    #     else:
    #         normalized_pockets_emb = self.embedding_norm(pockets_emb)
    #         pocket_emb_lm = self.projection(normalized_pockets_emb)

    #     input_emb_selfies = self.llm_model.get_input_embeddings()(selfies_batch.input_ids)
    #     input_emb = torch.cat([pocket_emb_lm,input_emb_selfies], dim=1)

    #     target_selfies = selfies_batch.input_ids.masked_fill(~selfies_batch.attention_mask.bool(), -100)
    #     batch_size, seq_len = pocket_emb_lm.shape[:2]
    #     empty_target = torch.full((batch_size, seq_len), -100, dtype=torch.long).to(pocket_emb_lm.device)
    #     targets = torch.cat([empty_target, target_selfies], dim=1)

    #     attention_mask = torch.cat([pock_attn_mask, selfies_batch.attention_mask], dim=1)

    #     outputs = self.llm_model(inputs_embeds=input_emb,
    #             attention_mask=attention_mask,
    #             return_dict=True,
    #             labels=targets,
    #             output_hidden_states=True)
    #     if torch.isnan(outputs.logits).any():
    #         print("!!! WARNING: NaN found in model logits !!!")
    #     if torch.isnan(outputs.loss):
    #         print("!!! WARNING: Loss is NaN. Logits min/max:", outputs.logits.min(), outputs.logits.max())

    #     lm_loss = outputs.loss

    #     # lm_loss, avg_nll = obtain_loss_and_ppl(outputs.logits, targets, selfies_batch.attention_mask, True, 0)
    #     return lm_loss


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group()
        # parser.add_argument('--llm_model', type=str, default="all_checkpoints/mollama")
        parser.add_argument('--llm_model', type=str, default="/home/mydata/MoLlama")
        parser.add_argument('--load_random_llm', action='store_true', default=False)
        parser.add_argument('--num_beams', type=int, default=1)
        parser.add_argument('--num_output', type=int, default=2)
        parser.add_argument('--do_sample', action='store_true', default=False)
        parser.add_argument('--llm_tune', type=str, default='freeze')
        parser.add_argument('--tune_embedding', action='store_true', default=True)
        parser.add_argument('--sample_num', type=int, default=10000)
        parser.add_argument('--temperature', type=float, default=0.7)
        parser.add_argument('--generate_eval_epoch', type=int, default=5)
        parser.add_argument('--conform_eval_epoch', type=int, default=2)
        # parser.add_argument('--eval_smiles_path', type=str, default=None)
        parser.add_argument('--bi_attend', action='store_true', default=False)
        parser.add_argument('--lm_loss', type=float, default=1.0)

        parser.add_argument('--aug_inv', type=float, default=0.0)

        ## llm config
        parser.add_argument('--attention_dropout', type=float, default=0)
        parser.add_argument('--unfreeze_epoch', type=int, default=10)

        ## lora config
        parser.add_argument('--lora_r', type=int, default=8)
        parser.add_argument('--lora_alpha', type=int, default=16)
        parser.add_argument('--lora_dropout', type=float, default=0.2)

        # optimization
        parser.add_argument('--weight_decay', type=float, default=0.05, help='optimizer weight decay')
        parser.add_argument('--init_lr', type=float, default=1e-4, help='optimizer init learning rate')
        parser.add_argument('--min_lr', type=float, default=1e-5, help='optimizer min learning rate')
        parser.add_argument('--warmup_lr', type=float, default=1e-6, help='optimizer warmup learning rate')
        parser.add_argument('--warmup_steps', type=int, default=1000, help='optimizer warmup steps')
        parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='optimizer lr decay rate')
        parser.add_argument('--scheduler', type=str, default='linear_warmup_cosine_lr', help='type of scheduler') # or
        parser.add_argument('--zero_embedding',action='store_true', default=False)
        parser.add_argument('--optimizer', type=str, default='adamw', help='type of scheduler')
        parser.add_argument('--init_checkpoint', type=str, default=None)
        parser.add_argument('--skip_eval', action='store_true', default=False)

        parser.add_argument('--epoch_without_eval', type=int, default=3)

        return parent_parser


    # @staticmethod
    # def generate_3d_mol(smiles: str, quality: str = 'fast') -> Chem.Mol:
    #     """
    #     统一的、只保留最优路径的3D构象生成函数。
    #     如果多构象生成失败，则直接返回 None。
    #     """
    #     try:
    #         mol = Chem.MolFromSmiles(smiles)
    #         if mol is None: return None
    #         mol = Chem.AddHs(mol)

    #         params = AllChem.ETKDGv3()
    #         params.randomSeed = 0xf00d
    #         params.numThreads = 1

    #         if quality == 'high':
    #             num_confs = 20
    #             optimizer = AllChem.MMFFOptimizeMoleculeConfs
    #         else: # fast 模式
    #             num_confs = 5
    #             optimizer = AllChem.UFFOptimizeMoleculeConfs

    #         # 尝试生成多个构象
    #         cids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
    #         if not cids:
    #             # 如果未能生成任何构象，则认为失败
    #             return None
            
    #         # 优化所有构象
    #         res = optimizer(mol, numThreads=1)
    #         # res 可能是 [(conf_id, energy), ...] 的列表或表示失败的整数
    #         if not isinstance(res, list) or not res:
    #             return None # 优化失败

    #         # 选取能量最低的构象
    #         min_energy_idx = np.argmin([e[1] for e in res])
            
    #         # 创建一个只包含最优构象的新分子对象并返回
    #         best_mol = Chem.Mol(mol)
    #         best_mol.RemoveAllConformers()
    #         best_conformer = mol.GetConformer(int(min_energy_idx))
    #         best_mol.AddConformer(best_conformer, assignId=True)
    #         return best_mol

    #     except Exception:
    #         # 任何其他意外错误都直接返回 None
    #         return None

    @staticmethod
    def generate_3d_mol(smiles: str, quality: str = 'fast') -> Chem.Mol:
        """
        一个更稳健且可调速的3D构象生成函数。
        quality: 'fast' 或 'high'
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: return None
            mol = Chem.AddHs(mol)
        except Exception:
            return None

        params = AllChem.ETKDGv3()
        params.randomSeed = 0xf00d
        params.numThreads = 1 # **关键优化点，见方案二**

        if quality == 'high':
            # 高质量模式：采样更多构象，使用MMFF力场
            num_confs = 20
            optimizer = AllChem.MMFFOptimizeMoleculeConfs
        else: # fast模式
            # 快速模式：采样更少构象，使用速度飞快的UFF力场
            num_confs = 5 
            optimizer = AllChem.UFFOptimizeMoleculeConfs

        try:
            cids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
            if not cids: raise ValueError("Conformer embedding failed.")
            
            res = optimizer(mol, numThreads=1) # **关键优化点，见方案二**
            if not res: raise ValueError("Optimization failed.")

            min_energy_idx = np.argmin([e[1] for e in res])
            
            best_mol = Chem.Mol(mol)
            best_mol.RemoveAllConformers()
            best_conformer = mol.GetConformer(int(min_energy_idx))
            best_mol.AddConformer(best_conformer, assignId=True)
            return best_mol
        except Exception:
            try:
                mol_fallback = Chem.Mol(mol)
                if AllChem.EmbedMolecule(mol_fallback, params) == -1:
                    return None
                
                if quality == 'fast':
                    AllChem.UFFOptimizeMolecule(mol_fallback)
                else:
                    AllChem.MMFFOptimizeMolecule(mol_fallback)
                return mol_fallback
            except Exception:
                return None
    @torch.no_grad()
    def debug_inspect_logits(self, logits, prompt_length, batch_idx, epoch, step):
        """
        一个全面的 logits 检查函数，用于诊断模型在 prompt 后的第一个 token 预测行为。

        Args:
            logits (torch.Tensor): 模型输出的原始 logits，形状为 [batch_size, seq_len, vocab_size]。
            prompt_length (int): 输入 prompt (pocket embedding) 的长度。
            batch_idx (int): 当前 batch 的索引。
            epoch (int): 当前的 epoch。
            step (int): 当前的 global_step。
        """
        # --- 准备工作 ---
        # 我们关心的是 prompt 之后紧接着的第一个预测 token
        # 其 logits 位于 prompt 的最后一个位置上 (索引为 prompt_length - 1)
        first_pred_logits = logits[:, prompt_length - 1, :]

        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        batch_size = logits.shape[0]

        # --- 开始打印详细的诊断信息 ---
        print("\n" + "="*80)
        print(f"|  <<<<<<<<<<<<<<<<< LOGITS INSPECTION REPORT >>>>>>>>>>>>>>>>>  |")
        print(f"|  Epoch: {epoch:<4} |  Global Step: {step:<7} |  Batch Index: {batch_idx:<5}  |")
        print("="*80)

        # 逐个样本分析
        for i in range(batch_size):
            sample_logits = first_pred_logits[i] # 当前样本的 logits, 形状: [vocab_size]

            # 1. 找出概率最高的 token
            pred_id = torch.argmax(sample_logits).item()
            pred_token_str = self.tokenizer.decode([pred_id])

            print(f"\n--- Sample {i+1}/{batch_size} ---")
            print(f"  [Top-1 Prediction]: Token='{pred_token_str}' (ID: {pred_id})")

            # 2. 检查这个 token 是否为特殊 token
            if pred_id == eos_token_id:
                print("  [!!!] DIAGNOSIS: The most likely next token is [EOS]. This is the suspected root cause of empty strings.")
            elif pred_id == pad_token_id:
                print("  [!!!] WARNING: The most likely next token is [PAD]. This is unusual and should be investigated.")
            else:
                print("  [OK]  DIAGNOSIS: The most likely token is a regular token, not a special one.")

            # 3. 查看 Top-5 预测，以了解概率分布
            top_5_logits, top_5_ids = torch.topk(sample_logits, 5)
            print("\n  [Top-5 Most Likely Tokens]:")
            print("  -------------------------------------------------------------")
            print(f"  | Rank | {'Token':<20} | {'ID':<7} | {'Logit Value':<15} |")
            print("  |------|----------------------|---------|-----------------|")
            for rank, (token_id, logit_val) in enumerate(zip(top_5_ids, top_5_logits)):
                token_id = token_id.item()
                token_str = self.tokenizer.decode([token_id])
                is_eos = "<- [EOS]" if token_id == eos_token_id else ""
                print(f"  |  #{rank+1}  | {token_str:<20} | {token_id:<7} | {logit_val.item():<15.4f} | {is_eos}")
            print("  -------------------------------------------------------------")

            # 4. 检查 EOS token 的具体排名和 logit 值
            eos_logit_val = sample_logits[eos_token_id].item()
            sorted_logits, sorted_indices = torch.sort(sample_logits, descending=True)
            eos_rank = (sorted_indices == eos_token_id).nonzero(as_tuple=True)[0].item()
            print(f"\n  [EOS Token Specifics]:")
            print(f"    - Logit Value: {eos_logit_val:.4f}")
            print(f"    - Overall Rank: #{eos_rank + 1} out of {len(sample_logits)} tokens")

        print("\n" + "="*80)
        print(f"|  <<<<<<<<<<<<<<<<<<<<< END OF REPORT >>>>>>>>>>>>>>>>>>>>>>>  |")
        print("="*80 + "\n")
        
def canonicalize_selfies(selfies):
    smiles = sf.decoder(selfies)
    try:
        canon_smiles = Chem.CanonSmiles(smiles)
        canon_selfies = sf.encoder(canon_smiles)
    except Exception:
        return '', '', ''
    return canon_selfies, canon_smiles, smiles

def reencode_selfies(selfies):
    decoded_smiles = sf.decoder(selfies)
    try:
        molecule = Chem.MolFromSmiles(decoded_smiles)
        smiles_with_chirality = Chem.MolToSmiles(molecule, kekuleSmiles=True)
        reencoded_selfies = sf.encoder(smiles_with_chirality)
        smiles_without_chirality = Chem.MolToSmiles(molecule, isomericSmiles=False)
    except Exception:
        return '', '', ''
    return reencoded_selfies, smiles_with_chirality, smiles_without_chirality

