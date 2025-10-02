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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import torch
import torch.nn as nn
from evaluation.eval_functions import run_full_evaluation

# class ResidualAdapter(nn.Module):
#     def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
#         super().__init__()
#         #1536 1024 2048
#         self.block = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.GELU(),

#             nn.Linear(hidden_dim, hidden_dim),
#             nn.GELU(),


#             nn.Linear(hidden_dim, output_dim)
#         )
#         if input_dim != output_dim:
#             self.shortcut = nn.Linear(input_dim, output_dim)
#         else:
#             self.shortcut = nn.Identity()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # H(x) = F(x) + x
#         return self.block(x) + self.shortcut(x)

# def print_grad_hook(module, grad_input, grad_output):
#     """A simple hook to print the norm of the output gradient."""
#     if grad_output[0] is not None:
#         grad_norm = torch.linalg.norm(grad_output[0])
#         print(f"\n[DEBUG GRADIENT] Grad norm for {module.__class__.__name__}: {grad_norm:.4f}\n")



def worker_generate_3d(data_pair):

    selfies_string, pocket_path = data_pair
    try:
        smiles = sf.decoder(selfies_string)
        if not smiles:
            return None, None, selfies_string

        mol_3d = LLMPL.generate_3d_mol(smiles,'fast')

        if mol_3d is None:
            return None, None, selfies_string

        return mol_3d, pocket_path, selfies_string

    except Exception:
        return None, None, selfies_string


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

    def on_fit_start(self):
            if self.unfreeze_epoch > 0 and self.args.llm_tune in ['lora', 'full']:
                if self.trainer.is_global_zero:
                    print("\n" + "="*80)
                    print(f"INFO (on_fit_start): Two-stage training initiated for '{self.args.llm_tune}' mode.")
                    print("STAGE 1: Training Adapter ONLY.")
                    print("="*80 + "\n")

                if self.args.llm_tune == 'lora':
                    for name, param in self.llm_model.named_parameters():
                        if 'lora' in name:
                            param.requires_grad = False
                elif self.args.llm_tune == 'full':
                    for param in self.llm_model.parameters():
                        param.requires_grad = False

                if self.trainer.is_global_zero:
                    print("Trainable parameters for STAGE 1 (set in on_fit_start):")
                    trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
                    total_params = sum(p.numel() for p in self.parameters())
                    print(f"Total trainable parameters: {trainable_params / 1e6:.2f}M")
                    print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.4f}%")
                    proj_params = sum(p.numel() for p in self.projection.parameters() if p.requires_grad)
                    print(f"Projection adapter has {proj_params/1e6:.2f}M trainable parameters.")


    def on_train_batch_start(self, batch, batch_idx):
        if self.is_in_llm_warmup:
            optimizer = self.trainer.optimizers[0]


            start_lr = self.llm_target_lr * 0.01
            current_step = self.llm_warmup_step_count
            total_steps = self.llm_warmup_steps

            lr_ratio = current_step / total_steps
            current_lr = start_lr + (self.llm_target_lr - start_lr) * lr_ratio


            optimizer.param_groups[0]['lr'] = current_lr

            self.llm_warmup_step_count += 1

            if self.llm_warmup_step_count >= self.llm_warmup_steps:
                self.is_in_llm_warmup = False
                optimizer.param_groups[0]['lr'] = self.llm_target_lr
                print("\n" + "="*80)
                print("INFO: LLM re-warmup finished. Handing LR control back to the main scheduler.")
                print(f"Final LLM LR set to: {optimizer.param_groups[0]['lr']}")
                print("="*80 + "\n")



    def on_train_epoch_start(self):

        if self.unfreeze_epoch <= 0 or self.args.llm_tune not in ['lora', 'full']:
            return

        if self.current_epoch == self.unfreeze_epoch:
            if self.trainer.is_global_zero:
                print("\n" + "="*80)
                print(f"INFO (on_train_epoch_start): Reached Epoch {self.current_epoch}. Transitioning to STAGE 2 for '{self.args.llm_tune}' mode.")
                print("="*80 + "\n")
            if not self.is_in_llm_warmup and self.llm_warmup_step_count == 0:
                print("\n" + "="*80)
                print(f"INFO: Epoch {self.current_epoch}. Unfreezing LLM and INITIATING a {self.llm_warmup_steps}-step re-warmup for LLM parameters.")
                print(f"Target LLM LR after warmup: {self.llm_target_lr}")
                print("="*80 + "\n")
                self.is_in_llm_warmup = True
            if self.args.llm_tune == 'lora':
                for name, param in self.llm_model.named_parameters():
                    if 'lora' in name:
                        param.requires_grad = True
            elif self.args.llm_tune == 'full':
                for param in self.llm_model.parameters():
                    param.requires_grad = True

            if self.args.tune_embedding:
                print("Unfreezing embed_tokens for STAGE 2...")
                set_embed_tokens_trainable(self.llm_model)

            if self.args.decay_projection_lr:
                print(f"Apply decay_projection_lr ")
                if not hasattr(self, 'adapter_lr_adjusted'):
                    # Get the current optimizer
                    optimizer = self.trainer.optimizers[0]

                    # The second parameter group (index 1) is our adapter_params
                    # We assume the first group (index 0) is llm_params
                    if len(optimizer.param_groups) > 1:
                        # Calculate the new, lower learning rate for the adapter
                        original_adapter_lr = optimizer.param_groups[1]['lr']
                        new_adapter_lr = original_adapter_lr / 10.0

                        # Directly set the new learning rate
                        optimizer.param_groups[1]['lr'] = new_adapter_lr

                        if self.trainer.is_global_zero:
                            print("\n" + "!"*80)
                            print(f"STRATEGY: Dynamically reducing Adapter learning rate for STAGE 2.")
                            print(f"--> Original Adapter LR: {original_adapter_lr}")
                            print(f"--> New Adapter LR: {new_adapter_lr}")
                            print("!"*80 + "\n")

                    # Set the flag so this block doesn't run again
                    self.adapter_lr_adjusted = True
            if self.trainer.is_global_zero:
                print("Trainable parameters for STAGE 2 (after unfreezing):")
                trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in self.parameters())
                print(f"Total trainable parameters: {trainable_params / 1e6:.2f}M")
                print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.4f}%")

    @classmethod
    def init_tokenizer(cls, args):
        tokenizer = AutoTokenizer.from_pretrained(
            args.llm_model,
            padding_side='left'
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("Warning: pad_token is not set. Setting it to eos_token.")

        tokenizer.add_bos_token = True
        tokenizer.add_eos_token = True
        return tokenizer



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
            # if args.tune_embedding:
            #     set_embed_tokens_trainable(llm_model)
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
        self.is_in_llm_warmup = False
        self.llm_warmup_steps = args.llm_warmup_steps
        self.llm_warmup_step_count = 0
        self.llm_target_lr = args.llm_target_lr
        self.adapter_lr_adjusted = False
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
        print(f"The model's hidden size is: {self.hidden_size}")

        self.embedding_norm = torch.nn.LayerNorm(1536,elementwise_affine=True)
        self.projection = torch.nn.Sequential(
        torch.nn.Linear(1536, 1024),
        torch.nn.GELU(),
        torch.nn.Linear(1024, self.hidden_size)
        )
        self.post_projection_norm = torch.nn.LayerNorm(self.llm_model.config.hidden_size,elementwise_affine=True)

               # --- 应用权重初始化 ---
        print("INFO: Applying Xavier uniform initialization to the projection adapter.")
        self.projection.apply(self._init_weights)

        # The last layer of your projection Sequential block is at index -1.
        # This will now print the gradient norm for this layer on every backward pass.
        # self.projection[-1].register_full_backward_hook(print_grad_hook)
        self.property_distribution = property_distribution


        # if self.delta_train:
        #     print("INFO: Staged training enabled. Freezing LoRA parameters initially.")
        #     for name, param in self.llm_model.named_parameters():
        #         if 'lora' in name:
        #             param.requires_grad = False


        self.save_hyperparameters(args)


    def setup(self, stage=None):
        """
        这个方法在数据加载器准备好后、训练开始前被调用一次。
        我们在这里计算所有依赖于数据长度的超参数。
        """
        if stage == 'fit' or stage is None:
            # 使用 self.trainer.datamodule 访问数据模块
            train_loader = self.trainer.datamodule.train_dataloader()

            # 计算总的训练步数
            # self.trainer.max_epochs 是从 Trainer 的参数中获取的
            self.total_steps = self.trainer.max_epochs * len(train_loader)

            # 计算预热步数
            self.warmup_steps = min(len(train_loader), self.args.warmup_steps)

            print(f"--- [INFO] Calculated total training steps: {self.total_steps} ---")
            print(f"--- [INFO] Calculated warmup steps: {self.warmup_steps} ---")

    def configure_optimizers(self):

        print(f"INFO: Configuring optimizer with differential learning rates...")


        # 定义一个列表来存放不同参数组的配置
        param_groups = []

        llm_params = {
            'params': self.llm_model.parameters(),
            'lr': self.llm_target_lr,
            'weight_decay': self.args.weight_decay
        }
        param_groups.append(llm_params)

        adapter_params = {
            'params': [
                *self.embedding_norm.parameters(),
                *self.projection.parameters(),
                *self.post_projection_norm.parameters()
            ],
            'lr': self.args.init_lr,
            'weight_decay': self.args.weight_decay
        }
        param_groups.append(adapter_params)

        optimizer = optim.AdamW(
            param_groups
        )

        if self.args.scheduler == 'linear_warmup_cosine_lr':
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.001,
                end_factor=1.0,
                total_iters=self.warmup_steps
            )
            main_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.total_steps - self.warmup_steps,
                eta_min=self.args.min_lr
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[self.warmup_steps]
            )


            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                }
            }
        else:
            return optimizer

    # def configure_optimizers(self):
    #     """
    #     一个干净的、使用 PyTorch 官方标准组件的优化器配置方法。
    #     """
    #     print(f"INFO: Configuring optimizer with official PyTorch schedulers...")

    #     optimizer = optim.AdamW(
    #         self.parameters(),
    #         lr=self.args.init_lr,
    #         weight_decay=self.args.weight_decay
    #     )

    #     if self.args.scheduler == 'linear_warmup_cosine_lr':
    #         # 1. 定义预热阶段的调度器
    #         # 从一个非常小的学习率线性增长到基础学习率 self.args.init_lr
    #         warmup_scheduler = LinearLR(
    #             optimizer,
    #             start_factor=0.001, # 你可以根据需要调整起始因子
    #             end_factor=1.0,
    #             total_iters=self.warmup_steps
    #         )

    #         # 2. 定义余弦退火阶段的调度器
    #         # 在预热结束后，从基础学习率衰减到最小学习率 self.args.min_lr
    #         main_scheduler = CosineAnnealingLR(
    #             optimizer,
    #             T_max=self.total_steps - self.warmup_steps,
    #             eta_min=self.args.min_lr
    #         )

    #         # 3. 使用 SequentialLR 将它们串联起来
    #         # milestones=[self.warmup_steps] 表示在第 warmup_steps 步时，从 warmup_scheduler 切换到 main_scheduler
    #         scheduler = SequentialLR(
    #             optimizer,
    #             schedulers=[warmup_scheduler, main_scheduler],
    #             milestones=[self.warmup_steps]
    #         )

    #         # 4. 以 Lightning 推荐的字典格式返回
    #         return {
    #             "optimizer": optimizer,
    #             "lr_scheduler": {
    #                 "scheduler": scheduler,
    #                 "interval": "step", # 表示每个训练步(step)都更新一次调度器
    #             }
    #         }
    #     else:
    #         # 如果不使用调度器，则只返回优化器
    #         return optimizer


    def training_step(self, batch, batch_idx):
        if batch[0] is None:
            return None
        # if self.scheduler:
        #     self.scheduler.step(self.trainer.global_step)

        selfies_batch, selfies2_batch, pockets_emb, pock_attn_mask, _, _ = batch
        batch_size = selfies_batch.input_ids.shape[0]

        # --- Standard Generative Loss (Default Behavior) ---

        lm_loss0 = self.forward(selfies_batch, pockets_emb, pock_attn_mask)
        lm_loss1 = self.forward(selfies2_batch, pockets_emb, pock_attn_mask)
        loss = (lm_loss0 + lm_loss1) / 2

        self.log('train_loss', loss, sync_dist=True, batch_size=batch_size)


        optimizer = self.trainer.optimizers[0]

        # Log the learning rate for the LLM parameter group (index 0)
        # Renamed to 'lr_llm' for clarity
        self.log('lr/lr_llm', optimizer.param_groups[0]['lr'], sync_dist=True, batch_size=batch_size)

        # Check if the second parameter group (adapter) exists and log its LR
        if len(optimizer.param_groups) > 1:
            # Log the learning rate for the Adapter parameter group (index 1)
            self.log('lr/lr_adapter', optimizer.param_groups[1]['lr'], sync_dist=True, batch_size=batch_size)



        # if self.global_step % 50 == 0:
        #     with torch.no_grad():
        #         # 新结构中，线性层分别在索引 0 和 3
        #         first_linear_layer_norm = torch.linalg.norm(self.projection[0].weight)
        #         second_linear_layer_norm = torch.linalg.norm(self.projection[3].weight)
        #         self.log('debug/proj_norm_L1', first_linear_layer_norm)
        #         self.log('debug/proj_norm_L2', second_linear_layer_norm)

        return loss

    @torch.no_grad()
    def on_validation_epoch_start(self):
        self.validation_outputs=[]
        self.sampled_seq = []

    # In LLMPL class
    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):

        if hasattr(self, 'global_rank'):
            print(f">>> [DEBUG] Rank {self.global_rank}: Entering validation_step for batch_idx {batch_idx}.")
        else:
            print(f">>> [DEBUG] Entering validation_step for batch_idx {batch_idx}.")
        selfies_batch, selfies2_batch, pockets_emb, pock_attn_mask, ground_truth_mols, pocket_paths = batch



        # --- 1. 深入的诊断逻辑 (仅在第一个batch执行) ---
        if self.trainer.is_global_zero and batch_idx == 0:
            print("\n" + "="*80)
            print(f"|  <<<<<<<<<<<<<<<<< DIAGNOSTIC REPORT (Epoch: {self.current_epoch}) >>>>>>>>>>>>>>>>>  |")
            print("="*80)

            normalized_pockets_emb_before = self.embedding_norm(pockets_emb)

            avg_emb_before = torch.mean(normalized_pockets_emb_before, dim=1)
            # Normalize for cosine similarity calculation
            avg_emb_before_norm = torch.nn.functional.normalize(avg_emb_before, p=2, dim=1)
            similarity_matrix_before = torch.matmul(avg_emb_before_norm, avg_emb_before_norm.T)

            mask = torch.triu(torch.ones_like(similarity_matrix_before), diagonal=1).bool()

            if mask.sum() > 0:
                avg_sim_before = similarity_matrix_before[mask].mean().item()
                max_sim_before = similarity_matrix_before[mask].max().item()
                print(f"|  [DIAGNOSIS - BEFORE Projection]")
                print(f"|  - Avg Cosine Similarity of Input Embeds:  {avg_sim_before:.4f}")
                print(f"|  - Max Cosine Similarity of Input Embeds:  {max_sim_before:.4f}")
                print("."*80)
                self.log('debug/Avg_Cosine_Similarity_of_Input_Embeds', avg_sim_before, rank_zero_only=True)
                self.log('debug/Max_Cosine_Similarity_of_Input_Embeds', max_sim_before, rank_zero_only=True)
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
                self.log('debug/Avg_Cosine_Similarity_of_Output_Embeds', avg_sim_after, rank_zero_only=True)
                self.log('debug/Max_Cosine_Similarity_of_Output_Embeds', max_sim_after, rank_zero_only=True)
            # --- 诊断 Part C: Logits 检查 ---
            prompt_embeds_for_logits = prompt_embeds_after
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



        is_2d_eval_epoch = (self.current_epoch + 1) % self.args.eval_2d_every_n_epochs == 0
        is_3d_eval_epoch = (self.current_epoch + 1) % self.args.eval_3d_every_n_epochs == 0
        if not is_2d_eval_epoch and not is_3d_eval_epoch:
            if self.trainer.is_global_zero:
                print(f"\n--- [INFO] Epoch {self.current_epoch}: Skipping molecule generation as per schedule. ---")
            return
        # ==========================================================

        # 只有在需要评估时，代码才会继续执行到这里
        if is_3d_eval_epoch:
            num_output_for_this_epoch = self.args.num_output_3d
        else: # 如果代码能到这里，is_2d_eval_epoch 必然为 True
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
        if not self.trainer.is_global_zero: # 只在主进程执行
            return

        is_2d_eval_epoch = (self.current_epoch + 1) % self.args.eval_2d_every_n_epochs == 0
        is_3d_eval_epoch = (self.current_epoch + 1) % self.args.eval_3d_every_n_epochs == 0
        if not is_2d_eval_epoch and not is_3d_eval_epoch:
            # 如果本周期不需要进行任何评估，则直接退出
            print(f"\n--- [INFO] Epoch {self.current_epoch}: Skipping on_validation_epoch_end metrics calculation as per schedule. ---")
            return
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

            print("\n" + "="*60)


            # --- SBDD 相似度评估 ---
            print(f"--- DEBUG: Preparing pairs for SBDD similarity evaluation... ---")
            generated_mols = []
            ground_truth_mols = []
            pocket_paths=[]
            for pair in self.validation_outputs:
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

            sbdd_metrics = SBDD_validation(generated_mols, ground_truth_mols)

            # --- 打印和记录新的SBDD指标 ---
            print("\n[--- SBDD Similarity Metrics vs. Ground Truth ---]")
            for key, value in sbdd_metrics.items():
                print(f"  - {key:<20}: {value:.4f}")
                self.log(key, value, rank_zero_only=True)

            print("="*60 + "\n")

            print("\n[--- Basic 2D Metrics ---]")
            for key, value in eval_results_2d.items():
                print(f"  - {key:<12}: {value:.4f}")
                self.log(key, value, rank_zero_only=True)
            print("\n[--- Advanced MOSES & Drug-likeness Metrics ---]")
            for key, value in moses_metrics.items():
                print(f"  - {key:<12}: {value:.4f}")
                self.log(key, value, rank_zero_only=True)
            print("="*60 + "\n")

        # print(f"--- DEBUG: Starting 3D conversion and docking for {len(self.validation_outputs)} generated sequences. ---")

        # run_3d_eval = (self.current_epoch + 1) % self.args.eval_3d_every_n_epochs == 0
        # if run_3d_eval:
        #     # --- 2. 准备3D分子并进行全面评估 (现在使用多进程并行) ---
        #     print(f"\n--- DEBUG: Starting PARALLEL 3D conversion for {len(self.validation_outputs)} generated sequences. ---")

        #     tasks = [(pair['generated'], pair['pocket_path']) for pair in self.validation_outputs]

        #     generated_mols_3d = []
        #     pocket_paths_for_vina = []
        #     failed_molecules_info = []
        #     successful_molecules_info = []

        #     # 获取CPU核心数，可以留一两个核心给系统
        #     import multiprocessing as mp
        #     import os

        #     slurm_cpus_str = os.getenv('SLURM_CPUS_PER_TASK')
        #     if slurm_cpus_str:
        #         cpu_cores = int(slurm_cpus_str)
        #         print(f"--- INFO: Detected Slurm allocation. Using {cpu_cores} allocated cores for 3D generation. ---")
        #     else:
        #         # 如果不在Slurm环境中，回退到使用mp.cpu_count()
        #         cpu_cores = mp.cpu_count()
        #         print(f"--- INFO: Not in a Slurm environment. Using total system cores: {cpu_cores} for 3D generation. ---")

        #     # 根据获取的核心数计算进程数，为操作系统和主进程留出余地
        #     num_processes = max(1, cpu_cores - 2)
        #     print(f"--- INFO: Using {num_processes} CPU cores for parallel processing. ---")

        #     with mp.Pool(processes=num_processes) as pool:
        #         results_iterator = pool.imap_unordered(worker_generate_3d, tasks)

        #         for mol_3d, pocket_path, original_selfies in tqdm(results_iterator, total=len(tasks), desc="Generating 3D conformers"):
        #             if mol_3d is not None and pocket_path is not None:
        #                 generated_mols_3d.append(mol_3d)
        #                 pocket_paths_for_vina.append(pocket_path)
        #                 try:
        #                     smiles = sf.decoder(original_selfies)
        #                 except:
        #                     smiles = "DECODING_ERROR"
        #                 successful_molecules_info.append({'selfies': original_selfies, 'smiles': smiles})
        #             else:
        #                 try:
        #                     smiles = sf.decoder(original_selfies) if original_selfies else "EMPTY_SELFIES"
        #                 except:
        #                     smiles = "DECODING_ERROR"
        #                 failed_molecules_info.append({'selfies': original_selfies, 'smiles': smiles})

        #     # --- 打印成功和失败的样本 ---
        #     if failed_molecules_info:
        #         print("\n--- [DEBUG] Molecules that failed 3D generation (sample) ---")
        #         for i, failure in enumerate(failed_molecules_info[:5]):
        #             print(f"  - Failed Mol {i+1}:")
        #             print(f"    - Original SELFIES: {failure['selfies']}")
        #             print(f"    - Decoded SMILES  : {failure['smiles']}")
        #         print("----------------------------------------------------------\n")

        #     if successful_molecules_info:
        #         print("\n--- [DEBUG] Sample of SUCCESSFULLY generated 3D molecules ---")
        #         for i, success in enumerate(successful_molecules_info[:5]):
        #             print(f"  - Success Mol {i+1}:")
        #             print(f"    - Original SELFIES: {success['selfies']}")
        #             print(f"    - Decoded SMILES  : {success['smiles']}")
        #         print("----------------------------------------------------------\n")

        #         num_generated = len(generated_mols_3d)
        #         num_failed = len(failed_molecules_info)
        #         total_attempted = num_generated + num_failed


        #         grouped_mols_for_eval = defaultdict(list)



        #         for mol, path in zip(generated_mols_3d, pocket_paths_for_vina):
        #             grouped_mols_for_eval[path].append(mol)
        #         # --- Final check and call to the evaluation function ---

        #         if not grouped_mols_for_eval:
        #             print("\n[WARNING] No valid 3D molecules to evaluate. Skipping comprehensive evaluation.")
        #         else:
        #             print("\n--- [INFO] 3D Molecule Generation Summary ---")
        #             print(f"  - Total sequences attempted  : {total_attempted}")
        #             print(f"  - Successfully generated     : {num_generated} ({num_generated/total_attempted*100:.2f}%)")
        #             print(f"  - Failed to generate         : {num_failed} ({num_failed/total_attempted*100:.2f}%)")
        #             print(f"  - Number of keys:", len(grouped_mols_for_eval))
        #             print("----------------------------------------------------------\n")

        #             # Call the single, powerful evaluation function
        #             eval_metrics = run_full_evaluation(
        #             grouped_mols=grouped_mols_for_eval
        #             )
        #             # Log the results
        #             print("\n" + "="*60)
        #             print(f"--- Logging Final Metrics for Epoch {self.current_epoch} ---")
        #             if eval_metrics:
        #                 for key, value in eval_metrics.items():
        #                     log_key = key.replace(' ', '_').replace('(%)', 'percent').replace('(', '').replace(')', '')
        #                     self.log(f'eval/{log_key}', value, sync_dist=True)
        #                     print(f"  - Logged eval/{log_key}: {value:.4f}")
        #             print("="*60 + "\n")


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
        parser.add_argument('--llm_target_lr', type=float, default=5e-7, help='Target learning rate for the LLM backbone in Stage 2.')
        parser.add_argument('--llm_warmup_steps', type=int, default=500, help='Number of re-warmup steps for the LLM in Stage 2.')

        parser.add_argument('--projection_dropout', type=float, default=0.1, help='projection_dropout')

        parser.add_argument('--weight_decay', type=float, default=0.05, help='optimizer weight decay')
        parser.add_argument('--init_lr', type=float, default=1e-4, help='optimizer init learning rate')
        parser.add_argument('--min_lr', type=float, default=1e-5, help='optimizer min learning rate')
        parser.add_argument('--warmup_lr', type=float, default=1e-6, help='optimizer warmup learning rate')
        parser.add_argument('--warmup_steps', type=int, default=1000, help='optimizer warmup steps')
        parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='optimizer lr decay rate')
        parser.add_argument('--scheduler', type=str, default='linear_warmup_cosine_lr', help='type of scheduler') # or
        parser.add_argument('--zero_embedding',action='store_true', default=False)
        parser.add_argument('--decay_projection_lr',action='store_true', default=False)
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
