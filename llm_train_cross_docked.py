import os
import argparse
import torch
import warnings
import sys
import lightning as L
import lightning.pytorch.callbacks as plc
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger, WandbLogger
import multiprocessing as mp
from model.llm_cross_docked_rewrite import LLMPL
from data_provider.cross_docked_data_module import PocketLigandDataModule
import traceback
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
torch.set_float32_matmul_precision('medium')
from datetime import timedelta
class LoraSaveCallback(plc.Callback):

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # 只在主进程(rank 0)执行保存操作
        if not trainer.is_global_zero:
            return

        # 获取刚刚保存的检查点路径
        # trainer.checkpoint_callback 是管理 .ckpt 文件的主要回调实例
        # 我们需要找到最新的检查点路径
        # 注意: ModelCheckpoint 的 `best_model_path` 可能不适用于 save_top_k=-1 的情况
        #      一个更稳妥的方法是直接从 trainer 的回调列表中找到它并获取其 last_model_path
        #      但更简单的方法是检查 trainer.ckpt_path

        last_ckpt_path = trainer.checkpoint_callback.last_model_path

        # 我们只关心按 epoch 保存的检查点，而不是 'last.ckpt'
        if not last_ckpt_path or "last.ckpt" in last_ckpt_path:
            return

        # 从检查点文件名中提取 epoch 编号
        # 例如: 从 ".../epoch_09.ckpt" -> 9
        try:
            # 简单但有效的文件名解析
            base_name = os.path.basename(last_ckpt_path) # e.g., "epoch=09.ckpt" or "{epoch:02d}.ckpt"
            # Lightning 格式通常是 "epoch=X-step=Y.ckpt" 或 "epoch_XX.ckpt"
            # 我们需要一个更灵活的解析方式
            current_epoch = trainer.current_epoch
        except (ValueError, IndexError):
            # 如果解析失败，则跳过
            print(f"\n--- [LoraSaveCallback] Could not determine epoch from ckpt path: {last_ckpt_path}. Skipping LoRA save. ---")
            return

        # 定义LoRA适配器的保存路径
        lora_save_path = os.path.join(
            trainer.checkpoint_callback.dirpath,
            f"epoch_{current_epoch:02d}_lora_adapters"
        )

        print(f"\n--- [LoraSaveCallback] Checkpoint saved for epoch {current_epoch}. Saving LoRA adapters to: {lora_save_path} ---")

        # 使用PEFT的 .save_pretrained() 方法来保存
        pl_module.llm_model.save_pretrained(lora_save_path)

def main(args):


    print(f"DEBUG: torch.cuda.device_count() = {torch.cuda.device_count()}")
    print(f"[RANK INFO] MASTER_ADDR={os.environ.get('MASTER_ADDR')}...")
    print(f"[RANK INFO] MASTER_ADDR={os.environ.get('MASTER_ADDR')} MASTER_PORT={os.environ.get('MASTER_PORT')}")
    current_epoch = 0
    L.seed_everything(args.seed)

    # Tokenizer
    tokenizer = LLMPL.init_tokenizer(args)
    print("Tokenizer vocab size:", len(tokenizer))

    # DataModule
    datamodule = PocketLigandDataModule(
        dataset_root=args.dataset_root,
        split_file=args.split_file,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        max_sf_tokens=args.max_sf_tokens,
        max_pocket_tokens=args.max_pocket_tokens,
    )


    model = LLMPL(
        args,
        tokenizer=tokenizer,
        max_sf_tokens=args.max_sf_tokens,
        max_pocket_tokens=args.max_pocket_tokens,
        property_distribution=None,
    )
    print("Total model parameters:", sum(p.numel() for p in model.parameters()))


    if args.lora_ckpt_path and os.path.isdir(args.lora_ckpt_path):
        print(f"\nINFO: Loading PEFT LoRA adapter weights from: {args.lora_ckpt_path}")
        # model.llm_model 是被PEFT包装过的模型实例
        model.llm_model.load_adapter(args.lora_ckpt_path)
        print("    -> LoRA adapter weights loaded successfully.\n")
    # =============================================================


    # Checkpoint directory
    ckpt_dir = args.output_dir

    callbacks = [
        plc.ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="{epoch:02d}",
            every_n_epochs=args.save_every_n_epochs,
            save_top_k=-1,
            save_on_train_epoch_end=True,
        ),
        plc.ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="last",
            save_last=True,
        ),
        LoraSaveCallback() # <-- 启用新的 LoRA 保存回调
    ]


    csv_logger = CSVLogger(save_dir=ckpt_dir, name=args.filename)
    tb_logger = TensorBoardLogger(save_dir=ckpt_dir, name=args.filename)
    wandb_logger = WandbLogger(project="NExT-Mol-SBDD", name=args.filename)


    strategy = 'auto'  # 默认值

    # 多卡场景
    is_multi_gpu = (isinstance(args.devices, int) and args.devices > 1) or \
                   (isinstance(args.devices, str) and len(args.devices.split(',')) > 1) or \
                   (args.devices == 'auto' and torch.cuda.device_count() > 1)

    if is_multi_gpu:
        print(f"Using multi-GPU training with devices: {args.devices}")
        if args.strategy_name == 'deepspeed':
            strategy = L.pytorch.strategies.DeepSpeedStrategy(stage=2)
            print(f"DeepSpeedStrategy has been Initialized")
        else:
            strategy = L.pytorch.strategies.DDPStrategy(find_unused_parameters=True)
            print(f"DDPStrategy has been Initialized")
    else:
        print("Using single-GPU or CPU")
        if args.ckpt_path and os.path.isdir(args.ckpt_path):
            print(f"Detected a directory checkpoint for single-device '{args.mode}' mode. Forcing DeepSpeedStrategy to load it.")
            strategy = L.pytorch.strategies.DeepSpeedStrategy(stage=2)
            print(f"DeepSpeedStrategy has been Initialized")

    print(f"Preparing the Trainer.. ")
    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=strategy,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        precision=args.precision,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        log_every_n_steps=20,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        logger=[csv_logger, wandb_logger],
        #gradient_clip_val=args.gradient_clip_val
    )
    print(f"The Trainer has been prepared!")


    if trainer.is_global_zero:
        log_dir = tb_logger.log_dir
        print("\n" + "="*80)
        print("TensorBoard Log Directory:")
        print(f"    {log_dir}")
        print("\nTo view logs, run the following command on a machine with access to this path:")
        print(f"    tensorboard --logdir='{log_dir}' --port=6006")
        print("="*80 + "\n")


    # Train / Eval
    if args.mode == 'train':
        print(f"Starting training...")
        if args.ckpt_path and args.ckpt_path.lower() != 'none':
             print(f"Resuming training from checkpoint: {args.ckpt_path}")
        else:
            print("Starting training from scratch.")

        trainer.fit(model, datamodule=datamodule, ckpt_path=args.ckpt_path if (args.ckpt_path and args.ckpt_path.lower() != 'none') else None)

    elif args.mode.startswith('eval'):
            if not args.ckpt_path or args.ckpt_path.lower() == 'none':
                raise ValueError("A valid --ckpt_path must be provided for evaluation mode.")

            print(f"Starting evaluation with checkpoint: {args.ckpt_path}")

            trainer.validate(model=model, datamodule=datamodule, ckpt_path=args.ckpt_path)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")
    print("INFO: Script finished successfully. Exiting.")
    sys.exit(0)
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()

    # 基本参数
    parser.add_argument('--filename', type=str, default='training_logs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'eval_gen', 'eval_conf', 'eval_1d_gen'], default='train')
    parser.add_argument('--strategy_name', type=str, default='deepspeed')
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=str, default='auto')
    parser.add_argument('--precision', type=str, default='bf16-mixed')
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--save_every_n_epochs', type=int, default=5)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--cache_epoch', type=int, default=2)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--use_flash_attention', action='store_true', help='Use Flash Attention if available')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Directory to save logs and checkpoints')
    parser.add_argument('--eval_3d_every_n_epochs', type=int, default=10, help='Frequency (in epochs) to run 3D metrics and docking.')
    parser.add_argument('--eval_2d_every_n_epochs', type=int, default=2, help='Frequency (in epochs) to run 2D metrics evaluation.')
    parser.add_argument('--num_output_2d', type=int, default=100, help='Number of molecules to generate for 2D evaluation.')
    parser.add_argument('--num_output_3d', type=int, default=5, help='Number of molecules to generate for 3D evaluation.')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0, help='Value for gradient clipping.')
    parser.add_argument('--deepspeed_config', type=str, default=None, help='Path to deepspeed config file.')

        # 数据参数
    parser.add_argument('--dataset_root', type=str, default="/home/mydata/nextmol/crossdocked_pocket")
    parser.add_argument('--split_file', type=str, default="/home/mydata/nextmol/crossdocked_pocket/split_by_name.pt")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=32, help='Batch size for validation/evaluation')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_sf_tokens', type=int, default=128)
    parser.add_argument('--max_pocket_tokens', type=int, default=128)
    parser.add_argument('--lora_ckpt_path', type=str, default=None, help='Path to the saved PEFT LoRA adapter directory')

    # 模型参数
    parser = LLMPL.add_model_specific_args(parser)

    args, unknown = parser.parse_known_args()
    print("\n======= Training Configuration =======")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("=====================================\n")

    main(args)
