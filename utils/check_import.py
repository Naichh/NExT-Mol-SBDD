print("start!!!")
import sys
print(f"Python executable: {sys.executable}")

print("\n[1/3] 正在尝试导入 torch...")
import torch
print("      torch 导入成功。")
print(f"      - PyTorch 版本: {torch.__version__}")
print(f"      - CUDA 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"      - CUDA 版本: {torch.version.cuda}")

print("\n[2/3] --------------------")

print("[Probe 1] 脚本开始执行...")

import os
import argparse
import torch
import warnings
import lightning as L
import lightning.pytorch.callbacks as plc
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
import multiprocessing as mp
from rdkit import RDLogger

# --- 探针 ---
print("[Probe 2] 核心库导入成功。正在导入自定义模块...")

from model.llm_cross_docked_rewrite import LLMPL
from data_provider.cross_docked_data_module import PocketLigandDataModule

# --- 探针 ---
print("[Probe 3] 所有模块导入成功。")


RDLogger.DisableLog('rdApp.warning')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
torch.set_float32_matmul_precision('medium')


print("\n测试完成，所有核心库都可以正常导入。")