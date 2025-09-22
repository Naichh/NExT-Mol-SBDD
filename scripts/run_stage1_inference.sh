#!/bin/bash

# --- 1. 设置项目根目录 ---
# 将此路径更改为你的NExT-Mol代码库所在的根目录
PROJECT_ROOT="/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/run_stage1_inference.sh"
cd $PROJECT_ROOT
echo "当前工作目录: $(pwd)"

# --- 2. 激活Conda环境 ---
# 确保你的conda环境名正确
source $(conda info --base)/etc/profile.d/conda.sh
conda activate nextmol
echo "已激活Conda环境: $CONDA_DEFAULT_ENV"

# --- 3. 配置路径和参数 ---
# !! 关键步骤 !!
# 将此路径更改为你训练好的Stage 1模型检查点(.ckpt)文件的完整路径
# 它通常在 all_checkpoints/你的运行filename/ 目录下
CKPT_PATH="/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/all_checkpoints/sbdd_llm_finetune_run/last_epoch_03.ckpt"

# 为这次推理运行命名，输出会保存在相关目录下
FILENAME="qm9_inference_run"

# 设置你想生成的分子数量
SAMPLE_NUM=100

# 设置使用的GPU (推理通常只需要一块)
export CUDA_VISIBLE_DEVICES='0'


# --- 4. 执行核心推理命令 ---
echo "开始执行Stage 1: 推理生成新分子..."
python llm_train.py \
    --mode "eval_1d_gen" \
    # 关键！将模式从'train'改为'eval_1d_gen'，告诉脚本只进行1D生成评估。
    --init_checkpoint "$CKPT_PATH" \
    # 关键！加载你训练好的模型“大脑”。注意这里用的是--init_checkpoint。
    --filename "$FILENAME" \
    # 为本次推理命名，生成的SMILES文件会保存在对应的日志文件夹下。

    --dataset "QM9" \
    # 仍需指定数据集，以便正确加载词汇表(Tokenizer)等配置。
    --llm_model "acharkq/MoLlama" \
    # 仍需指定基础模型架构。

    --sample_num $SAMPLE_NUM \
    # 控制生成分子的数量。
    --temperature 1.0 \
    # 控制采样的随机性。
    --num_beams 1
    # 使用贪心采样（greedy search）。

echo "Stage 1 推理脚本执行完毕。"
