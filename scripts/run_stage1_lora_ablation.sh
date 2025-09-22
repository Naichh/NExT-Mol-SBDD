#!/bin/bash
#set -x  # <-- 加入这一行
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# 你的项目代码根目录
PROJECT_ROOT="/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol"
cd $PROJECT_ROOT
echo "当前工作目录: $(pwd)"

# 激活你的Conda环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate nextmol
echo "已激活Conda环境: $CONDA_DEFAULT_ENV"


# SBDD数据集的根目录
SBDD_DATA_ROOT="${PROJECT_ROOT}/data/sbdd/crossdocked_pocket"
# 数据集划分文件的路径
SPLIT_FILE="${PROJECT_ROOT}/data/sbdd/crossdocked_pocket/split_by_name.pt"
# 预训练的MoLlama模型ID
LLM_MODEL_ID="acharkq/MoLlama"

RESUME_FROM_CKPT="/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/checkpoints/sbdd_lora-finetune_20250914_195210/last.ckpt" 

if [ -f "$RESUME_FROM_CKPT" ] || [ -d "$RESUME_FROM_CKPT" ]; then # 检查文件或目录是否存在
    echo "将从检查点恢复训练: $RESUME_FROM_CKPT"
fi

# 使用时间戳创建唯一运行名称
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="sbdd_lora-finetune_ablation_${TIMESTAMP}"

# 定义输出路径
RESULTS_DIR="${PROJECT_ROOT}/results/${RUN_NAME}"
CHECKPOINTS_DIR="${PROJECT_ROOT}/checkpoints/${RUN_NAME}"

# 设置GPU
export CUDA_VISIBLE_DEVICES='0'
# export NCCL_SOCKET_IFNAME=ib1
# export NCCL_IB_DISABLE=0      # 确保使用 IB


# If ib1 doesn't work:
# export NCCL_SOCKET_IFNAME=eno1
#export NCCL_P2P_DISABLE=1



# --- 3. 执行核心SBDD训练命令 ---
echo "开始执行SBDD Stage 1: 全量微调..."
echo "标准输出日志将保存在: ${RESULTS_DIR}/training_out.log"
echo "错误日志将保存在: ${RESULTS_DIR}/training_err.log"
echo "模型将保存在: ${CHECKPOINTS_DIR}"

# 确保输出目录存在
mkdir -p "$RESULTS_DIR"
mkdir -p "$CHECKPOINTS_DIR"


#torchrun --nproc_per_node=2
python llm_train_cross_docked.py \
    --output_dir "$CHECKPOINTS_DIR" \
    --filename "$RUN_NAME" \
    --seed 42 \
    --devices 'auto' \
    --mode 'train' \
    --max_epochs 100 \
    --generate_eval_epoch 2 \
    --save_every_n_epochs 5 \
    --check_val_every_n_epoch 1 \
    --dataset_root "$SBDD_DATA_ROOT" \
    --split_file "$SPLIT_FILE" \
    --batch_size 32 \
    --num_workers 2 \
    --no_embedding \
    --max_sf_tokens 128 \
    --max_pocket_tokens 128 \
    --llm_model "$LLM_MODEL_ID" \
    --llm_tune 'lora' \
    --unfreeze_epoch 3 \
    --accelerator 'gpu' \
    --precision 'bf16-mixed' \
    --init_lr 5e-7 \
    --epoch_without_eval 1 \
    --attention_dropout 0.15 \
    --weight_decay 0.01 \
    --strategy_name 'deepspeed' \
    > "${RESULTS_DIR}/training_out.log" \
    2> "${RESULTS_DIR}/training_err.log"

echo "SBDD Stage 1 运行执行完毕。"
