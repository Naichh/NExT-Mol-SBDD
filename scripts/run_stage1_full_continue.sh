#!/bin/bash
#----Debug/Environmental setting----
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1

# --- 1. 工作目录与环境配置 ---
PROJECT_ROOT="/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol"
cd $PROJECT_ROOT
echo "当前工作目录: $(pwd)"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate nextmol
echo "已激活Conda环境: $CONDA_DEFAULT_ENV"


# --- 2. 数据存储目录 ---
CKPT_PATH="/data/share/liuzhiyuan/nai/NExT-Mol/all_checkpoints/stage1/fft/sbdd_full-finetune_20250910_123652/last.ckpt"

SBDD_DATA_ROOT="/data/share/liuzhiyuan/nai/NExT-Mol/datasets/sbdd/crossdocked_pocket"
SPLIT_FILE="${SBDD_DATA_ROOT}/split_by_name.pt"
LLM_MODEL_ID="acharkq/MoLlama"

CHECKPOINTS_DIR="/data/share/liuzhiyuan/nai/NExT-Mol/all_checkpoints/stage1/fft/${RUN_NAME}"


if [ -f "$CKPT_PATH" ] || [ -d "$CKPT_PATH" ]; then # 检查文件或目录是否存在
    echo "将从检查点恢复训练: $CKPT_PATH"
fi


# --- 3. 日志目录 ---
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="sbdd_full-finetune_${TIMESTAMP}"
RESULTS_DIR="${PROJECT_ROOT}/results/training/stage1/$(date)/${RUN_NAME}"

mkdir -p "$RESULTS_DIR"
mkdir -p "$CHECKPOINTS_DIR"


# --- 4. 执行命令 ---
export CUDA_VISIBLE_DEVICES='1'
export MASTER_PORT=29500

echo "开始执行SBDD Stage 1: 全量微调..."
echo "标准输出日志将保存在: ${RESULTS_DIR}/training_out.log"
echo "错误日志将保存在: ${RESULTS_DIR}/training_err.log"
echo "模型将保存在: ${CHECKPOINTS_DIR}"

# 确保输出目录存在

torchrun  --nproc_per_node=2  llm_train_cross_docked.py \
    --output_dir "$CHECKPOINTS_DIR" \
    --filename "$RUN_NAME" \
    --seed 42 \
    --devices 'auto' \
    --mode 'train' \
    --max_epochs 100 \
    --generate_eval_epoch 1 \
    --save_every_n_epochs 5 \
    --check_val_every_n_epoch 1 \
    --temperature 0.2 \
    --dataset_root "$SBDD_DATA_ROOT" \
    --split_file "$SPLIT_FILE" \
    --batch_size 32 \
    --num_workers 2 \
    --max_sf_tokens 128 \
    --max_pocket_tokens 128 \
    --llm_model "$LLM_MODEL_ID" \
    --llm_tune 'full' \
    --unfreeze_epoch 3 \
    --accelerator 'gpu' \
    --precision 'bf16-mixed' \
    --strategy_name 'deepspeed' \
    --ckpt_path "$CKPT_PATH" \
    > "${RESULTS_DIR}/training_out.log" \
    2> "${RESULTS_DIR}/training_err.log"

echo "SBDD Stage 1 运行执行完毕。"
