#!/bin/bash
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1




export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
#增加超时时间
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=6000
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=ib1

# --- 1. 工作目录与环境配置 ---
PROJECT_ROOT="/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol"
cd $PROJECT_ROOT
echo "当前工作目录: $(pwd)"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate nextmol
echo "已激活Conda环境: $CONDA_DEFAULT_ENV"

# --- 2. 数据存储目录 ---

CKPT_PATH="/data/share/liuzhiyuan/nai/NExT-Mol/all_checkpoints/stage1/fft/sbdd_full-finetune_20250910_123652/epoch=64.ckpt"

SBDD_DATA_ROOT="/data/share/liuzhiyuan/nai/NExT-Mol/datasets/sbdd/crossdocked_pocket"
SPLIT_FILE="${SBDD_DATA_ROOT}/split_by_name.pt"
LLM_MODEL_ID="acharkq/MoLlama"


# --- 3. 日志目录 ---
CKPT_FILENAME=$(basename "$CKPT_PATH" .ckpt)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EVAL_RUN_NAME="eval_fft_${TIMESTAMP}"

RESULTS_DIR="${PROJECT_ROOT}/results/evaluation/$(date +'%Y%m%d_%H%M%S')/${EVAL_RUN_NAME}"
# 确保输出目录存在
mkdir -p "$RESULTS_DIR"


# --- 4. 执行命令 ---
echo "开始执行评估..."
echo "评估结果将保存在: ${RESULTS_DIR}"

export CUDA_VISIBLE_DEVICES='2'

torchrun --nproc_per_node=1 --master_port=29817 llm_train_cross_docked.py \
    --output_dir "$RESULTS_DIR" \
    --filename "$EVAL_RUN_NAME" \
    --seed 42 \
    --devices 1 \
    --mode 'eval' \
    --ckpt_path "$CKPT_PATH" \
    --dataset_root "$SBDD_DATA_ROOT" \
    --split_file "$SPLIT_FILE" \
    --llm_model "$LLM_MODEL_ID" \
    --batch_size 1 \
    --eval_batch_size 1 \
    --num_workers 10 \
    --max_sf_tokens 128 \
    --max_pocket_tokens 128 \
    --llm_tune 'full' \
    --precision 'bf16-mixed' \
    --eval_2d_every_n_epochs 1 \
    --eval_3d_every_n_epochs 999 \
    --num_output_2d 100 \
    --num_output_3d 5 \
    --do_sample \
    --old_ckpt \
    --num_beams 1 \
    --temperature 0.7 \
    > "${RESULTS_DIR}/eval_out.log" \
    2> "${RESULTS_DIR}/eval_err.log"

echo "评估运行执行完毕。"