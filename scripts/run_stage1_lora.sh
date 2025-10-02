#!/bin/bash
#set -x
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
#增加超时时间
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=8000
#export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=ib1
export NCCL_IB_HCA=ib1
export PL_TORCH_DISTRIBUTED_BACKEND_TIMEOUT=3600
export TOKENIZERS_PARALLELISM=false
ulimit -n 65535

# --- 1. 工作目录与环境配置 ---
PROJECT_ROOT="/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol"
cd $PROJECT_ROOT
echo "当前工作目录: $(pwd)"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate nextmol
echo "已激活Conda环境: $CONDA_DEFAULT_ENV"


# --- 2. 数据存储目录 ---

SBDD_DATA_ROOT="/data/share/liuzhiyuan/nai/NExT-Mol/datasets/sbdd/crossdocked_pocket"
SPLIT_FILE="${SBDD_DATA_ROOT}/split_by_name.pt"
LLM_MODEL_ID="acharkq/MoLlama"



if [ -f "$CKPT_PATH" ] || [ -d "$CKPT_PATH" ]; then # 检查文件或目录是否存在
    echo "将从检查点恢复训练: $CKPT_PATH"
fi


# --- 3. 日志目录 ---
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="sbdd_lora-finetune_${TIMESTAMP}"
RESULTS_DIR="${PROJECT_ROOT}/results/training/stage1/$(date +'%Y%m%d_%H%M%S')/${RUN_NAME}"

CHECKPOINTS_DIR="/data/share/liuzhiyuan/nai/NExT-Mol/all_checkpoints/stage1/lora/${RUN_NAME}"



mkdir -p "$RESULTS_DIR"
mkdir -p "$CHECKPOINTS_DIR"


# --- 4. 执行命令 ---
export CUDA_VISIBLE_DEVICES='0,1,2,3'

echo "开始执行SBDD Stage 1: lora微调..."
echo "标准输出日志将保存在: ${RESULTS_DIR}/training_out.log"
echo "错误日志将保存在: ${RESULTS_DIR}/training_err.log"
echo "模型将保存在: ${CHECKPOINTS_DIR}"

# 确保输出目录存在


torchrun --nproc_per_node=4 --master_port=54345  llm_train_cross_docked.py \
    --output_dir "$CHECKPOINTS_DIR" \
    --filename "$RUN_NAME" \
    --seed 42 \
    --devices 'auto' \
    --mode 'train' \
    --max_epochs 300 \
    --batch_size 80 \
    --eval_batch_size 32\
    --temperature 0.2 \
    --do_sample \
    --save_every_n_epochs 6 \
    --check_val_every_n_epoch 1 \
    --dataset_root "$SBDD_DATA_ROOT" \
    --split_file "$SPLIT_FILE" \
    --num_output_2d 32 \
    --num_output_3d 5 \
    --num_beams 1 \
    --num_workers 0 \
    --eval_2d_every_n_epochs 1 \
    --eval_3d_every_n_epochs 200 \
    --max_sf_tokens 128 \
    --max_pocket_tokens 128 \
    --llm_model "$LLM_MODEL_ID" \
    --accelerator 'gpu' \
    --precision 'bf16-mixed' \
    --accumulate_grad_batches 1\
    --llm_tune 'lora' \
    --unfreeze_epoch 10 \
    --warmup_steps 1000 \
    --init_lr 1e-4 \
    --min_lr 5e-6 \
    --gradient_clip_val 1.0 \
    --epoch_without_eval 1 \
    --attention_dropout 0.05 \
    --weight_decay 0.01 \
    --strategy_name 'deepspeed' \
    > "${RESULTS_DIR}/training_out.log" \
    2> "${RESULTS_DIR}/training_err.log"

echo "SBDD Stage 1 运行执行完毕。"
