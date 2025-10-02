#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 功能: 并行启动多个 ESM-3 embedding 提取进程

# 优雅退出函数
cleanup() {
    echo -e "\n\n捕获到退出信号... 正在停止所有后台处理进程..."
    pkill -P $$
    echo "所有子进程已停止。"
}
trap cleanup INT TERM EXIT

# --- 配置区 ---
NUM_GPUS=4
POCKET_DATA_ROOT="/data/share/liuzhiyuan/nai/NExT-Mol/datasets/sbdd/crossdocked_pocket"
FULL_PROTEIN_ROOT="/data/share/liuzhiyuan/nai/NExT-Mol/datasets/crossdocked_v1.1_rmsd1.0"
# 更新为新的Python脚本路径
SCRIPT_NAME="/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/utils/esm/pocket_embed_new.py"

# --- 启动区 ---
echo "即将启动 $NUM_GPUS 个并行进程来运行: $SCRIPT_NAME"

# 激活 Conda 环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate esm

# 循环启动并行进程
for (( i=0; i<$NUM_GPUS; i++ ))
do
    echo "  -> 正在启动分片 $i，使用 GPU $i..."
    CUDA_VISIBLE_DEVICES=$i python $SCRIPT_NAME \
        --dataset_root "$POCKET_DATA_ROOT" \
        --full_protein_root "$FULL_PROTEIN_ROOT" \
        --num-shards $NUM_GPUS \
        --shard-id $i &
done

# 等待所有后台进程执行完毕
wait

echo "所有分片已处理完毕！ Embedding 已直接保存在 $POCKET_DATA_ROOT 的相应子目录中。"

# 脚本正常结束，移除陷阱
trap - INT TERM EXIT
