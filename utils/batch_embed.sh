#!/bin/bash

# --- 新增：定义一个清理函数 ---
# 这个函数会在脚本退出时被调用
cleanup() {
    echo -e "\n\n捕获到退出信号... 正在停止所有后台处理进程..."
    # $$ 是当前脚本的进程ID (PID)
    # pkill -P $$ 会杀死所有父进程ID为当前脚本ID的子进程
    pkill -P $$
    echo "所有子进程已停止。"
}

# --- 新增：设置陷阱 ---
# trap 命令会在脚本收到特定信号时执行指定的函数
# INT: Ctrl+C
# TERM: kill 命令
# EXIT: 脚本正常或异常退出
trap cleanup INT TERM EXIT

# --- 配置 (保持不变) ---
NUM_GPUS=4 
DATASET_ROOT="/data/share/liuzhiyuan/nai/NExT-Mol/datasets/sbdd/crossdocked_pocket"
SCRIPT_NAME="/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/pocket_embed_new.py"

# --- 执行 (保持不变) ---
echo "即将启动 $NUM_GPUS 个并行进程来加速数据预处理..."
echo "现在，您可以随时按 Ctrl+C 来干净地终止所有进程。"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate esm3

for (( i=0; i<$NUM_GPUS; i++ ))
do
    echo "  -> 正在启动分片 $i，使用 GPU $i..."
    CUDA_VISIBLE_DEVICES=$i python $SCRIPT_NAME \
        --dataset_root "$DATASET_ROOT" \
        --num-shards $NUM_GPUS \
        --shard-id $i &
done

# 'wait' 命令会等待所有后台任务都执行完毕
# 如果在这里按 Ctrl+C，上面的 trap 会被触发
wait
echo "所有分片已处理完毕！"

# --- 新增：在脚本末尾移除陷阱 ---
# 这是一个好的实践，确保清理函数不会在脚本自然结束后再次被不必要地调用
trap - INT TERM EXIT