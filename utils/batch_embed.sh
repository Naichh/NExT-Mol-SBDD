#!/bin/bash

cleanup() {
    echo -e "\n\n捕获到退出信号... 正在停止所有后台处理进程..."
    pkill -P $$
    echo "所有子进程已停止。"
}
trap cleanup INT TERM EXIT

NUM_GPUS=4 
POCKET_DATA_ROOT="/data/share/liuzhiyuan/nai/NExT-Mol/datasets/sbdd/crossdocked_pocket"
FULL_PROTEIN_ROOT="/data/share/liuzhiyuan/nai/NExT-Mol/datasets/crossdocked_v1.1_rmsd1.0"
SCRIPT_NAME="/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/utils/pocket_embed_new.py" # 确保路径正确

echo "即将启动 $NUM_GPUS 个并行进程..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate esm3

for (( i=0; i<$NUM_GPUS; i++ ))
do
    echo "  -> 正在启动分片 $i，使用 GPU $i..."
    CUDA_VISIBLE_DEVICES=$i python $SCRIPT_NAME \
        --dataset_root "$POCKET_DATA_ROOT" \
        --full_protein_root "$FULL_PROTEIN_ROOT" \
        --num-shards $NUM_GPUS \
        --shard-id $i &
done

wait
echo "所有分片已处理完毕！"
trap - INT TERM EXIT