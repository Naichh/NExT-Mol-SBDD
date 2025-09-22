#!/bin/bash

# --- 配置 ---
# 设置您的数据集根目录
ROOT_DIR="/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/data/sbdd/crossdocked_pocket"
# 设置您的 Python 脚本路径
PYTHON_SCRIPT="/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/pocket_embed.py"
LOG_DIR="/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/logs"
TIMESTAMP=$(data+"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/embed_output_$TIMESTAMP.log"
{
# --- 主逻辑 ---
echo "Starting batch embedding process..."
echo "Mode: Generating FULL protein embeddings and OVERWRITING existing files."
echo "Dataset Root: $ROOT_DIR"

# 使用 find 命令安全地遍历所有 _pocket*.pdb 文件
find "$ROOT_DIR" -type f -name "*_pocket*.pdb" -print0 | while IFS= read -r -d $'\0' PDB_FILE; do
    
    # 获取 PDB 文件所在的目录
    DIR_PATH=$(dirname "$PDB_FILE")
    
    # 从 PDB 文件名推断出对应的 SDF 文件名
    SDF_BASENAME=$(basename "$PDB_FILE" | sed -E 's/_pocket[0-9]+\.pdb$//')
    SDF_FILE="$DIR_PATH/$SDF_BASENAME.sdf"

    # 输出目录就是文件所在的目录
    OUTPUT_DIR="$DIR_PATH"

    # 检查对应的 SDF 文件是否存在
    if [ -f "$SDF_FILE" ]; then
        echo "--------------------------------------------------"
        echo "Processing: $(basename "$PDB_FILE")"
        
        # 调用 Python 脚本
        python "$PYTHON_SCRIPT" \
            --pdb "$PDB_FILE" \
            --sdf "$SDF_FILE" \
            --output "$OUTPUT_DIR"
    else
        echo "--------------------------------------------------"
        echo "[!] Skipping $(basename "$PDB_FILE")"
        echo "    Reason: Corresponding SDF file not found at $SDF_FILE"
    fi
done

echo "--------------------------------------------------"
echo "Batch processing complete."

} 2>&1 | tee "LOG_FILE"