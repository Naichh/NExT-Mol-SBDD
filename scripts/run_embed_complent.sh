#!/bin/bash

# --- 1. 请在这里确认你的配置 ---

# a) 确认这是你那个包含 ESM 库的 Conda 环境的真实名称
ESM_ENV_NAME="esm3" # <--- 如果你的环境不叫 "esm"，请修改这里

# b) 确认这是你那个生成 .pt 文件的 Python 脚本的正确路径
PYTHON_SCRIPT="/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/pocket_embed.py"


# --- 日志文件配置 ---
LOG_DIR="/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/final_fix_pt_$TIMESTAMP.log"

# --- 需要修复的24个文件的列表 ---
FILES_TO_FIX=(
    "/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/data/sbdd/crossdocked_pocket/PPARA_HUMAN_197_468_0/2npa_A_rec_1k7l_544_lig_tt_min_0_pocket10.pkl"
    "/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/data/sbdd/crossdocked_pocket/PPARA_HUMAN_197_468_0/2rew_A_rec_3kdt_7ha_lig_tt_docked_0_pocket10.pkl"
    "/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/data/sbdd/crossdocked_pocket/PPARA_HUMAN_197_468_0/3kdt_B_rec_3kdu_nks_lig_tt_docked_3_pocket10.pkl"
    "/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/data/sbdd/crossdocked_pocket/PPARA_HUMAN_197_468_0/3kdt_B_rec_3sp6_il2_lig_tt_min_0_pocket10.pkl"
    "/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/data/sbdd/crossdocked_pocket/PPARA_HUMAN_197_468_0/2p54_A_rec_3kdt_7ha_lig_tt_docked_2_pocket10.pkl"
    "/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/data/sbdd/crossdocked_pocket/PPARA_HUMAN_197_468_0/2znn_A_rec_1i7g_az2_lig_tt_min_0_pocket10.pkl"
    "/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/data/sbdd/crossdocked_pocket/PPARA_HUMAN_197_468_0/3g8i_A_rec_3fei_ctm_lig_tt_min_0_pocket10.pkl"
    "/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/data/sbdd/crossdocked_pocket/PPARA_HUMAN_197_468_0/1i7g_A_rec_3sp6_il2_lig_tt_docked_1_pocket10.pkl"
    "/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/data/sbdd/crossdocked_pocket/PPARA_HUMAN_197_468_0/3kdu_B_rec_1k7l_544_lig_tt_min_0_pocket10.pkl"
    "/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/data/sbdd/crossdocked_pocket/PPARA_HUMAN_197_468_0/2p54_A_rec_3fei_ctm_lig_tt_docked_0_pocket10.pkl"
    "/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/data/sbdd/crossdocked_pocket/PPARA_HUMAN_197_468_0/4ci4_A_rec_3sp6_il2_lig_tt_min_0_pocket10.pkl"
    "/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/data/sbdd/crossdocked_pocket/PPARA_HUMAN_197_468_0/2znn_A_rec_3vi8_13m_lig_tt_docked_0_pocket10.pkl"
    "/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/data/sbdd/crossdocked_pocket/PPARA_HUMAN_197_468_0/1kkq_A_rec_1kkq_471_lig_tt_min_0_pocket10.pkl"
    "/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/data/sbdd/crossdocked_pocket/PPARA_HUMAN_197_468_0/4ci4_A_rec_3kdt_7ha_lig_tt_docked_1_pocket10.pkl"
    "/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/data/sbdd/crossdocked_pocket/PPARA_HUMAN_197_468_0/3g8i_A_rec_3fei_ctm_lig_tt_docked_0_pocket10.pkl"
    "/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/data/sbdd/crossdocked_pocket/PPARA_HUMAN_197_468_0/3vi8_A_rec_3sp6_il2_lig_tt_min_0_pocket10.pkl"
    "/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/data/sbdd/crossdocked_pocket/PPARA_HUMAN_197_468_0/2p54_A_rec_2p54_735_lig_tt_docked_1_pocket10.pkl"
    "/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/data/sbdd/crossdocked_pocket/PPARA_HUMAN_197_468_0/3kdt_B_rec_1k7l_544_lig_tt_min_0_pocket10.pkl"
    "/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/data/sbdd/crossdocked_pocket/PPARA_HUMAN_197_468_0/2znn_A_rec_2znn_s44_lig_tt_docked_0_pocket10.pkl"
    "/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/data/sbdd/crossdocked_pocket/PPARA_HUMAN_197_468_0/1i7g_A_rec_3et1_et1_lig_tt_min_0_pocket10.pkl"
    "/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/data/sbdd/crossdocked_pocket/PPARA_HUMAN_197_468_0/1i7g_A_rec_4ci4_y1n_lig_tt_min_0_pocket10.pkl"
    "/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/data/sbdd/crossdocked_pocket/PPARA_HUMAN_197_468_0/2znn_A_rec_3g8i_ro7_lig_tt_min_0_pocket10.pkl"
    "/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/data/sbdd/crossdocked_pocket/PPARA_HUMAN_197_468_0/2rew_A_rec_1i7g_az2_lig_tt_min_0_pocket10.pkl"
    "/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/data/sbdd/crossdocked_pocket/PPARA_HUMAN_197_468_0/1i7g_A_rec_1i7g_az2_lig_tt_docked_0_pocket10.pkl"
)

{
# --- 主逻辑 ---
echo "Starting FINAL FIX: Reprocessing 24 files to generate .pt embeddings..."

for PKL_FILE_PATH in "${FILES_TO_FIX[@]}"; do
    
    PDB_FILE="${PKL_FILE_PATH%.pkl}.pdb"
    
    if [ ! -f "$PDB_FILE" ]; then
        echo "--------------------------------------------------"
        echo "[!] Skipping $(basename "$PDB_FILE")... PDB file not found."
        continue
    fi

    DIR_PATH=$(dirname "$PDB_FILE")
    SDF_BASENAME=$(basename "$PDB_FILE" | sed -E 's/_pocket[0-9]+\.pdb$//')
    SDF_FILE="$DIR_PATH/$SDF_BASENAME.sdf"
    OUTPUT_DIR="$DIR_PATH"

    if [ -f "$SDF_FILE" ]; then
        echo "--------------------------------------------------"
        echo "Processing: $(basename "$PDB_FILE")"
        
        # --- 2. 关键改动：使用 conda run 来确保在正确的环境中执行 ---
        conda run -n "$ESM_ENV_NAME" python "$PYTHON_SCRIPT" \
            --pdb "$PDB_FILE" \
            --sdf "$SDF_FILE" \
            --output "$OUTPUT_DIR"
    else
        echo "--------------------------------------------------"
        echo "[!] Skipping $(basename "$PDB_FILE")... SDF file not found."
    fi
done

echo "--------------------------------------------------"
echo "Final fix script complete."

} 2>&1 | tee "$LOG_FILE"
