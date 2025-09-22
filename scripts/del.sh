#!/bin/bash
# 删除指定的 .pkl 文件

files=(
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

for f in "${files[@]}"; do
    if [ -f "$f" ]; then
        echo "Deleting $f"
        rm -f "$f"
    else
        echo "Not found: $f"
    fi
done
