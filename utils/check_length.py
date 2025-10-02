import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
import warnings

# --- 配置参数 (根据你的脚本和代码自动提取) ---

CACHE_PATH = "/data/share/liuzhiyuan/nai/NExT-Mol/datasets/cache/all_data_stage1_cache.pt"

LLM_MODEL_ID = "acharkq/MoLlama"

MAX_SF_TOKENS = 128
MAX_POCKET_TOKENS = 128


def analyze_dataset_lengths():
    """
    主函数，用于加载数据、分析长度并打印报告。
    """
    print("--- 开始数据集长度分析 ---")

    # 检查缓存文件是否存在
    if not os.path.exists(CACHE_PATH):
        print(f"\n[错误] 关键的缓存文件未找到: {CACHE_PATH}")

        return

    print(f"\n[1/4] 正在加载分词器 '{LLM_MODEL_ID}'...")
    try:
        # 禁用关于未使用权重的警告，因为我们只需要分词器
        warnings.filterwarnings("ignore", category=UserWarning, message=".*model weights.*")
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
        print("分词器加载成功。")
    except Exception as e:
        print(f"\n[错误] 加载分词器失败: {e}")
        print("请检查模型ID是否正确，以及你是否有网络连接来下载它。")
        return

    # 2. 加载预处理缓存数据
    print(f"\n[2/4] 正在从 '{CACHE_PATH}' 加载预处理数据...")
    print("这可能需要一些时间，取决于文件大小...")
    try:
        # 将数据加载到CPU以节省显存
        data_cache = torch.load(CACHE_PATH, map_location='cpu')
        if not isinstance(data_cache, dict) or not data_cache:
            print("[错误] 缓存文件格式不正确或为空。")
            return
        num_samples = len(data_cache)
        print(f"数据加载成功！共找到 {num_samples} 个样本。")
    except Exception as e:
        print(f"\n[错误] 加载缓存文件失败: {e}")
        print("请检查文件路径是否正确，以及文件是否损坏。")
        return

    # 3. 遍历数据并收集长度信息
    print("\n[3/4] 正在遍历样本并计算实际长度...")
    pocket_lengths = []
    selfies_lengths = []

    # data_cache.values() 包含了所有样本的字典
    for sample in tqdm(data_cache.values(), desc="分析样本"):
        if sample is None:
            continue

        # a. 获取 Pocket Embedding 长度
        if 'pdb_embedding' in sample and sample['pdb_embedding'] is not None:
            # 形状通常是 (sequence_length, embedding_dim)
            pocket_lengths.append(sample['pdb_embedding'].shape[0])

        # b. 获取 SELFIES token 长度
        # 你的代码为每个样本生成了 'selfies' 和 'selfies2'
        selfies1 = sample.get('selfies', '')
        selfies2 = sample.get('selfies2', '')

        # 使用分词器的 `encode` 方法可以准确计算出token数量，包含了特殊token
        len1 = len(tokenizer.encode(selfies1)) if selfies1 else 0
        len2 = len(tokenizer.encode(selfies2)) if selfies2 else 0

        # 我们关心的是两者中更长的那一个，因为它有被截断的风险
        selfies_lengths.append(max(len1, len2))

    print("长度计算完成。")

    # 4. 分析结果并打印报告
    print("\n[4/4] 生成分析报告...")
    print("\n" + "="*80)
    print("                 数 据 集 覆 盖 率 分 析 报 告")
    print("="*80)

    # --- 分析 Pocket ---
    print("\n--- 1. Protein Pocket (蛋白质口袋) ---")
    print(f"设定最大长度 (max_pocket_tokens): {MAX_POCKET_TOKENS}")
    if pocket_lengths:
        p_lens = np.array(pocket_lengths)
        print(f"  - 数据集中最大长度: {p_lens.max()}")
        print(f"  - 数据集中平均长度: {p_lens.mean():.2f}")
        print(f"  - 长度分布 (95th/99th percentile): {np.percentile(p_lens, 95):.0f} / {np.percentile(p_lens, 99):.0f}")

        exceeding_count = np.sum(p_lens > MAX_POCKET_TOKENS)
        exceeding_percent = (exceeding_count / num_samples) * 100
        print("\n  [结论]")
        if exceeding_count == 0:
            print(f"  ✅ 覆盖率 100%: 设定的 {MAX_POCKET_TOKENS} 长度足以覆盖所有蛋白质口袋。")
        else:
            print(f"  ⚠️  注意: 有 {exceeding_count} 个样本 ({exceeding_percent:.2f}%) 的蛋白质口袋长度超过 {MAX_POCKET_TOKENS}。")
            print("     这些样本的蛋白质信息在输入模型时会被截断。")
    else:
        print("  - 未能收集到任何蛋白质口袋长度信息。")


    # --- 分析 Selfies ---
    print("\n--- 2. Ligand SELFIES (配体分子) ---")
    print(f"设定最大长度 (max_sf_tokens): {MAX_SF_TOKENS}")
    if selfies_lengths:
        s_lens = np.array(selfies_lengths)
        print(f"  - 数据集中最大Token数: {s_lens.max()}")
        print(f"  - 数据集中平均Token数: {s_lens.mean():.2f}")
        print(f"  - Token数分布 (95th/99th percentile): {np.percentile(s_lens, 95):.0f} / {np.percentile(s_lens, 99):.0f}")

        exceeding_count = np.sum(s_lens > MAX_SF_TOKENS)
        exceeding_percent = (exceeding_count / num_samples) * 100
        print("\n  [结论]")
        if exceeding_count == 0:
            print(f"  ✅ 覆盖率 100%: 设定的 {MAX_SF_TOKENS} 长度足以覆盖所有分子的SELFIES表示。")
        else:
            print(f"  ⚠️  注意: 有 {exceeding_count} 个样本 ({exceeding_percent:.2f}%) 的分子SELFIES Token数超过 {MAX_SF_TOKENS}。")
            print("     这些分子的信息在输入模型时会被截断。")
    else:
        print("  - 未能收集到任何分子SELFIES长度信息。")

    print("\n" + "="*80)
    print("分析结束。")


if __name__ == '__main__':
    analyze_dataset_lengths()
