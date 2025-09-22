import argparse
import os
import sys
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)
from data_provider.data_module import QM9LMDataModule
from model.llm_pl import LLMPL
from data_provider.data_module import QM9LMDataset



def experience_the_data():
    # 1. 模拟命令行参数，这样我们就不需要运行完整的训练脚本
    args = argparse.Namespace(
        dataset='QM9',
        root='/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/data/GEOM-QM9', # <--- !!! 修改这里为你的GEOM-QM9文件夹路径 !!!
        num_workers=0,
        batch_size=4, # 我们只看4个样本
        llm_model='acharkq/MoLlama',
        rand_smiles='restricted',
        aug_inv=0,
        addHs=False,
        condition_property=None
    )

    print("--- 步骤 1: 初始化 Tokenizer ---")
    tokenizer = LLMPL.init_tokenizer(args)
    print(f"Tokenizer词汇表示量: {len(tokenizer)}\n")
    dataset = QM9LMDataset(root=args.root, selfies_tokenizer=tokenizer, rand_smiles=args.rand_smiles, aug_inv=args.aug_inv > 0)
    print("--- 步骤 2: 初始化 QM9LMDataModule (这将执行__init__中的所有逻辑) ---")
    dm = QM9LMDataModule(root=args.root, num_workers=args.num_workers,
                         batch_size=args.batch_size, selfies_tokenizer=tokenizer, args=args)
    print(f"数据模块初始化完成。训练集大小: {len(dm.train_dataset)}\n")

    print("--- 步骤 3: 创建训练数据加载器 (train_dataloader) ---")
    train_loader = dm.train_dataloader()
    print("数据加载器创建成功。\n")

    print("--- 步骤 4: 从加载器中取出一批(batch)数据 ---")
    # next(iter(...)) 是获取迭代器中第一个元素的标准方法
    batch = next(iter(train_loader))
    print("成功取出一批数据！\n")

    print("--- 步骤 5: 分析这批数据的构成 ---")
    print(f"这批数据的类型是: {type(batch)}")
    print("它是一个字典，包含了模型需要的所有信息：")

    for key, value in batch.items():
        print(f"\n键 (Key): '{key}'")
        print(f"  - 值的类型: {type(value)}")
        if hasattr(value, 'shape'):
            print(f"  - 值的形状 (Shape): {value.shape}")

        # 解释每个键的含义
        if key == 'input_ids':
            print("  - 含义: 这是分词后的数字ID序列，是送入模型的主要输入。形状代表(批次大小, 序列最大长度)。")
        elif key == 'attention_mask':
            print("  - 含义: 告诉模型注意力机制应该关注哪些部分。1代表真实token，0代表填充token。")
        elif key == 'labels':
            print("  - 含义: 这是用于计算损失的正确答案。对于语言模型，它通常是input_ids向左平移一位。")

    print("\n--- 体验结束 ---")
    print("现在你看到了数据是如何从原始文件一步步变成模型可以直接使用的、整齐的张量(Tensor)的。")


if __name__ == '__main__':
    experience_the_data()
