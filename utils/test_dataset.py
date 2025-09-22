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

    tokenizer = LLMPL.init_tokenizer(args)
    print(f"Tokenizer词汇表示量: {len(tokenizer)}\n")
    dataset = QM9LMDataset(root=args.root, selfies_tokenizer=tokenizer, rand_smiles=args.rand_smiles, aug_inv=args.aug_inv > 0)
    print("数据集加载成功！\n")

    print("--- 取出数据集中的第一个分子 (idx=0) ---")
    # 直接像访问列表一样，调用__getitem__(0)
    for i in range(20):
      first_molecule_data = dataset[i]

      print("成功获取一个Data对象！\n")

      print("--- 个Data对象 ---")
      # torch_geometric 会将Data对象打印得非常清晰
      print(first_molecule_data)

      print("\n--- 访问具体属性 ---")

      # 1. 3D坐标
      print(f"属性 'pos' (3D坐标) 的形状: {first_molecule_data.pos.shape}")
      print("这代表了 [原子数量, 3(x,y,z)]\n")

      # 2. 1D蓝图
      print(f"属性 'selfies' (1D蓝图): {first_molecule_data.selfies}\n")

      # 3. 目标属性
      print(f"属性 'y' (19个化学属性) 的形状: {first_molecule_data.y.shape}")
      print(f"第一个属性值(偶极矩): {first_molecule_data.y[0, 0].item():.4f}\n")

      # 4. 图连接性
      print(f"属性 'edge_index' (图连接性) 的形状: {first_molecule_data.edge_index.shape}")
      print("这代表了 [2, 化学键数量 * 2]\n")

      # 5. RDKit对象
      print(f"属性 'rdmol' 的类型: {type(first_molecule_data.rdmol)}")


if __name__ == '__main__':
    experience_the_data()
