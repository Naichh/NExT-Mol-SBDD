import pickle
import sys
import os
from pprint import pprint
#使用示例：python /mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/utils/pkl.py /mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/sampled_mol/20251006_203709_/epoch_0_3d_tasks.pkl
def view_pkl(file_path, max_items=5):
    if not os.path.isfile(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return

    print(f"📦 正在读取: {file_path}")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    print(f"\n✅ 读取成功，类型: {type(data)}")

    # 如果是字典
    if isinstance(data, dict):
        print(f"🔑 字典键数量: {len(data)}")
        print(f"前 {max_items} 个键:")
        pprint(list(data.keys())[:max_items])
        print("\n示例键值内容:")
        for k in list(data.keys())[:max_items]:
            print(f"── {k}: {type(data[k])}")
            pprint(data[k])
            print()
    # 如果是列表
    elif isinstance(data, list):
        print(f"📋 列表长度: {len(data)}")
        print(f"前 {max_items} 个元素示例:")
        for i, item in enumerate(data[:max_items]):
            print(f"[{i}] {type(item)}:")
            pprint(item)
            print()
    else:
        print("📘 数据内容:")
        pprint(data)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python view_pkl.py your_file.pkl [max_items]")
        sys.exit(1)

    file_path = sys.argv[1]
    max_items = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    view_pkl(file_path, max_items)
