import torch
import os
import pickle
# ---------- 配置 ----------
# 文件路径，请替换为你的 pkl 文件路径
file_path = "/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/temp_output/1m4n_A_rec_1m7y_ppg_lig_tt_min_0_pocket10.pkl"
file2_path='/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/temp_output/embed/1i7g_A_rec_1i7g_az2_lig_tt_docked_0_pocket10.pkl'
# ---------- 检查文件是否存在 ----------
if not os.path.exists(file_path):
    raise FileNotFoundError(f"文件不存在: {file_path}")


if not os.path.exists(file2_path):
    raise FileNotFoundError(f"文件不存在: {file2_path}")

with open(file_path, "rb") as f:
    data = pickle.load(f)
with open(file2_path, "rb") as f2:
    data2= pickle.load(f2)
print("数据类型:", type(data))
print(data)

# # ---------- 读取 pkl ----------
# data = torch.load(file_path, weights_only=False)
# # ---------- 打印信息 ----------
# print("数据类型:", type(data))
# if isinstance(data, torch.Tensor):
#     print("张量形状:", data.shape)
#     print("张量内容预览:\n", data)
# elif isinstance(data, dict):
#     print("字典键:", list(data.keys()))
#     print("字典内容预览:")
#     for k, v in data.items():
#         print(f"{k}: {v}")
# else:
#     print("数据内容:\n", data)
print(data.shape)
print("最大值:", data.max().item())
print("最小值:", data.min().item())
print("均值:", data.mean().item())

print("数据类型:", type(data2))
print(data2)

# # ---------- 读取 pkl ----------
# data = torch.load(file_path, weights_only=False)
# # ---------- 打印信息 ----------
# print("数据类型:", type(data))
# if isinstance(data, torch.Tensor):
#     print("张量形状:", data.shape)
#     print("张量内容预览:\n", data)
# elif isinstance(data, dict):
#     print("字典键:", list(data.keys()))
#     print("字典内容预览:")
#     for k, v in data.items():
#         print(f"{k}: {v}")
# else:
#     print("数据内容:\n", data)
print(data2.shape)
print("最大值:", data2.max().item())
print("最小值:", data2.min().item())
print("均值:", data2.mean().item())
