import os
import torch
import tempfile
import argparse
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, LogitsConfig

# 确保您可以从您的项目中导入这些函数
from data_provider.cross_docked_data import PDBProtein, parse_sdf_file


def standardize_pdb(pdb_path):
    """
    读取一个PDB文件，强制添加MODEL/ENDMDL记录，并返回一个标准化的临时文件路径。
    """
    with open(pdb_path, 'r', errors='ignore') as f:
        lines = f.readlines()

    lines = [line for line in lines if not line.startswith(('MODEL', 'ENDMDL'))]

    with tempfile.NamedTemporaryFile(mode='w', suffix=".pdb", delete=False, encoding='utf-8') as temp_f:
        temp_path = temp_f.name
        temp_f.write("MODEL        1\n")
        temp_f.writelines(lines)
        temp_f.write("ENDMDL\n")
    return temp_path


def main(pdb_path, sdf_path, output_dir, radius=10.0, device="cuda"):
    file_name = os.path.splitext(os.path.basename(pdb_path))[0]
    # --- 修改 1: 输出文件扩展名改为 .pt ---
    embed_path = os.path.join(output_dir, f"{file_name}.pt")

    if os.path.exists(embed_path):
        print(f"[*] Embedding file already exists, skipping: {os.path.basename(pdb_path)}")
        return  
    # 标准化 PDB 文件
    standardized_pdb_path = standardize_pdb(pdb_path)
    
    try:
        protein = ESMProtein.from_pdb(standardized_pdb_path)
    finally:
        os.remove(standardized_pdb_path) # 确保临时文件总能被删除

    model = ESM3.from_pretrained("esm3-sm-open-v1").to(device)
    
    protein_tensor = model.encode(protein)
    embedding_config = LogitsConfig(return_embeddings=True)
    output = model.logits(protein_tensor, embedding_config)
    
    full_embedding = output.embeddings.squeeze(0)
    
    print(f"[*] Full protein embedding extracted. Shape: {full_embedding.shape}")

    # --- 修改 2: 使用 torch.save 保存 ---
    print(f"[*] Saving FULL protein embedding to {embed_path}")
    # 使用 torch.save 保存，并且在保存前将张量移动到 CPU
    torch.save(full_embedding.cpu(), embed_path)

    print(f"[*] Done for {file_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract FULL protein embedding using ESM3")
    parser.add_argument("--pdb", type=str, required=True, help="Path to pocket .pdb file")
    parser.add_argument("--sdf", type=str, required=True, help="Path to ligand .sdf file (for file pairing)")
    parser.add_argument("--output", type=str, required=True, help="Directory to save embeddings")
    parser.add_argument("--radius", type=float, default=10.0, help="Pocket radius in Å (unused)")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    args = parser.parse_args()

    main(args.pdb, args.sdf, args.output, args.radius, args.device)