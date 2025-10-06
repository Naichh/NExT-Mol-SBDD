import time
from rdkit import Chem
from rdkit.Chem import AllChem

def generate_3d_mol(smiles: str, quality: str = 'fast') -> Chem.Mol:
    """
    从您的代码中复制过来的、完全相同的3D构象生成函数。
    """
    try:
        # 1. 从SMILES创建分子对象并加氢
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol, addCoords=True)

        # 2. 检查分子是否与MMFF94力场兼容
        AllChem.MMFFGetMoleculeProperties(mol)

        # 3. 根据质量要求设置参数
        if quality == 'high':
            num_confs = 50
        else:  # 'fast' 模式
            num_confs = 5

        params = AllChem.ETKDGv3()
        params.randomSeed = 0xf00d
        params.numThreads = 1  # 必须设置为1以保证线程安全

        # 4. 生成多个初始3D构象
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
        if len(cids) == 0:
            return None

        # 5. 使用MMFF94力场优化所有构象
        res = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=1)

        converged_res = [(i, e) for i, (flag, e) in enumerate(res) if flag == 0]
        if not converged_res:
            return None

        # 找到能量最低的构象
        min_energy_idx, _ = min(converged_res, key=lambda x: x[1])

        # 6. 创建一个只包含最优构象的新分子对象并返回
        best_mol = Chem.Mol(mol)
        best_mol.RemoveAllConformers()
        best_conformer = mol.GetConformer(int(min_energy_idx))
        best_mol.AddConformer(best_conformer, assignId=True)

        return best_mol

    except Exception:
        return None

# --- 主要测试逻辑 ---
if __name__ == "__main__":
    # 从您的输出中挑选了一些看起来最复杂的SMILES进行测试
    # 您可以随意添加或修改这个列表
    smiles_to_test = [
        # 一个相对正常的分子作为基准
        "Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O",
        # 一个长链、高柔性分子
        "Nc1ncnc2c1ncn2C(O)C(O)C(O)C(O)CCOP(=O)(O)OP(=O)(O)OC1C(COP(=O)(O)OP(=O)(O)O)OC(n2cnc3c(N)ncnc32)C1O",
        # 另一个长链、高柔性分子
        "CCCNc1nc2c(C(=O)N=CN=CNC3OC(CO)C(O)C3O)ncnc2n1C1OC(CO)C(O)C1O",
        # 一个非常巨大的分子
        "CN(C)c1nc2c(N)ncnc2n1C1C(C=C(C#N)C=c2c(N)nc3nc2=NC2=C4C5=C6C5N=CN=C(N(O)C2)C2(O)C(O)C(n5cnc7c(N)ncnc75)OC5COP(O)(O)(OP(=O)(O)O)OC6C4N3C52)C(O)C(O)C1COP(=O)(O)OP(=O)(O)CP(=O)(O)O",
        # 另一个非常巨大的分子
        "CC1(C)C(O[PH](=O)OCC2OC(n3cnc4c(N)nc(N)nc43)C(O)C2O)CN(C2N=CN=C3C2=NC(=O)N3C2C3=NC=NC45OP(O)(O)(OP(=O)(O)O)OC4C4OC(n6cnc7c(N)ncnc76)C(O)C(O)C42NCC5C3)CC(O)C(O)C1OP(=O)(O)OP(=O)(O)O",
        # 一个复杂的稠环分子
        "Nc1ncnc2c1ncn2C1OC2C(NCCCC34OC(CO)C(O)C(O)C3OP(=O)(O)OP(=O)(O)OC24)C(O)C1O"
    ]

    results = []
    total_start_time = time.time()

    print("="*80)
    print("--- Starting 3D Conformer Generation Benchmark ---")
    print(f"--- Testing {len(smiles_to_test)} molecules... ---")
    print("="*80 + "\n")

    for i, smiles in enumerate(smiles_to_test):
        print(f"--- Processing Molecule {i+1}/{len(smiles_to_test)} ---")
        print(f"SMILES: {smiles[:80]}...") # 打印SMILES的前80个字符

        start_time = time.time()
        mol_3d = generate_3d_mol(smiles, quality='fast')
        end_time = time.time()

        duration = end_time - start_time
        success = mol_3d is not None

        if success:
            print(f"  -> SUCCESS | Time taken: {duration:.4f} seconds")
        else:
            print(f"  -> FAILED  | Time taken: {duration:.4f} seconds")
        print("-" * 50)

        results.append({
            "index": i + 1,
            "smiles": smiles,
            "duration": duration,
            "success": success
        })

    total_duration = time.time() - total_start_time

    # --- 总结报告 ---
    print("\n" + "="*80)
    print("--- Benchmark Summary Report ---")
    print("="*80)

    # 按耗时降序排序
    slowest_molecules = sorted(results, key=lambda x: x['duration'], reverse=True)

    print("\n--- Top 5 Slowest Molecules ---")
    for result in slowest_molecules[:5]:
        print(f"  - Rank #{results.index(result)+1}: Took {result['duration']:.4f}s | SMILES: {result['smiles'][:60]}...")

    num_successful = sum(1 for r in results if r['success'])
    num_failed = len(results) - num_successful
    avg_time = total_duration / len(results) if results else 0

    print("\n--- Overall Statistics ---")
    print(f"  - Total molecules processed: {len(results)}")
    print(f"  - Successful generations: {num_successful}")
    print(f"  - Failed generations: {num_failed}")
    print(f"  - Total time taken: {total_duration:.4f} seconds")
    print(f"  - Average time per molecule: {avg_time:.4f} seconds")
    print("="*80)
