import os
import copy
import pickle
import numpy as np
import torch
import Levenshtein
from tqdm import tqdm
from itertools import combinations
import multiprocessing as mp
from functools import partial
import tempfile
import contextlib
import subprocess
from collections import defaultdict
import sys
# RDKit & Cheminformatics Imports
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, RDKFingerprint, Descriptors, QED, Crippen
import AutoDockTools

# Third-party scientific libraries
from vina import Vina
from openbabel import pybel
from meeko import MoleculePreparation
from fcd_torch import FCD as FCDMetric
from moses.metrics.metrics import SNNMetric, FragMetric, ScafMetric, internal_diversity, fraction_passes_filters
import time
# Local/Project-specific Imports
from .jodo.rdkit_metric import eval_rdmol
from .jodo.stability import stability_bonds, allowed_fc_bonds
from .jodo.mose_metric import compute_intermediate_statistics, mapper, get_smiles, reconstruct_mol, MeanProperty
from .utils.evaluation import sascorer

# ==============================================================================
# 1. 核心工作函数 (并行化单元)
# ==============================================================================

def _vina_docking_worker(mol, receptor_pdbqt_path, exhaustiveness):

    try:
        t_start = time.time()
        with tempfile.TemporaryDirectory() as tmpdir:
            ligand_sdf_path = os.path.join(tmpdir, 'ligand.sdf')
            ligand_pdbqt_path = os.path.join(tmpdir, 'ligand.pdbqt')
            mol_with_hs = Chem.AddHs(mol, addCoords=True)
            with Chem.SDWriter(ligand_sdf_path) as writer:
                writer.write(mol_with_hs)
            ob_mol = next(pybel.readfile('sdf', ligand_sdf_path))
            with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
                preparator = MoleculePreparation()
                preparator.prepare(ob_mol.OBMol)
                preparator.write_pdbqt_file(ligand_pdbqt_path)
            t_prep_done = time.time()
            pos = mol_with_hs.GetConformer(0).GetPositions()
            center = ((pos.max(0) + pos.min(0)) / 2).tolist()
            box_size = ((pos.max(0) - pos.min(0)) + 5.0).tolist()
            v = Vina(sf_name='vina', verbosity=0, cpu=1)
            v.set_receptor(receptor_pdbqt_path)
            v.set_ligand_from_file(ligand_pdbqt_path)
            t_setup_done = time.time()
            v.compute_vina_maps(center=center, box_size=box_size)
            t_maps_done = time.time()

            def get_score_safely(result):
                score = float('inf')
                if result is None: return score
                value = None
                if isinstance(result, np.ndarray) and result.size > 0:
                    value = result.flatten()[0]
                elif isinstance(result, list) and result:
                    nested_val = result[0]
                    if isinstance(nested_val, np.ndarray) and nested_val.size > 0:
                        value = nested_val.flatten()[0]
                    elif isinstance(nested_val, (float, int)):
                        value = nested_val
                elif isinstance(result, (float, int)):
                    value = result

                if value is not None:
                    score = float(value)
                return score

            scores_only = v.score()
            score_res = get_score_safely(scores_only)
            energies_min = v.optimize()
            min_res = get_score_safely(energies_min)
            v.dock(exhaustiveness=exhaustiveness, n_poses=1)
            energies_dock = v.energies(n_poses=1)
            dock_res = get_score_safely(energies_dock)
            t_dock_done = time.time()

        # 仅在成功时打印性能日志
        print(
            f"\n--- [PROFILING] Mol SMILES: {Chem.MolToSmiles(mol)}\n"
            f"    - Ligand Prep Time : {t_prep_done - t_start:.2f} s\n"
            f"    - Vina Setup Time  : {t_setup_done - t_prep_done:.2f} s\n"
            f"    - Map Compute Time : {t_maps_done - t_setup_done:.2f} s\n"
            f"    - Docking Time     : {t_dock_done - t_maps_done:.2f} s\n"
            f"    - Total Worker Time: {t_dock_done - t_start:.2f} s\n"
            f"    - Box Size (X,Y,Z) : {box_size[0]:.1f}, {box_size[1]:.1f}, {box_size[2]:.1f}\n",
            file=sys.stderr
        )
        return {'score': score_res, 'min': min_res, 'dock': dock_res}
    except Exception as e:
        print(f"\n---!!! VINA DOCKING WORKER FAILED !!!---", file=sys.stderr)
        print(f"Receptor Path: {receptor_pdbqt_path}", file=sys.stderr)
        try:
            smi = Chem.MolToSmiles(mol)
            print(f"Problematic SMILES: {smi}", file=sys.stderr)
        except:
            print("Could not generate SMILES for the problematic molecule.", file=sys.stderr)
        print(f"Error Type: {type(e).__name__}", file=sys.stderr)
        print(f"Error Details: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print(f"---!!! END VINA WORKER ERROR !!!---\n", file=sys.stderr)
        return None



# 放在 run_full_evaluation 外面
def ref_worker_wrapper(task):
    pocket_path, mol, rec_path, exh = task
    return pocket_path, _vina_docking_worker(mol, rec_path, exh)

def run_full_evaluation(grouped_mols):
    # --- 参数设定 ---
    protein_root = '/data/share/liuzhiyuan/nai/NExT-Mol/datasets/sbdd/crossdocked_pocket/'
    exhaustiveness = 4  # <<< 修改点1：降低耗尽度以加速

    slurm_cpus_str = os.getenv('SLURM_CPUS_PER_TASK')
    if slurm_cpus_str:
        cpu_cores = int(slurm_cpus_str)
        print(f"INFO: Detected Slurm allocation. Using {cpu_cores} allocated CPU cores.")
    else:
        # Fallback to mp.cpu_count() if not in a Slurm environment
        cpu_cores = mp.cpu_count()
        print(f"INFO: Not in a Slurm environment. Using total system CPU cores: {cpu_cores}.")

    # Calculate the number of workers, leaving one core for the main process and OS
    num_workers = max(1, cpu_cores - 1)
    #num_workers=29
    print(f"INFO: Setting num_workers for docking to {num_workers}.")



    # --- 初始化结果列表 ---
    all_qed_scores, all_sa_scores, per_pocket_diversities, per_pocket_high_affinity = [], [], [], []

    # --- 阶段一：任务收集 (串行但快速) ---
    ref_docking_tasks = []
    gen_docking_tasks = []
    task_to_pocket_map = []
    ref_qed_scores = []
    ref_sa_scores = []
    print("\n--- [Stage 1/4] Collecting tasks and calculating 2D metrics... ---")
    for pocket_path, mols_3d in tqdm(grouped_mols.items(), desc="Collecting Tasks"):
        receptor_path_full = os.path.join(protein_root, pocket_path)
        receptor_pdbqt_path = receptor_path_full.replace('.pdb', '.pdbqt')
        if not os.path.exists(receptor_pdbqt_path):
            continue

        try:
            # --- 处理参考配体 ---
            ref_sdf_path = os.path.join(protein_root, pocket_path.replace('_pocket10.pdb', '.sdf'))
            ref_mol = None
            if os.path.exists(ref_sdf_path):
                ref_supplier = Chem.SDMolSupplier(ref_sdf_path, removeHs=False, sanitize=True)
                for m in ref_supplier:
                    if m is not None:
                        ref_mol = m
                        break
                if ref_mol:
                    # 添加到参考对接任务列表
                    ref_docking_tasks.append((pocket_path, ref_mol, receptor_pdbqt_path, exhaustiveness))
                    # 计算2D指标
                    ref_qed_scores.append(QED.qed(ref_mol))
                    ref_sa_scores.append(sascorer.compute_sa_score(ref_mol))

            # --- 处理生成分子 ---
            valid_mols_3d = [mol for mol in mols_3d if mol is not None]
            all_qed_scores.extend([QED.qed(mol) for mol in valid_mols_3d])
            all_sa_scores.extend([sascorer.compute_sa_score(mol) for mol in valid_mols_3d])

            if len(valid_mols_3d) >= 2:
                fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in valid_mols_3d]
                sims = [DataStructs.TanimotoSimilarity(fps[i], fps[j])
                        for i, j in combinations(range(len(fps)), 2)]
                per_pocket_diversities.append(1 - (np.mean(sims) if sims else 1.0))

            for mol in valid_mols_3d:
                gen_docking_tasks.append((mol, receptor_pdbqt_path, exhaustiveness))
                task_to_pocket_map.append(pocket_path)

        except Exception as e:
            print(f"\n[WARNING] Failed during task collection for pocket {pocket_path}, skipping. Reason: {e}")
            continue

    # --- 阶段二：并行对接参考配体 ---
    ref_vina_scores, ref_vina_mins, ref_vina_docks = [], [], []
    ref_dock_scores_map = {}
    if ref_docking_tasks:
        print(f"\n--- [Stage 2/4] Starting parallel docking for {len(ref_docking_tasks)} reference molecules... ---")


        with mp.Pool(processes=num_workers) as pool:
            results_iterator = pool.imap_unordered(ref_worker_wrapper, ref_docking_tasks)
            for pocket_path, result in tqdm(results_iterator, total=len(ref_docking_tasks), desc="Docking Reference Mols"):
                if result:
                    ref_vina_scores.append(result['score'])
                    ref_vina_mins.append(result['min'])
                    ref_vina_docks.append(result['dock'])
                    ref_dock_scores_map[pocket_path] = result['dock']

    # --- 阶段三：并行对接生成分子 ---

    all_vina_scores, all_vina_mins, all_vina_docks = [], [], []
    if gen_docking_tasks:
        print(f"\n--- [Stage 3/4] Starting robust parallel docking for {len(gen_docking_tasks)} generated molecules... ---")


        INDIVIDUAL_TASK_TIMEOUT = 300

        raw_docking_results = []


        with mp.Pool(processes=num_workers) as pool:
            # 1. 使用 apply_async 异步地、非阻塞地提交所有任务。
            # 这会立即返回一个 AsyncResult 对象列表，程序不会在这里等待。
            # 注意：gen_docking_tasks 已经是 (mol, path, exhaustiveness) 的元组列表，正好符合 apply_async 的 args 格式。
            results_async = [pool.apply_async(_vina_docking_worker, args=task) for task in gen_docking_tasks]

            # 2. 逐个从 AsyncResult 列表中获取结果，并为每个 get() 操作设置超时。
            pbar = tqdm(total=len(gen_docking_tasks), desc="Docking Generated Mols")
            for async_res in results_async:
                try:
                    # 关键！为获取每个任务的结果设置超时。
                    result = async_res.get(timeout=INDIVIDUAL_TASK_TIMEOUT)
                    raw_docking_results.append(result)
                except mp.TimeoutError:
                    # 如果一个任务超时，捕获异常，打印警告，然后将结果记为 None。
                    print(f"\n[WARNING] A docking task timed out after {INDIVIDUAL_TASK_TIMEOUT} seconds and was skipped.", file=sys.stderr)
                    raw_docking_results.append(None)
                except Exception as e:
                    # 捕获其他在获取结果时可能发生的罕见错误。
                    print(f"\n[WARNING] A docking task failed with an unexpected error during result retrieval: {e}. Skipped.", file=sys.stderr)
                    raw_docking_results.append(None)
                finally:
                    # 无论成功、失败还是超时，都更新进度条。
                    pbar.update(1)
            pbar.close()

        #
        # (一个包含字典或None的列表)
        results_by_pocket = defaultdict(list)
        for i, result in enumerate(raw_docking_results):
            if result:
                pocket_id = task_to_pocket_map[i]
                results_by_pocket[pocket_id].append(result)

        for pocket_id, pocket_results in results_by_pocket.items():
            pocket_vina_docks = [res['dock'] for res in pocket_results]
            all_vina_docks.extend(pocket_vina_docks)
            all_vina_scores.extend([res['score'] for res in pocket_results])
            all_vina_mins.extend([res['min'] for res in pocket_results])

            ref_score = ref_dock_scores_map.get(pocket_id)
            if ref_score is not None and pocket_vina_docks:
                hit_count = sum(1 for score in pocket_vina_docks if score <= ref_score)
                per_pocket_high_affinity.append((hit_count / len(pocket_vina_docks)) * 100)



    # --- 阶段四：聚合与报告 ---
    print("\n--- [Stage 4/4] Aggregating final results... ---")

    def print_diagnostic_summary(score_list, score_name, total_tasks):
        successful_count = len(score_list)
        failed_count = total_tasks - successful_count

        print("\n" + "="*60)
        print(f"--- [ Diagnostic Summary for: {score_name} ] ---")
        print(f"  - Total Tasks Attempted: {total_tasks}")
        print(f"  - Successful Results   : {successful_count}")
        print(f"  - Failed/Timed-Out     : {failed_count} ({failed_count/total_tasks:.2%})")

        if successful_count > 0:
            scores_np = np.array(score_list)

            # 计算正、负、零分数的统计
            negative_count = np.sum(scores_np < 0)
            positive_count = np.sum(scores_np > 0)
            zero_count = np.sum(scores_np == 0)

            print(f"  - Favorable Scores (< 0): {negative_count} ({negative_count/successful_count:.2%})")
            print(f"  - Unfavorable Scores (> 0): {positive_count} ({positive_count/successful_count:.2%})")
            print(f"  - Zero Scores (== 0)    : {zero_count} ({zero_count/successful_count:.2%})")

            # 打印最好和最差的分数
            print(f"  - Best Score (Min)       : {np.min(scores_np):.4f}")
            print(f"  - Worst Score (Max)      : {np.max(scores_np):.4f}")
        print("="*60)

    # --- 对 Vina Dock, Vina Min, Vina Score 分别调用诊断函数 ---
    if gen_docking_tasks:
        total_tasks = len(gen_docking_tasks)
        print_diagnostic_summary(all_vina_docks, "Vina Dock Scores", total_tasks)
        print_diagnostic_summary(all_vina_mins, "Vina Min Scores (Local Optimization)", total_tasks)
        print_diagnostic_summary(all_vina_scores, "Vina Score Scores (Pre-Optimization)", total_tasks)


    results = {
        'Vina Score Avg': np.mean(all_vina_scores) if all_vina_scores else 0,
        'Vina Score Med': np.median(all_vina_scores) if all_vina_scores else 0,
        'Vina Min Avg': np.mean(all_vina_mins) if all_vina_mins else 0,
        'Vina Min Med': np.median(all_vina_mins) if all_vina_mins else 0,
        'Vina Dock Avg': np.mean(all_vina_docks) if all_vina_docks else 0,
        'Vina Dock Med': np.median(all_vina_docks) if all_vina_docks else 0,
        'High Affinity Avg (%)': np.mean(per_pocket_high_affinity) if per_pocket_high_affinity else 0,
        'High Affinity Med (%)': np.median(per_pocket_high_affinity) if per_pocket_high_affinity else 0,
        'QED Avg': np.mean(all_qed_scores) if all_qed_scores else 0,
        'QED Med': np.median(all_qed_scores) if all_qed_scores else 0,
        'SA Avg': np.mean(all_sa_scores) if all_sa_scores else 0,
        'SA Med': np.median(all_sa_scores) if all_sa_scores else 0,
        'Diversity Avg': np.mean(per_pocket_diversities) if per_pocket_diversities else 0,
        'Diversity Med': np.median(per_pocket_diversities) if per_pocket_diversities else 0,
        'Reference Vina Score Avg': np.mean(ref_vina_scores) if ref_vina_scores else 0,
        'Reference Vina Score Med': np.median(ref_vina_scores) if ref_vina_scores else 0,
        'Reference Vina Min Avg': np.mean(ref_vina_mins) if ref_vina_mins else 0,
        'Reference Vina Min Med': np.median(ref_vina_mins) if ref_vina_mins else 0,
        'Reference Vina Dock Avg': np.mean(ref_vina_docks) if ref_vina_docks else 0,
        'Reference Vina Dock Med': np.median(ref_vina_docks) if ref_vina_docks else 0,
        'Reference QED Avg': np.mean(ref_qed_scores) if ref_qed_scores else 0,
        'Reference QED Med': np.median(ref_qed_scores) if ref_qed_scores else 0,
        'Reference SA Avg': np.mean(ref_sa_scores) if ref_sa_scores else 0,
        'Reference SA Med': np.median(ref_sa_scores) if ref_sa_scores else 0,

    }

    print("\n--- Comprehensive Evaluation Report ---")
    for key, value in results.items():
        print(f"  - {key:<28}: {value:.4f}")

    return results


def SBDD_validation(generated_mols, ground_truth_mols):
    def get_exact_match(mol_gen, mol_gt):
        if not mol_gen or not mol_gt: return 0
        smiles_gen_canon = Chem.MolToSmiles(mol_gen, canonical=True)
        smiles_gt_canon = Chem.MolToSmiles(mol_gt, canonical=True)
        return 1 if smiles_gen_canon == smiles_gt_canon else 0

    def get_levenshtein_distance(mol_gen, mol_gt):
        if not mol_gen or not mol_gt: return 1000
        smiles_gen = Chem.MolToSmiles(mol_gen, canonical=True)
        smiles_gt = Chem.MolToSmiles(mol_gt, canonical=True)
        return Levenshtein.distance(smiles_gen, smiles_gt)

    def get_maccs_fts(mol_gen, mol_gt):
        if not mol_gen or not mol_gt: return 0.0
        fp_gen = MACCSkeys.GenMACCSKeys(mol_gen)
        fp_gt = MACCSkeys.GenMACCSKeys(mol_gt)
        return DataStructs.TanimotoSimilarity(fp_gen, fp_gt)

    def get_rdk_fts(mol_gen, mol_gt):
        if not mol_gen or not mol_gt: return 0.0
        fp_gen = RDKFingerprint(mol_gen)
        fp_gt = RDKFingerprint(mol_gt)
        return DataStructs.TanimotoSimilarity(fp_gen, fp_gt)

    def get_morgan_fts(mol_gen, mol_gt, radius=2, nBits=2048):
        if not mol_gen or not mol_gt: return 0.0
        fp_gen = AllChem.GetMorganFingerprintAsBitVect(mol_gen, radius, nBits=nBits)
        fp_gt = AllChem.GetMorganFingerprintAsBitVect(mol_gt, radius, nBits=nBits)
        return DataStructs.TanimotoSimilarity(fp_gen, fp_gt)

    scores = {
        'exact_match': [],
        'levenshtein': [],
        'maccs_fts': [],
        'rdk_fts': [],
        'morgan_fts': []
    }

    num_pairs = min(len(generated_mols), len(ground_truth_mols))
    for i in range(num_pairs):
        mol_gen = generated_mols[i]
        mol_gt = ground_truth_mols[i]

        scores['exact_match'].append(get_exact_match(mol_gen, mol_gt))
        scores['levenshtein'].append(get_levenshtein_distance(mol_gen, mol_gt))
        scores['maccs_fts'].append(get_maccs_fts(mol_gen, mol_gt))
        scores['rdk_fts'].append(get_rdk_fts(mol_gen, mol_gt))
        scores['morgan_fts'].append(get_morgan_fts(mol_gen, mol_gt))

    if num_pairs == 0:
        return {key: 0.0 for key in scores}

    avg_scores = {
        'SBDD/ExactMatch': np.mean(scores['exact_match']),
        'SBDD/Levenshtein': np.mean(scores['levenshtein']),
        'SBDD/MACCS_FTS': np.mean(scores['maccs_fts']),
        'SBDD/RDK_FTS': np.mean(scores['rdk_fts']),
        'SBDD/Morgan_FTS': np.mean(scores['morgan_fts']),
    }

    return avg_scores

def check_2D_stability(rdmol):
    """Convert the generated tensors to rdkit mols and check stability"""
    rdmol = Chem.AddHs(rdmol)
    atom_num = rdmol.GetNumAtoms()
    # kekulize mol and iterate bonds for stability
    new_mol = copy.deepcopy(rdmol)
    try:
        Chem.Kekulize(new_mol)
    except:
        print('Can\'t Kekulize mol.')
        pass

    nr_bonds = np.zeros(atom_num, dtype='int')
    for bond in new_mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        # if bond_type in stability_bonds:
        #     order = stability_bonds[bond_type]
        # else:
        #     order = 0
        order = stability_bonds[bond_type]
        nr_bonds[start] += order
        nr_bonds[end] += order

    # stability
    nr_stable_bonds = 0
    atom_types_str = [atom.GetSymbol() for atom in rdmol.GetAtoms()]
    formal_charges = [atom.GetFormalCharge() for atom in rdmol.GetAtoms()]
    for atom_type_i, nr_bonds_i, fc_i in zip(atom_types_str, nr_bonds, formal_charges):
        # fc_i = fc_i.item()
        possible_bonds = allowed_fc_bonds[atom_type_i]
        if type(possible_bonds) == int:
            is_stable = possible_bonds == nr_bonds_i
        elif type(possible_bonds) == dict:
            expected_bonds = possible_bonds[fc_i] if fc_i in possible_bonds.keys() else possible_bonds[0]
            is_stable = expected_bonds == nr_bonds_i if type(expected_bonds) == int else nr_bonds_i in expected_bonds
        else:
            is_stable = nr_bonds_i in possible_bonds
        nr_stable_bonds += int(is_stable)

    molecule_stable = nr_stable_bonds == atom_num
    return molecule_stable, nr_stable_bonds, atom_num


def get_2D_edm_metric(predict_mols, train_mols=None):
    train_smiles = None
    if train_mols is not None:
        train_smiles = [Chem.MolToSmiles(mol) for mol in train_mols]
        train_smiles = [Chem.CanonSmiles(s) for s in train_smiles]

    molecule_stable = 0
    nr_stable_bonds = 0
    n_atoms = 0

    for mol in tqdm(predict_mols):
        try:
            validity_res = check_2D_stability(mol)
        except:
            print('Check stability failed.')
            validity_res = [0, 0, mol.GetNumAtoms()]
        molecule_stable += int(validity_res[0])
        nr_stable_bonds += int(validity_res[1])
        n_atoms += int(validity_res[2])

    # Stability
    fraction_mol_stable = molecule_stable / float(len(predict_mols))
    fraction_atm_stable = nr_stable_bonds / float(n_atoms)

    output_dict = {
        'mol_stable': fraction_mol_stable,
        'atom_stable': fraction_atm_stable,
    }

    # Basic rdkit metric result (Validity, Fragment, Unique)
    rdkit_dict = eval_rdmol(predict_mols, train_smiles)

    ## union the two dicts
    output_dict.update(rdkit_dict)
    return output_dict


def check_3D_stability(positions, atoms, dataset_name, debug=False, rdmol=None, use_mmff=False):
    """Look up for bond types and construct a Rdkit Mol"""
    assert len(positions.shape) == 2
    assert positions.shape[1] == 3
    if use_mmff:
        try:
            AllChem.MMFFOptimizeMolecule(rdmol, confId=0, maxIters=200)
            positions = rdmol.GetConformer(0).GetPositions()
        except:
            print('MMFF failed, use original coordinates.')

    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    nr_bonds = np.zeros(len(x), dtype='int')

    # atoms
    # convert to RDKit Mol, add atom first
    mol = Chem.RWMol()
    for atom in atoms:
        a = Chem.Atom(atom)
        mol.AddAtom(a)

    # add positions to Mol
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(positions[i][0].item(), positions[i][1].item(), positions[i][2].item()))
    mol.AddConformer(conf)

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom1, atom2 = atoms[i], atoms[j]
            pair = sorted([atoms[i], atoms[j]])
            if 'QM9' in dataset_name:
                order = get_bond_order(atom1, atom2, dist)

            elif 'Geom' in dataset_name:
                order = geom_predictor((pair[0], pair[1]), dist)
            else:
                raise ValueError('Fail to get dataset bond info.')
            nr_bonds[i] += order
            nr_bonds[j] += order
            # add bond to RDKIT Mol
            if order > 0:
                mol.AddBond(i, j, bond_list[order])

    nr_stable_bonds = 0
    for atom_type_i, nr_bonds_i in zip(atoms, nr_bonds):
        possible_bonds = allowed_bonds[atom_type_i]
        if type(possible_bonds) == int:
            is_stable = possible_bonds == nr_bonds_i
        else:
            is_stable = nr_bonds_i in possible_bonds
        if not is_stable and debug:
            print("Invalid bonds for molecule %s with %d bonds" % (atom_type_i, nr_bonds_i))
        nr_stable_bonds += int(is_stable)

    molecule_stable = nr_stable_bonds == len(x)
    return molecule_stable, nr_stable_bonds, len(x), mol


def get_3D_edm_metric(predict_mols, train_mols=None, dataset_name='QM9', use_mmff=False):
    train_smiles = None
    if train_mols is not None:
        train_smiles = [Chem.MolToSmiles(mol) for mol in train_mols]

    molecule_stable = 0
    nr_stable_bonds = 0
    n_atoms = 0

    rd_mols = []
    for mol in tqdm(predict_mols):
        pos = mol.GetConformer(0).GetPositions()
        pos = pos - pos.mean(axis=0)
        atom_type = [atom.GetSymbol() for atom in mol.GetAtoms()]
        try:
            validity_res = check_3D_stability(pos, atom_type, dataset_name, rdmol=mol, use_mmff=use_mmff, debug=False)
        except:
            print('Check stability failed.')
            validity_res = [0, 0, mol.GetNumAtoms(), mol]

        molecule_stable += int(validity_res[0])
        nr_stable_bonds += int(validity_res[1])
        n_atoms += int(validity_res[2])
        rd_mols.append(validity_res[3])

    # Stability
    fraction_mol_stable = molecule_stable / float(len(predict_mols))
    fraction_atm_stable = nr_stable_bonds / float(n_atoms)
    output_dict = {
        'mol_stable': fraction_mol_stable,
        'atom_stable': fraction_atm_stable,
    }

    # Basic rdkit metric result (Validity, Fragment, Unique)
    rdkit_dict = eval_rdmol(rd_mols, train_smiles)
    output_dict.update(rdkit_dict)
    return output_dict, rd_mols


def get_3D_edm_metric_batch(predict_mols, train_mols=None, dataset_name='QM9'):
    train_smiles = None
    if train_mols is not None:
        train_smiles = [Chem.MolToSmiles(mol) for mol in train_mols]

    molecule_stable = 0
    nr_stable_bonds = 0
    n_atoms = 0

    rd_mols = []
    predict_mols = [predict_mols[i:i+10] for i in range(0, len(predict_mols), 10)]
    for mol_list in tqdm(predict_mols):
        validity_res_list = []
        # sanity check
        smiles = [Chem.MolToSmiles(mol) for mol in mol_list]
        assert len(set(smiles)) == 1

        for mol in mol_list:
            pos = mol.GetConformer(0).GetPositions()
            pos = pos - pos.mean(axis=0)
            atom_type = [atom.GetSymbol() for atom in mol.GetAtoms()]
            validity_res = check_3D_stability(pos, atom_type, dataset_name, rdmol=mol)
            validity_res_list.append(validity_res)
        max_validity_res = max(validity_res_list, key=lambda x: x[0])
        molecule_stable += int(max_validity_res[0])
        nr_stable_bonds += int(max_validity_res[1])
        n_atoms += int(max_validity_res[2])
        rd_mols.append(max_validity_res[3])

    # Stability
    fraction_mol_stable = molecule_stable / float(len(predict_mols))
    fraction_atm_stable = nr_stable_bonds / float(n_atoms)
    output_dict = {
        'mol_stable': fraction_mol_stable,
        'atom_stable': fraction_atm_stable,
    }

    # Basic rdkit metric result (Validity, Fragment, Unique)
    rdkit_dict = eval_rdmol(rd_mols, train_smiles)
    output_dict.update(rdkit_dict)
    return output_dict
from multiprocessing import Pool

# def get_moses_metrics(test_mols, n_jobs=1, device='cpu', batch_size=2000, ptest_pool=None, cache_path=None):
#     n_jobs = 1
#     # compute intermediate statistics for test rdmols
#     if cache_path is not None and os.path.exists(cache_path):
#         with open(cache_path, 'rb') as f:
#             ptest = pickle.load(f)
#     else:
#         ptest = compute_intermediate_statistics(test_mols, n_jobs=n_jobs, device=device,
#                                                 batch_size=batch_size, pool=ptest_pool)
#         if cache_path is not None:
#             with open(cache_path, 'wb') as f:
#                 pickle.dump(ptest, f)

#     def moses_metrics(gen_mols, pool=None):
#         metrics = {}
#         if pool is None:
#             if n_jobs != 1:
#                 pool = Pool(n_jobs)
#                 close_pool = True
#             else:
#                 pool = 1
#                 close_pool = False
#         kwargs = {'n_jobs': pool, 'device': device, 'batch_size': batch_size}
#         kwargs_fcd = {'n_jobs': n_jobs, 'device': device, 'batch_size': batch_size}
#         gen_smiles = mapper(pool)(get_smiles, gen_mols)
#         gen_smiles = list(set(gen_smiles) - {None})
#         re_mols = mapper(pool)(reconstruct_mol, gen_smiles)
#         metrics['FCD'] = FCDMetric(**kwargs_fcd)(gen=gen_smiles, pref=ptest['FCD'])
#         metrics['SNN'] = SNNMetric(**kwargs)(gen=re_mols, pref=ptest['SNN'])
#         metrics['Frag'] = FragMetric(**kwargs)(gen=re_mols, pref=ptest['Frag'])
#         metrics['Scaf'] = ScafMetric(**kwargs)(gen=re_mols, pref=ptest['Scaf'])
#         metrics['IntDiv'] = internal_diversity(re_mols, pool, device=device)
#         metrics['Filters'] = fraction_passes_filters(re_mols, pool)

#         # drug properties
#         metrics['QED'] = MeanProperty(re_mols, QED.qed, n_jobs)
#         metrics['SA'] = MeanProperty(re_mols, sascorer.calculateScore, n_jobs)
#         metrics['logP'] = MeanProperty(re_mols, Crippen.MolLogP, n_jobs)
#         metrics['weight'] = MeanProperty(re_mols, Descriptors.ExactMolWt, n_jobs)

#         if close_pool:
#             pool.close()
#             pool.join()
#         return metrics

#     return moses_metrics

def get_moses_metrics(test_mols, n_jobs=1, device='cpu', batch_size=2000, ptest_pool=None, cache_path=None):
    n_jobs = 1
    if cache_path is not None and os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            ptest = pickle.load(f)
    else:
        ptest = compute_intermediate_statistics(test_mols, n_jobs=n_jobs, device=device, batch_size=batch_size)
        if cache_path is not None:
            with open(cache_path, 'wb') as f:
                pickle.dump(ptest, f)

    def moses_metrics_single_process(gen_mols):
        metrics = {}
        gen_smiles = [get_smiles(m) for m in gen_mols]
        gen_smiles = list(set(gen_smiles) - {None})
        re_mols = [reconstruct_mol(s) for s in gen_smiles]
        kwargs = {'n_jobs': n_jobs, 'device': device, 'batch_size': batch_size}
        metrics['FCD'] = FCDMetric(**kwargs)(gen=gen_smiles, pref=ptest['FCD'])
        metrics['SNN'] = SNNMetric(**kwargs)(gen=re_mols, pref=ptest['SNN'])
        metrics['Frag'] = FragMetric(**kwargs)(gen=re_mols, pref=ptest['Frag'])
        metrics['Scaf'] = ScafMetric(**kwargs)(gen=re_mols, pref=ptest['Scaf'])

        # 将 n_jobs=1 传递给其他工具函数
        metrics['IntDiv'] = internal_diversity(re_mols, n_jobs=n_jobs, device=device)
        metrics['Filters'] = fraction_passes_filters(re_mols, n_jobs=n_jobs)

        # drug properties
        metrics['QED'] = MeanProperty(re_mols, QED.qed, n_jobs=n_jobs)
        metrics['SA'] = MeanProperty(re_mols, sascorer.calculateScore, n_jobs=n_jobs)
        metrics['logP'] = MeanProperty(re_mols, Crippen.MolLogP, n_jobs=n_jobs)
        metrics['weight'] = MeanProperty(re_mols, Descriptors.ExactMolWt, n_jobs=n_jobs)

        return metrics

    return moses_metrics_single_process

def get_sub_geometry_metric(test_mols, dataset_info, root_path):
    tar_geo_stat = load_target_geometry(test_mols, dataset_info, root_path)

    def sub_geometry_metric(gen_mols):
        bond_length_dict = compute_geo_mmd(gen_mols, tar_geo_stat, cal_bond_distance,
                                           dataset_info['top_bond_sym'], mean_name='bond_length_mean')
        bond_angle_dict = compute_geo_mmd(gen_mols, tar_geo_stat, cal_bond_angle,
                                          dataset_info['top_angle_sym'], mean_name='bond_angle_mean')
        dihedral_angle_dict = compute_geo_mmd(gen_mols, tar_geo_stat, cal_dihedral_angle,
                                              dataset_info['top_dihedral_sym'], mean_name='dihedral_angle_mean')
        metric = {**bond_length_dict, **bond_angle_dict, **dihedral_angle_dict}

        return metric

    return sub_geometry_metric
import AutoDockTools
import subprocess
