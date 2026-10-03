#!/usr/bin/env python
import os
import sys
import warnings

os.environ['PYTHONHASHSEED'] = '2026'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
WORKSPACE_ROOT = os.path.dirname(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from DuoAlign.preprocess import fix_seed, construct_duoalign_graphs
from DuoAlign.train_duoalign import Train_DuoAlign
from DuoAlign.utils import evaluate_clustering


class Args:
    data_root = os.environ.get('DUOALIGN_DATA_ROOT', os.path.join(WORKSPACE_ROOT, 'data'))
    dataset_dir = os.path.join(data_root, 'Human_placenta')
    label_key = 'cell_type'
    random_state = 2022

    use_dense_adj = True

    dim_out = 64
    n_prototypes = 32
    tau_pcl = 0.1
    sinkhorn_eps = 0.05
    sinkhorn_iters = 50
    use_saf = True
    spatial_k = 10
    feature_k = 20

    lr = 1e-3
    weight_decay = 5e-4
    pretrain_epochs = 350
    phi = 0.5
    lambda_ortho = 0.01

    n_top_genes = 3000
    pca_comps_rna = 200
    pca_comps_pro = 50

    omics2_is_counts = False


args = Args()
fix_seed(args.random_state)


def discover_slices(dataset_dir):
    slices = {}
    if os.path.isdir(dataset_dir):
        for fn in sorted(os.listdir(dataset_dir)):
            if fn.endswith('_rna.h5ad'):
                stem = fn[:-len('_rna.h5ad')]
                atac_fn = stem + '_ATAC.h5ad'
                rna_path = os.path.join(dataset_dir, fn)
                atac_path = os.path.join(dataset_dir, atac_fn)
                if os.path.isfile(atac_path):
                    slices[stem] = (rna_path, atac_path)
    return slices


def _coords_to_float(array):
    coords = np.asarray(array)
    if not np.issubdtype(coords.dtype, np.floating):
        coords = coords.astype(np.float64)
    return coords


def load_and_preprocess(rna_path, atac_path):
    warnings.filterwarnings('ignore', category=UserWarning,
                            message=".*Variable names are not unique.*")

    adata_rna = sc.read_h5ad(rna_path)
    adata_pro = sc.read_h5ad(atac_path)
    adata_rna.var_names_make_unique()
    adata_pro.var_names_make_unique()

    if args.label_key not in adata_rna.obs.columns:
        raise KeyError(f"标签列 '{args.label_key}' 不在 obs 中, 可用列: {list(adata_rna.obs.columns)}")
    adata_rna.obs['manual-anno'] = adata_rna.obs[args.label_key].astype(str)
    adata_pro = adata_pro[adata_rna.obs_names].copy()

    le = LabelEncoder()
    adata_rna.obs['Group'] = le.fit_transform(adata_rna.obs['manual-anno'].astype(str))
    n_clusters = len(le.classes_)
    print(f"  ✅ Cells: {adata_rna.n_obs} | Clusters: {n_clusters} | "
          f"Labels: {list(le.classes_)}")

    spatial_coords = _coords_to_float(adata_rna.obsm['spatial'])
    adata_rna.obsm['spatial'] = spatial_coords
    coords_norm = ((spatial_coords - spatial_coords.min(axis=0)) /
                   (spatial_coords.max(axis=0) - spatial_coords.min(axis=0)))

    sc.pp.highly_variable_genes(adata_rna, flavor="seurat_v3",
                                n_top_genes=args.n_top_genes)
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    sc.pp.scale(adata_rna, zero_center=False, max_value=10)
    sc.pp.pca(adata_rna, n_comps=args.pca_comps_rna, use_highly_variable=True,
              svd_solver='arpack', random_state=args.random_state)
    adata_rna.obsm['feat'] = adata_rna.obsm['X_pca'].copy()

    if args.omics2_is_counts:
        sc.pp.normalize_total(adata_pro, target_sum=1e4)
        sc.pp.log1p(adata_pro)
        sc.pp.scale(adata_pro, zero_center=False, max_value=10)
    else:
        sc.pp.scale(adata_pro, zero_center=True)

    n_comps_pro = min(args.pca_comps_pro, adata_pro.n_vars - 1)
    if n_comps_pro > 0:
        sc.pp.pca(adata_pro, n_comps=n_comps_pro, svd_solver='arpack',
                  random_state=args.random_state)
        adata_pro.obsm['feat'] = adata_pro.obsm['X_pca'].copy()
    else:
        adata_pro.obsm['feat'] = adata_pro.X.copy()

    return adata_rna, adata_pro, coords_norm, n_clusters


def run_slice(slice_name, rna_path, atac_path):
    print("\n" + "=" * 62)
    print(f"🧬 Section: {slice_name}")
    print(f"  RNA:  {rna_path}")
    print(f"  ATAC: {atac_path}")
    print("=" * 62)

    adata_rna, adata_pro, coords_norm, n_clusters = load_and_preprocess(rna_path, atac_path)

    print("\nBuilding DuoAlign graphs (shared spatial + per-omics feature)...")
    graphs = construct_duoalign_graphs(coords_norm,
                                   adata_rna.obsm['feat'], adata_pro.obsm['feat'],
                                   spatial_k=args.spatial_k, feature_k=args.feature_k)

    data_dict = {'adata_omics1': adata_rna, 'adata_omics2': adata_pro, 'graphs': graphs}

    print("Initializing DuoAlign Trainer...")
    trainer = Train_DuoAlign(data_dict, args)

    print(f"\n=== Stage 1: Single-omics Pretraining ({args.pretrain_epochs} epochs) ===")
    init_z = trainer.pretrain()

    from sklearn.cluster import KMeans
    print("\n=== Stage-1 Protocol: KMeans on stage-1 z_cross ===")
    kmeans = KMeans(n_clusters=n_clusters, n_init=20,
                    random_state=args.random_state)
    final_labels = kmeans.fit_predict(init_z.cpu().numpy())

    y_true = adata_rna.obs['Group'].values
    metrics = evaluate_clustering(y_true, final_labels)
    print("\n" + "=" * 62)
    print(f"🏆 DuoAlign {slice_name} Final Performance (10 metrics) 🏆")
    print("=" * 62)
    for name, val in metrics.items():
        print(f"{name:<12} | {val:>10.4f}")
    print("-" * 30)
    docx_order = ['MI', 'NMI', 'AMI', 'FMI', 'ARI', 'V-Measure', 'F1', 'Jaccard', 'Compl']
    vals9 = [metrics[k] * 100 for k in docx_order]
    print("docx参考格式 (" + ",".join(docx_order) + ", %): "
          + " ".join(f"{v:.2f}" for v in vals9))
    print("=" * 62 + "\n")

    with torch.no_grad():
        trainer.model.eval()
        res_final = trainer._forward()
    adata_rna.obsm['MHALGlue_emb'] = res_final['z_cross'].cpu().numpy()
    adata_rna.obs['MHALGlue_cluster'] = pd.Categorical(final_labels)
    if 'alpha_cross' in res_final and res_final['alpha_cross'] is not None:
        adata_rna.obsm['MHALGlue_alpha_cross'] = res_final['alpha_cross'].cpu().numpy()
    if res_final.get('eta1') is not None:
        adata_rna.obs['MHALGlue_eta1'] = float(np.asarray(res_final['eta1'].cpu()))
        adata_rna.obs['MHALGlue_eta2'] = float(np.asarray(res_final['eta2'].cpu()))

    result_dir = os.path.join(PROJECT_ROOT, 'results', 'Human_placenta', slice_name)
    os.makedirs(result_dir, exist_ok=True)

    plt.rcParams['figure.figsize'] = (10, 10)
    ax = sc.pl.embedding(adata_rna, basis='spatial', color='MHALGlue_cluster',
                         title=f'DuoAlign {slice_name} (ARI: {metrics["ARI"]:.4f})',
                         size=250, show=False)
    if isinstance(ax, list):
        ax[0].set_aspect('auto')
    else:
        ax.set_aspect('auto')
    plt.savefig(os.path.join(result_dir, f'{slice_name}_duoalign.png'),
                dpi=300, bbox_inches='tight')
    plt.close('all')

    adata_rna.write_h5ad(os.path.join(result_dir, f'{slice_name}_rna.h5ad'))
    print(f"✅ Results saved to '{result_dir}/'\n")

    return slice_name, metrics


if __name__ == '__main__':
    slices = discover_slices(args.dataset_dir)
    if not slices:
        raise FileNotFoundError(
            f"未在 {args.dataset_dir} 下发现任何 '*_rna.h5ad + *_ATAC.h5ad' 配对。"
            f"请检查 DUOALIGN_DATA_ROOT / dataset_dir 配置。")

    print(f"📁 Dataset: {args.dataset_dir}")
    print(f"🔬 Sections to run: {list(slices.keys())}")

    summary = {}
    for name, (rna_p, atac_p) in slices.items():
        try:
            name, metrics = run_slice(name, rna_p, atac_p)
            summary[name] = metrics
        except Exception as e:
            print(f"❌ Section {name} 失败: {e}")
            import traceback
            traceback.print_exc()

    if summary:
        print("\n" + "=" * 62)
        print("📊 Human_placenta 汇总 (DuoAlign, 10 metrics)")
        print("=" * 62)
        rows = []
        for name, m in summary.items():
            row = {'Section': name}
            row.update({k: round(v, 4) for k, v in m.items()})
            rows.append(row)
        df = pd.DataFrame(rows).set_index('Section')
        print(df.to_string())
        out_csv = os.path.join(PROJECT_ROOT, 'results', 'Human_placenta', 'summary.csv')
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df.to_csv(out_csv)
        print(f"\n✅ Summary saved to '{out_csv}'")
        print("=" * 62)
