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

import torch
import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from DuoAlign.preprocess import fix_seed, pca, construct_duoalign_graphs
from DuoAlign.train_duoalign import Train_DuoAlign
from DuoAlign.utils import evaluate_clustering


class Args:


    path_rna = '/mnt/data/jzhang/project/SpatioDAG/data/dataset/HLN3/adata_RNA.h5ad'
    path_pro = '/mnt/data/jzhang/project/SpatioDAG/data/dataset/HLN3/adata_ADT.h5ad'
    path_labels = '/mnt/data/jzhang/project/SpatioDAG/data/dataset/HLN3/HLN3_10.txt'


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
    pretrain_epochs = 351
    phi = 0.5
    lambda_ortho = 0.01


    n_top_genes = 3000
    pca_comps_rna = 200
    pca_comps_pro = 50

    save_prefix = 'lymph_node_duoalign'


args = Args()
fix_seed(args.random_state)


print("Loading data...")
warnings.filterwarnings('ignore', category=UserWarning, message=".*Variable names are not unique.*")

adata_rna = sc.read_h5ad(args.path_rna)
adata_pro = sc.read_h5ad(args.path_pro)
adata_rna.var_names_make_unique()
adata_pro.var_names_make_unique()

df_labels = pd.read_csv(args.path_labels, header=None)
adata_rna.obs['manual-anno'] = df_labels[0].values.astype(str)
sc.pp.filter_cells(adata_rna, min_counts=1)
adata_pro = adata_pro[adata_rna.obs_names].copy()

le = LabelEncoder()
adata_rna.obs['Group'] = le.fit_transform(adata_rna.obs['manual-anno'].astype(str))
n_clusters = len(le.classes_)
print(f"✅ Cells: {adata_rna.n_obs} | Clusters: {n_clusters}")

sc.pp.highly_variable_genes(adata_rna, flavor="seurat_v3", n_top_genes=args.n_top_genes)
sc.pp.normalize_total(adata_rna, target_sum=1e4)
sc.pp.log1p(adata_rna)
sc.pp.scale(adata_rna, zero_center=False, max_value=10)
sc.pp.pca(adata_rna, n_comps=args.pca_comps_rna, use_highly_variable=True,
              svd_solver='arpack', random_state=args.random_state)
adata_rna.obsm['feat'] = adata_rna.obsm['X_pca'].copy()

sc.pp.normalize_total(adata_pro, target_sum=1e4)
sc.pp.log1p(adata_pro)
sc.pp.scale(adata_pro, zero_center=False, max_value=10)
n_comps_pro = min(args.pca_comps_pro, adata_pro.n_vars - 1)
if n_comps_pro > 0:
    sc.pp.pca(adata_pro, n_comps=n_comps_pro, svd_solver='arpack', random_state=args.random_state)
    adata_pro.obsm['feat'] = adata_pro.obsm['X_pca'].copy()
else:
    adata_pro.obsm['feat'] = adata_pro.X.copy()

spatial_coords = adata_rna.obsm['spatial'].astype(float)
coords_norm = (spatial_coords - spatial_coords.min(axis=0)) / (spatial_coords.max(axis=0) - spatial_coords.min(axis=0))

print("Building DuoAlign graphs (shared spatial + per-omics feature)...")
graphs = construct_duoalign_graphs(coords_norm,
                               adata_rna.obsm['feat'], adata_pro.obsm['feat'],
                               spatial_k=args.spatial_k, feature_k=args.feature_k)

data_dict = {'adata_omics1': adata_rna, 'adata_omics2': adata_pro, 'graphs': graphs}

print("Initializing DuoAlign Trainer...")
trainer = Train_DuoAlign(data_dict, args)

print("\n=== Stage 1: Single-omics Pretraining (仅阶段一, 阶段二已移除) ===")
init_z = trainer.pretrain()

from sklearn.cluster import KMeans
print("\n=== Stage-1 Protocol: KMeans on stage-1 z_cross ===")
kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=args.random_state)
final_labels = kmeans.fit_predict(init_z.cpu().numpy())

y_true = adata_rna.obs['Group'].values
metrics = evaluate_clustering(y_true, final_labels)

print("\n" + "=" * 62)
print("🏆 DuoAlign Final Performance (10 metrics) 🏆")
print("=" * 62)
for name, val in metrics.items():
    print(f"{name:<12} | {val:>10.4f}")
print("-" * 30)
docx_order = ['MI', 'NMI', 'AMI', 'FMI', 'ARI', 'V-Measure', 'F1', 'Jaccard', 'Compl']
vals9 = [metrics[k] * 100 for k in docx_order]
print("docx参考格式 (" + ",".join(docx_order) + ", %): " + " ".join(f"{v:.2f}" for v in vals9))
print("=" * 62 + "\n")

with torch.no_grad():
    trainer.model.eval()
    res_final = trainer._forward()
adata_rna.obsm['MHALGlue_emb'] = res_final['z_cross'].cpu().numpy()
adata_rna.obs['MHALGlue_cluster'] = pd.Categorical(final_labels)

result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(result_dir, exist_ok=True)

plt.rcParams['figure.figsize'] = (10, 10)
ax = sc.pl.embedding(adata_rna, basis='spatial', color='MHALGlue_cluster',
                     title=f'DuoAlign (ARI: {metrics["ARI"]:.4f})', size=250, show=False)
if isinstance(ax, list):
    ax[0].set_aspect('auto')
else:
    ax.set_aspect('auto')
plt.savefig(os.path.join(result_dir, args.save_prefix + '.png'), dpi=300, bbox_inches='tight')
adata_rna.write_h5ad(os.path.join(result_dir, args.save_prefix + '_rna.h5ad'))
print(f"✅ Results saved to '{result_dir}/'")
