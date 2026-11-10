DuoAlign package — DuoAlign spatial multi-omics model (dual-view front end × SpatialGlue cross-modal attention fusion).
Modules:
  model.py               new model DuoAlign_Overall (shared spatial GCN / per-omics feature GCNs / SAF / weighted fusion / shared prototype pool / cross-modal attention / DEC)
  preprocess.py          preprocessing + construct_duoalign_graphs (shared spatial graph + per-omics feature graphs)
  utils.py               Sinkhorn / PSA / PCL / orthogonality / spatial ordering / CellNiche / 10-metric evaluation
  train_duoalign.py      trainer (stage one only; finetune archived)
Usage: run `python run_DuoAlign.py` from the project root. See ../README.md for details.
