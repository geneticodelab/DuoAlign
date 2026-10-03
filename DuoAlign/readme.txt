DuoAlign package — DuoAlign 空间多组学模型（双视图前端 × SpatialGlue 跨模态注意力融合）。
模块：
  model.py               新模型 DuoAlign_Overall（共享空间GCN/独立特征GCN/SAF/加权融合/共享原型池/跨模态注意力/DEC）
  model_spatialglue.py   SpatialGlue 原版模型（基线对比）
  preprocess.py          预处理 + construct_duoalign_graphs（共享空间图 + 独立特征图）
  utils.py               Sinkhorn/PSA/PCL/正交/空间排序/CellNiche/10指标评估
  train_duoalign.py      训练器（仅阶段一；finetune 存档）
  train_spatialglue.py   SpatialGlue 原版训练器（基线）
使用：项目根目录运行 `python run_DuoAlign.py`。详见 ../../README.md。
