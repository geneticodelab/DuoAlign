from setuptools import Command, find_packages, setup

__lib_name__ = "DuoAlign"
__lib_version__ = "0.1.0"
__description__ = "DuoAlign: Multi-View Hierarchical Alignment Learning x SpatialGlue for spatial multi-omics"
__url__ = ""
__author__ = "SpatioDAG-Glue team"
__author_email__ = ""
__license__ = "MIT"
__keywords__ = ["Spatial multi-omics", "Cross-omics integration", "Deep learning", "Graph neural networks",
                "Prototype contrastive learning", "Optimal transport"]
__requires__ = ["requests",]

with open("README.rst", "r", encoding="utf-8") as f:
    __long_description__ = f.read()

setup(
    name = __lib_name__,
    version = __lib_version__,
    description = __description__,
    url = __url__,
    author = __author__,
    author_email = __author_email__,
    license = __license__,
    packages = ["DuoAlign"],
    install_requires = __requires__,
    zip_safe = False,
    include_package_data = True,
    long_description = """DuoAlign: a duo-view hierarchical alignment model for spatial multi-omics domain
identification, built on a duo-view front-end with SpatialGlue-style cross-omics fusion.
- Per-omics front-ends: shared spatial GCN + independent feature-graph refinement GCN
  + intra-modality positive-sample alignment + prototype contrastive learning (Sinkhorn OT).
- Cross-omics: cross-modality attention + shared prototype pool with swapped prediction.
- Two-stage training: single-omics pretraining -> fusion fine-tuning.""",
    long_description_content_type="text/markdown"
)
