from typing import Collection

import anndata as ad
import scanpy as sc


def average_by(adata: ad.AnnData, by: str | Collection[str], **kwargs) -> ad.AnnData:
    adata = sc.get.aggregate(adata, by=by, func="mean", **kwargs)
    adata.X = adata.layers["mean"]
    del adata.layers["mean"]
    return adata
