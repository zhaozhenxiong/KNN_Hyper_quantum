"""CUDA inference for frozen, numeric, symmetric CatBoost trees."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from catboost import CatBoostRegressor


class FrozenCudaUnavailable(RuntimeError):
    """The model format or runtime cannot be evaluated by this CUDA path."""


def predict_frozen_on_cuda(model: CatBoostRegressor, features: pd.DataFrame) -> np.ndarray:
    """Return raw predictions in frozen feature order; the caller checks native parity."""
    try:
        import torch
    except ImportError as error:
        raise FrozenCudaUnavailable("PyTorch is unavailable") from error
    if not torch.cuda.is_available():
        raise FrozenCudaUnavailable("PyTorch CUDA is unavailable")

    try:
        with TemporaryDirectory() as temp:
            path = Path(temp) / "catboost.json"
            model.save_model(str(path), format="json")
            exported = json.loads(path.read_text(encoding="utf-8"))
        float_features = exported["features_info"]["float_features"]
        feature_indices = {feature["feature_index"]: feature["flat_feature_index"] for feature in float_features}
        trees = exported["oblivious_trees"]
        scale, bias = exported["scale_and_bias"]
        values = features.to_numpy(dtype=np.float32, copy=True)
        if np.isinf(values).any() or list(features.columns) != model.feature_names_:
            raise FrozenCudaUnavailable("CUDA tree evaluation requires frozen feature order and disallows infinity")
        if set(exported["features_info"]) != {"float_features"}:
            raise FrozenCudaUnavailable("CUDA tree evaluation supports only numeric features")
        nan_rules = {feature["feature_index"]: feature["nan_value_treatment"] for feature in float_features}
        if set(nan_rules.values()) - {"AsIs", "AsFalse", "AsTrue"}:
            raise FrozenCudaUnavailable("CUDA tree evaluation does not recognize the model's missing-value rule")

        by_depth = defaultdict(list)
        for tree in trees:
            splits = tree["splits"] or []
            if any(split["split_type"] != "FloatFeature" for split in splits):
                raise FrozenCudaUnavailable("CUDA tree evaluation supports only float splits")
            by_depth[len(splits)].append(tree)

        with torch.inference_mode():
            inputs = torch.as_tensor(values, dtype=torch.float32, device="cuda")
            output = torch.zeros((len(values), len(bias)), dtype=torch.float64, device=inputs.device)
            for depth, group in by_depth.items():
                splits = [tree["splits"] or [] for tree in group]
                float_ids = [[split["float_feature_index"] for split in row] for row in splits]
                indices = torch.as_tensor([[feature_indices[index] for index in row] for row in float_ids],
                                          dtype=torch.long, device=inputs.device)
                borders = torch.as_tensor([[split["border"] for split in row] for row in splits],
                                          dtype=torch.float32, device=inputs.device)
                nan_right = torch.as_tensor([[nan_rules[index] == "AsTrue" for index in row] for row in float_ids],
                                            dtype=torch.bool, device=inputs.device)
                weights = 1 << torch.arange(depth, device=inputs.device, dtype=torch.long)
                selected = inputs[:, indices]
                go_right = torch.where(selected.isnan(), nan_right, selected > borders)
                terminal = (go_right * weights).sum(dim=-1)
                leaves = torch.as_tensor([tree["leaf_values"] for tree in group], dtype=torch.float64, device=inputs.device)
                leaves = leaves.reshape(len(group), 1 << depth, len(bias))
                output += leaves[torch.arange(len(group), device=inputs.device), terminal].sum(dim=1)
            result = (output * scale + torch.as_tensor(bias, dtype=output.dtype, device=inputs.device)).cpu().numpy()
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        if isinstance(error, FrozenCudaUnavailable):
            raise
        raise FrozenCudaUnavailable(str(error)) from error
    return result[:, 0] if result.shape[1] == 1 else result
