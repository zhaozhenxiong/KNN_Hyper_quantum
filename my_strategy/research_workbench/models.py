"""Candidate models for the A-share research workbench."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from my_strategy.deep_learning.models import select_torch_device
from my_strategy.validation.leakage import validate_feature_columns_no_lookahead


_LIGHTGBM = {
    "n_estimators": 150, "learning_rate": 0.05, "num_leaves": 15,
    "min_child_samples": 30, "early_stopping_rounds": 15,
    "n_jobs": 4, "max_depth": -1, "reg_lambda": 0.0,
}
_LSTM = {
    "lookback": 20, "hidden_size": 32, "num_layers": 1, "dropout": 0.1,
    "embedding_dim": 4, "batch_size": 256, "epochs": 12, "patience": 4,
    "learning_rate": 0.001, "weight_decay": 0.0001,
}
_SEQLSTM = {**_LSTM, "batch_size": 512, "max_train_samples": 200_000}
_STATIC_COLUMNS = {"board", "exchange", "industry", "concepts"}
_RESERVED_COLUMNS = {"target", "label_end", "date", "stock", *_STATIC_COLUMNS}


@dataclass(frozen=True)
class ModelResult:
    predictions: pd.Series
    metadata: dict[str, Any]


def _positions(frame: pd.DataFrame, mask: Any) -> np.ndarray:
    if isinstance(mask, pd.Series) and not mask.index.equals(frame.index):
        raise ValueError("model mask index must match the input frame")
    values = np.asarray(mask)
    if values.shape != (len(frame),) or values.dtype != np.dtype(bool):
        raise ValueError("model masks must contain one Boolean per input row")
    return np.flatnonzero(values)


def _numeric(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"missing model features: {sorted(missing)}")
    features = frame[columns]
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes):
        features = features.apply(pd.to_numeric, errors="coerce")
    return features.to_numpy(dtype=np.float32, na_value=np.nan, copy=True)


def _ordered_history(frame: pd.DataFrame, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    order = frame[["stock", "date"]].reset_index(drop=True).sort_values(["stock", "date"]).index.to_numpy()
    stocks = frame["stock"].to_numpy()[order]
    positions = np.arange(len(order))
    first_in_stock = np.r_[True, stocks[1:] != stocks[:-1]][:len(order)]
    stock_starts = np.maximum.accumulate(np.where(first_in_stock, positions, 0))
    ready = np.zeros(len(order), dtype=bool)
    ready[order] = positions - stock_starts + 1 >= lookback
    return order, ready


def sequence_ready(frame: pd.DataFrame, lookback: int) -> pd.Series:
    """Mark rows with enough same-stock observations up to and including that date."""
    return pd.Series(_ordered_history(frame, lookback)[1], index=frame.index)


def sample_training_rows(frame: pd.DataFrame, rows: np.ndarray, limit: int, seed: int) -> np.ndarray:
    """Select deterministic, approximately balanced training rows across stocks."""
    if not limit or len(rows) <= limit:
        return rows
    keys = frame[["stock", "date"]].iloc[rows].reset_index(drop=True)
    keys = keys.sort_values(["stock", "date"])
    row_offsets = keys.index.to_numpy()
    random = np.random.default_rng(seed)
    priorities = np.empty(len(rows), dtype=float)
    for positions in keys.groupby("stock", sort=True).indices.values():
        members = row_offsets[positions]
        priorities[members] = random.permutation(len(members)) + random.random(len(members))
    chosen = np.argpartition(priorities, limit - 1)[:limit]
    return np.sort(rows[chosen])


def _category(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    name = value.strip()
    return "" if name.lower() == "unknown" else name


def _concepts(value: Any) -> set[str]:
    if not isinstance(value, (list, tuple, set, np.ndarray)):
        return set()
    return {_category(item) for item in value} - {""}


def _vocabularies(frame: pd.DataFrame, columns: Sequence[str]) -> dict[str, list[str]]:
    vocabularies = {}
    for column in columns:
        values = frame.get(column, pd.Series(dtype=object))
        if column == "concepts":
            vocabularies[column] = sorted({concept for value in values for concept in _concepts(value)})
        else:
            vocabularies[column] = sorted(set(values.map(_category)) - {""})
    return vocabularies


def _static(frame: pd.DataFrame, metadata: dict[str, Any]) -> tuple[np.ndarray, sparse.csr_matrix]:
    vocabularies = metadata["vocabularies"]
    categorical = np.zeros((len(frame), len(metadata["categorical_columns"])), dtype=np.int32)
    for index, column in enumerate(metadata["categorical_columns"]):
        if column in frame:
            ids = {name: code for code, name in enumerate(vocabularies[column], start=1)}
            categorical[:, index] = frame[column].map(_category).map(ids).fillna(0).to_numpy(dtype=np.int32)
    concept_ids = {name: code for code, name in enumerate(vocabularies.get("concepts", ()))}
    rows, columns = [], []
    if concept_ids and "concepts" in frame:
        for row, value in enumerate(frame["concepts"]):
            for name in _concepts(value).intersection(concept_ids):
                rows.append(row)
                columns.append(concept_ids[name])
    concepts = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, columns)),
        shape=(len(frame), len(concept_ids)),
    )
    return categorical, concepts


def _tree_input(frame: pd.DataFrame, metadata: dict[str, Any], rows: np.ndarray) -> Any:
    columns = [*metadata["feature_columns"], *(column for column in metadata["static_columns"] if column in frame)]
    frame = frame[columns].iloc[rows]
    values = _numeric(frame, metadata["feature_columns"])
    values[~np.isfinite(values)] = np.nan
    categorical, concepts = _static(frame, metadata)
    ordinary = np.column_stack((values, categorical.astype(np.float32)))
    if concepts.shape[1]:
        return sparse.hstack([sparse.csr_matrix(ordinary), concepts], format="csr")
    return ordinary


def _fit_tree(frame: pd.DataFrame, train: np.ndarray, valid: np.ndarray, y: np.ndarray, metadata: dict[str, Any]) -> Any:
    import lightgbm as lgb

    metadata["libraries"]["lightgbm"] = lgb.__version__
    params = dict(metadata["params"])
    rounds = params.pop("n_estimators")
    early_stop = params.pop("early_stopping_rounds")
    start = len(metadata["feature_columns"])
    category_indexes = list(range(start, start + len(metadata["categorical_columns"])))
    training = lgb.Dataset(_tree_input(frame, metadata, train), label=y[train], categorical_feature=category_indexes)
    validation = lgb.Dataset(_tree_input(frame, metadata, valid), label=y[valid], reference=training)
    binary = metadata["task"] == "classification"
    model = lgb.train(
        {**params, "objective": "binary" if binary else "regression", "metric": "binary_logloss" if binary else "l2",
         "seed": metadata["seed"], "verbosity": -1, "deterministic": True, "force_col_wise": True, "device_type": "cpu"},
        training, num_boost_round=rounds, valid_sets=[validation],
        callbacks=[lgb.early_stopping(early_stop, verbose=False)],
    )
    metadata["best_iteration"] = model.best_iteration
    metadata["validation_loss"] = float(next(iter(model.best_score["valid_0"].values())))
    return model


def _make_lstm(metadata: dict[str, Any]) -> Any:
    import torch
    from torch import nn

    options = metadata["params"]
    hidden = options["hidden_size"]
    embedding_dim = options["embedding_dim"]
    vocabularies = metadata["vocabularies"]

    class ResearchLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.LSTM(
                len(metadata["feature_columns"]), hidden, num_layers=options["num_layers"], batch_first=True,
                dropout=options["dropout"] if options["num_layers"] > 1 else 0.0,
            )
            self.embeddings = nn.ModuleList(
                nn.Embedding(len(vocabularies[column]) + 1, embedding_dim, padding_idx=0)
                for column in metadata["categorical_columns"]
            )
            concept_count = len(vocabularies.get("concepts", ()))
            self.concept_projection = nn.Linear(concept_count, embedding_dim, bias=False) if concept_count else None
            input_size = hidden + embedding_dim * (len(self.embeddings) + int(concept_count > 0))
            self.head = nn.Sequential(nn.LayerNorm(input_size), nn.Dropout(options["dropout"]), nn.Linear(input_size, 1))

        def forward(self, x: Any, categorical: Any, concepts: Any) -> Any:
            history, _ = self.encoder(x)
            branches = [history[:, -1, :]]
            for index, embedding in enumerate(self.embeddings):
                branches.append(embedding(categorical[:, index]))
            if self.concept_projection is not None:
                branches.append(self.concept_projection(concepts))
            return self.head(torch.cat(branches, dim=1)).squeeze(1)

    return ResearchLSTM()


class _SequenceInput:
    def __init__(
        self, frame: pd.DataFrame, metadata: dict[str, Any], values: np.ndarray,
        history: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        self.lookback = metadata["params"]["lookback"]
        order, self.has_history = history if history is not None else _ordered_history(frame, self.lookback)
        self.position = np.empty(len(order), dtype=np.int64)
        self.position[order] = np.arange(len(order))
        center = np.asarray(metadata["feature_center"], dtype=np.float32)
        scale = np.asarray(metadata["feature_scale"], dtype=np.float32)
        self.values = values[order]
        for index, feature in enumerate(self.values.T):
            feature[~np.isfinite(feature)] = center[index]
        self.values -= center
        self.values /= scale
        self.categorical, self.concepts = _static(frame, metadata)

    def batch(self, rows: np.ndarray, device: str) -> tuple[Any, Any, Any]:
        import torch

        indexes = self.position[rows, None] + np.arange(1 - self.lookback, 1)
        return (
            torch.from_numpy(self.values[indexes]).to(device),
            torch.from_numpy(self.categorical[rows]).to(device),
            torch.from_numpy(self.concepts[rows].toarray()).to(device),
        )


def _batches(rows: np.ndarray, size: int):
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def _sequence_predictions(network: Any, inputs: _SequenceInput, rows: np.ndarray, metadata: dict[str, Any], device: str) -> np.ndarray:
    import torch

    size = metadata["params"]["batch_size"]
    ready = inputs.has_history[rows]
    output = np.full(len(rows), np.nan, dtype=float)
    network.eval()
    with torch.no_grad():
        predictions = []
        for batch in _batches(rows[ready], size):
            logits = network(*inputs.batch(batch, device))
            values = logits.sigmoid() if metadata["task"] == "classification" else logits
            predictions.append(values.cpu().numpy())
        if predictions:
            output[ready] = np.concatenate(predictions) * metadata["target_scale"] + metadata["target_center"]
    return output


def _fit_sequence(
    frame: pd.DataFrame, train: np.ndarray, valid: np.ndarray, y: np.ndarray, metadata: dict[str, Any], requested_device: str,
    history: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[Any, _SequenceInput]:
    import torch
    from torch import nn

    metadata["libraries"]["torch"] = str(torch.__version__)
    values = _numeric(frame, metadata["feature_columns"])
    centers, scales = [], []
    for column in range(values.shape[1]):
        values_in_train = values[train, column]
        observed = values_in_train[np.isfinite(values_in_train)].astype(float)
        centers.append(float(np.median(observed)) if observed.size else 0.0)
        scales.append((float(observed.std()) or 1.0) if observed.size else 1.0)
    metadata.update(feature_center=centers, feature_scale=scales)
    inputs = _SequenceInput(frame, metadata, values, history)
    del values
    train = train[inputs.has_history[train]]
    valid = valid[inputs.has_history[valid]]
    if not len(train) or not len(valid):
        raise ValueError("LSTM lookback leaves no training or validation samples")
    binary = metadata["task"] == "classification"
    if binary and np.unique(y[train]).size < 2:
        raise ValueError("classification needs both outcomes in the training sequence samples")
    target_center = 0.0 if binary else float(y[train].mean())
    target_scale = 1.0 if binary else (float(y[train].std()) or 1.0)
    transformed_y = ((y - target_center) / target_scale).astype(np.float32)
    device = select_torch_device(torch, requested_device)
    metadata.update(device=device, target_center=target_center, target_scale=target_scale,
                    train_sequences=len(train), validation_sequences=len(valid))
    options = metadata["params"]
    rng = np.random.default_rng(metadata["seed"])
    device_index = torch.device(device).index
    cuda_index = [torch.cuda.current_device() if device_index is None else device_index] if device.startswith("cuda") else []
    with torch.random.fork_rng(devices=cuda_index):
        torch.random.default_generator.manual_seed(metadata["seed"])
        if cuda_index:
            with torch.cuda.device(cuda_index[0]):
                torch.cuda.manual_seed(metadata["seed"])
        network = _make_lstm(metadata).to(device)
        optimizer = torch.optim.AdamW(network.parameters(), lr=options["learning_rate"], weight_decay=options["weight_decay"])
        loss_fn = nn.BCEWithLogitsLoss() if binary else nn.SmoothL1Loss()
        best_state, best_loss, best_epoch = None, float("inf"), 0
        history = []
        for epoch in range(1, options["epochs"] + 1):
            network.train()
            for batch in _batches(rng.permutation(train), options["batch_size"]):
                optimizer.zero_grad(set_to_none=True)
                observed = torch.from_numpy(transformed_y[batch]).to(device)
                loss = loss_fn(network(*inputs.batch(batch, device)), observed)
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0, error_if_nonfinite=True)
                optimizer.step()
            network.eval()
            with torch.no_grad():
                validation_sum = 0.0
                for batch in _batches(valid, options["batch_size"]):
                    observed = torch.from_numpy(transformed_y[batch]).to(device)
                    loss = loss_fn(network(*inputs.batch(batch, device)), observed)
                    validation_sum += loss.item() * len(batch)
            val_loss = validation_sum / len(valid)
            history.append(val_loss)
            if not math.isfinite(val_loss):
                raise ValueError("LSTM encountered a non-finite validation loss")
            if val_loss < best_loss:
                best_state = {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}
                best_loss, best_epoch = val_loss, epoch
            if epoch - best_epoch >= options["patience"]:
                break
    network.load_state_dict(best_state)
    metadata.update(best_iteration=best_epoch, validation_loss=best_loss, validation_history=history)
    return network, inputs


def _options(model: str, params: Mapping[str, Any] | None) -> tuple[str, dict[str, Any]]:
    overrides = dict(params or {})
    task = overrides.pop("task", "regression")
    if task not in {"regression", "classification"}:
        raise ValueError(f"unknown model task: {task}")
    defaults = {"lightgbm": _LIGHTGBM, "lstm": _LSTM, "seqlstm": _SEQLSTM}[model]
    if model != "lightgbm" and "lr" in overrides and "learning_rate" not in overrides:
        overrides["learning_rate"] = overrides.pop("lr")
    unsupported = set(overrides).difference(defaults)
    if unsupported:
        raise ValueError(f"unsupported {model} parameters: {sorted(unsupported)}")
    options = {**defaults, **overrides}
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) for value in options.values()):
        raise ValueError("model parameters must be finite numbers")
    for name, default in defaults.items():
        value = options[name]
        minimum = -1 if name == "max_depth" else 0 if name == "max_train_samples" else 1
        if isinstance(default, int) and (not isinstance(value, int) or value < minimum):
            raise ValueError(f"invalid integer model parameter: {name}={value}")
    if options["learning_rate"] <= 0:
        raise ValueError("learning rate must be positive")
    if model != "lightgbm" and (not 0 <= options["dropout"] < 1 or options["weight_decay"] < 0):
        raise ValueError("LSTM dropout must be in [0, 1) and weight_decay must be nonnegative")
    if model == "lightgbm" and (options["num_leaves"] < 2 or options["reg_lambda"] < 0):
        raise ValueError("LightGBM requires at least two leaves and nonnegative reg_lambda")
    return task, options


def _write_bundle(path: Path, model: Any, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with TemporaryDirectory(prefix=".model-", dir=path.parent) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        if metadata["model"] == "lightgbm":
            model.save_model(str(staging / "model.txt"))
        else:
            import torch

            torch.save({key: tensor.detach().cpu() for key, tensor in model.state_dict().items()}, staging / "model.pt")
        (staging / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        staging.rename(path)


def fit_predict(
    frame: pd.DataFrame, *, feature_columns: Sequence[str], train_mask: Any, valid_mask: Any,
    predict_mask: Any, model: str, params: Mapping[str, Any] | None = None, static_columns: Sequence[str] = (),
    seed: int = 42, device: str = "auto", bundle_path: Path | None = None,
) -> ModelResult:
    """Fit on a purged chronological split and return decisions indexed to prediction rows."""
    if model not in {"lightgbm", "lstm", "seqlstm"}:
        raise ValueError(f"unknown research model: {model}")
    if not frame.index.is_unique or frame[["stock", "date"]].duplicated().any():
        raise ValueError("model input requires unique row indexes and stock/date keys")
    features = list(feature_columns)
    static = list(static_columns)
    if not features or len(set(features)) != len(features) or set(features).intersection(_RESERVED_COLUMNS):
        raise ValueError("feature columns must be distinct numeric inputs")
    if not validate_feature_columns_no_lookahead(features)["ok"]:
        raise ValueError("future-looking features are prohibited")
    if len(set(static)) != len(static) or set(static).difference(_STATIC_COLUMNS):
        raise ValueError("static inputs must be selected from board, exchange, industry, and concepts")
    task, options = _options(model, params)
    y = pd.to_numeric(frame["target"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    train = _positions(frame, train_mask)
    valid = _positions(frame, valid_mask)
    requested = _positions(frame, predict_mask)
    train, valid = train[np.isfinite(y[train])], valid[np.isfinite(y[valid])]
    if not len(train) or not len(valid):
        raise ValueError("missing labeled training or validation rows")
    dates = pd.to_datetime(frame["date"])
    label_ends = pd.to_datetime(frame["label_end"])
    labeled = np.concatenate((train, valid))
    if dates.isna().any() or label_ends.iloc[labeled].isna().any() or (label_ends.iloc[labeled] <= dates.iloc[labeled]).any():
        raise ValueError("labeled rows require label_end strictly after their signal date")
    if label_ends.iloc[train].max() >= dates.iloc[valid].min():
        raise ValueError("training labels must end before the first validation date")
    history = None
    if model == "seqlstm":
        history = _ordered_history(frame, options["lookback"])
        ready = history[1]
        train, valid = train[ready[train]], valid[ready[valid]]
        train = sample_training_rows(frame, train, options["max_train_samples"], seed)
        if not len(train) or not len(valid):
            raise ValueError("SeqLSTM lookback leaves no training or validation samples")
    if task == "classification":
        y = (y > 0).astype(float)
        if np.unique(y[train]).size < 2:
            raise ValueError("classification needs both outcomes in the training split")
    metadata = {
        "format_version": 1, "model": model, "status": "candidate", "task": task,
        "prediction_kind": "probability_positive" if task == "classification" else "expected_return",
        "params": options, "feature_columns": features, "static_columns": static,
        "categorical_columns": [column for column in static if column != "concepts"],
        "vocabularies": _vocabularies(frame[[column for column in static if column in frame]].iloc[train], static), "seed": seed,
        "libraries": {"numpy": np.__version__, "pandas": pd.__version__},
        "device_requested": device, "device": "cpu", "train_rows": len(train), "validation_rows": len(valid),
        "train_start": dates.iloc[train].min().isoformat(), "train_end": dates.iloc[train].max().isoformat(),
        "train_label_end": label_ends.iloc[train].max().isoformat(),
        "validation_start": dates.iloc[valid].min().isoformat(), "validation_end": dates.iloc[valid].max().isoformat(),
        "validation_label_end": label_ends.iloc[valid].max().isoformat(),
    }
    if model == "lightgbm":
        fitted = _fit_tree(frame, train, valid, y, metadata)
        values = fitted.predict(_tree_input(frame, metadata, requested), num_threads=options["n_jobs"]) if len(requested) else np.empty(0)
    else:
        fitted, inputs = _fit_sequence(frame, train, valid, y, metadata, device, history)
        values = _sequence_predictions(fitted, inputs, requested, metadata, metadata["device"])
    if bundle_path is not None:
        _write_bundle(Path(bundle_path), fitted, metadata)
    return ModelResult(pd.Series(values, index=frame.index[requested], name="prediction", dtype=float), metadata)


def load_predict(frame: pd.DataFrame, *, bundle_path: Path, predict_mask: Any, device: str = "cpu") -> ModelResult:
    """Replay a native model bundle with its original feature order and preprocessing."""
    bundle_path = Path(bundle_path)
    metadata = json.loads((bundle_path / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("format_version") != 1:
        raise ValueError(f"unknown research model bundle version: {metadata.get('format_version')}")
    requested = _positions(frame, predict_mask)
    if not len(requested):
        return ModelResult(pd.Series(dtype=float, index=frame.index[requested], name="prediction"), metadata)
    if metadata["model"] == "lightgbm":
        import lightgbm as lgb

        model = lgb.Booster(model_file=str(bundle_path / "model.txt"))
        values = model.predict(_tree_input(frame, metadata, requested), num_threads=metadata["params"]["n_jobs"])
    elif metadata["model"] in {"lstm", "seqlstm"}:
        import torch

        runtime = select_torch_device(torch, device)
        model = _make_lstm(metadata)
        model.load_state_dict(torch.load(bundle_path / "model.pt", map_location="cpu", weights_only=True))
        model.to(runtime)
        inputs = _SequenceInput(frame, metadata, _numeric(frame, metadata["feature_columns"]))
        values = _sequence_predictions(model, inputs, requested, metadata, runtime)
    else:
        raise ValueError(f"unknown research model in bundle: {metadata['model']}")
    return ModelResult(pd.Series(values, index=frame.index[requested], name="prediction", dtype=float), metadata)
