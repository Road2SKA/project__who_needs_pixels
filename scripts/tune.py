import argparse
import json
import time
from pathlib import Path
from typing import Any
from pprint import pformat

import numpy as np
import optuna
import torch
from tqdm.auto import tqdm

from configs import load_config
from train import Siren, load_dataset, train


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def suggest_param(trial: optuna.Trial, name: str, spec: Any) -> Any:
    if isinstance(spec, list):
        return trial.suggest_categorical(name, spec)
    if isinstance(spec, dict):
        ptype = spec.get("type", "float")
        if ptype == "categorical":
            return trial.suggest_categorical(name, spec["choices"])
        low = spec["low"]
        high = spec["high"]
        log = bool(spec.get("log", False))
        step = spec.get("step", None)
        if ptype == "int":
            return trial.suggest_int(name, int(low), int(high), step=step or 1, log=log)
        if ptype == "float":
            return trial.suggest_float(
                name, float(low), float(high), step=step, log=log
            )
    raise ValueError(
        "Search spec must be a list or dict with type/low/high (or categorical)"
    )


def estimate_trials(search: dict[str, Any]) -> int:
    total = 1
    for spec in search.values():
        if isinstance(spec, list):
            total *= max(len(spec), 1)
        elif isinstance(spec, dict) and spec.get("type") == "categorical":
            total *= max(len(spec.get("choices", [])), 1)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for SIREN")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_path = Path(cfg.paths.data)
    out_dir = Path(cfg.tune.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    search = cfg.tune.search
    if not search:
        raise ValueError("Config must include a non-empty 'search' section")

    # Required search params for omega values
    if "first_omega" not in search or "hidden_omega" not in search:
        raise ValueError("'search' must include first_omega and hidden_omega")

    fixed = cfg.tune.fixed
    seed = cfg.tune.seed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_mode = cfg.tune.amp
    if device.type != "cuda":
        amp_mode = "none"

    if cfg.tune.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    data_dtype_name = cfg.tune.data_dtype
    if data_dtype_name == "float16":
        data_dtype = torch.float16
    elif data_dtype_name == "bfloat16":
        data_dtype = torch.bfloat16
    else:
        data_dtype = torch.float32

    x_data, y_data, _ = load_dataset(data_path, device, data_dtype)

    results_path = out_dir / "results.jsonl"
    best_path = out_dir / "best_model.pt"
    best_loss = float("inf")
    best_params: dict[str, Any] | None = None

    n_trials = int(cfg.tune.n_trials or estimate_trials(search))
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_loss, best_params
        set_seed(seed)

        params = {
            name: suggest_param(trial, name, spec) for name, spec in search.items()
        }
        model_defaults = {
            "in_features": cfg.model.in_features,
            "out_features": cfg.model.out_features,
            "hidden_features": cfg.model.hidden_features,
            "hidden_layers": cfg.model.hidden_layers,
            "outermost_linear": cfg.model.outermost_linear,
            "first_omega": cfg.model.first_omega,
            "hidden_omega": cfg.model.hidden_omega,
        }
        base = {**model_defaults, **fixed}
        merged = {**base, **params}

        model = Siren(
            in_features=merged.get("in_features", 3),
            out_features=merged.get("out_features", 1),
            hidden_features=int(merged.get("hidden_features", 256)),
            hidden_layers=int(merged.get("hidden_layers", 3)),
            outermost_linear=bool(merged.get("outermost_linear", True)),
            first_omega=float(merged["first_omega"]),
            hidden_omega=float(merged["hidden_omega"]),
        ).to(device)

        if merged.get("compile", cfg.tune.compile):
            model = torch.compile(model)

        start_time = time.time()
        final_loss = train(
            model,
            x_data,
            y_data,
            steps=int(merged.get("steps", cfg.tune.steps)),
            batch_size_pixels=int(merged.get("batch_size", cfg.tune.batch_size)),
            lr=float(merged.get("lr", cfg.tune.lr)),
            amp_mode=merged.get("amp", amp_mode),
        )
        elapsed = time.time() - start_time

        record = {
            "trial": trial.number + 1,
            "params": merged,
            "final_loss": final_loss,
            "elapsed_sec": elapsed,
        }
        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        if final_loss < best_loss:
            best_loss = final_loss
            best_params = merged
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "final_loss": final_loss,
                    "params": merged,
                },
                best_path,
            )

        return final_loss

    def on_trial(study: optuna.Study, _trial: optuna.Trial) -> None:
        if study.best_trial is None:
            return
        best_trial = study.best_trial
        tqdm.write(
            "Best params so far "
            f"(trial {best_trial.number + 1}, loss {best_trial.value:.6f}):\n"
            f"{pformat(best_trial.params)}"
        )

    study.optimize(objective, n_trials=n_trials, callbacks=[on_trial])

    if best_params is not None:
        summary_path = out_dir / "best_params.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump({"best_loss": best_loss, "params": best_params}, f, indent=2)


if __name__ == "__main__":
    main()
