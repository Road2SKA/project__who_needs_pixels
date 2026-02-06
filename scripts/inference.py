import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from train import Siren
from configs import load_config


def strip_compile_prefix(state_dict, prefix="_orig_mod."):
    """Strip torch.compile prefix from state dict keys if present."""
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {
        k[len(prefix) :] if k.startswith(prefix) else k: v
        for k, v in state_dict.items()
    }


def load_siren_checkpoint(
    checkpoint_path,
    device=None,
    in_features=2,
    out_features=1,
    hidden_features=256,
    hidden_layers=3,
    outermost_linear=True,
    first_omega=None,
    hidden_omega=None,
):
    """Load a trained SIREN model from checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    params = checkpoint.get("params", {})

    def _get_param(name, fallback):
        if name in params:
            return params[name]
        if name in checkpoint:
            return checkpoint[name]
        return fallback

    in_features = int(_get_param("in_features", in_features))
    out_features = int(_get_param("out_features", out_features))
    hidden_features = int(_get_param("hidden_features", hidden_features))
    hidden_layers = int(_get_param("hidden_layers", hidden_layers))
    outermost_linear = bool(_get_param("outermost_linear", outermost_linear))
    first_omega = float(_get_param("first_omega", first_omega))
    hidden_omega = float(_get_param("hidden_omega", hidden_omega))

    Console().log(
        f"{first_omega=}, {hidden_omega=}, {hidden_features=}, {hidden_layers=}"
    )

    state_dict = strip_compile_prefix(state_dict)

    model = Siren(
        in_features=in_features,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        out_features=out_features,
        outermost_linear=outermost_linear,
        first_omega=first_omega,
        hidden_omega=hidden_omega,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, checkpoint


def load_prepared_dataset(npz_path, device=None, dtype=torch.float32):
    """Load prepared dataset from npz file."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load(npz_path)
    coords = torch.from_numpy(data["coords"]).unsqueeze(0).to(device, dtype=dtype)
    pixels = torch.from_numpy(data["pixels"]).unsqueeze(0).to(device, dtype=dtype)
    height = int(data["height"]) if "height" in data else None
    width = int(data["width"]) if "width" in data else None
    rows = data["rows"] if "rows" in data else None
    cols = data["cols"] if "cols" in data else None
    return coords, pixels, height, width, rows, cols


def render_model_image(model, coords, height, width, batch_size, rows=None, cols=None):
    """Render the full image from the model in batches."""
    model.eval()
    num_pixels = coords.shape[1]
    num_batches = (num_pixels + batch_size - 1) // batch_size

    outputs = []
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_pixels)
            batch_coords = coords[:, start_idx:end_idx, :]
            batch_output, _ = model(batch_coords)
            outputs.append(batch_output)

    output = torch.cat(outputs, dim=1)
    flat = output.squeeze(0).detach().cpu().numpy()
    if rows is not None and cols is not None:
        image = np.full((height, width), np.nan, dtype=flat.dtype)
        image[rows, cols] = flat.reshape(-1)
    else:
        if flat.size != height * width:
            raise ValueError(
                "Cannot reshape output to full image. "
                "Provide rows/cols in the dataset to reconstruct masked pixels."
            )
        image = flat.view(height, width)
    return image


def calculate_statistics(pred_image, truth_image):
    """Calculate and return various statistics between prediction and truth."""
    residual = pred_image - truth_image

    # Basic statistics
    mse = np.mean(residual**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residual))
    max_abs_error = np.max(np.abs(residual))

    # PSNR (Peak Signal-to-Noise Ratio)
    data_range = truth_image.max() - truth_image.min()
    if mse > 0:
        psnr = 20 * np.log10(data_range / rmse)
    else:
        psnr = float("inf")

    # Relative error
    relative_error = np.mean(np.abs(residual) / (np.abs(truth_image) + 1e-10))

    stats = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "Max Absolute Error": max_abs_error,
        "PSNR (dB)": psnr,
        "Relative Error": relative_error,
        "Truth Min": truth_image.min(),
        "Truth Max": truth_image.max(),
        "Truth Mean": truth_image.mean(),
        "Truth Std": truth_image.std(),
        "Pred Min": pred_image.min(),
        "Pred Max": pred_image.max(),
        "Pred Mean": pred_image.mean(),
        "Pred Std": pred_image.std(),
    }

    return stats


def print_statistics(stats, console: Console):
    """Print statistics in a formatted manner."""
    metrics = Table(title="Inference Statistics", show_lines=False)
    metrics.add_column("Metric", style="cyan", no_wrap=True)
    metrics.add_column("Value", style="white")
    metrics.add_row("MSE", f"{stats['MSE']:.6e}")
    metrics.add_row("RMSE", f"{stats['RMSE']:.6e}")
    metrics.add_row("MAE", f"{stats['MAE']:.6e}")
    metrics.add_row("Max Abs Error", f"{stats['Max Absolute Error']:.6e}")
    metrics.add_row("PSNR (dB)", f"{stats['PSNR (dB)']:.2f}")
    metrics.add_row("Relative Error", f"{stats['Relative Error']:.6f}")
    metrics.add_row("Truth Min", f"{stats['Truth Min']:.6e}")
    metrics.add_row("Truth Max", f"{stats['Truth Max']:.6e}")
    metrics.add_row("Truth Mean", f"{stats['Truth Mean']:.6e}")
    metrics.add_row("Truth Std", f"{stats['Truth Std']:.6e}")
    metrics.add_row("Pred Min", f"{stats['Pred Min']:.6e}")
    metrics.add_row("Pred Max", f"{stats['Pred Max']:.6e}")
    metrics.add_row("Pred Mean", f"{stats['Pred Mean']:.6e}")
    metrics.add_row("Pred Std", f"{stats['Pred Std']:.6e}")
    console.print(metrics)


def plot_prediction_vs_truth(pred_image, truth_image, title_prefix="", save_path=None):
    """Plot truth, prediction, and residual side by side."""
    residual = pred_image - truth_image
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=120)

    axes[0].imshow(truth_image, cmap="gray")
    axes[0].set_title(f"{title_prefix}Truth")

    axes[1].imshow(pred_image, cmap="gray")
    axes[1].set_title(f"{title_prefix}Prediction")

    axes[2].imshow(residual, cmap="coolwarm")
    axes[2].set_title(f"{title_prefix}Residual")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Plot saved to {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Run inference on trained SIREN model")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to YAML config file",
    )

    args = parser.parse_args()

    console = Console()
    cfg = load_config(args.config)

    model_path = Path(cfg.paths.model)
    data_path = Path(cfg.paths.data)
    hidden_features = cfg.model.hidden_features
    hidden_layers = cfg.model.hidden_layers
    in_features = cfg.model.in_features
    out_features = cfg.model.out_features
    outermost_linear = cfg.model.outermost_linear
    first_omega = cfg.model.first_omega
    hidden_omega = cfg.model.hidden_omega
    batch_size = cfg.common.batch_size
    title_prefix = cfg.inference.title_prefix
    save_plot_path = Path(cfg.inference.save_plot) if cfg.inference.save_plot else None
    no_plot = cfg.inference.no_plot

    summary = Table(title="Inference Setup", show_lines=False)
    summary.add_column("Key", style="cyan", no_wrap=True)
    summary.add_column("Value", style="white")
    summary.add_row("model_path", str(model_path))
    summary.add_row("data_path", str(data_path))
    summary.add_row("device", "cuda" if torch.cuda.is_available() else "cpu")
    summary.add_row("batch_size", str(batch_size))
    summary.add_row("title_prefix", str(title_prefix))
    summary.add_row("save_plot", str(save_plot_path))
    summary.add_row("no_plot", str(no_plot))
    summary.add_row("in_features", str(in_features))
    summary.add_row("out_features", str(out_features))
    summary.add_row("hidden_features", str(hidden_features))
    summary.add_row("hidden_layers", str(hidden_layers))
    summary.add_row("outermost_linear", str(outermost_linear))
    summary.add_row("first_omega", str(first_omega))
    summary.add_row("hidden_omega", str(hidden_omega))
    console.print(Panel(summary))

    # Check if files exist
    if model_path is None or not model_path.exists():
        console.print(Panel(f"Model file not found: {model_path}", title="Error"))
        return

    if data_path is None or not data_path.exists():
        console.print(Panel(f"Data file not found: {data_path}", title="Error"))
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.log(f"Using device: {device}")

    # Load dataset
    console.log(f"Loading dataset from {data_path}...")
    coords, pixels, height, width, rows, cols = load_prepared_dataset(data_path, device)
    if height is None or width is None:
        console.print(Panel(f"Invalid shape: {height=}, {width=}", title="Error"))
        exit(1)

    console.log(f"Dataset shape: {coords.shape}, (h, w) = ({height}, {width})")

    # Load model
    console.log(f"Loading model from {model_path}...")
    model, checkpoint = load_siren_checkpoint(
        model_path,
        device=device,
        in_features=in_features,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        out_features=out_features,
        outermost_linear=outermost_linear,
        first_omega=first_omega,
        hidden_omega=hidden_omega,
    )

    if "final_loss" in checkpoint:
        console.log(f"Model training final loss: {checkpoint['final_loss']:.6e}")

    # Generate prediction
    console.log("Generating prediction...")
    pred_image = render_model_image(
        model, coords, height, width, batch_size, rows=rows, cols=cols
    )
    truth_flat = pixels.squeeze(0).cpu().numpy()
    if rows is not None and cols is not None:
        truth_image = np.full((height, width), np.nan, dtype=truth_flat.dtype)
        truth_image[rows, cols] = truth_flat.reshape(-1)
    else:
        truth_image = truth_flat.view(height, width)

    # Calculate statistics
    stats = calculate_statistics(pred_image, truth_image)
    print_statistics(stats, console)

    # Plot results
    if not no_plot:
        plot_prediction_vs_truth(pred_image, truth_image, title_prefix, save_plot_path)


if __name__ == "__main__":
    main()
