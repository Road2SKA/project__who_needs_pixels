import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from rich import print
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
):
    """Load a trained SIREN model from checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    first_omega = checkpoint.get("first_omega")
    hidden_omega = checkpoint.get("hidden_omega")
    print(f"{first_omega=}, {hidden_omega=}, {hidden_features=}, {hidden_layers=}")

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
    return coords, pixels, height, width


def render_model_image(model, coords, height, width, batch_size):
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
    image = output.squeeze(0).view(height, width).detach().cpu().numpy()
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


def print_statistics(stats):
    """Print statistics in a formatted manner."""
    print("\n" + "=" * 60)
    print("INFERENCE STATISTICS")
    print("=" * 60)

    print("\nError Metrics:")
    print(f"  MSE:                 {stats['MSE']:.6e}")
    print(f"  RMSE:                {stats['RMSE']:.6e}")
    print(f"  MAE:                 {stats['MAE']:.6e}")
    print(f"  Max Absolute Error:  {stats['Max Absolute Error']:.6e}")
    print(f"  PSNR:                {stats['PSNR (dB)']:.2f} dB")
    print(f"  Relative Error:      {stats['Relative Error']:.6f}")

    print("\nTruth Image Statistics:")
    print(f"  Min:   {stats['Truth Min']:.6e}")
    print(f"  Max:   {stats['Truth Max']:.6e}")
    print(f"  Mean:  {stats['Truth Mean']:.6e}")
    print(f"  Std:   {stats['Truth Std']:.6e}")

    print("\nPrediction Image Statistics:")
    print(f"  Min:   {stats['Pred Min']:.6e}")
    print(f"  Max:   {stats['Pred Max']:.6e}")
    print(f"  Mean:  {stats['Pred Mean']:.6e}")
    print(f"  Std:   {stats['Pred Std']:.6e}")

    print("=" * 60 + "\n")


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

    cfg = load_config(args.config)

    model_path = Path(cfg.paths.model)
    data_path = Path(cfg.paths.data)
    hidden_features = cfg.model.hidden_features
    hidden_layers = cfg.model.hidden_layers
    in_features = cfg.model.in_features
    batch_size = cfg.inference.batch_size
    title_prefix = cfg.inference.title_prefix
    save_plot_path = Path(cfg.inference.save_plot) if cfg.inference.save_plot else None
    no_plot = cfg.inference.no_plot

    # Check if files exist
    if model_path is None or not model_path.exists():
        print(f"Error: Model file {model_path} not found")
        return

    if data_path is None or not data_path.exists():
        print(f"Error: Data file {data_path} not found")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    print(f"Loading dataset from {data_path}...")
    coords, pixels, height, width = load_prepared_dataset(data_path, device)
    if height is None or width is None:
        print(f"ERROR: {height=}, {width=}")
        exit(1)

    print(f"Dataset shape: {coords.shape}, (h, w) = ({height}, {width})")

    # Load model
    print(f"Loading model from {model_path}...")
    model, checkpoint = load_siren_checkpoint(
        model_path,
        device=device,
        in_features=in_features,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
    )

    if "final_loss" in checkpoint:
        print(f"Model training final loss: {checkpoint['final_loss']:.6e}")

    # Generate prediction
    print("Generating prediction...")
    pred_image = render_model_image(model, coords, height, width, batch_size)
    truth_image = pixels.squeeze(0).view(height, width).cpu().numpy()

    # Calculate statistics
    stats = calculate_statistics(pred_image, truth_image)
    print_statistics(stats)

    # Plot results
    if not no_plot:
        plot_prediction_vs_truth(pred_image, truth_image, title_prefix, save_plot_path)


if __name__ == "__main__":
    main()
