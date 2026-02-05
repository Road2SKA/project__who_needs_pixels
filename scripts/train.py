
import argparse
from datetime import datetime
from pathlib import Path
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from tqdm.auto import tqdm
from configs import load_config
import ptwt
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


class SineLayer(nn.Module):
    def __init__(
        self, in_features, out_features, bias=True, is_first=False, omega=30.0
    ):
        super().__init__()
        self.omega = omega
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega,
                    np.sqrt(6 / self.in_features) / self.omega,
                )

    def forward(self, input):
        return torch.sin(self.omega * self.linear(input))

    def forward_with_intermediate(self, input):
        intermediate = self.omega * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        outermost_linear=False,
        first_omega=30.0,
        hidden_omega=30.0,
    ):
        super().__init__()
        self.net = []
        self.net.append(
            SineLayer(in_features, hidden_features, is_first=True, omega=first_omega)
        )

        for _ in range(hidden_layers):
            self.net.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega=hidden_omega,
                )
            )

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / hidden_omega,
                    np.sqrt(6 / hidden_features) / hidden_omega,
                )
            self.net.append(final_linear)
        else:
            self.net.append(
                SineLayer(
                    hidden_features,
                    out_features,
                    is_first=False,
                    omega=hidden_omega,
                )
            )

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)
        output = self.net(coords)
        return output, coords

    def forward_with_activations(self, coords, retain_grad=False):
        activations = OrderedDict()
        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations["input"] = x
        for layer in self.net:
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()
                activations[
                    "_".join((str(layer.__class__), "%d" % activation_count))
                ] = intermed
                activation_count += 1
            else:
                x = layer(x)
                if retain_grad:
                    x.retain_grad()
            activations["_".join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1
        return activations


def load_dataset(npz_path: Path, device: torch.device, dtype: torch.dtype):
    data = np.load(npz_path)
    coords = torch.from_numpy(data["coords"]).unsqueeze(0).to(device, dtype=dtype)
    pixels = torch.from_numpy(data["pixels"]).unsqueeze(0).to(device, dtype=dtype)
    height = int(data["height"])
    width = int(data["width"])
    return coords, pixels, height, width


def wavelet_reg_term(img, wavelet="db1", level=2, mode="constant") -> torch.Tensor:
    if img.dtype in (torch.float16, torch.bfloat16):
        img = img.float()
    coeffs = ptwt.wavedec2(img, wavelet, level=level, mode=mode)
    details = coeffs[1:]

    total = 0.0
    n = 0
    for cH, cV, cD in details:
        for c in (cH, cV, cD):
            total += torch.log1p(torch.abs(c)).mean()
            n += 1

    return total / max(n, 1)  # pyright: ignore[reportReturnType]


def render_full_image(
    model,
    x_data,
    height,
    width,
    batch_size_pixels,
):
    num_pixels = max(x_data.shape[1], 1)
    num_batches = (num_pixels + batch_size_pixels - 1) // batch_size_pixels
    model_was_training = model.training
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size_pixels
            end_idx = min((batch_idx + 1) * batch_size_pixels, num_pixels)
            batch_x = x_data[:, start_idx:end_idx, :]
            batch_output, _ = model(batch_x)
            outputs.append(batch_output)
    output = torch.cat(outputs, dim=1)
    image = output.squeeze(0).view(height, width).detach().cpu().numpy()
    if model_was_training:
        model.train()
    return image


def train(
    model,
    x_data,
    y_data,
    height,
    width,
    device,
    steps=500,
    batch_size_pixels=4096,
    lr=1e-4,
    amp_mode: str = "fp16",
    wavelet: str = "db1",
    level: int = 2,
    mode: str = "constant",
    lambda_wavelet: float = 0.001,
    checkpoint_every: int = 0,
    checkpoint_dir: Path | None = None,
    save_image_every: int = 0,
    image_dir: Path | None = None,
    early_stop: bool = False,
    early_stop_every: int = 0,
    early_stop_patience: int = 3,
    early_stop_tolerance: float = 1e-6,
    verbose: bool = True,
):
    console = Console() if verbose else None
    optimizer = torch.optim.Adam(lr=lr, params=model.parameters())
    use_cuda = x_data.is_cuda
    use_scaler = use_cuda and amp_mode == "fp16"
    scaler = GradScaler(enabled=use_scaler)
    num_pixels = max(x_data.shape[1], 1)
    num_batches = (num_pixels + batch_size_pixels - 1) // batch_size_pixels

    if amp_mode == "bf16":
        amp_dtype = torch.bfloat16
    elif amp_mode == "fp16":
        amp_dtype = torch.float16
    else:
        amp_dtype = None

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if image_dir is not None:
        image_dir.mkdir(parents=True, exist_ok=True)

    last_check_loss = None
    plateau_count = 0

    pbar = tqdm(range(steps), desc="Training", disable=not verbose)
    for step in pbar:
        optimizer.zero_grad(set_to_none=True)
        current_loss = 0.0
        mse_loss = 0.0
        weighted_wave_loss = 0.0

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size_pixels
            end_idx = min((batch_idx + 1) * batch_size_pixels, num_pixels)
            batch_x = x_data[:, start_idx:end_idx, :]
            batch_y = y_data[:, start_idx:end_idx, :]
            batch_weight = (end_idx - start_idx) / num_pixels

            with autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_dtype is not None,
            ):
                batch_output, _ = model(batch_x)
                batch_mse_loss = F.mse_loss(batch_output, batch_y)
                batch_img2d = batch_output.view(1, 1, -1)
                batch_wav_loss = wavelet_reg_term(
                    batch_img2d,
                    wavelet=wavelet,
                    level=level,
                    mode=mode,
                )
                batch_weighted_wave_loss = lambda_wavelet * batch_wav_loss
                batch_total_loss = (
                    batch_mse_loss + batch_weighted_wave_loss
                ) * batch_weight

            if use_scaler:
                scaler.scale(batch_total_loss).backward()
            else:
                batch_total_loss.backward()

            mse_loss += batch_mse_loss.detach().item() * batch_weight
            weighted_wave_loss += (
                batch_weighted_wave_loss.detach().item() * batch_weight
            )
            current_loss += batch_total_loss.detach().item()

        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if verbose:
            pbar.set_postfix(
                {
                    "mse loss": f"{mse_loss:.6f}",
                    "wav loss": f"{weighted_wave_loss:.6f}",
                    "total loss": f"{current_loss:.6f}",
                    "batches": num_batches,
                    "gpu_mem": (
                        f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB"
                        if device.type == "cuda"
                        else "N/A"
                    ),
                }
            )

        step_num = step + 1

        if save_image_every and (step_num % save_image_every == 0) and image_dir:
            image = render_full_image(
                model,
                x_data,
                height,
                width,
                batch_size_pixels,
            )
            image_path = image_dir / f"train_pred_step_{step_num:06d}.npy"
            np.save(image_path, image)
            if console is not None:
                timestamp = datetime.now().isoformat(timespec="seconds")
                console.log(
                    f"[{timestamp}] Saved image: {image_path} | "
                    f"mse={mse_loss:.6f}, wave={weighted_wave_loss:.6f}, "
                    f"total={current_loss:.6f}"
                )

        if checkpoint_every and (step_num % checkpoint_every == 0) and checkpoint_dir:
            checkpoint_path = checkpoint_dir / f"checkpoint_step_{step_num:06d}.pt"
            torch.save(
                {
                    "step": step_num,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "mse_loss": mse_loss,
                    "wave_loss": weighted_wave_loss,
                    "total_loss": current_loss,
                },
                checkpoint_path,
            )
            if console is not None:
                timestamp = datetime.now().isoformat(timespec="seconds")
                console.log(
                    f"[{timestamp}] Saved checkpoint: {checkpoint_path} | "
                    f"mse={mse_loss:.6f}, wave={weighted_wave_loss:.6f}, "
                    f"total={current_loss:.6f}"
                )

        if early_stop and early_stop_every and (step_num % early_stop_every == 0):
            if last_check_loss is not None:
                denom = max(1.0, abs(last_check_loss))
                if abs(current_loss - last_check_loss) <= early_stop_tolerance * denom:
                    plateau_count += 1
                else:
                    plateau_count = 0
                if plateau_count >= early_stop_patience:
                    if console is not None:
                        console.log(
                            "Early stopping: loss plateaued "
                            f"(step {step_num}, loss {current_loss:.6f})."
                        )
                    break
            last_check_loss = current_loss

    return current_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SIREN on prepared dataset")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    data_path = Path(cfg.paths.data)
    steps = cfg.train.steps
    batch_size = cfg.common.batch_size
    lr = cfg.common.lr
    hidden_features = cfg.model.hidden_features
    hidden_layers = cfg.model.hidden_layers
    out_path = Path(cfg.train.out or cfg.paths.model)
    amp = cfg.common.amp
    data_dtype_name = cfg.common.data_dtype
    tf32 = cfg.common.tf32
    compile_model =  cfg.common.compile
    first_omega = cfg.model.first_omega
    hidden_omega = cfg.model.hidden_omega
    in_features = cfg.model.in_features
    out_features = cfg.model.out_features
    outermost_linear = cfg.model.outermost_linear
    lambda_wavelet = cfg.model.lambda_wavelet
    wavelet = cfg.train.wavelet
    level = cfg.train.level
    mode = cfg.train.mode
    cfg_device = cfg.train.device

    checkpoint_every = int(getattr(cfg.train, "checkpoint_every", 0))
    checkpoint_dir = getattr(cfg.paths, "checkpoints_dir", None)
    save_image_every = int(getattr(cfg.train, "save_image_every", 0))
    image_dir = getattr(cfg.paths, "images_dir", "data")
    early_stop = bool(getattr(cfg.train, "early_stop", False))
    early_stop_every = int(getattr(cfg.train, "early_stop_every", 0))
    early_stop_patience = int(getattr(cfg.train, "early_stop_patience", 3))
    early_stop_tolerance = float(getattr(cfg.train, "early_stop_tolerance", 1e-6))

    if cfg_device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(cfg_device)

    if device.type != "cuda":
        amp = "none"

    if tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    if data_dtype_name == "float16":
        data_dtype = torch.float16
    elif data_dtype_name == "bfloat16":
        data_dtype = torch.bfloat16
    else:
        data_dtype = torch.float32

    (x_data, y_data, height, width) = load_dataset(data_path, device, data_dtype)
    console = Console()
    params_table = Table(title="Training Parameters", show_lines=False)
    params_table.add_column("Key", style="cyan", no_wrap=True)
    params_table.add_column("Value", style="white")
    params_table.add_row("data_path", str(data_path))
    params_table.add_row("device", str(device))
    params_table.add_row("steps", str(steps))
    params_table.add_row("batch_size", str(batch_size))
    params_table.add_row("lr", str(lr))
    params_table.add_row("amp", str(amp))
    params_table.add_row("data_dtype", str(data_dtype_name))
    params_table.add_row("tf32", str(tf32))
    params_table.add_row("compile", str(compile_model))
    params_table.add_row("wavelet", str(wavelet))
    params_table.add_row("level", str(level))
    params_table.add_row("mode", str(mode))
    params_table.add_row("lambda_wavelet", str(lambda_wavelet))
    params_table.add_row("in_features", str(in_features))
    params_table.add_row("out_features", str(out_features))
    params_table.add_row("hidden_features", str(hidden_features))
    params_table.add_row("hidden_layers", str(hidden_layers))
    params_table.add_row("outermost_linear", str(outermost_linear))
    params_table.add_row("first_omega", str(first_omega))
    params_table.add_row("hidden_omega", str(hidden_omega))
    params_table.add_row("checkpoint_every", str(checkpoint_every))
    params_table.add_row("checkpoint_dir", str(checkpoint_dir))
    params_table.add_row("save_image_every", str(save_image_every))
    params_table.add_row("image_dir", str(image_dir))
    params_table.add_row("early_stop", str(early_stop))
    params_table.add_row("early_stop_every", str(early_stop_every))
    params_table.add_row("early_stop_patience", str(early_stop_patience))
    params_table.add_row("early_stop_tolerance", str(early_stop_tolerance))
    console.print(Panel(params_table))
    model = Siren(
        in_features=in_features,
        out_features=out_features,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        outermost_linear=outermost_linear,
        first_omega=first_omega,
        hidden_omega=hidden_omega,
    ).to(device)

    if compile_model:
        model = torch.compile(model)

    final_loss = train(
        model,
        x_data,
        y_data,
        height,
        width,
        device,
        steps=steps,
        batch_size_pixels=batch_size,
        lr=lr,
        amp_mode=amp,
        wavelet=wavelet,
        level=level,
        mode=mode,
        lambda_wavelet=lambda_wavelet,
        checkpoint_every=checkpoint_every,
        checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else None,
        save_image_every=save_image_every,
        image_dir=Path(image_dir) if image_dir else None,
        early_stop=early_stop,
        early_stop_every=early_stop_every,
        early_stop_patience=early_stop_patience,
        early_stop_tolerance=early_stop_tolerance,
        verbose=True,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "final_loss": final_loss,
            "first_omega": first_omega,
            "hidden_omega": hidden_omega,
        },
        out_path,
    )
    console.print(Panel(f"Saved model to {out_path}", title="Done"))


if __name__ == "__main__":
    main()
