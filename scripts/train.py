#!/usr/bin/env python3
import argparse
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
    sidelength = int(data["sidelength"]) if "sidelength" in data else None
    return coords, pixels, sidelength


def wavelet_reg_term(img, wavelet="db1" ,level=2, mode="constant"):
  coeffs = ptwt.wavedec2(img, wavelet, level=level, mode=mode)
  details = coeffs[1:]

  total = 0.0
  n = 0
  for (cH, cV, cD) in details:
    for c in (cH, cV, cD):
      total += torch.log1p(torch.abs(c)).mean()
      n += 1

  return total /  max(n, 1)


def train(
    model,
    x_data,
    y_data,
    steps=500,
    batch_size_pixels=4096,
    lr=1e-4,
    amp_mode: str = "fp16",
    wavelet: str = "db1",
    level: int = 2,
    mode: str = "constant",
    lambda_wavelet: float = 0.001,
):
    optimizer = torch.optim.Adam(lr=lr, params=model.parameters())
    use_cuda = x_data.is_cuda
    use_scaler = use_cuda and amp_mode == "fp16"
    scaler = GradScaler(enabled=use_scaler)
    num_pixels = x_data.shape[1]
    num_batches = (num_pixels + batch_size_pixels - 1) // batch_size_pixels
    sidelength_val = int(np.sqrt(num_pixels))

    pbar = tqdm(range(steps), desc="Training")
    for step in pbar:
        optimizer.zero_grad(set_to_none=True)
        current_loss = 0.0
        mse_loss = 0.0
        outputs = []

        for i in range(num_batches):
            start_idx = i * batch_size_pixels
            end_idx = min((i + 1) * batch_size_pixels, num_pixels)

            batch_x_data = x_data[:, start_idx:end_idx, :]
            batch_y_data = y_data[:, start_idx:end_idx, :]

            if amp_mode == "bf16":
                amp_dtype = torch.bfloat16
            elif amp_mode == "fp16":
                amp_dtype = torch.float16
            else:
                amp_dtype = None

            with autocast(
                "cuda" if x_data.is_cuda else "cpu",
                dtype=amp_dtype,
                enabled=amp_dtype is not None,
            ):
                batch_output, _ = model(batch_x_data)
                loss = F.mse_loss(batch_output, batch_y_data)
                outputs.append(batch_output)

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            mse_loss += loss.item() * (end_idx - start_idx)

        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

            
        full_model_output = torch.cat(outputs, dim=1)  # (1, num_pixels, 1), with graph
        mse_loss /= num_pixels

        img2d = full_model_output.view(1, sidelength_val, sidelength_val)

        wav_loss = wavelet_reg_term(
            img2d,
            wavelet=wavelet,
            level=level,
            mode=mode,
        )

        current_loss = mse_loss + lambda_wavelet * wav_loss
        pbar.set_postfix({"mse loss": f"{mse_loss:.6f}", "wav loss": f"{wav_loss:.6f}", "total loss": f"{current_loss:.6f}", "batches": num_batches})

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
    batch_size = cfg.train.batch_size
    lr = cfg.train.lr
    hidden_features = cfg.model.hidden_features
    hidden_layers = cfg.model.hidden_layers
    out_path = Path(cfg.train.out or cfg.paths.model)
    amp = cfg.train.amp
    data_dtype_name = cfg.train.data_dtype
    tf32 = cfg.train.tf32
    compile_model = cfg.train.compile
    first_omega = cfg.model.first_omega
    hidden_omega = cfg.model.hidden_omega
    in_features = cfg.model.in_features
    out_features = cfg.model.out_features
    outermost_linear = cfg.model.outermost_linear
    lambda_wavelet = cfg.train.lambda_wavelet
    wavelet = cfg.train.wavelet
    level = cfg.train.level
    mode = cfg.train.mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    (x_data, y_data, _) = load_dataset(data_path, device, data_dtype)
    print(f"{first_omega=}, {hidden_omega=}")
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
        steps=steps,
        batch_size_pixels=batch_size,
        lr=lr,
        amp_mode=amp,
        wavelet=wavelet,
        level=level,
        mode=mode,
        lambda_wavelet=lambda_wavelet,
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
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    main()
