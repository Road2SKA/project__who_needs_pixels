#!/usr/bin/env python3
"""
SIREN Image Fitting for ILIFU
Simplified version focused on image fitting with radio astronomy data

Based on: "Implicit Neural Representations with Periodic Activation Functions"
https://vsitzmann.github.io/siren

Usage:
    python siren_fits.py --fits_file <path> --steps 500 --output_dir ./results
    python siren_fits.py --steps 1000 --output_dir ./results  # Uses cameraman image
"""

import argparse
import os
import sys
import time
from pathlib import Path

# ============================================================================
# IMPORTS
# ============================================================================

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
#import skimage
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for ILIFU
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from astropy.io import fits
import ptwt
# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

def setup_device():
    """Automatically detect and configure the appropriate device"""
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print('✓ Using Metal Performance Shaders (MPS) - Mac GPU acceleration')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print('✓ Using CUDA - NVIDIA GPU acceleration')
    else:
        device = torch.device('cpu')
        print('⚠ Using CPU - no GPU acceleration available')
    
    print(f'Device: {device}')
    return device

# ============================================================================
# UTILITIES
# ============================================================================

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class SineLayer(nn.Module):
    """
    Sine activation layer with proper initialisation.
    See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    """
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class Siren(nn.Module):
    """
    SIREN network with sine activations
    """
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, 
                 outermost_linear=False, first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)
        output = self.net(coords)
        return output, coords

# ============================================================================
# DIFFERENTIAL OPERATORS
# ============================================================================

def gradient(y, x, grad_outputs=None):
    """Compute gradient of y with respect to x"""
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def divergence(y, x):
    """Compute divergence of y with respect to x"""
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), 
                                   create_graph=True)[0][..., i:i+1]
    return div


def laplace(y, x):
    """Compute Laplacian of y with respect to x"""
    grad = gradient(y, x)
    return divergence(grad, x)

# ============================================================================
# DATA LOADING
# ============================================================================


def get_fits_square_crop(fits_file, crop_size, sidelength, center_x=None, center_y=None):
    """
    Extract a square crop from a FITS file and resize to sidelength.
    Returns a tensor of shape [1, sidelength, sidelength] normalised to [-1, 1].
    """
    with fits.open(fits_file) as hdul:
        data = hdul[0].data
        
        # Handle different FITS dimensions
        if data.ndim == 4:
            data = data[0, 0]  # [freq, pol, y, x] -> [y, x]
        elif data.ndim == 3:
            data = data[0]     # [channel, y, x] -> [y, x]
        elif data.ndim != 2:
            raise ValueError(f"Unexpected FITS dimensions: {data.ndim}")
        
        height, width = data.shape
        print(f"Original: {height} x {width}")
        
        # Default to center
        if center_x is None:
            center_x = width // 2
        if center_y is None:
            center_y = height // 2
        
        # Calculate crop boundaries
        half = crop_size // 2
        x_start = max(0, center_x - half)
        x_end = min(width, center_x + half)
        y_start = max(0, center_y - half)
        y_end = min(height, center_y + half)
        
        data = data[y_start:y_end, x_start:x_end]
        print(f"Cropped to: {data.shape}")
        
        # Handle NaN values
        data = np.nan_to_num(data, nan=0.0)
        
        # Convert to float32
        data = np.asarray(data, dtype=np.float32)
        
        # Resize to sidelength using PIL
        img = Image.fromarray(data)
        img = img.resize((sidelength, sidelength), Image.BILINEAR)
        data = np.array(img)
        
        # Convert to tensor [1, H, W]
        data_tensor = torch.from_numpy(data).unsqueeze(0)
        
        # Normalize to [-1, 1]
        data_min = data_tensor.min()
        data_max = data_tensor.max()
        if data_max > data_min:
            data_tensor = 2 * (data_tensor - data_min) / (data_max - data_min) - 1
        
        return data_tensor


class RadioImageFitting(Dataset):
    """
    Dataset for fitting a single-channel image tensor shaped [1, H, W].
    Returns (coords, pixels) for all pixels in the image.
    """
    def __init__(self, image_tensor: torch.Tensor):
        super().__init__()
        if not isinstance(image_tensor, torch.Tensor):
            raise TypeError("image_tensor must be a torch.Tensor")
        if image_tensor.ndim != 3 or image_tensor.shape[0] != 1:
            raise ValueError(f"Expected image_tensor shape [1, H, W], got {tuple(image_tensor.shape)}")
        
        _, H, W = image_tensor.shape
        if H != W:
            raise ValueError(f"Expected square image, got H={H}, W={W}")
        
        # pixels: [H*W, 1]
        self.pixels = image_tensor.permute(1, 2, 0).contiguous().view(-1, 1)
        # coords: [H*W, 2]
        self.coords = get_mgrid(H, 2)
    
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError
        return self.coords, self.pixels

# ============================================================================
# TRAINING
# ============================================================================

def train_siren(device, image_tensor, args):
    """
    Train SIREN to fit an image with mini-batch gradient accumulation
    """
    print("\n" + "="*80)
    print("TRAINING: SIREN Image Fitting")
    print("="*80)
    
    # Create dataset
    dataset = RadioImageFitting(image_tensor)
    dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=0)
    
    # Create model
    img_siren = Siren(
        in_features=2, 
        out_features=1, 
        hidden_features=args.hidden_features,
        hidden_layers=args.hidden_layers, 
        outermost_linear=True
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in img_siren.parameters()):,}")
    print(f"Hidden features: {args.hidden_features}")
    print(f"Hidden layers: {args.hidden_layers}")
    
    # Setup optimizer
    optim = torch.optim.Adam(lr=args.lr, params=img_siren.parameters())
    
    # Get full data
    model_input_full, ground_truth_full = next(iter(dataloader))
    model_input_full = model_input_full.to(device)
    ground_truth_full = ground_truth_full.to(device)
    
    # Determine sidelength
    sidelength = int(np.sqrt(model_input_full.shape[1]))
    print(f"Image size: {sidelength}x{sidelength}")
    
    # Mini-batch parameters
    num_pixels = model_input_full.shape[1]
    num_batches = (num_pixels + args.batch_size_pixels - 1) // args.batch_size_pixels
    
    print(f"Training for {args.steps} steps with mini-batches of {args.batch_size_pixels} pixels")
    print(f"Number of mini-batches per step: {num_batches}")
    
    # Training loop
    for step in range(args.steps):
        optim.zero_grad()
        current_loss = 0.0
        
        # Process in mini-batches
        for i in range(num_batches):
            start_idx = i * args.batch_size_pixels
            end_idx = min((i + 1) * args.batch_size_pixels, num_pixels)
            
            # Extract batch
            batch_model_input = model_input_full[:, start_idx:end_idx, :]
            batch_ground_truth = ground_truth_full[:, start_idx:end_idx, :]
            
            # Forward pass
            model_output_batch, _ = img_siren(batch_model_input)
            
            # Calculate loss
            loss = ((model_output_batch - batch_ground_truth)**2).mean()
            
            # Accumulate gradients
            loss.backward()
            
            current_loss += loss.item() * (end_idx - start_idx)
        
        # Update parameters
        optim.step()
        current_loss /= num_pixels
        
        if step % args.steps_til_summary == 0:
            print(f"Step {step:5d}, Loss {current_loss:.6f}")
    
    print(f"\n✓ Training complete. Final loss: {current_loss:.6f}")
    
    return img_siren, model_input_full, ground_truth_full, sidelength

# ============================================================================
# VISUALIsATION
# ============================================================================

def visualise_results(img_siren, model_input_full, ground_truth_full, sidelength, 
                     device, output_dir, batch_size_pixels=2048):
    """
    Generate comprehensive visualisation of SIREN results
    """
    print("\n" + "="*80)
    print("GENERATING visualisationS")
    print("="*80)
    
    num_pixels = model_input_full.shape[1]
    num_batches = (num_pixels + batch_size_pixels - 1) // batch_size_pixels
    
    # Initialize output tensors
    full_model_output = torch.empty_like(ground_truth_full)
    full_grad_output = torch.zeros(ground_truth_full.shape[0], ground_truth_full.shape[1], 2, 
                                   device=ground_truth_full.device)
    full_laplace_output = torch.empty_like(ground_truth_full)
    
    # Process in batches
    print(f"Processing {num_batches} batches for visualisation...")
    for i in range(num_batches):
        start_idx = i * batch_size_pixels
        end_idx = min((i + 1) * batch_size_pixels, num_pixels)
        
        batch_model_input = model_input_full[:, start_idx:end_idx, :]
        batch_output, coords_for_grad = img_siren(batch_model_input)
        
        # Calculate gradients and Laplacian
        img_grad = gradient(batch_output, coords_for_grad)
        img_laplacian = laplace(batch_output, coords_for_grad)
        
        # Store results
        full_model_output[:, start_idx:end_idx, :] = batch_output.detach()
        full_grad_output[:, start_idx:end_idx, :] = img_grad.detach()
        full_laplace_output[:, start_idx:end_idx, :] = img_laplacian.detach()
    
    # Create figure with 5 subplots
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    # Convert to numpy for plotting
    ground_truth_np = ground_truth_full.cpu().view(sidelength, sidelength).detach().numpy()
    reconstructed_np = full_model_output.cpu().view(sidelength, sidelength).detach().numpy()
    grad_magnitude_np = full_grad_output.norm(dim=-1).cpu().view(sidelength, sidelength).detach().numpy()
    laplacian_np = full_laplace_output.cpu().view(sidelength, sidelength).detach().numpy()
    residual_np = (ground_truth_full - full_model_output).cpu().view(sidelength, sidelength).detach().numpy()
    
    # Plot Original Image
    c0 = axes[0].imshow(ground_truth_np, cmap='magma')
    axes[0].set_title("Original Image", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(c0, ax=axes[0], shrink=0.8)
    
    # Plot Reconstructed Image
    c1 = axes[1].imshow(reconstructed_np, cmap='magma')
    axes[1].set_title("Reconstructed Image", fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(c1, ax=axes[1], shrink=0.8)
    
    # Plot Gradient Magnitude
    c2 = axes[2].imshow(grad_magnitude_np, cmap='magma')
    axes[2].set_title("Gradient Magnitude", fontsize=12, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(c2, ax=axes[2], shrink=0.8)
    
    # Plot Laplacian
    c3 = axes[3].imshow(laplacian_np, cmap='seismic', norm=colors.CenteredNorm())
    axes[3].set_title("Laplacian", fontsize=12, fontweight='bold')
    axes[3].axis('off')
    plt.colorbar(c3, ax=axes[3], shrink=0.8)
    
    # Plot Residual Error
    c4 = axes[4].imshow(residual_np, cmap='seismic', norm=colors.CenteredNorm())
    axes[4].set_title("Residual Error", fontsize=12, fontweight='bold')
    axes[4].axis('off')
    plt.colorbar(c4, ax=axes[4], shrink=0.8)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'siren_results.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ visualisation saved to: {output_path}")
    
    # Calculate and print metrics
    mse = ((ground_truth_full - full_model_output)**2).mean().item()
    psnr = 10 * np.log10(4.0 / mse)  # Range is [-1, 1], so max value is 2, max^2 = 4
    
    print(f"\nMetrics:")
    print(f"  MSE:  {mse:.6f}")
    print(f"  PSNR: {psnr:.2f} dB")

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='SIREN Image Fitting for ILIFU')
    
    # Data arguments
    parser.add_argument('--fits_file', type=str, default=None,
                        help='Path to FITS file (optional, uses cameraman if not provided)')
    parser.add_argument('--crop_size', type=int, default=1500,
                        help='Size of crop to extract from FITS file')
    parser.add_argument('--sidelength', type=int, default=256,
                        help='Size to resize image to')
    parser.add_argument('--center_x', type=int, default=None,
                        help='X coordinate for crop center (default: image center)')
    parser.add_argument('--center_y', type=int, default=None,
                        help='Y coordinate for crop center (default: image center)')
    
    # Model arguments
    parser.add_argument('--hidden_features', type=int, default=256,
                        help='Number of hidden features in SIREN')
    parser.add_argument('--hidden_layers', type=int, default=3,
                        help='Number of hidden layers in SIREN')
    
    # Training arguments
    parser.add_argument('--steps', type=int, default=500,
                        help='Number of training steps')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--batch_size_pixels', type=int, default=2048,
                        help='Number of pixels per mini-batch')
    parser.add_argument('--steps_til_summary', type=int, default=20,
                        help='Steps between progress prints')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./siren_results',
                        help='Output directory for results')
    parser.add_argument('--save_model', action='store_true',
                        help='Save trained model weights')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nOutput directory: {args.output_dir}")
    
    # Setup device
    device = setup_device()
    
    # Load image
    print("\n" + "="*80)
    print("LOADING IMAGE")
    print("="*80)
    
    if args.fits_file and os.path.exists(args.fits_file):
        print(f"Loading FITS file: {args.fits_file}")
        image_tensor = get_fits_square_crop(
            args.fits_file,
            crop_size=args.crop_size,
            sidelength=args.sidelength,
            center_x=args.center_x,
            center_y=args.center_y
        )
    else:
        if args.fits_file:
            print(f"Warning: FITS file not found: {args.fits_file}")
        print("Using cameraman image")
        image_tensor = get_cameraman_tensor(args.sidelength)
    
    print(f"Image tensor shape: {image_tensor.shape}")
    
    # Train model
    img_siren, model_input_full, ground_truth_full, sidelength = train_siren(
        device, image_tensor, args
    )
    
    # Generate visualisations
    visualise_results(
        img_siren, model_input_full, ground_truth_full, sidelength,
        device, args.output_dir, args.batch_size_pixels
    )
    
    # Save model if requested
    if args.save_model:
        model_path = os.path.join(args.output_dir, 'siren_model.pth')
        torch.save(img_siren.state_dict(), model_path)
        print(f"\n✓ Model saved to: {model_path}")
    
    print("\n" + "="*80)
    print("✓ DONE AND WINNING :)")
    print(f"✓ Results saved to: {args.output_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
