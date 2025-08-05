"""
Gaussian Process models for CO2 reduction predictions.
Implements both standard GP and physics-informed hybrid GP models.
"""
import torch
import gpytorch
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pathlib import Path

# Import our physics model for hybrid GP
from .physics_model import PhModel, PhysicsConfig


@dataclass
class GPConfig:
    """Configuration for Gaussian Process models."""
    num_tasks: int = 2  # FE_Eth and FE_CO
    kernel_type: str = "rbf"  # rbf, matern, linear
    rank: int = 1  # For MultitaskKernel
    learning_rate: float = 0.1
    num_iterations: int = 400


class MultitaskGPModel(gpytorch.models.ExactGP):
    """
    Standard Multi-task Gaussian Process model.

    Uses a constant mean function and RBF kernel to learn the relationship
    between experimental inputs and Faradaic Efficiencies directly.
    """

    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor,
                 likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
                 config: GPConfig = None):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)

        self.config = config or GPConfig()

        # Mean function - constant mean for each task
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(),
            num_tasks=self.config.num_tasks
        )

        # Covariance function - RBF kernel with multi-task structure
        base_kernel = self._get_base_kernel()
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            base_kernel,
            num_tasks=self.config.num_tasks,
            rank=self.config.rank
        )

    def _get_base_kernel(self):
        """Get the base kernel based on configuration."""
        if self.config.kernel_type == "rbf":
            return gpytorch.kernels.RBFKernel()
        elif self.config.kernel_type == "matern":
            return gpytorch.kernels.MaternKernel()
        elif self.config.kernel_type == "linear":
            return gpytorch.kernels.LinearKernel()
        else:
            raise ValueError(f"Unknown kernel type: {self.config.kernel_type}")

    def forward(self, x: torch.Tensor):
        """Forward pass through the GP."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class PhysicsInformedMean(gpytorch.means.Mean):
    """
    Custom mean function that uses a PhModel for physics-informed predictions.

    This allows the GP to learn residuals/corrections on top of physics-based predictions.
    """

    def __init__(self, ph_model: Optional[PhModel] = None,
                 zlt_mu_stds: Optional[Tuple[float, float]] = None,
                 freeze_model: bool = False):
        super().__init__()

        if ph_model is not None:
            self.ph_model = ph_model
        else:
            # Create default PhModel if none provided
            ph_config = PhysicsConfig(current_target=233)  # Match original
            self.ph_model = PhModel(config=ph_config, zlt_mu_stds=zlt_mu_stds)

        if freeze_model:
            self._freeze_physics_model()

    def _freeze_physics_model(self):
        """Freeze the physics model parameters and disable dropout."""
        # Freeze all parameters
        for param in self.ph_model.parameters():
            param.requires_grad = False

        # Disable dropout by setting p=0
        def remove_dropout(module):
            for child in module.children():
                if isinstance(child, torch.nn.Dropout):
                    child.p = 0
                else:
                    remove_dropout(child)

        remove_dropout(self.ph_model)

    def forward(self, x: torch.Tensor):
        """Forward pass using physics model."""
        # Ensure physics model is in eval mode for consistent predictions
        self.ph_model.eval()
        return self.ph_model(x)


class MultitaskGPhysModel(gpytorch.models.ExactGP):
    """
    Hybrid Physics-Informed Gaussian Process model.

    Uses a PhModel as the mean function and a GP to learn residuals/uncertainties
    on top of the physics-based predictions.
    """

    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor,
                 likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
                 ph_model: Optional[PhModel] = None,
                 zlt_mu_stds: Optional[Tuple[float, float]] = None,
                 freeze_model: bool = False,
                 config: GPConfig = None):
        super(MultitaskGPhysModel, self).__init__(train_x, train_y, likelihood)

        self.config = config or GPConfig()

        # Physics-informed mean function
        self.mean_module = PhysicsInformedMean(
            ph_model=ph_model,
            zlt_mu_stds=zlt_mu_stds,
            freeze_model=freeze_model
        )

        # Covariance function - same as standard GP
        base_kernel = self._get_base_kernel()
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            base_kernel,
            num_tasks=self.config.num_tasks,
            rank=self.config.rank
        )

    def _get_base_kernel(self):
        """Get the base kernel based on configuration."""
        if self.config.kernel_type == "rbf":
            return gpytorch.kernels.RBFKernel()
        elif self.config.kernel_type == "matern":
            return gpytorch.kernels.MaternKernel()
        elif self.config.kernel_type == "linear":
            return gpytorch.kernels.LinearKernel()
        else:
            raise ValueError(f"Unknown kernel type: {self.config.kernel_type}")

    def forward(self, x: torch.Tensor):
        """Forward pass through the hybrid GP."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


def train_gp_model(X_train: torch.Tensor, y_train: torch.Tensor,
                   model_type: str = "standard",
                   config: GPConfig = None,
                   ph_model: Optional[PhModel] = None,
                   zlt_mu_stds: Optional[Tuple[float, float]] = None,
                   freeze_physics: bool = False,
                   verbose: bool = True) -> Tuple[gpytorch.models.ExactGP, gpytorch.likelihoods.MultitaskGaussianLikelihood, Dict]:
    """
    Train a Gaussian Process model.

    Args:
        X_train: Training features (N, 5) - normalized
        y_train: Training targets (N, 2) - FE values in [0,1]
        model_type: "standard" for MultitaskGPModel or "physics" for MultitaskGPhysModel
        config: GP configuration
        ph_model: Pre-trained PhModel for physics-informed GP (optional)
        zlt_mu_stds: Zero layer thickness normalization parameters
        freeze_physics: Whether to freeze the physics model parameters
        verbose: Whether to print training progress

    Returns:
        Trained GP model, likelihood, and training statistics
    """
    config = config or GPConfig()

    # Validate inputs
    assert X_train.ndim == 2 and X_train.shape[1] == 5, f"Expected X shape (N, 5), got {X_train.shape}"
    assert y_train.ndim == 2 and y_train.shape[1] == 2, f"Expected y shape (N, 2), got {y_train.shape}"

    # Set up likelihood
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=config.num_tasks)

    # Create model based on type
    if model_type == "standard":
        model = MultitaskGPModel(X_train, y_train, likelihood, config)
    elif model_type == "physics":
        model = MultitaskGPhysModel(
            X_train, y_train, likelihood,
            ph_model=ph_model,
            zlt_mu_stds=zlt_mu_stds,
            freeze_model=freeze_physics,
            config=config
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Set up optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Training loop
    model.train()
    likelihood.train()

    training_stats = {'losses': [], 'iterations': list(range(config.num_iterations))}

    if verbose:
        print(f"Training {model_type} GP model for {config.num_iterations} iterations...")

    for iteration in range(config.num_iterations):
        optimizer.zero_grad()

        # Forward pass
        output = model(X_train)

        # Calculate loss (negative marginal log likelihood)
        loss = -mll(output, y_train)

        # Backward pass
        loss.backward()
        optimizer.step()

        training_stats['losses'].append(loss.item())

        # Print progress
        if verbose and (iteration % 100 == 0 or iteration == config.num_iterations - 1):
            print(f"  Iteration {iteration:3d}/{config.num_iterations} - Loss: {loss.item():.6f}")

    # Set to eval mode
    model.eval()
    likelihood.eval()

    if verbose:
        final_loss = training_stats['losses'][-1]
        print(f"GP training complete. Final loss: {final_loss:.6f}")

    return model, likelihood, training_stats


def predict_with_gp(model: gpytorch.models.ExactGP,
                   likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
                   X_test: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Make predictions with a trained GP model.

    Args:
        model: Trained GP model
        likelihood: Trained likelihood
        X_test: Test features (N, 5) - normalized

    Returns:
        mean_pred: Mean predictions (N, 2)
        std_pred: Standard deviations (N, 2)
    """
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = model(X_test)
        pred_dist = likelihood(posterior)

        mean_pred = pred_dist.mean  # (N, 2)
        std_pred = pred_dist.stddev  # (N, 2)

    return mean_pred, std_pred
