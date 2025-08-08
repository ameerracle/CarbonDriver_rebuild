"""
Gaussian Process Ensemble implementations.
Provides both standard GP and physics-informed GP ensembles.
"""
import torch
import gpytorch
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import copy
import logging

from models.gp_model import MultitaskGPModel, MultitaskGPhysModel, GPConfig
from models.physics_model import PhModel, PhysicsConfig

logger = logging.getLogger(__name__)


@dataclass
class GPEnsembleConfig:
    """Configuration for GP ensemble."""
    ensemble_size: int = 10  # Smaller than MLP ensembles due to computational cost
    bootstrap_fraction: float = 0.8
    gp_config: GPConfig = None

    def __post_init__(self):
        if self.gp_config is None:
            self.gp_config = GPConfig()


class GPEnsemble:
    """
    Ensemble of standard Gaussian Process models.
    Each GP in the ensemble is trained on a bootstrap sample of the data.
    """

    def __init__(self, config: GPEnsembleConfig):
        self.config = config
        self.models: List[MultitaskGPModel] = []
        self.likelihoods: List[gpytorch.likelihoods.MultitaskGaussianLikelihood] = []
        self.is_trained = False

    def train(self, X: torch.Tensor, y: torch.Tensor, num_epochs: int = 400, verbose: bool = False):
        """Train the GP ensemble using bootstrap sampling."""
        self.models = []
        self.likelihoods = []

        n_samples = X.shape[0]
        bootstrap_size = int(n_samples * self.config.bootstrap_fraction)

        for i in range(self.config.ensemble_size):
            if verbose and i % 5 == 0:
                logger.info(f"Training GP {i+1}/{self.config.ensemble_size}")

            # Bootstrap sampling
            indices = torch.randint(0, n_samples, (bootstrap_size,))
            X_boot = X[indices]
            y_boot = y[indices]

            # Create likelihood and model
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                num_tasks=self.config.gp_config.num_tasks
            )
            model = MultitaskGPModel(X_boot, y_boot, likelihood, self.config.gp_config)

            # Train the GP
            self._train_single_gp(model, likelihood, X_boot, y_boot, num_epochs)

            # Store trained model and likelihood
            model.eval()
            likelihood.eval()
            self.models.append(model)
            self.likelihoods.append(likelihood)

        self.is_trained = True

    def _train_single_gp(self, model, likelihood, X, y, num_epochs):
        """Train a single GP model with improved numerical stability."""
        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.gp_config.learning_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for epoch in range(num_epochs):
            try:
                optimizer.zero_grad()

                # Check for NaN or infinite values in inputs
                if torch.isnan(X).any() or torch.isinf(X).any():
                    logger.warning("NaN or infinite values detected in training inputs")
                    break

                if torch.isnan(y).any() or torch.isinf(y).any():
                    logger.warning("NaN or infinite values detected in training targets")
                    break

                # Add jitter for numerical stability
                with gpytorch.settings.cholesky_jitter(1e-4):
                    output = model(X)
                    loss = -mll(output, y)

                    # Check for NaN loss
                    if torch.isnan(loss):
                        logger.warning(f"NaN loss detected at epoch {epoch}, stopping training")
                        break

                    loss.backward()

                    # Clip gradients to prevent explosion
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()

            except RuntimeError as e:
                if "cholesky" in str(e).lower():
                    logger.warning(f"Cholesky decomposition failed at epoch {epoch}: {e}")
                    # Try with more jitter
                    try:
                        with gpytorch.settings.cholesky_jitter(1e-3):
                            output = model(X)
                            loss = -mll(output, y)
                            if not torch.isnan(loss):
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                                optimizer.step()
                    except RuntimeError:
                        logger.warning("Training failed even with increased jitter, stopping")
                        break
                else:
                    raise e

    def predict(self, X: torch.Tensor, return_std: bool = False) -> torch.Tensor:
        """
        Make predictions using the ensemble.
        Returns mean if return_std=False, else (mean, std).
        """
        if not self.is_trained:
            raise RuntimeError("Ensemble must be trained before making predictions")

        predictions = []

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for model, likelihood in zip(self.models, self.likelihoods):
                model.eval()
                likelihood.eval()
                pred = likelihood(model(X))
                predictions.append(pred.mean)

        # Stack predictions and compute ensemble statistics
        pred_stack = torch.stack(predictions)  # [ensemble_size, n_samples, n_tasks]
        mean = pred_stack.mean(dim=0)
        std = pred_stack.std(dim=0)

        if return_std:
            return mean, std
        else:
            return mean


class PhysicsGPEnsemble:
    """
    Ensemble of physics-informed Gaussian Process models.
    Each GP uses a PhModel as the mean function and learns residuals.
    """

    def __init__(self, config: GPEnsembleConfig, zlt_mu_stds: Optional[Tuple[float, float]] = None):
        self.config = config
        self.zlt_mu_stds = zlt_mu_stds or (5e-6, 1e-6)
        self.models: List[MultitaskGPhysModel] = []
        self.likelihoods: List[gpytorch.likelihoods.MultitaskGaussianLikelihood] = []
        self.is_trained = False

    def train(self, X: torch.Tensor, y: torch.Tensor, num_epochs: int = 400, verbose: bool = False):
        """Train the physics-informed GP ensemble using bootstrap sampling."""
        self.models = []
        self.likelihoods = []

        n_samples = X.shape[0]
        bootstrap_size = int(n_samples * self.config.bootstrap_fraction)

        for i in range(self.config.ensemble_size):
            if verbose and i % 5 == 0:
                logger.info(f"Training Physics-GP {i+1}/{self.config.ensemble_size}")

            # Bootstrap sampling
            indices = torch.randint(0, n_samples, (bootstrap_size,))
            X_boot = X[indices]
            y_boot = y[indices]

            # Create a fresh PhModel for each ensemble member
            ph_config = PhysicsConfig(current_target=233)
            ph_model = PhModel(config=ph_config, zlt_mu_stds=self.zlt_mu_stds)

            # Create likelihood and model
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                num_tasks=self.config.gp_config.num_tasks
            )
            model = MultitaskGPhysModel(
                X_boot, y_boot, likelihood,
                ph_model=ph_model,
                zlt_mu_stds=self.zlt_mu_stds,
                freeze_model=False,  # Allow physics model to adapt
                config=self.config.gp_config
            )

            # Train the hybrid model
            self._train_single_gp(model, likelihood, X_boot, y_boot, num_epochs)

            # Store trained model and likelihood
            model.eval()
            likelihood.eval()
            self.models.append(model)
            self.likelihoods.append(likelihood)

        self.is_trained = True

    def _train_single_gp(self, model, likelihood, X, y, num_epochs):
        """Train a single physics-informed GP model with improved numerical stability."""
        model.train()
        likelihood.train()

        # Include physics model parameters in optimization
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.gp_config.learning_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for epoch in range(num_epochs):
            try:
                optimizer.zero_grad()

                # Check for NaN or infinite values in inputs
                if torch.isnan(X).any() or torch.isinf(X).any():
                    logger.warning("NaN or infinite values detected in training inputs")
                    break

                if torch.isnan(y).any() or torch.isinf(y).any():
                    logger.warning("NaN or infinite values detected in training targets")
                    break

                # Add jitter for numerical stability
                with gpytorch.settings.cholesky_jitter(1e-4):
                    output = model(X)
                    loss = -mll(output, y)

                    # Check for NaN loss
                    if torch.isnan(loss):
                        logger.warning(f"NaN loss detected at epoch {epoch}, stopping training")
                        break

                    loss.backward()

                    # Clip gradients to prevent explosion
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()

            except RuntimeError as e:
                if "cholesky" in str(e).lower():
                    logger.warning(f"Cholesky decomposition failed at epoch {epoch}: {e}")
                    # Try with more jitter
                    try:
                        with gpytorch.settings.cholesky_jitter(1e-3):
                            output = model(X)
                            loss = -mll(output, y)
                            if not torch.isnan(loss):
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                                optimizer.step()
                    except RuntimeError:
                        logger.warning("Training failed even with increased jitter, stopping")
                        break
                else:
                    raise e

    def predict(self, X: torch.Tensor, return_std: bool = False) -> torch.Tensor:
        """
        Make predictions using the physics-informed ensemble.
        Returns mean if return_std=False, else (mean, std).
        """
        if not self.is_trained:
            raise RuntimeError("Ensemble must be trained before making predictions")

        predictions = []

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for model, likelihood in zip(self.models, self.likelihoods):
                model.eval()
                likelihood.eval()
                pred = likelihood(model(X))
                predictions.append(pred.mean)

        # Stack predictions and compute ensemble statistics
        pred_stack = torch.stack(predictions)  # [ensemble_size, n_samples, n_tasks]
        mean = pred_stack.mean(dim=0)
        std = pred_stack.std(dim=0)

        if return_std:
            return mean, std
        else:
            return mean
