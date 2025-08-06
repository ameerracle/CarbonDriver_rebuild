"""
PhModel Ensemble for physics-informed predictions with uncertainty.
Uses the same ensemble approach as MLP but with physics-informed models.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import copy
from .physics_model import PhModel, PhysicsConfig

# Add BoTorch imports for proper integration
try:
    from botorch.models.ensemble import EnsembleModel
    from botorch.posteriors import Posterior
    from gpytorch.distributions import MultitaskMultivariateNormal
    BOTORCH_AVAILABLE = True
except ImportError:
    BOTORCH_AVAILABLE = False
    print("Warning: BoTorch not available. Install with 'pip install botorch'")


@dataclass
class PhEnsembleConfig:
    """Configuration for PhModel ensemble."""
    ensemble_size: int = 50
    hidden_dim: int = 64
    dropout_rate: float = 0.1
    learning_rate: float = 0.001
    bootstrap_fraction: float = 0.5  # Fraction of data each model sees
    current_target: float = 200.0    # Target current density [A/m^2]
    grid_size: int = 1000
    voltage_bounds: Tuple[float, float] = (-1.25, 0.0)


class PhModelEnsemble:
    """
    Ensemble of Physics-Informed Models for uncertainty-aware predictions.

    Combines the ensemble training approach (like MLP) with physics-informed models.
    Each model in the ensemble is a complete PhModel with the full physics engine.
    """

    def __init__(self, config: PhEnsembleConfig = None, zlt_mu_stds: Optional[Tuple[float, float]] = None):
        self.config = config or PhEnsembleConfig()
        self.zlt_mu_stds = zlt_mu_stds or (5e-6, 1e-6)  # Default ZLT normalization
        self.models: List[PhModel] = []
        self.is_trained = False

        # BoTorch compatibility attributes
        self.num_outputs = 2  # FE_Eth and FE_CO
        self._num_outputs = 2

        # Create ensemble of PhModels
        for _ in range(self.config.ensemble_size):
            ph_config = PhysicsConfig(
                hidden_dim=self.config.hidden_dim,
                dropout_rate=self.config.dropout_rate,
                current_target=self.config.current_target,
                grid_size=self.config.grid_size,
                voltage_bounds=self.config.voltage_bounds
            )
            model = PhModel(config=ph_config, zlt_mu_stds=self.zlt_mu_stds)
            self.models.append(model)

    def _create_bootstrap_data(self, X: torch.Tensor, y: torch.Tensor,
                              model_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create bootstrap sample for a specific model using sampling WITH replacement."""
        n_samples = int(len(X) * self.config.bootstrap_fraction)

        # Use model index as seed for reproducible bootstrapping
        generator = torch.Generator().manual_seed(model_idx)
        # Sample WITH replacement (true bootstrapping)
        indices = torch.randint(0, len(X), (n_samples,), generator=generator)

        return X[indices], y[indices]

    def train(self, X_train: torch.Tensor, y_train: torch.Tensor,
              num_epochs: int = 400, verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the ensemble of PhModels.

        Args:
            X_train: Training features (N, 5) - normalized
            y_train: Training targets (N, 2) - FE values in [0,1]
            num_epochs: Number of training epochs
            verbose: Whether to print training progress

        Returns:
            Dictionary with training statistics
        """
        # Validate inputs
        assert X_train.ndim == 2 and X_train.shape[1] == 5, f"Expected X shape (N, 5), got {X_train.shape}"
        assert y_train.ndim == 2 and y_train.shape[1] == 2, f"Expected y shape (N, 2), got {y_train.shape}"
        assert len(X_train) == len(y_train), "X and y must have same number of samples"

        training_stats = {'losses': [], 'model_losses': [[] for _ in range(self.config.ensemble_size)]}

        # Train each PhModel in the ensemble
        for model_idx, model in enumerate(self.models):
            if verbose:
                print(f"Training PhModel {model_idx + 1}/{self.config.ensemble_size}")

            # Create bootstrap sample for this model
            X_boot, y_boot = self._create_bootstrap_data(X_train, y_train, model_idx)

            # Setup optimizer and loss
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
            criterion = nn.MSELoss()

            model.train()
            model_losses = []

            for epoch in range(num_epochs):
                optimizer.zero_grad()

                predictions = model(X_boot)
                loss = criterion(predictions, y_boot)

                loss.backward()
                optimizer.step()

                model_losses.append(loss.item())

                # Print progress occasionally
                if verbose and epoch % 100 == 0:
                    print(f"  Epoch {epoch}, Loss: {loss.item():.6f}")

            training_stats['model_losses'][model_idx] = model_losses

        self.is_trained = True

        # Calculate ensemble training loss
        if verbose:
            final_losses = [losses[-1] for losses in training_stats['model_losses']]
            avg_loss = np.mean(final_losses)
            print(f"PhModel ensemble training complete. Average final loss: {avg_loss:.6f}")

        return training_stats

    def predict(self, X: torch.Tensor, return_std: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Make predictions with the PhModel ensemble.

        Args:
            X: Input features (N, 5) - normalized
            return_std: Whether to return uncertainty estimates

        Returns:
            mean_pred: Mean predictions (N, 2)
            std_pred: Standard deviations (N, 2) if return_std=True
        """
        assert self.is_trained, "PhModel ensemble must be trained before making predictions"
        assert X.ndim == 2 and X.shape[1] == 5, f"Expected input shape (N, 5), got {X.shape}"

        # Collect predictions from all PhModels
        all_predictions = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(X)
                all_predictions.append(pred)

        # Stack predictions: (ensemble_size, N, 2)
        predictions = torch.stack(all_predictions, dim=0)

        # Calculate statistics
        mean_pred = predictions.mean(dim=0)  # (N, 2)

        if return_std:
            std_pred = predictions.std(dim=0)  # (N, 2)
            return mean_pred, std_pred
        else:
            return mean_pred, None

    def get_prediction_samples(self, X: torch.Tensor, n_samples: int = 1000) -> torch.Tensor:
        """
        Get prediction samples for uncertainty quantification.

        Args:
            X: Input features (N, 5)
            n_samples: Number of samples to draw

        Returns:
            Samples of shape (n_samples, N, 2)
        """
        assert self.is_trained, "PhModel ensemble must be trained before sampling"

        # For an ensemble, we can sample by randomly selecting models
        samples = []

        for _ in range(n_samples):
            # Randomly select a model from the ensemble
            model_idx = torch.randint(0, len(self.models), (1,)).item()
            model = self.models[model_idx]

            model.eval()
            with torch.no_grad():
                pred = model(X)
                samples.append(pred)

        return torch.stack(samples, dim=0)  # (n_samples, N, 2)

    def get_botorch_posterior(self, X: torch.Tensor) -> 'Posterior':
        """
        Get BoTorch-compatible posterior for the PhModel ensemble.

        Args:
            X: Input features (batch_size, 5) - normalized

        Returns:
            BoTorch Posterior object with correct tensor dimensions
        """
        if not BOTORCH_AVAILABLE:
            raise ImportError("BoTorch not available. Install with 'pip install botorch'")

        assert self.is_trained, "PhModel ensemble must be trained before getting posterior"
        assert X.ndim == 2 and X.shape[1] == 5, f"Expected input shape (N, 5), got {X.shape}"

        # Get ensemble predictions: (ensemble_size, batch_size, 2)
        all_predictions = []
        for model in self.models:
            model.eval()
            # Remove torch.no_grad() to enable gradients for optimization
            pred = model(X)  # (batch_size, 2)
            all_predictions.append(pred)

        predictions = torch.stack(all_predictions, dim=0)  # (ensemble_size, batch_size, 2)

        # Calculate mean and covariance
        mean = predictions.mean(dim=0)  # (batch_size, 2)
        mean = mean.unsqueeze(1)  # (batch_size, 1, 2) for BoTorch

        # Calculate covariance across ensemble
        batch_size = X.shape[0]
        covariance_matrices = []

        for b in range(batch_size):
            # Get predictions for this batch element across all ensemble members
            batch_preds = predictions[:, b, :]  # (ensemble_size, 2)
            # Calculate covariance matrix
            cov = torch.cov(batch_preds.T)  # (2, 2)
            # Add small diagonal for numerical stability
            cov = cov + 1e-6 * torch.eye(2)
            covariance_matrices.append(cov)

        # Stack covariance matrices: (batch_size, 2, 2)
        covariance = torch.stack(covariance_matrices, dim=0)

        # Create MultitaskMultivariateNormal distribution
        mvn = MultitaskMultivariateNormal(mean, covariance)

        # Return as BoTorch Posterior
        from botorch.posteriors.gpytorch import GPyTorchPosterior
        return GPyTorchPosterior(mvn)

    def forward(self, X: torch.Tensor) -> 'Posterior':
        """BoTorch-style forward method."""
        return self.get_botorch_posterior(X)

    def get_prediction_samples_botorch(self, X: torch.Tensor, n_samples: int = 1000) -> torch.Tensor:
        """
        Get prediction samples in BoTorch format.

        Args:
            X: Input features (batch_size, 5)
            n_samples: Number of samples to draw

        Returns:
            Samples of shape (n_samples, batch_size, 1, 2) for BoTorch compatibility
        """
        # Get samples in our format: (n_samples, batch_size, 2)
        samples = self.get_prediction_samples(X, n_samples)

        # Reshape for BoTorch: (n_samples, batch_size, 1, 2)
        return samples.unsqueeze(2)  # Add task dimension

    def save(self, filepath: str):
        """Save the trained PhModel ensemble."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained PhModel ensemble")

        state_dict = {
            'config': self.config,
            'zlt_mu_stds': self.zlt_mu_stds,
            'model_states': [model.state_dict() for model in self.models],
            'is_trained': self.is_trained
        }
        torch.save(state_dict, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'PhModelEnsemble':
        """Load a trained PhModel ensemble."""
        state_dict = torch.load(filepath)

        ensemble = cls(config=state_dict['config'], zlt_mu_stds=state_dict['zlt_mu_stds'])

        for model, model_state in zip(ensemble.models, state_dict['model_states']):
            model.load_state_dict(model_state)

        ensemble.is_trained = state_dict['is_trained']
        return ensemble
