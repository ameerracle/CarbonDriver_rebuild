"""
Unified optimizer interface for all model types (MLP, PhModel, GP, GP+Ph).
Provides both within-data and free optimization capabilities.
"""
import torch
import pandas as pd
import numpy as np
import os
from typing import Optional, Tuple, Union
from data.loader import load_data
from models.mlp_ensemble import MLPEnsemble, EnsembleConfig
from models.ph_ensemble import PhModelEnsemble, PhEnsembleConfig
from models.gp_model import MultitaskGPModel, MultitaskGPhysModel, GPConfig, train_gp_model
from models.physics_model import PhModel, PhysicsConfig
import gpytorch

# BoTorch imports for acquisition functions
try:
    from botorch.acquisition.analytic import ExpectedImprovement
    from botorch.optim import optimize_acqf
    from botorch.posteriors.gpytorch import GPyTorchPosterior
    from gpytorch.distributions import MultitaskMultivariateNormal
    BOTORCH_AVAILABLE = True
except ImportError:
    BOTORCH_AVAILABLE = False


def get_data_file_path(filename):
    """Return absolute path to a file in the data folder."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', filename))


class SingleOutputEnsembleWrapper:
    """
    Wrapper to make multi-output ensembles work with single-output acquisition functions.
    Extracts a single output dimension for optimization.
    """

    def __init__(self, ensemble, output_index=0):
        self.ensemble = ensemble
        self.output_index = output_index
        self.num_outputs = 1
        self._num_outputs = 1

    def posterior(self, X: torch.Tensor, **kwargs):
        """Get posterior distribution for the specified output dimension."""
        if X.ndim == 3 and X.shape[1] == 1:
            X = X.squeeze(1)

        full_posterior = self.ensemble.get_botorch_posterior(X)
        mean = full_posterior.mean[..., self.output_index:self.output_index+1]

        batch_size = X.shape[0]
        covariance_single = []
        for b in range(batch_size):
            var = full_posterior.distribution.covariance_matrix[b, self.output_index, self.output_index]
            cov_matrix = var.unsqueeze(0).unsqueeze(0)
            covariance_single.append(cov_matrix)

        covariance = torch.stack(covariance_single, dim=0)
        mvn_single = MultitaskMultivariateNormal(mean, covariance)
        return GPyTorchPosterior(mvn_single)

    def forward(self, X: torch.Tensor):
        """Forward pass that returns posterior (for compatibility)."""
        return self.posterior(X)


class SingleOutputGPWrapper:
    """
    Wrapper to make multi-output GP models work with single-output acquisition functions.
    Extracts a single output dimension for optimization.
    """

    def __init__(self, gp_model, likelihood, output_index=0):
        self.gp_model = gp_model
        self.likelihood = likelihood
        self.output_index = output_index
        self.num_outputs = 1
        self._num_outputs = 1

    def posterior(self, X: torch.Tensor, **kwargs):
        """Get posterior distribution for the specified output dimension."""
        if X.ndim == 3 and X.shape[1] == 1:
            X = X.squeeze(1)

        self.gp_model.eval()
        self.likelihood.eval()

        # Enable gradients for optimization - remove torch.no_grad()
        with gpytorch.settings.fast_pred_var():
            gp_posterior = self.gp_model(X)
            full_posterior = self.likelihood(gp_posterior)

            mean = full_posterior.mean[..., self.output_index:self.output_index+1]
            var = full_posterior.variance[..., self.output_index:self.output_index+1]
            mean = mean.unsqueeze(1)

            batch_size = X.shape[0]
            covariance = torch.zeros(batch_size, 1, 1, dtype=mean.dtype, device=mean.device)
            for b in range(batch_size):
                covariance[b, 0, 0] = var[b, 0]

            mvn_single = MultitaskMultivariateNormal(mean, covariance)
            return GPyTorchPosterior(mvn_single)

    def forward(self, X: torch.Tensor):
        """Forward pass that returns posterior (for compatibility)."""
        return self.posterior(X)


class EnsembleOptimizer:
    """
    Unified optimizer for all model types: MLP, PhModel, GP, and GP+Ph.

    Provides both:
    - step_within_data(): Pick from existing data points
    - step(): Optimize over continuous feature space
    """

    def __init__(self, model_type="MLP", quantity="FE (Eth)", maximize=True, output_dir="./out"):
        """
        Initialize the optimizer with the specified model type.

        Args:
            model_type: "MLP", "PhModel", "GP", or "GP+Ph"
            quantity: The quantity to optimize (e.g., 'FE (Eth)')
            maximize: Whether to maximize or minimize the quantity
            output_dir: Directory to save output files
        """
        if not BOTORCH_AVAILABLE:
            raise ImportError("BoTorch required for optimization. Install with: pip install botorch")

        self.model_type = model_type
        self.quantity = quantity
        self.maximize = maximize
        self.output_dir = output_dir
        self.model = None
        self.likelihood = None  # For GP models
        self.is_trained = False

        # Determine output index for single-output wrapper
        self.output_index = 0 if quantity == "FE (Eth)" else 1
        self.best_f = None

    def _train_model(self, df: pd.DataFrame):
        """Train the model on the provided data."""
        data_tensors, norm_params, _ = load_data()

        # Filter to match the provided DataFrame indices
        indices = df.index.tolist()
        train_mask = torch.tensor([i in indices for i in range(len(data_tensors.X))])
        X_train = data_tensors.X[train_mask]
        y_train = data_tensors.y[train_mask]

        if self.model_type == "MLP":
            config = EnsembleConfig(
                ensemble_size=5,
                hidden_dim=64,
                dropout_rate=0.1,
                learning_rate=0.001,
                bootstrap_fraction=0.8
            )
            self.model = MLPEnsemble(config)
            self.model.train(X_train, y_train, num_epochs=100, verbose=False)

        elif self.model_type == "PhModel":
            # Get original ZLT stats for PhModel
            original_data_path = get_data_file_path('Characterization_data.xlsx')
            original_data = pd.read_excel(original_data_path, skiprows=[1], index_col=0)
            original_data = original_data.dropna()

            # Add thickness calculation
            dens_Ag = 10490
            dens_Cu = 8935
            dens_avg = (1 - original_data['AgCu Ratio']) * dens_Cu + original_data['AgCu Ratio'] * dens_Ag
            mass = original_data['Catalyst mass loading'] * 1e-6
            area = 1.85**2
            A = area * 1e-4
            thickness = (mass / dens_avg) / A
            original_data.insert(3, column='Zero_eps_thickness', value=thickness)

            original_zlt_mean = original_data['Zero_eps_thickness'].mean()
            original_zlt_std = original_data['Zero_eps_thickness'].std(ddof=0)

            config = PhEnsembleConfig(
                ensemble_size=5,
                hidden_dim=64,
                dropout_rate=0.1,
                learning_rate=0.001,
                bootstrap_fraction=0.8,
                current_target=200.0,
                grid_size=100
            )
            self.model = PhModelEnsemble(config, zlt_mu_stds=(original_zlt_mean, original_zlt_std))
            self.model.train(X_train, y_train, num_epochs=100, verbose=False)

        elif self.model_type == "GP":
            config = GPConfig(
                num_tasks=2,
                kernel_type="rbf",
                rank=1,
                learning_rate=0.1,
                num_iterations=200
            )
            self.model, self.likelihood, _ = train_gp_model(
                X_train, y_train,
                model_type="standard",
                config=config,
                verbose=False
            )

        elif self.model_type == "GP+Ph":
            # Get original ZLT stats for PhModel
            original_data_path = get_data_file_path('Characterization_data.xlsx')
            original_data = pd.read_excel(original_data_path, skiprows=[1], index_col=0)
            original_data = original_data.dropna()

            dens_Ag = 10490
            dens_Cu = 8935
            dens_avg = (1 - original_data['AgCu Ratio']) * dens_Cu + original_data['AgCu Ratio'] * dens_Ag
            mass = original_data['Catalyst mass loading'] * 1e-6
            area = 1.85**2
            A = area * 1e-4
            thickness = (mass / dens_avg) / A
            original_data.insert(3, column='Zero_eps_thickness', value=thickness)

            original_zlt_mean = original_data['Zero_eps_thickness'].mean()
            original_zlt_std = original_data['Zero_eps_thickness'].std(ddof=0)

            # Pre-train a PhModel for the mean function
            ph_config = PhysicsConfig(
                hidden_dim=64,
                current_target=233,
                grid_size=100
            )
            ph_model = PhModel(config=ph_config, zlt_mu_stds=(original_zlt_mean, original_zlt_std))

            # Quick training of PhModel
            ph_model.train()
            optimizer = torch.optim.Adam(ph_model.parameters(), lr=0.001)
            criterion = torch.nn.MSELoss()

            for epoch in range(50):
                optimizer.zero_grad()
                predictions = ph_model(X_train)
                loss = criterion(predictions, y_train)
                loss.backward()
                optimizer.step()

            config = GPConfig(
                num_tasks=2,
                kernel_type="rbf",
                rank=1,
                learning_rate=0.1,
                num_iterations=200
            )
            self.model, self.likelihood, _ = train_gp_model(
                X_train, y_train,
                model_type="physics",
                config=config,
                ph_model=ph_model,
                zlt_mu_stds=(original_zlt_mean, original_zlt_std),
                freeze_physics=True,
                verbose=False
            )

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.is_trained = True

        # Update best value for acquisition function
        target_col_idx = 0 if self.quantity == "FE (Eth)" else 1
        if self.maximize:
            self.best_f = y_train[:, target_col_idx].max()
        else:
            self.best_f = y_train[:, target_col_idx].min()

    def _get_acquisition_function(self):
        """Get the Expected Improvement acquisition function."""
        # Create appropriate wrapper based on model type
        if self.model_type in ["MLP", "PhModel"]:
            single_output_model = SingleOutputEnsembleWrapper(self.model, self.output_index)
        elif self.model_type in ["GP", "GP+Ph"]:
            single_output_model = SingleOutputGPWrapper(self.model, self.likelihood, self.output_index)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        return ExpectedImprovement(
            model=single_output_model,
            best_f=self.best_f,
            maximize=self.maximize
        )

    def step_within_data(self, df_train: pd.DataFrame, df_explore: pd.DataFrame) -> Tuple[torch.Tensor, int]:
        """
        Perform acquisition function evaluation within the exploration data.

        Args:
            df_train: Training data to fit the model
            df_explore: Exploration data to evaluate acquisition function

        Returns:
            ei_values: Expected Improvement values for exploration points
            next_pick: Index of the best point to pick next
        """
        self._train_model(df_train)

        # Prepare exploration data
        data_tensors, _, full_df = load_data()

        # Get exploration indices and corresponding features
        explore_indices = df_explore.index.tolist()
        explore_mask = torch.tensor([i in explore_indices for i in range(len(data_tensors.X))])
        X_explore = data_tensors.X[explore_mask]

        if len(X_explore) == 0:
            raise ValueError("No exploration points found")

        # Get acquisition function
        acquisition_func = self._get_acquisition_function()

        # Evaluate acquisition function
        with torch.no_grad():
            ei_values = acquisition_func(X_explore.unsqueeze(1))  # Add batch dimension for BoTorch
            ei_values = ei_values.squeeze()  # Remove extra dimensions

        # Find best point
        if self.maximize:
            best_idx = ei_values.argmax().item()
        else:
            best_idx = ei_values.argmin().item()

        # Map back to original DataFrame index
        next_pick = explore_indices[best_idx]

        return ei_values, next_pick

    def step(self, new_data: pd.DataFrame, bounds: Optional[torch.Tensor] = None) -> Tuple[float, np.ndarray]:
        """
        Perform a step in the optimization process using new data and bounds.
        Finds optimal point in continuous feature space (not limited to existing data).

        Args:
            new_data: New data to be added to existing data for training the model
            bounds: Optional bounds for optimization. If None, uses data bounds.

        Returns:
            af_value: Acquisition function value at optimal point
            next_experiment: Optimal feature values as numpy array
        """
        self._train_model(new_data)

        # Get bounds for optimization
        if bounds is None:
            data_tensors, _, _ = load_data()
            X_all = data_tensors.X
            bounds_min = X_all.min(dim=0)[0]
            bounds_max = X_all.max(dim=0)[0]
            # Ensure bounds are float and detached (no gradients needed for bounds)
            bounds = torch.stack([bounds_min.detach(), bounds_max.detach()]).float()

        # Get acquisition function
        acquisition_func = self._get_acquisition_function()

        # Optimize acquisition function over continuous space
        next_experiment, af_value = optimize_acqf(
            acq_function=acquisition_func,
            bounds=bounds,
            q=1,
            num_restarts=10,  # Reduced for faster testing
            raw_samples=20,   # Reduced for faster testing
            options={}
        )

        return af_value.item(), next_experiment.detach().cpu().numpy().flatten()
