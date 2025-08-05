"""
API tests for MLP and PhModel ensembles matching the original test_api.py structure.
Tests the same acquisition function and step methods as the original.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import pandas as pd
import numpy as np
from data.loader import load_data
from models.mlp_ensemble import MLPEnsemble, EnsembleConfig
from models.ph_ensemble import PhModelEnsemble, PhEnsembleConfig
from models.gp_model import MultitaskGPModel, MultitaskGPhysModel, GPConfig, train_gp_model, predict_with_gp
from models.physics_model import PhModel, PhysicsConfig
import gpytorch

# BoTorch imports for acquisition functions
try:
    from botorch.acquisition.analytic import ExpectedImprovement
    from botorch.optim import optimize_acqf
    BOTORCH_AVAILABLE = True
except ImportError:
    BOTORCH_AVAILABLE = False
    print("Warning: BoTorch not available. Install with 'pip install botorch'")


class SingleOutputEnsembleWrapper:
    """
    Wrapper to make multi-output ensembles work with single-output acquisition functions.
    Extracts a single output dimension for optimization.
    """

    def __init__(self, ensemble, output_index=0):
        self.ensemble = ensemble
        self.output_index = output_index
        self.num_outputs = 1  # Single output for acquisition function
        self._num_outputs = 1

    def posterior(self, X: torch.Tensor, **kwargs):
        """Get posterior distribution for the specified output dimension."""
        # Handle BoTorch's batch dimension - reshape (batch, 1, features) to (batch, features)
        if X.ndim == 3 and X.shape[1] == 1:
            X = X.squeeze(1)  # Remove the extra dimension

        # Get full posterior from ensemble
        full_posterior = self.ensemble.get_botorch_posterior(X)

        # Extract only the specified output dimension
        mean = full_posterior.mean[..., self.output_index:self.output_index+1]  # (batch, 1, 1)

        # Extract corresponding covariance for this output dimension
        batch_size = X.shape[0]
        covariance_single = []
        for b in range(batch_size):
            # Get the variance for this output dimension
            var = full_posterior.distribution.covariance_matrix[b, self.output_index, self.output_index]
            cov_matrix = var.unsqueeze(0).unsqueeze(0)  # (1, 1)
            covariance_single.append(cov_matrix)

        covariance = torch.stack(covariance_single, dim=0)  # (batch, 1, 1)

        # Create single-output distribution
        from gpytorch.distributions import MultitaskMultivariateNormal
        mvn_single = MultitaskMultivariateNormal(mean, covariance)

        from botorch.posteriors.gpytorch import GPyTorchPosterior
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
        self.num_outputs = 1  # Single output for acquisition function
        self._num_outputs = 1

    def posterior(self, X: torch.Tensor, **kwargs):
        """Get posterior distribution for the specified output dimension."""
        # Handle BoTorch's batch dimension
        if X.ndim == 3 and X.shape[1] == 1:
            X = X.squeeze(1)

        self.gp_model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Get full posterior from GP
            gp_posterior = self.gp_model(X)
            full_posterior = self.likelihood(gp_posterior)

            # Extract only the specified output dimension
            mean = full_posterior.mean[..., self.output_index:self.output_index+1]  # (batch, 1)
            var = full_posterior.variance[..., self.output_index:self.output_index+1]  # (batch, 1)

            # Reshape for BoTorch: (batch, 1, 1)
            mean = mean.unsqueeze(1)  # (batch, 1, 1)

            # Create covariance matrices (diagonal)
            batch_size = X.shape[0]
            covariance = torch.zeros(batch_size, 1, 1)
            for b in range(batch_size):
                covariance[b, 0, 0] = var[b, 0]

            # Create single-output distribution
            from gpytorch.distributions import MultitaskMultivariateNormal
            mvn_single = MultitaskMultivariateNormal(mean, covariance)

            from botorch.posteriors.gpytorch import GPyTorchPosterior
            return GPyTorchPosterior(mvn_single)

    def forward(self, X: torch.Tensor):
        """Forward pass that returns posterior (for compatibility)."""
        return self.posterior(X)


class EnsembleOptimizer:
    """
    Optimizer class that wraps our MLP, PhModel, and GP models for Bayesian optimization.
    Mimics the API structure from the original GDEOptimizer.
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
        self.model_type = model_type
        self.quantity = quantity
        self.maximize = maximize
        self.output_dir = output_dir
        self.model = None
        self.likelihood = None  # For GP models
        self.is_trained = False

        # Determine output index for single-output wrapper
        self.output_index = 0 if quantity == "FE (Eth)" else 1

        # For tracking current best value
        self.best_f = None

    def _train_model(self, df: pd.DataFrame):
        """Train the model on the provided data."""
        # Load and prepare data
        data_tensors, norm_params, _ = load_data()

        # Filter to match the provided DataFrame indices
        indices = df.index.tolist()
        train_mask = torch.tensor([i in indices for i in range(len(data_tensors.X))])
        X_train = data_tensors.X[train_mask]
        y_train = data_tensors.y[train_mask]

        if self.model_type == "MLP":
            config = EnsembleConfig(
                ensemble_size=5,  # Smaller for testing
                hidden_dim=32,
                dropout_rate=0.1,
                learning_rate=0.001,
                bootstrap_fraction=0.5
            )
            self.model = MLPEnsemble(config)
            print(f"Training {self.model_type} ensemble on {len(X_train)} samples...")
            self.model.train(X_train, y_train, num_epochs=50, verbose=False)

        elif self.model_type == "PhModel":
            # Get original ZLT stats for PhModel
            original_data = pd.read_excel('Characterization_data.xlsx', skiprows=[1], index_col=0)
            original_data = original_data.dropna()
            # Add thickness calculation (matching data loader)
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
                ensemble_size=3,  # Smaller for testing
                hidden_dim=32,
                dropout_rate=0.1,
                learning_rate=0.001,
                bootstrap_fraction=0.5,
                current_target=200.0,
                grid_size=50  # Smaller for faster testing
            )
            self.model = PhModelEnsemble(config, zlt_mu_stds=(original_zlt_mean, original_zlt_std))
            print(f"Training {self.model_type} ensemble on {len(X_train)} samples...")
            self.model.train(X_train, y_train, num_epochs=50, verbose=False)

        elif self.model_type == "GP":
            config = GPConfig(
                num_tasks=2,
                kernel_type="rbf",
                rank=1,
                learning_rate=0.1,
                num_iterations=100  # Reduced for testing
            )
            print(f"Training {self.model_type} model on {len(X_train)} samples...")
            self.model, self.likelihood, _ = train_gp_model(
                X_train, y_train,
                model_type="standard",
                config=config,
                verbose=False
            )

        elif self.model_type == "GP+Ph":
            # Get original ZLT stats for PhModel
            original_data = pd.read_excel('Characterization_data.xlsx', skiprows=[1], index_col=0)
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
                hidden_dim=32,
                current_target=233,
                grid_size=50
            )
            ph_model = PhModel(config=ph_config, zlt_mu_stds=(original_zlt_mean, original_zlt_std))

            # Quick training of PhModel
            ph_model.train()
            optimizer = torch.optim.Adam(ph_model.parameters(), lr=0.001)
            criterion = torch.nn.MSELoss()

            for epoch in range(20):  # Quick training
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
                num_iterations=100
            )
            print(f"Training {self.model_type} model on {len(X_train)} samples...")
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
        if not BOTORCH_AVAILABLE:
            raise ImportError("BoTorch not available for acquisition functions")

        # Create appropriate wrapper based on model type
        if self.model_type in ["MLP", "PhModel"]:
            # Ensemble models use SingleOutputEnsembleWrapper
            single_output_model = SingleOutputEnsembleWrapper(self.model, self.output_index)
        elif self.model_type in ["GP", "GP+Ph"]:
            # GP models use SingleOutputGPWrapper
            single_output_model = SingleOutputGPWrapper(self.model, self.likelihood, self.output_index)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        return ExpectedImprovement(
            model=single_output_model,
            best_f=self.best_f,
            maximize=self.maximize
        )

    def step_within_data(self, df_train: pd.DataFrame, df_explore: pd.DataFrame):
        """
        Perform acquisition function evaluation within the exploration data.

        Args:
            df_train: Training data to fit the model
            df_explore: Exploration data to evaluate acquisition function

        Returns:
            ei_values: Expected Improvement values for exploration points
            next_pick: Index of the best point to pick next
        """
        # Train model on training data
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


def test_mlp_optimizer():
    """Test the MLP ensemble optimizer matching the original test_api.py structure."""
    print("Testing MLP Ensemble Optimizer API...")

    # Load data
    _, _, df = load_data()

    # Create optimizer
    optimizer = EnsembleOptimizer(model_type="MLP", quantity="FE (Eth)", maximize=True)

    # Split data like in original test
    df_train = df.iloc[:30]
    df_explore = df.iloc[31:]

    print(f"Training set: {len(df_train)} samples")
    print(f"Exploration set: {len(df_explore)} samples")

    # First step
    ei, next_pick = optimizer.step_within_data(df_train, df_explore)
    print(f"First pick - EI max: {ei.max():.6f}, Next pick index: {next_pick}")

    # Add picked point to training and remove from exploration
    df_new = df_explore.loc[next_pick:next_pick]  # Single row DataFrame
    df_train_updated = pd.concat([df_train, df_new])
    df_explore_updated = df_explore.drop(index=next_pick)

    # Second step
    optimizer_2 = EnsembleOptimizer(model_type="MLP", quantity="FE (Eth)", maximize=True)
    ei_2, next_pick_2 = optimizer_2.step_within_data(df_train_updated, df_explore_updated)
    print(f"Second pick - EI max: {ei_2.max():.6f}, Next pick index: {next_pick_2}")

    print("✅ MLP optimizer API test completed")
    return True


def test_phmodel_optimizer():
    """Test the PhModel ensemble optimizer matching the original test_api.py structure."""
    print("\nTesting PhModel Ensemble Optimizer API...")

    # Load data
    _, _, df = load_data()

    # Create optimizer
    optimizer = EnsembleOptimizer(model_type="PhModel", quantity="FE (Eth)", maximize=True)

    # Split data like in original test
    df_train = df.iloc[:30]
    df_explore = df.iloc[31:]

    print(f"Training set: {len(df_train)} samples")
    print(f"Exploration set: {len(df_explore)} samples")

    # First step
    ei, next_pick = optimizer.step_within_data(df_train, df_explore)
    print(f"First pick - EI max: {ei.max():.6f}, Next pick index: {next_pick}")

    # Add picked point to training and remove from exploration
    df_new = df_explore.loc[next_pick:next_pick]  # Single row DataFrame
    df_train_updated = pd.concat([df_train, df_new])
    df_explore_updated = df_explore.drop(index=next_pick)

    # Second step
    optimizer_2 = EnsembleOptimizer(model_type="PhModel", quantity="FE (Eth)", maximize=True)
    ei_2, next_pick_2 = optimizer_2.step_within_data(df_train_updated, df_explore_updated)
    print(f"Second pick - EI max: {ei_2.max():.6f}, Next pick index: {next_pick_2}")

    print("✅ PhModel optimizer API test completed")
    return True


def test_gp_optimizer():
    """Test the standard GP optimizer matching the original test_api.py structure."""
    print("\nTesting Standard GP Optimizer API...")

    # Load data
    _, _, df = load_data()

    # Create optimizer
    optimizer = EnsembleOptimizer(model_type="GP", quantity="FE (Eth)", maximize=True)

    # Split data like in original test
    df_train = df.iloc[:30]
    df_explore = df.iloc[31:]

    print(f"Training set: {len(df_train)} samples")
    print(f"Exploration set: {len(df_explore)} samples")

    # First step
    ei, next_pick = optimizer.step_within_data(df_train, df_explore)
    print(f"First pick - EI max: {ei.max():.6f}, Next pick index: {next_pick}")

    # Add picked point to training and remove from exploration
    df_new = df_explore.loc[next_pick:next_pick]  # Single row DataFrame
    df_train_updated = pd.concat([df_train, df_new])
    df_explore_updated = df_explore.drop(index=next_pick)

    # Second step
    optimizer_2 = EnsembleOptimizer(model_type="GP", quantity="FE (Eth)", maximize=True)
    ei_2, next_pick_2 = optimizer_2.step_within_data(df_train_updated, df_explore_updated)
    print(f"Second pick - EI max: {ei_2.max():.6f}, Next pick index: {next_pick_2}")

    print("✅ GP optimizer API test completed")
    return True


def test_gp_physics_optimizer():
    """Test the physics-informed GP optimizer matching the original test_api.py structure."""
    print("\nTesting Physics-Informed GP (GP+Ph) Optimizer API...")

    # Load data
    _, _, df = load_data()

    # Create optimizer
    optimizer = EnsembleOptimizer(model_type="GP+Ph", quantity="FE (Eth)", maximize=True)

    # Split data like in original test
    df_train = df.iloc[:30]
    df_explore = df.iloc[31:]

    print(f"Training set: {len(df_train)} samples")
    print(f"Exploration set: {len(df_explore)} samples")

    # First step
    ei, next_pick = optimizer.step_within_data(df_train, df_explore)
    print(f"First pick - EI max: {ei.max():.6f}, Next pick index: {next_pick}")

    # Add picked point to training and remove from exploration
    df_new = df_explore.loc[next_pick:next_pick]  # Single row DataFrame
    df_train_updated = pd.concat([df_train, df_new])
    df_explore_updated = df_explore.drop(index=next_pick)

    # Second step
    optimizer_2 = EnsembleOptimizer(model_type="GP+Ph", quantity="FE (Eth)", maximize=True)
    ei_2, next_pick_2 = optimizer_2.step_within_data(df_train_updated, df_explore_updated)
    print(f"Second pick - EI max: {ei_2.max():.6f}, Next pick index: {next_pick_2}")

    print("✅ GP+Ph optimizer API test completed")
    return True


def test_comparison():
    """Compare MLP vs PhModel acquisition function behavior."""
    print("\nComparing MLP vs PhModel Acquisition Functions...")

    # Load data
    _, _, df = load_data()
    df_train = df.iloc[:20]  # Smaller training set for comparison
    df_explore = df.iloc[21:30]  # Smaller exploration set

    # Test both optimizers on the same data
    mlp_opt = EnsembleOptimizer(model_type="MLP", quantity="FE (Eth)", maximize=True)
    ph_opt = EnsembleOptimizer(model_type="PhModel", quantity="FE (Eth)", maximize=True)

    ei_mlp, pick_mlp = mlp_opt.step_within_data(df_train, df_explore)
    ei_ph, pick_ph = ph_opt.step_within_data(df_train, df_explore)

    print(f"MLP - Best EI: {ei_mlp.max():.6f}, Pick: {pick_mlp}")
    print(f"PhModel - Best EI: {ei_ph.max():.6f}, Pick: {pick_ph}")
    print(f"Same pick: {pick_mlp == pick_ph}")

    return pick_mlp, pick_ph


def test_all_models_comparison():
    """Compare all four model types: MLP, PhModel, GP, and GP+Ph."""
    print("\nComparing ALL Models: MLP vs PhModel vs GP vs GP+Ph...")

    # Load data
    _, _, df = load_data()
    df_train = df.iloc[:20]  # Smaller training set for comparison
    df_explore = df.iloc[21:30]  # Smaller exploration set

    # Test all optimizers on the same data
    model_types = ["MLP", "PhModel", "GP", "GP+Ph"]
    results = {}

    for model_type in model_types:
        print(f"\nTesting {model_type}...")
        optimizer = EnsembleOptimizer(model_type=model_type, quantity="FE (Eth)", maximize=True)
        ei, pick = optimizer.step_within_data(df_train, df_explore)
        results[model_type] = {
            'best_ei': ei.max().item(),
            'pick': pick
        }
        print(f"{model_type} - Best EI: {results[model_type]['best_ei']:.6f}, Pick: {results[model_type]['pick']}")

    print(f"\nComparison Summary:")
    for model_type, result in results.items():
        print(f"  {model_type:8}: EI={result['best_ei']:.6f}, Pick={result['pick']}")

    # Find model with highest EI (most exploratory)
    best_model = max(results.keys(), key=lambda k: results[k]['best_ei'])
    print(f"\nMost exploratory model: {best_model} (highest EI)")

    return results


if __name__ == "__main__":
    if not BOTORCH_AVAILABLE:
        print("❌ BoTorch not available - cannot run acquisition function tests")
        print("Install with: pip install botorch")
    else:
        try:
            test_mlp_optimizer()
            test_phmodel_optimizer()
            test_gp_optimizer()
            test_gp_physics_optimizer()
            test_all_models_comparison()
            print("\n✅ All API tests completed successfully!")
            print(f"Summary:")
            print(f"  - MLP Ensemble: ✓")
            print(f"  - PhModel Ensemble: ✓")
            print(f"  - Standard GP: ✓")
            print(f"  - Physics-Informed GP: ✓")
            print(f"  - All models BoTorch compatible: ✓")
        except Exception as e:
            print(f"❌ API test failed with error: {e}")
            import traceback
            traceback.print_exc()
