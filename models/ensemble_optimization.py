"""
Ensemble optimization and evaluation following CarbonDriver preprint methods.
Implements calibration, uncertainty evaluation, and active learning as described in the paper.
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy import stats
import logging

from models.mlp_ensemble import MLPEnsemble, EnsembleConfig
from models.ph_ensemble import PhModelEnsemble, PhEnsembleConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Store evaluation metrics as reported in the preprint."""
    nll: float  # Negative Log Likelihood
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Square Error
    calibration_score: float  # Uncertainty calibration

    def __str__(self):
        return f"NLL: {self.nll:.3f}, MAE: {self.mae:.3f}, RMSE: {self.rmse:.3f}, Cal: {self.calibration_score:.3f}"


@dataclass
class ActiveLearningConfig:
    """Configuration for active learning experiments."""
    initial_pool_size: int = 9  # Start with 3 triplets (3x3=9)
    acquisition_batch_size: int = 3  # Add triplets at each step
    max_iterations: int = 20
    random_seed: int = 42


class EnsembleEvaluator:
    """
    Evaluate ensemble models following CarbonDriver preprint methodology.
    Implements metrics and calibration procedures from Section 3.1.
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    def calculate_nll(self, y_true: torch.Tensor, mean_pred: torch.Tensor,
                     std_pred: torch.Tensor) -> float:
        """
        Calculate Negative Log Likelihood for uncertainty-aware predictions.

        Args:
            y_true: True values (N, 2)
            mean_pred: Predicted means (N, 2)
            std_pred: Predicted standard deviations (N, 2)

        Returns:
            Average NLL across all predictions
        """
        # Ensure minimum std to prevent numerical issues
        std_pred = torch.clamp(std_pred, min=1e-6)

        # Calculate log likelihood for each output dimension
        log_likelihood = -0.5 * torch.log(2 * np.pi * std_pred**2) - \
                        (y_true - mean_pred)**2 / (2 * std_pred**2)

        # Sum across output dimensions and average across samples
        nll = -log_likelihood.sum(dim=1).mean().item()
        return float(nll)

    def calculate_mae(self, y_true: torch.Tensor, mean_pred: torch.Tensor) -> float:
        """Calculate Mean Absolute Error."""
        return float(torch.abs(y_true - mean_pred).mean().item())

    def calculate_rmse(self, y_true: torch.Tensor, mean_pred: torch.Tensor) -> float:
        """Calculate Root Mean Square Error."""
        return float(torch.sqrt(torch.mean((y_true - mean_pred)**2)).item())

    def calculate_calibration_score(self, y_true: torch.Tensor, mean_pred: torch.Tensor,
                                  std_pred: torch.Tensor, n_bins: int = 10) -> float:
        """
        Calculate uncertainty calibration score.
        A well-calibrated model should have prediction intervals that contain
        the true values at the expected frequency.
        """
        # Flatten for calibration analysis
        y_true_flat = y_true.flatten()
        mean_flat = mean_pred.flatten()
        std_flat = std_pred.flatten()

        # Calculate prediction intervals at different confidence levels
        confidence_levels = np.linspace(0.1, 0.9, n_bins)
        calibration_errors = []

        for conf in confidence_levels:
            # Calculate prediction interval
            z_score = stats.norm.ppf((1 + conf) / 2)
            lower = mean_flat - z_score * std_flat
            upper = mean_flat + z_score * std_flat

            # Check how many true values fall within interval
            within_interval = ((y_true_flat >= lower) & (y_true_flat <= upper)).float().mean()

            # Calibration error is difference from expected coverage
            calibration_errors.append(abs(within_interval.item() - conf))

        return float(np.mean(calibration_errors))

    def evaluate_ensemble(self, ensemble, X_test: torch.Tensor,
                         y_test: torch.Tensor) -> EvaluationMetrics:
        """
        Comprehensive evaluation of an ensemble model.

        Args:
            ensemble: Trained ensemble model
            X_test: Test features (N, input_dim)
            y_test: Test targets (N, 2)

        Returns:
            EvaluationMetrics object with all computed metrics
        """
        logger.info("Evaluating ensemble performance...")

        # Get predictions with uncertainty
        mean_pred, std_pred = ensemble.predict(X_test, return_std=True)

        # Calculate metrics
        nll = self.calculate_nll(y_test, mean_pred, std_pred)
        mae = self.calculate_mae(y_test, mean_pred)
        rmse = self.calculate_rmse(y_test, mean_pred)
        calibration = self.calculate_calibration_score(y_test, mean_pred, std_pred)

        metrics = EvaluationMetrics(nll=nll, mae=mae, rmse=rmse, calibration_score=calibration)

        logger.info(f"Evaluation complete: {metrics}")
        return metrics

    def plot_parity(self, ensemble, X_test: torch.Tensor, y_test: torch.Tensor,
                   save_path: Optional[str] = None):
        """
        Create parity plots showing predicted vs true values with uncertainty.
        Similar to Figure in the preprint.
        """
        mean_pred, std_pred = ensemble.predict(X_test, return_std=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        output_names = ['FE_Ethylene', 'FE_CO']

        for i, (ax, name) in enumerate(zip(axes, output_names)):
            y_true_i = y_test[:, i].numpy()
            y_pred_i = mean_pred[:, i].numpy()
            y_std_i = std_pred[:, i].numpy()

            # Scatter plot with error bars
            ax.errorbar(y_true_i, y_pred_i, yerr=y_std_i, fmt='o', alpha=0.6, capsize=3)

            # Perfect prediction line
            min_val = min(y_true_i.min(), y_pred_i.min())
            max_val = max(y_true_i.max(), y_pred_i.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect prediction')

            ax.set_xlabel(f'True {name}')
            ax.set_ylabel(f'Predicted {name}')
            ax.set_title(f'{name} Parity Plot')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Parity plot saved to {save_path}")

        return fig


class EnsembleOptimizer:
    """
    Optimize ensemble hyperparameters following preprint methodology.
    Implements hyperparameter sweep from Section A.3.
    """

    def __init__(self, evaluator: EnsembleEvaluator):
        self.evaluator = evaluator

    def hyperparameter_sweep(self, X_train: torch.Tensor, y_train: torch.Tensor,
                           X_val: torch.Tensor, y_val: torch.Tensor,
                           ensemble_sizes: List[int] = None,
                           data_fractions: List[float] = None,
                           model_type: str = 'mlp',
                           norm_params=None) -> Dict[str, Any]:
        """
        Perform hyperparameter sweep for ensemble configuration.

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            ensemble_sizes: List of ensemble sizes to test
            data_fractions: List of bootstrap fractions to test
            model_type: 'mlp' or 'physics'
            norm_params: Normalization parameters for physics models

        Returns:
            Dictionary with sweep results and best configuration
        """
        if ensemble_sizes is None:
            ensemble_sizes = [10, 20, 40, 60]  # Limit to max 60

        if data_fractions is None:
            data_fractions = [0.2, 0.33, 0.5, 0.6]  # Limit to max 0.6

        logger.info(f"Starting hyperparameter sweep for {model_type} ensemble")
        logger.info(f"Testing {len(ensemble_sizes)} ensemble sizes and {len(data_fractions)} data fractions")

        results = []
        best_nll = float('inf')
        best_config = None

        for ensemble_size in ensemble_sizes:
            for data_fraction in data_fractions:
                logger.info(f"Testing: ensemble_size={ensemble_size}, data_fraction={data_fraction}")

                try:
                    # Create and train ensemble
                    if model_type == 'mlp':
                        config = EnsembleConfig(
                            ensemble_size=ensemble_size,
                            bootstrap_fraction=data_fraction
                        )
                        ensemble = MLPEnsemble(config)
                    else:
                        config = PhEnsembleConfig(
                            ensemble_size=ensemble_size,
                            bootstrap_fraction=data_fraction
                        )
                        # Pass correct normalization parameters for physics models
                        if norm_params is not None:
                            # Extract thickness normalization (feature index 3 is Zero_eps_thickness)
                            zlt_mu_stds = (norm_params.feature_means[3].item(), norm_params.feature_stds[3].item())
                        else:
                            zlt_mu_stds = (5e-6, 1e-6)  # Default fallback

                        ensemble = PhModelEnsemble(config, zlt_mu_stds=zlt_mu_stds)

                    # Train with reduced epochs for sweep
                    ensemble.train(X_train, y_train, num_epochs=200, verbose=False)

                    # Evaluate on validation set
                    metrics = self.evaluator.evaluate_ensemble(ensemble, X_val, y_val)

                    result = {
                        'ensemble_size': ensemble_size,
                        'data_fraction': data_fraction,
                        'nll': metrics.nll,
                        'mae': metrics.mae,
                        'rmse': metrics.rmse,
                        'calibration': metrics.calibration_score
                    }
                    results.append(result)

                    # Track best configuration based on NLL
                    if metrics.nll < best_nll:
                        best_nll = metrics.nll
                        best_config = config

                    logger.info(f"Result: {metrics}")

                except Exception as e:
                    logger.warning(f"Failed for ensemble_size={ensemble_size}, data_fraction={data_fraction}: {e}")

        sweep_results = {
            'results': results,
            'best_config': best_config,
            'best_nll': best_nll
        }

        logger.info(f"Hyperparameter sweep complete. Best NLL: {best_nll:.3f}")
        return sweep_results

    def plot_sweep_results(self, sweep_results: Dict[str, Any], save_path: Optional[str] = None):
        """Plot heatmap of hyperparameter sweep results."""
        results = sweep_results['results']

        # Extract unique values
        ensemble_sizes = sorted(list(set([r['ensemble_size'] for r in results])))
        data_fractions = sorted(list(set([r['data_fraction'] for r in results])))

        # Create NLL matrix
        nll_matrix = np.full((len(data_fractions), len(ensemble_sizes)), np.nan)

        for result in results:
            i = data_fractions.index(result['data_fraction'])
            j = ensemble_sizes.index(result['ensemble_size'])
            nll_matrix[i, j] = result['nll']

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(nll_matrix, cmap='viridis', aspect='auto')

        # Set ticks and labels
        ax.set_xticks(range(len(ensemble_sizes)))
        ax.set_xticklabels(ensemble_sizes)
        ax.set_yticks(range(len(data_fractions)))
        ax.set_yticklabels([f'{f:.2f}' for f in data_fractions])

        ax.set_xlabel('Ensemble Size')
        ax.set_ylabel('Data Fraction')
        ax.set_title('Negative Log Likelihood Heatmap')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('NLL')

        # Add text annotations
        for i in range(len(data_fractions)):
            for j in range(len(ensemble_sizes)):
                if not np.isnan(nll_matrix[i, j]):
                    text = ax.text(j, i, f'{nll_matrix[i, j]:.2f}',
                                 ha="center", va="center", color="white", fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Sweep results plot saved to {save_path}")

        return fig


class ActiveLearningSimulator:
    """
    Simulate pool-based active learning following preprint methodology.
    Implements the active learning procedure from Section A.4.
    """

    def __init__(self, config: ActiveLearningConfig = None, evaluator: EnsembleEvaluator = None):
        self.config = config or ActiveLearningConfig()
        self.evaluator = evaluator or EnsembleEvaluator()

    def expected_improvement_acquisition(self, ensemble, X_candidates: torch.Tensor,
                                       current_best: float) -> torch.Tensor:
        """
        Calculate Expected Improvement acquisition function.

        Args:
            ensemble: Trained ensemble model
            X_candidates: Candidate points (N, input_dim)
            current_best: Current best function value

        Returns:
            Expected improvement scores for each candidate
        """
        mean_pred, std_pred = ensemble.predict(X_candidates, return_std=True)

        # Calculate EI for multi-objective case (sum of FE values)
        f_pred = mean_pred.sum(dim=1)  # Sum FE_Ethylene + FE_CO
        s_pred = torch.sqrt((std_pred**2).sum(dim=1))  # Combined uncertainty

        # Expected Improvement calculation
        improvement = f_pred - current_best
        z = improvement / (s_pred + 1e-9)

        # EI = improvement * Φ(z) + s * φ(z)
        normal = torch.distributions.Normal(0, 1)
        ei = improvement * normal.cdf(z) + s_pred * torch.exp(normal.log_prob(z))

        return ei

    def simulate_active_learning(self, X_pool: torch.Tensor, y_pool: torch.Tensor,
                               ensemble_class, ensemble_config,
                               objective_fn=None) -> Dict[str, Any]:
        """
        Simulate pool-based active learning experiment.

        Args:
            X_pool: Full dataset features
            y_pool: Full dataset targets
            ensemble_class: MLPEnsemble or PhModelEnsemble class
            ensemble_config: Configuration for the ensemble
            objective_fn: Function to calculate objective (default: sum of FE values)

        Returns:
            Dictionary with active learning results and acceleration factor
        """
        if objective_fn is None:
            objective_fn = lambda y: y.sum(dim=1)  # Sum of FE values

        logger.info("Starting active learning simulation...")

        # Split into training and candidate pools
        n_total = len(X_pool)
        indices = torch.randperm(n_total)

        # Initial training pool (3 triplets = 9 samples)
        train_indices = indices[:self.config.initial_pool_size].clone()
        candidate_indices = indices[self.config.initial_pool_size:].clone()

        # Track results
        iteration_results = []
        best_objectives = []

        for iteration in range(self.config.max_iterations):
            logger.info(f"Active learning iteration {iteration + 1}/{self.config.max_iterations}")

            # Get current training data
            X_train = X_pool[train_indices]
            y_train = y_pool[train_indices]

            # Train ensemble
            ensemble = ensemble_class(ensemble_config)
            ensemble.train(X_train, y_train, num_epochs=300, verbose=False)

            # Calculate current best
            train_objectives = objective_fn(y_train)
            current_best = train_objectives.max().item()
            best_objectives.append(current_best)

            # Evaluate on remaining candidates
            X_candidates = X_pool[candidate_indices]

            if len(candidate_indices) == 0:
                logger.info("No more candidates available")
                break

            # Calculate acquisition function
            ei_scores = self.expected_improvement_acquisition(ensemble, X_candidates, current_best)

            # Select next batch (triplet)
            batch_size = min(self.config.acquisition_batch_size, len(candidate_indices))
            _, top_indices = torch.topk(ei_scores, batch_size)

            # Move selected candidates to training pool
            selected_candidates = candidate_indices[top_indices]
            train_indices = torch.cat([train_indices, selected_candidates])

            # Remove from candidate pool
            mask = torch.ones(len(candidate_indices), dtype=torch.bool)
            mask[top_indices] = False
            candidate_indices = candidate_indices[mask]

            # Store iteration results
            iteration_results.append({
                'iteration': iteration + 1,
                'training_size': len(train_indices),
                'best_objective': current_best,
                'mean_ei': ei_scores.mean().item(),
                'max_ei': ei_scores.max().item()
            })

            logger.info(f"Training size: {len(train_indices)}, Best objective: {current_best:.4f}")

        # Calculate acceleration factor compared to random sampling
        af = self._calculate_acceleration_factor(best_objectives, X_pool, y_pool, objective_fn)

        results = {
            'iteration_results': iteration_results,
            'best_objectives': best_objectives,
            'acceleration_factor': af,
            'final_training_indices': train_indices
        }

        logger.info(f"Active learning complete. Acceleration factor: {af:.2f}")
        return results

    def _calculate_acceleration_factor(self, al_objectives: List[float],
                                     X_pool: torch.Tensor, y_pool: torch.Tensor,
                                     objective_fn, n_random_runs: int = 10) -> float:
        """Calculate acceleration factor compared to random sampling."""
        logger.info("Calculating acceleration factor...")

        # Simulate random sampling
        random_objectives = []

        for run in range(n_random_runs):
            torch.manual_seed(run)  # Different seed for each run
            indices = torch.randperm(len(X_pool))

            run_objectives = []
            for i in range(len(al_objectives)):
                n_samples = self.config.initial_pool_size + i * self.config.acquisition_batch_size
                if n_samples <= len(indices):
                    selected_indices = indices[:n_samples]
                    selected_y = y_pool[selected_indices]
                    best_obj = objective_fn(selected_y).max().item()
                    run_objectives.append(best_obj)
                else:
                    run_objectives.append(run_objectives[-1])  # No more samples

            random_objectives.append(run_objectives)

        # Average across random runs
        avg_random_objectives = np.mean(random_objectives, axis=0)

        # Calculate steps to reach final AL performance
        final_al_objective = al_objectives[-1]

        # Find where random sampling reaches this performance
        al_steps = len(al_objectives)
        random_steps = len(avg_random_objectives)

        for i, obj in enumerate(avg_random_objectives):
            if obj >= final_al_objective:
                random_steps = i + 1
                break

        # Acceleration factor
        af = random_steps / al_steps if al_steps > 0 else 1.0
        return af

    def plot_active_learning_results(self, results: Dict[str, Any], save_path: Optional[str] = None):
        """Plot active learning performance over iterations."""
        iteration_results = results['iteration_results']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot best objective over iterations
        iterations = [r['iteration'] for r in iteration_results]
        best_objs = [r['best_objective'] for r in iteration_results]
        training_sizes = [r['training_size'] for r in iteration_results]

        ax1.plot(iterations, best_objs, 'b-o', label='Active Learning')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Best Objective Value')
        ax1.set_title('Active Learning Progress')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot training set size
        ax2.plot(iterations, training_sizes, 'g-s')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Training Set Size')
        ax2.set_title('Training Set Growth')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Active learning plot saved to {save_path}")

        return fig
