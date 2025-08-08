"""
Comprehensive debugging test for Ph+GP ensemble issues.
This test isolates and diagnoses numerical stability problems.
"""
import torch
import numpy as np
import pytest
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.ph_ensemble import PhModelEnsemble, PhEnsembleConfig
from models.gp_ensemble import GPEnsemble, GPEnsembleConfig
from models.physics_model import PhysicsConfig
from models.gp_model import GPConfig
from data.loader import load_data

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PhGPDebugger:
    """Debug Ph+GP ensemble numerical issues."""

    def __init__(self):
        self.data_tensors, self.norm_params, self.df_raw = load_data()
        self.X = self.data_tensors.X
        self.y = self.data_tensors.y

    def analyze_data_properties(self):
        """Analyze data for potential numerical issues."""
        logger.info("=== DATA ANALYSIS ===")

        # Check for NaN/inf values
        x_nan = torch.isnan(self.X).any()
        y_nan = torch.isnan(self.y).any()
        x_inf = torch.isinf(self.X).any()
        y_inf = torch.isinf(self.y).any()

        logger.info(f"X contains NaN: {x_nan}, inf: {x_inf}")
        logger.info(f"y contains NaN: {y_nan}, inf: {y_inf}")

        # Check data ranges and scaling
        logger.info(f"X shape: {self.X.shape}, y shape: {self.y.shape}")
        logger.info(f"X range: [{self.X.min().item():.6f}, {self.X.max().item():.6f}]")
        logger.info(f"y range: [{self.y.min().item():.6f}, {self.y.max().item():.6f}]")
        logger.info(f"X std: {self.X.std().item():.6f}")
        logger.info(f"y std: {self.y.std().item():.6f}")

        # Check condition number of X
        try:
            cond_num = torch.linalg.cond(self.X.T @ self.X)
            logger.info(f"X^T X condition number: {cond_num.item():.2e}")
        except:
            logger.warning("Could not compute condition number")

        # Check for duplicate or near-duplicate rows
        unique_rows = torch.unique(self.X, dim=0).shape[0]
        logger.info(f"Unique X rows: {unique_rows}/{self.X.shape[0]}")

        return {
            'has_nan': x_nan or y_nan,
            'has_inf': x_inf or y_inf,
            'x_range': (self.X.min().item(), self.X.max().item()),
            'y_range': (self.y.min().item(), self.y.max().item()),
            'x_std': self.X.std().item(),
            'y_std': self.y.std().item(),
            'unique_rows': unique_rows
        }

    def test_physics_model_alone(self, subset_size=50):
        """Test physics model in isolation."""
        logger.info("=== TESTING PHYSICS MODEL ALONE ===")

        # Use small subset for debugging
        indices = torch.randperm(self.X.shape[0])[:subset_size]
        X_subset = self.X[indices]
        y_subset = self.y[indices]

        try:
            ph_config = PhEnsembleConfig(
                ensemble_size=3,  # Small ensemble for debugging
                learning_rate=0.001,
                dropout_rate=0.1
            )

            ph_ensemble = PhModelEnsemble(ph_config)
            ph_ensemble.train(X_subset, y_subset, num_epochs=50, verbose=True)

            # Test prediction
            with torch.no_grad():
                pred = ph_ensemble.predict(X_subset[:10])
                logger.info(f"Physics prediction shape: {pred.shape}")
                logger.info(f"Physics prediction range: [{pred.min().item():.6f}, {pred.max().item():.6f}]")

            return True, pred

        except Exception as e:
            logger.error(f"Physics model failed: {e}")
            return False, None

    def test_gp_model_alone(self, subset_size=50):
        """Test GP model in isolation."""
        logger.info("=== TESTING GP MODEL ALONE ===")

        # Use small subset for debugging
        indices = torch.randperm(self.X.shape[0])[:subset_size]
        X_subset = self.X[indices]
        y_subset = self.y[indices]

        try:
            gp_config = GPConfig(
                learning_rate=0.01,
                num_tasks=y_subset.shape[1]
            )

            ensemble_config = GPEnsembleConfig(
                ensemble_size=2,  # Very small for debugging
                gp_config=gp_config
            )

            gp_ensemble = GPEnsemble(ensemble_config)
            gp_ensemble.train(X_subset, y_subset, num_epochs=50, verbose=True)

            # Test prediction
            with torch.no_grad():
                pred = gp_ensemble.predict(X_subset[:10])
                logger.info(f"GP prediction shape: {pred.shape}")
                logger.info(f"GP prediction range: [{pred.min().item():.6f}, {pred.max().item():.6f}]")

            return True, pred

        except Exception as e:
            logger.error(f"GP model failed: {e}")
            return False, None

    def test_covariance_matrix_properties(self, subset_size=30):
        """Analyze GP covariance matrix properties."""
        logger.info("=== ANALYZING COVARIANCE MATRIX ===")

        indices = torch.randperm(self.X.shape[0])[:subset_size]
        X_subset = self.X[indices]
        y_subset = self.y[indices]

        # Create a simple GP model for analysis
        import gpytorch
        from models.gp_model import MultitaskGPModel

        try:
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=y_subset.shape[1])
            gp_config = GPConfig(num_tasks=y_subset.shape[1])
            model = MultitaskGPModel(X_subset, y_subset, likelihood, gp_config)

            model.eval()
            likelihood.eval()

            with torch.no_grad():
                # Get covariance matrix
                covar = model.covar_module(X_subset)
                K = covar.evaluate()

                logger.info(f"Covariance matrix shape: {K.shape}")
                logger.info(f"Covariance matrix range: [{K.min().item():.6f}, {K.max().item():.6f}]")

                # Check condition number
                eigenvals = torch.linalg.eigvals(K)
                eigenvals_real = eigenvals.real
                cond_num = eigenvals_real.max() / eigenvals_real.min()
                logger.info(f"Covariance condition number: {cond_num.item():.2e}")

                # Check for near-zero eigenvalues
                min_eigval = eigenvals_real.min()
                logger.info(f"Minimum eigenvalue: {min_eigval.item():.2e}")

                # Check diagonal dominance
                diag_vals = torch.diag(K)
                off_diag_max = (K - torch.diag(diag_vals)).abs().max()
                logger.info(f"Max diagonal: {diag_vals.max().item():.6f}")
                logger.info(f"Max off-diagonal: {off_diag_max.item():.6f}")

                return {
                    'condition_number': cond_num.item(),
                    'min_eigenvalue': min_eigval.item(),
                    'max_diagonal': diag_vals.max().item(),
                    'max_off_diagonal': off_diag_max.item()
                }

        except Exception as e:
            logger.error(f"Covariance analysis failed: {e}")
            return None

    def test_with_different_data_fractions(self):
        """Test stability with different data subset sizes."""
        logger.info("=== TESTING DIFFERENT DATA FRACTIONS ===")

        fractions = [0.1, 0.2, 0.3, 0.5]
        results = {}

        for frac in fractions:
            subset_size = int(self.X.shape[0] * frac)
            logger.info(f"\nTesting with {subset_size} samples (fraction {frac})")

            indices = torch.randperm(self.X.shape[0])[:subset_size]
            X_subset = self.X[indices]
            y_subset = self.y[indices]

            # Test covariance properties
            cov_results = self.test_covariance_matrix_properties(subset_size)
            results[frac] = {
                'subset_size': subset_size,
                'covariance_analysis': cov_results
            }

            # Try to train a small GP
            try:
                gp_config = GPConfig(num_tasks=y_subset.shape[1])
                ensemble_config = GPEnsembleConfig(ensemble_size=1, gp_config=gp_config)
                gp_ensemble = GPEnsemble(ensemble_config)
                gp_ensemble.train(X_subset, y_subset, num_epochs=10, verbose=False)
                results[frac]['gp_training'] = 'success'
            except Exception as e:
                results[frac]['gp_training'] = f'failed: {str(e)[:100]}'

        return results

    def suggest_fixes(self, analysis_results):
        """Suggest potential fixes based on analysis."""
        logger.info("=== SUGGESTED FIXES ===")

        suggestions = []

        if analysis_results.get('has_nan') or analysis_results.get('has_inf'):
            suggestions.append("1. Clean data: Remove NaN/inf values before training")

        if analysis_results.get('x_std', 0) > 10 or analysis_results.get('y_std', 0) > 10:
            suggestions.append("2. Improve normalization: Data appears to have large scale differences")

        if analysis_results.get('unique_rows', 0) < self.X.shape[0] * 0.9:
            suggestions.append("3. Remove duplicate data points that can cause singular matrices")

        suggestions.extend([
            "4. Use more conservative GP settings (smaller learning rate, more jitter)",
            "5. Implement robust covariance matrix regularization",
            "6. Use inducing points or sparse GP methods for large datasets",
            "7. Pre-condition data by removing highly correlated features",
            "8. Implement early stopping when numerical issues are detected"
        ])

        for i, suggestion in enumerate(suggestions, 1):
            logger.info(suggestion)

        return suggestions


def run_full_debug():
    """Run complete debugging analysis."""
    debugger = PhGPDebugger()

    # Analyze data properties
    data_analysis = debugger.analyze_data_properties()

    # Test individual models
    ph_success, ph_pred = debugger.test_physics_model_alone()
    gp_success, gp_pred = debugger.test_gp_model_alone()

    # Analyze covariance properties
    cov_analysis = debugger.test_covariance_matrix_properties()

    # Test different data sizes
    fraction_results = debugger.test_with_different_data_fractions()

    # Get suggestions
    suggestions = debugger.suggest_fixes(data_analysis)

    return {
        'data_analysis': data_analysis,
        'physics_success': ph_success,
        'gp_success': gp_success,
        'covariance_analysis': cov_analysis,
        'fraction_results': fraction_results,
        'suggestions': suggestions
    }


if __name__ == "__main__":
    results = run_full_debug()
    print("\n=== DEBUG COMPLETE ===")
    print("Check the log output above for detailed analysis.")
