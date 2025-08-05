"""
Comprehensive evaluation script following CarbonDriver preprint methodology.
Demonstrates ensemble optimization, calibration, and active learning.
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Any, Tuple

from models.ensemble_optimization import (
    EnsembleEvaluator, EnsembleOptimizer, ActiveLearningSimulator,
    EvaluationMetrics, ActiveLearningConfig
)
from models.mlp_ensemble import MLPEnsemble, EnsembleConfig
from models.ph_ensemble import PhModelEnsemble, PhEnsembleConfig
from data.loader import DataLoader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CarbonDriverEvaluation:
    """
    Main evaluation class implementing the CarbonDriver preprint methodology.
    Reproduces Table 1 results and active learning experiments.
    """

    def __init__(self, data_path: str = "data/Characterization_data.xlsx", random_seed: int = 42):
        self.data_path = data_path
        self.random_seed = random_seed
        self.results_dir = Path("evaluation_results")
        self.results_dir.mkdir(exist_ok=True)

        # Initialize components
        self.evaluator = EnsembleEvaluator(random_seed)
        self.optimizer = EnsembleOptimizer(self.evaluator)
        self.al_simulator = ActiveLearningSimulator()

        # Load and prepare data
        self.X_train, self.X_test, self.y_train, self.y_test = self._load_and_split_data()

        logger.info(f"Data loaded: {len(self.X_train)} training, {len(self.X_test)} test samples")

    def _load_and_split_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load and split data following preprint methodology."""
        try:
            # Load data using your existing loader
            loader = DataLoader(excel_path=self.data_path, random_seed=self.random_seed)

            # Get full dataset
            X_full, y_full = loader.get_tensors()

            # Store normalization parameters for physics models
            from data.loader import load_data
            _, norm_params, _ = load_data(normalize_features=True, normalize_targets=False)
            self.norm_params = norm_params  # Store for physics models

            # Split into train/test (80/20 split is common for small datasets)
            n_total = len(X_full)
            n_train = int(0.8 * n_total)

            torch.manual_seed(self.random_seed)
            indices = torch.randperm(n_total)

            train_indices = indices[:n_train]
            test_indices = indices[n_train:]

            X_train = X_full[train_indices]
            X_test = X_full[test_indices]
            y_train = y_full[train_indices]
            y_test = y_full[test_indices]

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            # Create dummy data for demonstration
            logger.warning("Creating dummy data for demonstration")
            n_samples = 90  # As mentioned in preprint
            X_full = torch.randn(n_samples, 5)
            y_full = torch.rand(n_samples, 2)  # FE values in [0,1]

            # Set dummy normalization params
            self.norm_params = None

            n_train = int(0.8 * n_samples)
            return X_full[:n_train], X_full[n_train:], y_full[:n_train], y_full[n_train:]

    def run_hyperparameter_optimization(self) -> Dict[str, Any]:
        """
        Run hyperparameter optimization following Section A.3 of the preprint.
        Tests ensemble sizes up to 50+ and data fractions around 0.33.
        """
        logger.info("=== HYPERPARAMETER OPTIMIZATION ===")

        # Split training data for validation
        n_train = len(self.X_train)
        n_val = int(0.2 * n_train)

        torch.manual_seed(self.random_seed)
        indices = torch.randperm(n_train)

        X_train_opt = self.X_train[indices[n_val:]]
        y_train_opt = self.y_train[indices[n_val:]]
        X_val = self.X_train[indices[:n_val]]
        y_val = self.y_train[indices[:n_val]]

        # Test ensemble sizes and data fractions as in preprint
        ensemble_sizes = [10, 25, 50, 75]  # Up to 50+ as in preprint
        data_fractions = [0.2, 0.33, 0.5, 0.67]  # Include 0.33 from preprint

        results = {}

        # Test MLP ensemble
        logger.info("Optimizing MLP ensemble...")
        mlp_results = self.optimizer.hyperparameter_sweep(
            X_train_opt, y_train_opt, X_val, y_val,
            ensemble_sizes=ensemble_sizes,
            data_fractions=data_fractions,
            model_type='mlp'
        )
        results['mlp'] = mlp_results

        # Test Physics ensemble if available
        try:
            logger.info("Optimizing Physics ensemble...")
            ph_results = self.optimizer.hyperparameter_sweep(
                X_train_opt, y_train_opt, X_val, y_val,
                ensemble_sizes=ensemble_sizes,
                data_fractions=data_fractions,
                model_type='physics'
            )
            results['physics'] = ph_results
        except Exception as e:
            logger.warning(f"Physics ensemble optimization failed: {e}")

        # Save and plot results
        self._save_optimization_results(results)

        return results

    def evaluate_model_performance(self, optimization_results: Dict[str, Any] = None) -> Dict[str, EvaluationMetrics]:
        """
        Evaluate model performance following Table 1 methodology.
        Compare MLP ensemble, Physics ensemble, and combinations.
        """
        logger.info("=== MODEL PERFORMANCE EVALUATION ===")

        results = {}

        # Use optimized configurations if available
        if optimization_results:
            mlp_config = optimization_results.get('mlp', {}).get('best_config')
            ph_config = optimization_results.get('physics', {}).get('best_config')
        else:
            # Use preprint defaults
            mlp_config = EnsembleConfig(ensemble_size=50, bootstrap_fraction=0.33)
            ph_config = PhEnsembleConfig(ensemble_size=50, bootstrap_fraction=0.33)

        # Evaluate MLP Ensemble
        logger.info("Evaluating MLP ensemble...")
        mlp_ensemble = MLPEnsemble(mlp_config or EnsembleConfig(ensemble_size=50, bootstrap_fraction=0.33))
        mlp_ensemble.train(self.X_train, self.y_train, num_epochs=400, verbose=True)
        mlp_metrics = self.evaluator.evaluate_ensemble(mlp_ensemble, self.X_test, self.y_test)
        results['MLP_ensemble'] = mlp_metrics

        # Create parity plot for MLP
        fig_mlp = self.evaluator.plot_parity(
            mlp_ensemble, self.X_test, self.y_test,
            save_path=self.results_dir / "mlp_parity_plot.png"
        )

        # Evaluate Physics Ensemble if available
        try:
            logger.info("Evaluating Physics ensemble...")
            ph_ensemble = PhModelEnsemble(ph_config or PhEnsembleConfig(ensemble_size=50, bootstrap_fraction=0.33))
            ph_ensemble.train(self.X_train, self.y_train, num_epochs=400, verbose=True)
            ph_metrics = self.evaluator.evaluate_ensemble(ph_ensemble, self.X_test, self.y_test)
            results['Physics_ensemble'] = ph_metrics

            # Create parity plot for Physics
            fig_ph = self.evaluator.plot_parity(
                ph_ensemble, self.X_test, self.y_test,
                save_path=self.results_dir / "physics_parity_plot.png"
            )

        except Exception as e:
            logger.warning(f"Physics ensemble evaluation failed: {e}")

        # Print results table (similar to Table 1 in preprint)
        self._print_results_table(results)

        return results

    def run_active_learning_simulation(self, model_configs: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run simulated pool-based active learning following Section A.4.
        Compare different model types and calculate acceleration factors.
        """
        logger.info("=== ACTIVE LEARNING SIMULATION ===")

        # Combine all data for pool-based learning
        X_pool = torch.cat([self.X_train, self.X_test], dim=0)
        y_pool = torch.cat([self.y_train, self.y_test], dim=0)

        al_config = ActiveLearningConfig(
            initial_pool_size=9,  # 3 triplets as in preprint
            acquisition_batch_size=3,  # Add triplets
            max_iterations=15
        )

        al_results = {}

        # Test MLP ensemble
        logger.info("Running active learning with MLP ensemble...")
        mlp_config = EnsembleConfig(ensemble_size=25, bootstrap_fraction=0.33)  # Smaller for faster AL
        mlp_al_results = self.al_simulator.simulate_active_learning(
            X_pool, y_pool, MLPEnsemble, mlp_config
        )
        al_results['MLP_ensemble'] = mlp_al_results

        # Test Physics ensemble if available
        try:
            logger.info("Running active learning with Physics ensemble...")
            ph_config = PhEnsembleConfig(ensemble_size=25, bootstrap_fraction=0.33)
            ph_al_results = self.al_simulator.simulate_active_learning(
                X_pool, y_pool, PhModelEnsemble, ph_config
            )
            al_results['Physics_ensemble'] = ph_al_results
        except Exception as e:
            logger.warning(f"Physics active learning failed: {e}")

        # Plot active learning results
        for model_name, results in al_results.items():
            self.al_simulator.plot_active_learning_results(
                results,
                save_path=self.results_dir / f"{model_name}_active_learning.png"
            )

        # Print acceleration factors
        self._print_acceleration_factors(al_results)

        return al_results

    def run_complete_evaluation(self) -> Dict[str, Any]:
        """
        Run the complete evaluation pipeline following the preprint methodology.
        """
        logger.info("=== STARTING COMPLETE CARBONDRIVER EVALUATION ===")

        all_results = {}

        # 1. Hyperparameter optimization
        optimization_results = self.run_hyperparameter_optimization()
        all_results['optimization'] = optimization_results

        # 2. Model performance evaluation
        performance_results = self.evaluate_model_performance(optimization_results)
        all_results['performance'] = performance_results

        # 3. Active learning simulation
        al_results = self.run_active_learning_simulation()
        all_results['active_learning'] = al_results

        # 4. Save complete results
        self._save_complete_results(all_results)

        logger.info("=== EVALUATION COMPLETE ===")
        logger.info(f"Results saved to {self.results_dir}")

        return all_results

    def _save_optimization_results(self, results: Dict[str, Any]):
        """Save hyperparameter optimization results."""
        for model_type, model_results in results.items():
            if 'results' in model_results:
                df = pd.DataFrame(model_results['results'])
                df.to_csv(self.results_dir / f"{model_type}_optimization.csv", index=False)

                # Plot heatmap
                self.optimizer.plot_sweep_results(
                    model_results,
                    save_path=self.results_dir / f"{model_type}_optimization_heatmap.png"
                )

    def _print_results_table(self, results: Dict[str, EvaluationMetrics]):
        """Print results table similar to Table 1 in preprint."""
        logger.info("\\n=== PERFORMANCE RESULTS (Table 1 Style) ===")
        logger.info(f"{'Model':<20} {'NLL':<8} {'MAE':<8} {'RMSE':<8} {'Calibration':<12}")
        logger.info("-" * 56)

        for model_name, metrics in results.items():
            logger.info(f"{model_name:<20} {metrics.nll:<8.3f} {metrics.mae:<8.3f} {metrics.rmse:<8.3f} {metrics.calibration_score:<12.3f}")

    def _print_acceleration_factors(self, al_results: Dict[str, Any]):
        """Print acceleration factors for active learning."""
        logger.info("\\n=== ACCELERATION FACTORS ===")
        for model_name, results in al_results.items():
            af = results['acceleration_factor']
            logger.info(f"{model_name}: {af:.1f}x acceleration")

    def _save_complete_results(self, results: Dict[str, Any]):
        """Save all results to files."""
        import json
        import pickle

        # Save as pickle for Python objects
        with open(self.results_dir / "complete_results.pkl", 'wb') as f:
            pickle.dump(results, f)

        # Save summary as JSON
        summary = {
            'hyperparameter_optimization': {
                model_type: {
                    'best_nll': result.get('best_nll', None),
                    'num_tested': len(result.get('results', []))
                }
                for model_type, result in results.get('optimization', {}).items()
            },
            'performance_evaluation': {
                model_name: {
                    'nll': metrics.nll,
                    'mae': metrics.mae,
                    'rmse': metrics.rmse,
                    'calibration': metrics.calibration_score
                }
                for model_name, metrics in results.get('performance', {}).items()
            },
            'active_learning': {
                model_name: {
                    'acceleration_factor': result['acceleration_factor'],
                    'final_objective': result['best_objectives'][-1] if result['best_objectives'] else None
                }
                for model_name, result in results.get('active_learning', {}).items()
            }
        }

        with open(self.results_dir / "evaluation_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)


def main():
    """Run the complete CarbonDriver evaluation."""
    evaluator = CarbonDriverEvaluation()
    results = evaluator.run_complete_evaluation()

    # Print final summary
    print("\\n" + "="*50)
    print("CARBONDRIVER EVALUATION COMPLETE")
    print("="*50)
    print(f"Results saved to: {evaluator.results_dir}")
    print("\\nKey files:")
    print("- evaluation_summary.json: Summary of all results")
    print("- *_parity_plot.png: Parity plots showing prediction quality")
    print("- *_optimization_heatmap.png: Hyperparameter sweep results")
    print("- *_active_learning.png: Active learning progress")


if __name__ == "__main__":
    main()
