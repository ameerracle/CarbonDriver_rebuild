"""
Ensemble optimization and evaluation following CarbonDriver preprint methods.
Implements calibration, uncertainty evaluation, and active learning as described in the paper.
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
from multiprocessing import Pool, cpu_count
from models.mlp_ensemble import MLPEnsemble, EnsembleConfig
from models.ph_ensemble import PhModelEnsemble, PhEnsembleConfig
from models.gp_model import MultitaskGPModel, MultitaskGPhysModel, GPConfig
from models.gp_ensemble import GPEnsemble, PhysicsGPEnsemble, GPEnsembleConfig
from models.ensemble_optimization import EnsembleEvaluator
import gpytorch
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SweepJob:
    ensemble_size: int
    data_fraction: float
    model_type: str
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_val: torch.Tensor
    y_val: torch.Tensor
    norm_params: Any
    evaluator: EnsembleEvaluator

    def run(self):
        try:
            if self.model_type == 'mlp':
                config = EnsembleConfig(
                    ensemble_size=self.ensemble_size,
                    bootstrap_fraction=self.data_fraction
                )
                ensemble = MLPEnsemble(config)
            elif self.model_type == 'ph':
                config = PhEnsembleConfig(
                    ensemble_size=self.ensemble_size,
                    bootstrap_fraction=self.data_fraction
                )
                if self.norm_params is not None:
                    zlt_mu_stds = (self.norm_params.feature_means[3].item(), self.norm_params.feature_stds[3].item())
                else:
                    zlt_mu_stds = (5e-6, 1e-6)
                ensemble = PhModelEnsemble(config, zlt_mu_stds=zlt_mu_stds)
            elif self.model_type == 'gp':
                # Create a proper GP ensemble
                ensemble = self._create_gp_ensemble()
            elif self.model_type == 'ph+gp':
                # Create a proper hybrid physics-informed GP ensemble
                ensemble = self._create_hybrid_gp_ensemble()
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

            ensemble.train(self.X_train, self.y_train, num_epochs=200, verbose=False)
            metrics = self.evaluator.evaluate_ensemble(ensemble, self.X_val, self.y_val)
            result = {
                'ensemble_size': self.ensemble_size,
                'data_fraction': self.data_fraction,
                'nll': metrics.nll,
                'mae': metrics.mae,
                'rmse': metrics.rmse,
                'calibration': metrics.calibration_score
            }
            return result
        except Exception as e:
            logger.warning(f"Failed for ensemble_size={self.ensemble_size}, data_fraction={self.data_fraction}, model_type={self.model_type}: {e}")
            return None

    def _create_gp_ensemble(self):
        """Create a proper GP ensemble."""
        # Use smaller ensemble size for GPs due to computational cost
        gp_ensemble_size = min(self.ensemble_size, 20)  # Cap at 20 for computational efficiency

        config = GPEnsembleConfig(
            ensemble_size=gp_ensemble_size,
            bootstrap_fraction=self.data_fraction,
            gp_config=GPConfig(
                num_tasks=2,
                kernel_type="rbf",
                learning_rate=0.1,
                num_iterations=400
            )
        )
        return GPEnsemble(config)

    def _create_hybrid_gp_ensemble(self):
        """Create a proper hybrid physics-informed GP ensemble."""
        # Use smaller ensemble size for GPs due to computational cost
        gp_ensemble_size = min(self.ensemble_size, 20)  # Cap at 20 for computational efficiency

        config = GPEnsembleConfig(
            ensemble_size=gp_ensemble_size,
            bootstrap_fraction=self.data_fraction,
            gp_config=GPConfig(
                num_tasks=2,
                kernel_type="rbf",
                learning_rate=0.1,
                num_iterations=400
            )
        )

        if self.norm_params is not None:
            zlt_mu_stds = (self.norm_params.feature_means[3].item(), self.norm_params.feature_stds[3].item())
        else:
            zlt_mu_stds = (5e-6, 1e-6)

        return PhysicsGPEnsemble(config, zlt_mu_stds=zlt_mu_stds)

def run_sweep_job(job):
    return job.run()

def hyperparameter_sweep_parallel(X_train, y_train, X_val, y_val, ensemble_sizes=None, data_fractions=None, model_type='mlp', norm_params=None, skip_sweeps=0):
    evaluator = EnsembleEvaluator()
    if ensemble_sizes is None:
        ensemble_sizes = [10, 20, 40, 50, 60]
    if data_fractions is None:
        data_fractions = [0.33, 0.5, 0.6, 0.7]
    logger.info(f"Starting parallel hyperparameter sweep for {model_type} ensemble")
    jobs = []
    for ensemble_size in ensemble_sizes:
        for data_fraction in data_fractions:
            jobs.append(SweepJob(
                ensemble_size=ensemble_size,
                data_fraction=data_fraction,
                model_type=model_type,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                norm_params=norm_params,
                evaluator=evaluator
            ))
    # Skip the first N sweeps
    jobs = jobs[skip_sweeps:]
    with Pool(processes=min(cpu_count(), len(jobs))) as pool:
        results = pool.map(run_sweep_job, jobs)
    results = [r for r in results if r is not None]
    best_nll = float('inf')
    best_config = None
    for r in results:
        if r['nll'] < best_nll:
            best_nll = r['nll']
            best_config = r
    sweep_results = {
        'results': results,
        'best_config': best_config,
        'best_nll': best_nll
    }
    logger.info(f"Parallel hyperparameter sweep complete. Best NLL: {best_nll:.3f}")
    return sweep_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip_sweeps', type=int, default=0, help='Number of sweeps to skip at the start')
    parser.add_argument('--model_type', type=str, default=None, help='Model type to run (mlp, ph, gp, ph+gp)')
    args = parser.parse_args()
    import os, json, pandas as pd
    from data.loader import load_data
    # Load and preprocess data from Excel
    data_tensors, norm_params, df_raw = load_data()
    X, y = data_tensors.X, data_tensors.y
    # Split into training, validation, and test sets (recommended 70/15/15 split)
    N = X.shape[0]
    train_frac, val_frac, test_frac = 0.7, 0.15, 0.15
    train_end = int(train_frac * N)
    val_end = train_end + int(val_frac * N)
    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

    model_types = ['mlp', 'ph', 'gp', 'ph+gp']
    if args.model_type:
        model_types = [args.model_type]
    all_sweep_results = {}
    best_summary = {}
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)

    for model_type in model_types:
        logger.info(f"\n===== Running sweep for model: {model_type} =====")
        sweep_results = hyperparameter_sweep_parallel(
            X_train, y_train, X_val, y_val,
            model_type=model_type,
            norm_params=norm_params,
            skip_sweeps=args.skip_sweeps
        )
        all_sweep_results[model_type] = sweep_results
        # Save as JSON
        with open(os.path.join(output_dir, f"parallel_sweep_results_{model_type}.json"), "w") as f:
            json.dump(sweep_results, f, indent=2)
        # Save as CSV
        df = pd.DataFrame(sweep_results["results"])
        df.to_csv(os.path.join(output_dir, f"parallel_sweep_results_{model_type}.csv"), index=False)
        # Store best NLL and MAE
        best_summary[model_type] = {
            'best_nll': sweep_results['best_config']['nll'],
            'best_mae': sweep_results['best_config']['mae'],
            'best_config': sweep_results['best_config']
        }
        logger.info(f"Best for {model_type}: NLL={sweep_results['best_config']['nll']:.4f}, MAE={sweep_results['best_config']['mae']:.4f}")

    print("\n===== Best settings for each model =====")
    for model_type, summary in best_summary.items():
        print(f"Model: {model_type}")
        print(f"  Best NLL: {summary['best_nll']:.4f}")
        print(f"  Best MAE: {summary['best_mae']:.4f}")
        print(f"  Config: ensemble_size={summary['best_config']['ensemble_size']}, data_fraction={summary['best_config']['data_fraction']}")
        print()
    logger.info("All sweeps complete. Best settings printed above.")
