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
from models.ensemble_optimization import EnsembleEvaluator

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
            else:
                config = PhEnsembleConfig(
                    ensemble_size=self.ensemble_size,
                    bootstrap_fraction=self.data_fraction
                )
                if self.norm_params is not None:
                    zlt_mu_stds = (self.norm_params.feature_means[3].item(), self.norm_params.feature_stds[3].item())
                else:
                    zlt_mu_stds = (5e-6, 1e-6)
                ensemble = PhModelEnsemble(config, zlt_mu_stds=zlt_mu_stds)
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
            logger.warning(f"Failed for ensemble_size={self.ensemble_size}, data_fraction={self.data_fraction}: {e}")
            return None

def hyperparameter_sweep_parallel(X_train, y_train, X_val, y_val, ensemble_sizes=None, data_fractions=None, model_type='mlp', norm_params=None):
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
    with Pool(processes=min(cpu_count(), len(jobs))) as pool:
        results = pool.map(lambda job: job.run(), jobs)
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
    import os
    import json
    import pandas as pd
    # Example: load your data here
    # X_train, y_train, X_val, y_val, norm_params = ...
    # model_type = 'mlp' or 'ph'
    # Run the sweep
    sweep_results = hyperparameter_sweep_parallel(X_train, y_train, X_val, y_val, model_type=model_type, norm_params=norm_params)
    # Ensure output folder exists
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)
    # Save as JSON
    with open(os.path.join(output_dir, "parallel_sweep_results.json"), "w") as f:
        json.dump(sweep_results, f, indent=2)
    # Save as CSV
    df = pd.DataFrame(sweep_results["results"])
    df.to_csv(os.path.join(output_dir, "parallel_sweep_results.csv"), index=False)
    logger.info(f"Saved sweep results to {output_dir}")
