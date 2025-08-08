"""
Deep investigation of Ph+GP ensemble dimension issues.
Focus on the exact tensor shapes at every step of the ensemble pipeline.
"""
import torch
import gpytorch
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.gp_ensemble import PhysicsGPEnsemble, GPEnsembleConfig
from models.gp_model import MultitaskGPhysModel, GPConfig, PhysicsInformedMean
from models.physics_model import PhModel, PhysicsConfig
from data.loader import load_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def debug_ph_gp_ensemble_dimensions():
    """Deep dive into Ph+GP ensemble dimension issues."""
    logger.info("=== DEEP PH+GP ENSEMBLE DIMENSION DEBUG ===")

    # Load data
    data_tensors, norm_params, df_raw = load_data()
    X = data_tensors.X
    y = data_tensors.y

    logger.info(f"Original data: X={X.shape}, y={y.shape}")

    # Test with problematic size (74 points like in error message)
    test_size = 74
    X_test = X[:test_size]
    y_test = y[:test_size]

    logger.info(f"Test data: X={X_test.shape}, y={y_test.shape}")

    # Step 1: Test standalone PhModel
    logger.info("\n--- Step 1: Standalone PhModel Test ---")
    try:
        if norm_params is not None:
            zlt_mu_stds = (norm_params.feature_means[3].item(), norm_params.feature_stds[3].item())
        else:
            zlt_mu_stds = (5e-6, 1e-6)

        ph_config = PhysicsConfig(current_target=233)
        ph_model = PhModel(config=ph_config, zlt_mu_stds=zlt_mu_stds)

        ph_model.eval()
        with torch.no_grad():
            ph_output = ph_model(X_test[:5])
            logger.info(f"✓ PhModel output shape: {ph_output.shape}")
            logger.info(f"  PhModel output range: [{ph_output.min().item():.6f}, {ph_output.max().item():.6f}]")

    except Exception as e:
        logger.error(f"✗ PhModel test failed: {e}")
        return

    # Step 2: Test PhysicsInformedMean
    logger.info("\n--- Step 2: PhysicsInformedMean Test ---")
    try:
        mean_fn = PhysicsInformedMean(ph_model=ph_model, zlt_mu_stds=zlt_mu_stds)

        with torch.no_grad():
            mean_output = mean_fn(X_test[:5])
            logger.info(f"PhysicsInformedMean output shape: {mean_output.shape}")
            logger.info(f"PhysicsInformedMean output range: [{mean_output.min().item():.6f}, {mean_output.max().item():.6f}]")

            # Test the problematic .squeeze() behavior
            raw_ph_output = ph_model(X_test[:5])
            squeezed_output = raw_ph_output.squeeze()
            logger.info(f"Raw PhModel output: {raw_ph_output.shape}")
            logger.info(f"After .squeeze(): {squeezed_output.shape}")

            if raw_ph_output.shape != squeezed_output.shape:
                logger.error(f"✗ SQUEEZE CHANGES SHAPE! {raw_ph_output.shape} → {squeezed_output.shape}")
            else:
                logger.info("✓ Squeeze doesn't change shape")

    except Exception as e:
        logger.error(f"✗ PhysicsInformedMean test failed: {e}")
        return

    # Step 3: Test MultitaskGPhysModel creation
    logger.info("\n--- Step 3: MultitaskGPhysModel Creation Test ---")
    try:
        # Use small subset first
        X_small = X_test[:10]
        y_small = y_test[:10]

        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
        gp_config = GPConfig(num_tasks=2)

        model = MultitaskGPhysModel(
            X_small, y_small, likelihood,
            ph_model=ph_model,
            zlt_mu_stds=zlt_mu_stds,
            config=gp_config
        )

        logger.info(f"✓ MultitaskGPhysModel created")
        logger.info(f"  Train inputs shape: {model.train_inputs[0].shape}")
        logger.info(f"  Train targets shape: {model.train_targets.shape}")

        # Test forward pass
        model.eval()
        with torch.no_grad():
            test_input = X_small[:3]
            logger.info(f"Test input shape: {test_input.shape}")

            # Test mean function separately
            mean_out = model.mean_module(test_input)
            logger.info(f"Mean module output shape: {mean_out.shape}")

            # Test covariance function
            covar_out = model.covar_module(test_input)
            K = covar_out.evaluate()
            logger.info(f"Covariance matrix shape: {K.shape}")

            # Test full forward pass
            output = model(test_input)
            logger.info(f"Full model output mean shape: {output.mean.shape}")
            logger.info(f"Full model covariance shape: {output.covariance_matrix.shape}")

    except Exception as e:
        logger.error(f"✗ MultitaskGPhysModel test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

    # Step 4: Test with exact problematic size (74 points)
    logger.info(f"\n--- Step 4: Test with Problematic Size ({test_size} points) ---")
    try:
        likelihood_74 = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
        model_74 = MultitaskGPhysModel(
            X_test, y_test, likelihood_74,
            ph_model=ph_model,
            zlt_mu_stds=zlt_mu_stds,
            config=gp_config
        )

        logger.info(f"✓ Created model with {test_size} points")
        logger.info(f"  Train targets shape: {model_74.train_targets.shape}")

        # Test covariance matrix formation
        model_74.eval()
        with torch.no_grad():
            # Test mean function on full dataset
            mean_74 = model_74.mean_module(X_test)
            logger.info(f"Mean output for {test_size} points: {mean_74.shape}")

            # This is where the problem likely occurs
            covar_74 = model_74.covar_module(X_test)
            K_74 = covar_74.evaluate()
            logger.info(f"Covariance matrix shape: {K_74.shape}")
            logger.info(f"Expected covariance shape: ({test_size * 2}, {test_size * 2}) = ({test_size * 2}, {test_size * 2})")

            # Check for NaN/inf in covariance
            nan_count = torch.isnan(K_74).sum()
            inf_count = torch.isinf(K_74).sum()
            logger.info(f"NaN values in covariance: {nan_count}")
            logger.info(f"Inf values in covariance: {inf_count}")

            if nan_count > 0:
                logger.error(f"✗ FOUND NaN VALUES IN COVARIANCE MATRIX!")
                # Find where NaN values are
                nan_locations = torch.isnan(K_74).nonzero()
                logger.error(f"First few NaN locations: {nan_locations[:10]}")

                # Check diagonal vs off-diagonal
                diag_nans = torch.isnan(torch.diag(K_74)).sum()
                logger.error(f"NaN values on diagonal: {diag_nans}")

            # Try Cholesky decomposition
            try:
                L = torch.linalg.cholesky(K_74)
                logger.info("✓ Cholesky decomposition successful")
            except RuntimeError as e:
                logger.error(f"✗ Cholesky failed: {e}")

                # Add jitter and try again
                jitter_values = [1e-6, 1e-5, 1e-4, 1e-3]
                for jitter in jitter_values:
                    try:
                        K_jittered = K_74 + jitter * torch.eye(K_74.shape[0])
                        L = torch.linalg.cholesky(K_jittered)
                        logger.info(f"✓ Cholesky successful with jitter {jitter}")
                        break
                    except RuntimeError:
                        continue
                else:
                    logger.error("✗ Cholesky failed even with jitter")

    except Exception as e:
        logger.error(f"✗ Test with {test_size} points failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

    # Step 5: Test ensemble bootstrap sampling
    logger.info(f"\n--- Step 5: Ensemble Bootstrap Sampling Test ---")
    try:
        config = GPEnsembleConfig(
            ensemble_size=2,  # Small for testing
            bootstrap_fraction=0.6,
            gp_config=GPConfig(num_tasks=2)
        )

        # Simulate what happens in PhysicsGPEnsemble
        n_samples = X_test.shape[0]
        bootstrap_size = int(n_samples * config.bootstrap_fraction)

        logger.info(f"Bootstrap sampling: {n_samples} → {bootstrap_size} samples")

        for i in range(2):
            indices = torch.randint(0, n_samples, (bootstrap_size,))
            X_boot = X_test[indices]
            y_boot = y_test[indices]

            logger.info(f"Bootstrap {i+1}: X={X_boot.shape}, y={y_boot.shape}")

            # Test creating model with bootstrap sample
            try:
                likelihood_boot = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
                model_boot = MultitaskGPhysModel(
                    X_boot, y_boot, likelihood_boot,
                    ph_model=ph_model,
                    zlt_mu_stds=zlt_mu_stds,
                    config=gp_config
                )
                logger.info(f"✓ Bootstrap {i+1} model created successfully")

                # Test covariance matrix
                model_boot.eval()
                with torch.no_grad():
                    covar_boot = model_boot.covar_module(X_boot)
                    K_boot = covar_boot.evaluate()

                    nan_count_boot = torch.isnan(K_boot).sum()
                    if nan_count_boot > 0:
                        logger.error(f"✗ Bootstrap {i+1} has {nan_count_boot} NaN values in covariance")
                    else:
                        logger.info(f"✓ Bootstrap {i+1} covariance is clean")

            except Exception as e:
                logger.error(f"✗ Bootstrap {i+1} failed: {e}")

    except Exception as e:
        logger.error(f"✗ Bootstrap sampling test failed: {e}")


if __name__ == "__main__":
    debug_ph_gp_ensemble_dimensions()
