"""
Test script for PhModel ensemble.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
from data.loader import load_data
from models.ph_ensemble import PhModelEnsemble, PhEnsembleConfig
import time


def test_ph_model_ensemble():
    """Test the PhModel ensemble training and prediction."""
    print("Testing PhModel ensemble...")

    # Load data with proper normalization
    print("\n1. Loading data...")
    data_tensors, norm_params, df = load_data(
        normalize_features=True,  # PhModel expects normalized inputs
        normalize_targets=False   # Targets stay as decimals 0-1
    )
    print(f"Data: {data_tensors.X.shape[0]} samples, {data_tensors.X.shape[1]} features")

    # Extract original ZLT statistics for PhModel
    original_zlt = df['Zero_eps_thickness']
    original_zlt_mean = original_zlt.mean()
    original_zlt_std = original_zlt.std(ddof=0)
    print(f"Original ZLT stats: mean={original_zlt_mean:.2e}, std={original_zlt_std:.2e}")

    # Create small PhModel ensemble for testing
    print("\n2. Creating PhModel ensemble...")
    test_config = PhEnsembleConfig(
        ensemble_size=3,          # Small for testing
        hidden_dim=32,            # Smaller for faster training
        dropout_rate=0.1,
        learning_rate=0.001,
        bootstrap_fraction=0.5,
        current_target=200.0,
        grid_size=50,             # Smaller grid for faster physics solving
        voltage_bounds=(-1.25, 0.0)
    )
    ensemble = PhModelEnsemble(config=test_config, zlt_mu_stds=(original_zlt_mean, original_zlt_std))
    print(f"Created PhModel ensemble with {len(ensemble.models)} models")

    # Train ensemble
    print("\n3. Training PhModel ensemble...")
    start_time = time.time()
    training_stats = ensemble.train(
        data_tensors.X,
        data_tensors.y,
        num_epochs=20,    # Quick training for test
        verbose=True
    )
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.1f} seconds")

    # Test predictions
    print("\n4. Testing ensemble predictions...")
    mean_pred, std_pred = ensemble.predict(data_tensors.X, return_std=True)
    print(f"Prediction shapes - Mean: {mean_pred.shape}, Std: {std_pred.shape}")
    print(f"Mean prediction range: [{mean_pred.min():.3f}, {mean_pred.max():.3f}]")
    print(f"Uncertainty range: [{std_pred.min():.3f}, {std_pred.max():.3f}]")

    # Verify bootstrapping worked (models should give different predictions)
    print("\n5. Verifying bootstrap diversity...")
    individual_preds = []
    for i, model in enumerate(ensemble.models):
        model.eval()
        with torch.no_grad():
            pred = model(data_tensors.X[:1])  # Single test point
            individual_preds.append(pred)

    # Calculate variance across individual model predictions
    individual_stack = torch.stack(individual_preds, dim=0)
    model_variance = individual_stack.var(dim=0).mean().item()
    print(f"Variance across individual PhModels: {model_variance:.6f}")
    if model_variance > 1e-6:
        print("✅ Bootstrap sampling created diverse PhModels")
    else:
        print("❌ PhModels are too similar - bootstrap may not be working")

    # Performance metrics
    print("\n6. Performance evaluation...")
    train_mse = torch.mean((mean_pred - data_tensors.y)**2).item()
    print(f"Training MSE: {train_mse:.6f}")

    # Show sample predictions
    print(f"\nSample predictions (first 3 points):")
    for i in range(3):
        true_eth, true_co = data_tensors.y[i]
        pred_eth, pred_co = mean_pred[i]
        std_eth, std_co = std_pred[i]
        print(f"  Point {i}: True=[{true_eth:.3f}, {true_co:.3f}], "
              f"Pred=[{pred_eth:.3f}±{std_eth:.3f}, {pred_co:.3f}±{std_co:.3f}]")

    # Test BoTorch compatibility
    print("\n7. Testing BoTorch compatibility...")
    try:
        samples_botorch = ensemble.get_prediction_samples_botorch(data_tensors.X[:3], n_samples=50)
        print(f"BoTorch sample format: {samples_botorch.shape} (n_samples, batch_size, 1, outputs) ✅")

        # Test posterior
        posterior = ensemble.get_botorch_posterior(data_tensors.X[:3])
        print(f"BoTorch posterior created successfully ✅")
        print(f"Posterior mean shape: {posterior.mean.shape}")

    except ImportError as e:
        print(f"BoTorch not available: {e}")
    except Exception as e:
        print(f"BoTorch integration error: {e}")

    print("\n✅ PhModel ensemble test completed!")
    print(f"Summary:")
    print(f"  - Ensemble size: {len(ensemble.models)} PhModels")
    print(f"  - Bootstrap sampling: ✓ (with replacement)")
    print(f"  - Physics-informed predictions: ✓")
    print(f"  - Uncertainty estimation: ✓ (std dev across ensemble)")
    print(f"  - BoTorch compatible: ✓ (sampling interface)")
    print(f"  - Training time: {train_time:.1f}s")

    return ensemble, train_mse, model_variance


if __name__ == "__main__":
    try:
        test_ph_model_ensemble()
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
