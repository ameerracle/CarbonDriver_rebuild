"""
Test script for MLP ensemble model.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
from data.loader import load_data
from models.mlp_ensemble import MLPEnsemble, EnsembleConfig
from config import default_config


def test_mlp_ensemble():
    """Test the MLP ensemble training and prediction."""
    print("Testing MLP ensemble...")

    # Load data
    print("\n1. Loading data...")
    data_tensors, norm_params, df = load_data(
        normalize_features=True,  # Always normalize features for MLP training
        normalize_targets=False
    )
    print(f"Data loaded: X shape {data_tensors.X.shape}, y shape {data_tensors.y.shape}")
    print(f"Features normalized: {norm_params is not None}")
    print(f"Target range: [{data_tensors.y.min():.3f}, {data_tensors.y.max():.3f}]")

    # Create small ensemble for testing
    print("\n2. Creating MLP ensemble...")
    test_config = EnsembleConfig(
        ensemble_size=5,  # Small for testing
        hidden_dim=32,    # Smaller for faster training
        dropout_rate=0.1,
        learning_rate=0.001,
        bootstrap_fraction=0.5
    )
    ensemble = MLPEnsemble(test_config)
    print(f"Created ensemble with {len(ensemble.models)} models")

    # Train ensemble
    print("\n3. Training ensemble...")
    training_stats = ensemble.train(
        data_tensors.X,
        data_tensors.y,
        num_epochs=50,  # Quick training for test
        verbose=True
    )

    # Test predictions
    print("\n4. Testing predictions...")
    mean_pred, std_pred = ensemble.predict(data_tensors.X, return_std=True)
    print(f"Prediction shapes - Mean: {mean_pred.shape}, Std: {std_pred.shape}")
    print(f"Mean prediction range: [{mean_pred.min():.3f}, {mean_pred.max():.3f}]")
    print(f"Uncertainty range: [{std_pred.min():.3f}, {std_pred.max():.3f}]")

    # Test sampling for BoTorch compatibility
    print("\n5. Testing BoTorch compatibility...")
    samples = ensemble.get_prediction_samples(data_tensors.X[:5], n_samples=100)
    print(f"Sample shape: {samples.shape}")
    sample_mean = samples.mean(dim=0)
    sample_std = samples.std(dim=0)
    print(f"Sample statistics match predictions: Mean diff {torch.abs(sample_mean - mean_pred[:5]).max():.3e}")

    # Calculate training error
    print("\n6. Training performance...")
    train_error = torch.mean((mean_pred - data_tensors.y)**2).item()
    print(f"Final training MSE: {train_error:.6f}")

    # Test individual outputs
    print(f"\nSample predictions for first 3 points:")
    for i in range(3):
        true_eth, true_co = data_tensors.y[i]
        pred_eth, pred_co = mean_pred[i]
        std_eth, std_co = std_pred[i]
        print(f"  Point {i}: True=[{true_eth:.3f}, {true_co:.3f}], "
              f"Pred=[{pred_eth:.3f}±{std_eth:.3f}, {pred_co:.3f}±{std_co:.3f}]")

    print("\n✅ MLP ensemble test completed successfully!")
    return ensemble, data_tensors, norm_params


if __name__ == "__main__":
    try:
        test_mlp_ensemble()
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
