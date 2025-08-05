"""
Test MLP ensemble with non-normalized inputs to check robustness.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
from data.loader import load_data
from models.mlp_ensemble import MLPEnsemble, EnsembleConfig


def test_mlp_with_unnormalized_inputs():
    """Test if MLP can work with non-normalized inputs."""
    print("Testing MLP ensemble with non-normalized inputs...")

    # Load data without normalization
    print("\n1. Loading data without normalization...")
    data_tensors, norm_params, df = load_data(
        normalize_features=False,  # Test without normalization
        normalize_targets=False
    )
    print(f"Data loaded: X shape {data_tensors.X.shape}, y shape {data_tensors.y.shape}")
    print(f"Features normalized: {norm_params is not None}")
    print(f"X range: [{data_tensors.X.min():.3f}, {data_tensors.X.max():.3f}]")
    print(f"Sample X values:\n{data_tensors.X[:3]}")

    # Create small ensemble for testing
    print("\n2. Creating MLP ensemble...")
    test_config = EnsembleConfig(
        ensemble_size=3,  # Very small for quick test
        hidden_dim=32,
        dropout_rate=0.1,
        learning_rate=0.01,  # Higher learning rate for unnormalized data
        bootstrap_fraction=0.5
    )
    ensemble = MLPEnsemble(test_config)

    # Train ensemble
    print("\n3. Training ensemble with unnormalized inputs...")
    try:
        training_stats = ensemble.train(
            data_tensors.X,
            data_tensors.y,
            num_epochs=100,  # More epochs for harder training
            verbose=True
        )

        # Test predictions
        mean_pred, std_pred = ensemble.predict(data_tensors.X, return_std=True)
        print(f"\nPrediction range: [{mean_pred.min():.3f}, {mean_pred.max():.3f}]")

        # Calculate training error
        train_error = torch.mean((mean_pred - data_tensors.y)**2).item()
        print(f"Final training MSE: {train_error:.6f}")

        print("✅ MLP can work with non-normalized inputs!")
        return True

    except Exception as e:
        print(f"❌ MLP failed with non-normalized inputs: {e}")
        return False


if __name__ == "__main__":
    test_mlp_with_unnormalized_inputs()
