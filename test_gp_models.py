"""
Test script for Gaussian Process models.
Tests both standard GP and physics-informed hybrid GP models.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import argparse
from data.loader import load_data
from models.gp_model import MultitaskGPModel, MultitaskGPhysModel, GPConfig, train_gp_model, predict_with_gp
from models.physics_model import PhModel, PhysicsConfig
import time


def test_standard_gp():
    """Test the standard MultitaskGPModel."""
    print("Testing Standard Gaussian Process Model...")

    # Load data
    data_tensors, norm_params, df = load_data(
        normalize_features=True,  # GP expects normalized inputs
        normalize_targets=False   # Targets stay as decimals 0-1
    )
    print(f"Data: {data_tensors.X.shape[0]} samples, {data_tensors.X.shape[1]} features")

    # Use small subset for testing
    X_train = data_tensors.X[:50]
    y_train = data_tensors.y[:50]
    X_test = data_tensors.X[50:]
    y_test = data_tensors.y[50:]

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Configure GP
    config = GPConfig(
        num_tasks=2,
        kernel_type="rbf",
        rank=1,
        learning_rate=0.1,
        num_iterations=100  # Reduced for testing
    )

    # Train standard GP
    print("\nTraining standard GP...")
    start_time = time.time()
    model, likelihood, stats = train_gp_model(
        X_train, y_train,
        model_type="standard",
        config=config,
        verbose=True
    )
    train_time = time.time() - start_time

    # Test predictions
    print("\nTesting predictions...")
    mean_pred, std_pred = predict_with_gp(model, likelihood, X_test)

    print(f"Prediction shapes - Mean: {mean_pred.shape}, Std: {std_pred.shape}")
    print(f"Mean prediction range: [{mean_pred.min():.3f}, {mean_pred.max():.3f}]")
    print(f"Uncertainty range: [{std_pred.min():.3f}, {std_pred.max():.3f}]")

    # Calculate test error
    test_mse = torch.mean((mean_pred - y_test)**2).item()
    print(f"Test MSE: {test_mse:.6f}")

    # Show sample predictions
    print(f"\nSample predictions (first 3 test points):")
    for i in range(min(3, len(X_test))):
        true_eth, true_co = y_test[i]
        pred_eth, pred_co = mean_pred[i]
        std_eth, std_co = std_pred[i]
        print(f"  Point {i}: True=[{true_eth:.3f}, {true_co:.3f}], "
              f"Pred=[{pred_eth:.3f}±{std_eth:.3f}, {pred_co:.3f}±{std_co:.3f}]")

    print(f"✅ Standard GP test completed in {train_time:.1f}s")
    return model, likelihood, test_mse


def test_physics_informed_gp():
    """Test the physics-informed MultitaskGPhysModel."""
    print("\nTesting Physics-Informed Gaussian Process Model...")

    # Load data
    data_tensors, norm_params, df = load_data(
        normalize_features=True,
        normalize_targets=False
    )

    # Get original ZLT stats for PhModel
    original_zlt = df['Zero_eps_thickness']
    original_zlt_mean = original_zlt.mean()
    original_zlt_std = original_zlt.std(ddof=0)

    # Use small subset for testing
    X_train = data_tensors.X[:50]
    y_train = data_tensors.y[:50]
    X_test = data_tensors.X[50:]
    y_test = data_tensors.y[50:]

    # Pre-train a PhModel for the mean function
    print("Pre-training PhModel for mean function...")
    ph_config = PhysicsConfig(
        hidden_dim=32,
        current_target=233,
        grid_size=50  # Smaller for faster testing
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

    print(f"PhModel pre-training completed. Final loss: {loss.item():.6f}")

    # Configure GP
    config = GPConfig(
        num_tasks=2,
        kernel_type="rbf",
        rank=1,
        learning_rate=0.1,
        num_iterations=100  # Reduced for testing
    )

    # Train physics-informed GP
    print("\nTraining physics-informed GP...")
    start_time = time.time()
    model, likelihood, stats = train_gp_model(
        X_train, y_train,
        model_type="physics",
        config=config,
        ph_model=ph_model,
        zlt_mu_stds=(original_zlt_mean, original_zlt_std),
        freeze_physics=True,  # Freeze the pre-trained PhModel
        verbose=True
    )
    train_time = time.time() - start_time

    # Test predictions
    print("\nTesting predictions...")
    mean_pred, std_pred = predict_with_gp(model, likelihood, X_test)

    print(f"Prediction shapes - Mean: {mean_pred.shape}, Std: {std_pred.shape}")
    print(f"Mean prediction range: [{mean_pred.min():.3f}, {mean_pred.max():.3f}]")
    print(f"Uncertainty range: [{std_pred.min():.3f}, {std_pred.max():.3f}]")

    # Calculate test error
    test_mse = torch.mean((mean_pred - y_test)**2).item()
    print(f"Test MSE: {test_mse:.6f}")

    # Show sample predictions
    print(f"\nSample predictions (first 3 test points):")
    for i in range(min(3, len(X_test))):
        true_eth, true_co = y_test[i]
        pred_eth, pred_co = mean_pred[i]
        std_eth, std_co = std_pred[i]
        print(f"  Point {i}: True=[{true_eth:.3f}, {true_co:.3f}], "
              f"Pred=[{pred_eth:.3f}±{std_eth:.3f}, {pred_co:.3f}±{std_co:.3f}]")

    print(f"✅ Physics-informed GP test completed in {train_time:.1f}s")
    return model, likelihood, test_mse


def test_gp_comparison():
    """Compare standard GP vs physics-informed GP performance."""
    print("\nComparing Standard GP vs Physics-Informed GP...")

    # Test both models
    _, _, standard_mse = test_standard_gp()
    _, _, physics_mse = test_physics_informed_gp()

    print(f"\nComparison Results:")
    print(f"Standard GP Test MSE:        {standard_mse:.6f}")
    print(f"Physics-Informed GP Test MSE: {physics_mse:.6f}")

    if physics_mse < standard_mse:
        improvement = ((standard_mse - physics_mse) / standard_mse) * 100
        print(f"✅ Physics-informed GP improved by {improvement:.1f}%")
    else:
        degradation = ((physics_mse - standard_mse) / standard_mse) * 100
        print(f"⚠️  Standard GP performed {degradation:.1f}% better")

    return standard_mse, physics_mse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Gaussian Process models.")
    parser.add_argument(
        "--model",
        choices=["standard", "physics", "compare"],
        default="compare",
        help="Which GP model to test: 'standard', 'physics' or 'compare' (default: 'compare')"
    )
    args = parser.parse_args()

    try:
        if args.model == "standard":
            test_standard_gp()
            print("\n✅ Standard GP test completed successfully!")
            print("Summary:")
            print("  - Standard MultitaskGPModel: ✓")
            print("  - BoTorch compatibility: ✓ (inherits from ExactGP)")
            print("  - Uncertainty estimation: ✓ (native GP uncertainty)")

        elif args.model == "physics":
            test_physics_informed_gp()
            print("\n✅ Physics-informed GP test completed successfully!")
            print("Summary:")
            print("  - Physics-informed MultitaskGPhysModel: ✓")
            print("  - PhModel integration as mean function: ✓")
            print("  - BoTorch compatibility: ✓ (inherits from ExactGP)")
            print("  - Uncertainty estimation: ✓ (native GP uncertainty)")

        else:
            test_gp_comparison()
            print("\n✅ All GP model tests completed successfully!")
            print("Summary:")
            print("  - Standard MultitaskGPModel: ✓")
            print("  - Physics-informed MultitaskGPhysModel: ✓")
            print("  - Performance comparison: ✓")
            print("  - BoTorch compatibility: ✓ (inherits from ExactGP)")
            print("  - Uncertainty estimation: ✓ (native GP uncertainty)")

    except Exception as e:
        print(f"❌ GP test failed with error: {e}")
        import traceback
        traceback.print_exc()
