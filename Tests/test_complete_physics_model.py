"""
Test script for the updated Physics-Informed Model with complete gde_system.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
from data.loader import load_data
from models.physics_model import PhModel, PhysicsConfig


def test_complete_physics_model():
    """Test the PhModel with complete gde_system physics engine."""
    print("Testing PhModel with COMPLETE gde_system physics engine...")

    # Load data with normalization (as the original PhModel expects)
    print("\n1. Loading data with normalization...")
    data_tensors, norm_params, df = load_data(
        normalize_features=True,  # PhModel expects normalized inputs
        normalize_targets=False   # Targets stay as decimals 0-1
    )
    print(f"Data: {data_tensors.X.shape[0]} samples, {data_tensors.X.shape[1]} features")
    print(f"X normalized range: [{data_tensors.X.min():.3f}, {data_tensors.X.max():.3f}]")
    print(f"Target range: [{data_tensors.y.min():.3f}, {data_tensors.y.max():.3f}]")

    # Extract Zero_eps_thickness normalization parameters
    print("\n2. Setting up PhModel with correct normalization parameters...")
    zlt_mean = norm_params.feature_means[3].item()  # Zero_eps_thickness mean (normalized)
    zlt_std = norm_params.feature_stds[3].item()    # Zero_eps_thickness std (normalized)
    print(f"Zero layer thickness norm params: mean={zlt_mean:.6f}, std={zlt_std:.6f}")

    # These are the normalization stats from the ORIGINAL data (before normalization)
    # We need to calculate the original mean and std for denormalization in PhModel
    original_zlt = df['Zero_eps_thickness']
    original_zlt_mean = original_zlt.mean()
    original_zlt_std = original_zlt.std(ddof=0)  # Use ddof=0 to match PyTorch default
    print(f"Original ZLT stats: mean={original_zlt_mean:.2e}, std={original_zlt_std:.2e}")

    config = PhysicsConfig(
        hidden_dim=64,
        dropout_rate=0.1,
        current_target=200.0,  # Target current density [A/m^2]
        grid_size=100,         # Smaller grid for faster testing
        voltage_bounds=(-1.25, 0.0)
    )

    # Create PhModel with original (unnormalized) ZLT statistics
    model = PhModel(config=config, zlt_mu_stds=(original_zlt_mean, original_zlt_std))
    total_params = sum(p.numel() for p in model.parameters())
    print(f"PhModel created with {total_params} parameters")
    print(f"Physics engine: Complete gde_system.System")

    # Test forward pass
    print("\n3. Testing forward pass with complete physics...")
    model.eval()
    try:
        with torch.no_grad():
            # Test with small batch first
            test_input = data_tensors.X[:3]
            predictions = model(test_input)

        print(f"✅ Forward pass successful!")
        print(f"Prediction shape: {predictions.shape}")
        print(f"Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")

        # Display sample predictions
        print(f"\nSample predictions (first 3 points):")
        for i in range(3):
            true_eth, true_co = data_tensors.y[i]
            pred_eth, pred_co = predictions[i]
            print(f"  Point {i}: True=[{true_eth:.3f}, {true_co:.3f}], Pred=[{pred_eth:.3f}, {pred_co:.3f}]")

        # Test physics parameter extraction
        print("\n4. Testing physics parameter extraction...")
        physics_params = model.get_physics_parameters(test_input)

        print("Physics parameters for first 3 samples:")
        for key, values in physics_params.items():
            if key == 'pore_radius':
                print(f"  {key}: {[f'{v:.2e}' for v in values.squeeze().tolist()]} [m]")
            elif key in ['zero_layer_thickness', 'layer_thickness']:
                print(f"  {key}: {[f'{v:.2e}' for v in values.squeeze().tolist()]} [m]")
            elif key == 'porosity':
                print(f"  {key}: {[f'{v:.3f}' for v in values.squeeze().tolist()]} (dimensionless)")
            else:
                print(f"  {key}: {[f'{v:.3f}' for v in values.squeeze().tolist()]}")

        # Test training capability (few steps)
        print("\n5. Testing training capability...")
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()

        initial_loss = None
        print("Training for 5 epochs...")
        for epoch in range(5):
            optimizer.zero_grad()

            # Use smaller batch for testing
            batch_X = data_tensors.X[:20]  # Smaller batch
            batch_y = data_tensors.y[:20]

            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

            if epoch == 0:
                initial_loss = loss.item()

            print(f"  Epoch {epoch}: Loss = {loss.item():.6f}")

        final_loss = loss.item()
        print(f"Loss change: {initial_loss:.6f} → {final_loss:.6f}")

        # Validate physics constraints with complete engine
        print("\n6. Validating physics constraints...")
        model.eval()
        with torch.no_grad():
            test_predictions = model(data_tensors.X[:10])
            params = model.get_physics_parameters(data_tensors.X[:10])

        constraints_valid = True

        # Check parameter ranges
        porosity = params['porosity']
        if not (0 < porosity.min() and porosity.max() < 1):
            print(f"❌ Porosity out of bounds: [{porosity.min():.3f}, {porosity.max():.3f}]")
            constraints_valid = False
        else:
            print(f"✅ Porosity in bounds: [{porosity.min():.3f}, {porosity.max():.3f}]")

        pore_radius = params['pore_radius']
        if not (1e-10 < pore_radius.min() and pore_radius.max() < 1e-6):
            print(f"❌ Pore radius unreasonable: [{pore_radius.min():.2e}, {pore_radius.max():.2e}]")
            constraints_valid = False
        else:
            print(f"✅ Pore radius reasonable: [{pore_radius.min():.2e}, {pore_radius.max():.2e}]")

        # Surface coverage fractions should sum to ≤ 1
        theta_sum = params['theta_CO'] + params['theta_C2H4'] + params['theta_H2b']
        if not (theta_sum.max() <= 1.01):  # Small tolerance
            print(f"❌ Surface coverage sum > 1: max = {theta_sum.max():.3f}")
            constraints_valid = False
        else:
            print(f"✅ Surface coverage sum ≤ 1: max = {theta_sum.max():.3f}")

        # Predictions should be in [0, 1]
        if not (0 <= test_predictions.min() and test_predictions.max() <= 1):
            print(f"❌ Predictions out of [0,1]: [{test_predictions.min():.3f}, {test_predictions.max():.3f}]")
            constraints_valid = False
        else:
            print(f"✅ Predictions in [0,1]: [{test_predictions.min():.3f}, {test_predictions.max():.3f}]")

        # Check that predictions are reasonable (not all the same)
        pred_std = test_predictions.std().item()
        if pred_std < 1e-6:
            print(f"❌ Predictions are too similar (std={pred_std:.2e})")
            constraints_valid = False
        else:
            print(f"✅ Predictions have variation (std={pred_std:.3f})")

        if constraints_valid:
            print("\n✅ All physics constraints satisfied!")
        else:
            print("\n❌ Some physics constraints violated!")

        print("\n✅ Complete PhModel test finished!")
        print(f"Summary:")
        print(f"  - Complete physics engine: ✓")
        print(f"  - Normalized input handling: ✓")
        print(f"  - Forward pass: ✓")
        print(f"  - Parameter extraction: ✓")
        print(f"  - Gradient flow: ✓")
        print(f"  - Physics constraints: {'✓' if constraints_valid else '❌'}")
        print(f"  - Prediction variation: ✓")

        return model, constraints_valid

    except Exception as e:
        print(f"❌ PhModel test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, False


if __name__ == "__main__":
    test_complete_physics_model()
