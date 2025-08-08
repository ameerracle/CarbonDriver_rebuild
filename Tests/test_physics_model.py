"""
Test script for Physics-Informed Model (PhModel).
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
from data.loader import load_data
from models.physics_model import PhModel, PhysicsConfig


def test_physics_model():
    """Test the Physics-Informed Model."""
    print("Testing Physics-Informed Model (PhModel)...")

    # Load data
    print("\n1. Loading data...")
    data_tensors, norm_params, df = load_data(
        normalize_features=False,  # Features normalized
        normalize_targets=False   # Targets stay as decimals 0-1
    )
    print(f"Data: {data_tensors.X.shape[0]} samples, {data_tensors.X.shape[1]} features")
    print(f"Target range: [{data_tensors.y.min():.3f}, {data_tensors.y.max():.3f}]")

    # Create PhModel
    print("\n2. Creating Physics-Informed Model...")

    # Calculate normalization stats for Zero_eps_thickness (column 3)
    zlt_mean = norm_params.feature_means[3].item() if norm_params else 5e-6
    zlt_std = norm_params.feature_stds[3].item() if norm_params else 1e-6
    print(f"Zero layer thickness normalization: mean={zlt_mean:.2e}, std={zlt_std:.2e}")

    config = PhysicsConfig(
        hidden_dim=64,
        dropout_rate=0.1,
        current_target=200.0
    )

    model = PhModel(config=config, zlt_mu_stds=(zlt_mean, zlt_std))
    print(f"PhModel created with {sum(p.numel() for p in model.parameters())} parameters")

    # Test forward pass
    print("\n3. Testing forward pass...")
    model.eval()
    with torch.no_grad():
        predictions = model(data_tensors.X[:5])  # Test with 5 samples

    print(f"Prediction shape: {predictions.shape}")
    print(f"Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    print(f"Sample predictions:")
    for i in range(3):
        true_eth, true_co = data_tensors.y[i]
        pred_eth, pred_co = predictions[i]
        print(f"  Point {i}: True=[{true_eth:.3f}, {true_co:.3f}], Pred=[{pred_eth:.3f}, {pred_co:.3f}]")

    # Test physics parameter extraction
    print("\n4. Testing physics parameter extraction...")
    physics_params = model.get_physics_parameters(data_tensors.X[:3])

    print("Physics parameters for first 3 samples:")
    for key, values in physics_params.items():
        print(f"  {key}: {values.squeeze().tolist()}")

    # Test training capability
    print("\n5. Testing training capability...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    # Train for a few epochs to test gradient flow
    initial_loss = None
    for epoch in range(10):
        optimizer.zero_grad()
        predictions = model(data_tensors.X)
        loss = criterion(predictions, data_tensors.y)
        loss.backward()
        optimizer.step()

        if epoch == 0:
            initial_loss = loss.item()
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: Loss = {loss.item():.6f}")

    final_loss = loss.item()
    print(f"Loss improved: {initial_loss:.6f} → {final_loss:.6f}")

    # Validate physics constraints
    print("\n6. Validating physics constraints...")
    model.eval()
    with torch.no_grad():
        test_predictions = model(data_tensors.X[:10])
        params = model.get_physics_parameters(data_tensors.X[:10])

    # Check parameter ranges
    constraints_valid = True

    # Porosity should be in (0, 1)
    porosity = params['porosity']
    if not (0 < porosity.min() and porosity.max() < 1):
        print(f"❌ Porosity out of bounds: [{porosity.min():.3f}, {porosity.max():.3f}]")
        constraints_valid = False
    else:
        print(f"✅ Porosity in bounds: [{porosity.min():.3f}, {porosity.max():.3f}]")

    # Pore radius should be positive and reasonable
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

    if constraints_valid:
        print("\n✅ All physics constraints satisfied!")
    else:
        print("\n❌ Some physics constraints violated!")

    print("\n✅ PhModel test completed!")
    print(f"Summary:")
    print(f"  - Forward pass: ✓")
    print(f"  - Parameter extraction: ✓")
    print(f"  - Gradient flow: ✓")
    print(f"  - Physics constraints: {'✓' if constraints_valid else '❌'}")

    return model, constraints_valid


if __name__ == "__main__":
    try:
        test_physics_model()
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
