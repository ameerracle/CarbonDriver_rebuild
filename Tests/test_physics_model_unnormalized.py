"""
Test PhModel with non-normalized inputs to check robustness.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
from data.loader import load_data
from models.physics_model import PhModel, PhysicsConfig


def test_physics_model_unnormalized():
    """Test if PhModel can work with non-normalized inputs."""
    print("Testing PhModel with non-normalized inputs...")

    # Load data without normalization
    print("\n1. Loading data without normalization...")
    data_tensors_norm, norm_params_norm, df_norm = load_data(
        normalize_features=True,   # Normalized version for comparison
        normalize_targets=False
    )

    data_tensors_raw, norm_params_raw, df_raw = load_data(
        normalize_features=False,  # Non-normalized version to test
        normalize_targets=False
    )

    print(f"Normalized X range: [{data_tensors_norm.X.min():.3f}, {data_tensors_norm.X.max():.3f}]")
    print(f"Raw X range: [{data_tensors_raw.X.min():.3f}, {data_tensors_raw.X.max():.3f}]")
    print(f"Sample raw X values:\n{data_tensors_raw.X[:2]}")

    # Test PhModel with NORMALIZED inputs (should work)
    print("\n2. Testing PhModel with normalized inputs...")
    zlt_mean_norm = norm_params_norm.feature_means[3].item()
    zlt_std_norm = norm_params_norm.feature_stds[3].item()

    config = PhysicsConfig(hidden_dim=32, dropout_rate=0.1)
    model_norm = PhModel(config=config, zlt_mu_stds=(zlt_mean_norm, zlt_std_norm))

    model_norm.eval()
    with torch.no_grad():
        pred_norm = model_norm(data_tensors_norm.X[:5])
        physics_params_norm = model_norm.get_physics_parameters(data_tensors_norm.X[:5])

    print(f"Normalized input predictions: {pred_norm.shape}")
    print(f"Sample normalized prediction: {pred_norm[0]}")
    print(f"Porosity range (normalized): [{physics_params_norm['porosity'].min():.3f}, {physics_params_norm['porosity'].max():.3f}]")

    # Test PhModel with RAW inputs (the main test)
    print("\n3. Testing PhModel with RAW inputs...")

    # For raw inputs, we need to use raw normalization stats for Zero_eps_thickness
    # Or better yet, use the raw values directly
    raw_zlt_mean = data_tensors_raw.X[:, 3].mean().item()  # Direct calculation from raw data
    raw_zlt_std = data_tensors_raw.X[:, 3].std().item()

    print(f"Raw ZLT stats: mean={raw_zlt_mean:.2e}, std={raw_zlt_std:.2e}")

    model_raw = PhModel(config=config, zlt_mu_stds=(raw_zlt_mean, raw_zlt_std))

    try:
        model_raw.eval()
        with torch.no_grad():
            pred_raw = model_raw(data_tensors_raw.X[:5])
            physics_params_raw = model_raw.get_physics_parameters(data_tensors_raw.X[:5])

        print(f"Raw input predictions: {pred_raw.shape}")
        print(f"Sample raw prediction: {pred_raw[0]}")
        print(f"Prediction range: [{pred_raw.min():.3f}, {pred_raw.max():.3f}]")

        # Check if physics constraints are still satisfied
        print("\n4. Validating physics constraints with raw inputs...")

        porosity_raw = physics_params_raw['porosity']
        pore_radius_raw = physics_params_raw['pore_radius']
        theta_sum_raw = (physics_params_raw['theta_CO'] +
                        physics_params_raw['theta_C2H4'] +
                        physics_params_raw['theta_H2b'])

        constraints_ok = True

        # Check porosity bounds
        if not (0 < porosity_raw.min() and porosity_raw.max() < 1):
            print(f"❌ Porosity out of bounds: [{porosity_raw.min():.3f}, {porosity_raw.max():.3f}]")
            constraints_ok = False
        else:
            print(f"✅ Porosity in bounds: [{porosity_raw.min():.3f}, {porosity_raw.max():.3f}]")

        # Check pore radius
        if not (1e-10 < pore_radius_raw.min() and pore_radius_raw.max() < 1e-6):
            print(f"❌ Pore radius unreasonable: [{pore_radius_raw.min():.2e}, {pore_radius_raw.max():.2e}]")
            constraints_ok = False
        else:
            print(f"✅ Pore radius reasonable: [{pore_radius_raw.min():.2e}, {pore_radius_raw.max():.2e}]")

        # Check surface coverage sum
        if not (theta_sum_raw.max() <= 1.01):
            print(f"❌ Surface coverage sum > 1: max = {theta_sum_raw.max():.3f}")
            constraints_ok = False
        else:
            print(f"✅ Surface coverage sum ≤ 1: max = {theta_sum_raw.max():.3f}")

        # Check prediction bounds
        if not (0 <= pred_raw.min() and pred_raw.max() <= 1):
            print(f"❌ Predictions out of [0,1]: [{pred_raw.min():.3f}, {pred_raw.max():.3f}]")
            constraints_ok = False
        else:
            print(f"✅ Predictions in [0,1]: [{pred_raw.min():.3f}, {pred_raw.max():.3f}]")

        if constraints_ok:
            print("\n✅ PhModel works with non-normalized inputs!")
            print("Physics constraints are maintained even with raw input scales.")
        else:
            print("\n❌ PhModel struggles with non-normalized inputs - constraints violated")

        # Compare predictions
        print("\n5. Comparing normalized vs raw input predictions...")
        print(f"Normalized pred sample: {pred_norm[0]}")
        print(f"Raw pred sample:        {pred_raw[0]}")
        pred_diff = torch.abs(pred_norm[0] - pred_raw[0]).max()
        print(f"Max prediction difference: {pred_diff:.3f}")

        return True, constraints_ok

    except Exception as e:
        print(f"❌ PhModel failed with raw inputs: {e}")
        import traceback
        traceback.print_exc()
        return False, False


if __name__ == "__main__":
    test_physics_model_unnormalized()
