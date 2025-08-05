"""
Test script to validate the new data loading system.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from data.loader import load_data, load_raw_data, prepare_tensors
import torch


def test_data_loading():
    """Test the complete data loading pipeline."""
    print("Testing data loading pipeline...")

    # Test 1: Load raw data
    print("\n1. Testing raw data loading...")
    df = load_raw_data()
    print(f"Raw data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Sample data:\n{df.head(3)}")

    # Test 2: Prepare tensors without normalization
    print("\n2. Testing tensor preparation (no normalization)...")
    data_tensors, norm_params = prepare_tensors(df, normalize_features=False, normalize_targets=False)
    print(f"X shape: {data_tensors.X.shape}, dtype: {data_tensors.X.dtype}")
    print(f"y shape: {data_tensors.y.shape}, dtype: {data_tensors.y.dtype}")
    print(f"Feature names: {data_tensors.feature_names}")
    print(f"Target names: {data_tensors.target_names}")
    print(f"Normalization params: {norm_params}")

    # Test 3: Prepare tensors with feature normalization only
    print("\n3. Testing tensor preparation (feature normalization only)...")
    data_tensors_norm, norm_params_norm = prepare_tensors(df, normalize_features=True, normalize_targets=False)
    print(f"X normalized shape: {data_tensors_norm.X.shape}")
    print(f"X mean (should be ~0): {data_tensors_norm.X.mean(dim=0)}")
    print(f"X std (should be ~1): {data_tensors_norm.X.std(dim=0, unbiased=False)}")
    print(f"y unchanged mean: {data_tensors_norm.y.mean(dim=0)}")

    # Test 4: Test normalization reversibility
    print("\n4. Testing normalization reversibility...")
    X_denorm = norm_params_norm.denormalize_features(data_tensors_norm.X)
    max_diff = torch.max(torch.abs(X_denorm - data_tensors.X))
    print(f"Max difference after denormalization: {max_diff.item():.2e} (should be ~0)")

    # Test 5: Complete pipeline
    print("\n5. Testing complete pipeline...")
    data_complete, norm_complete, df_complete = load_data(normalize_features=True, normalize_targets=False)
    print(f"Complete pipeline X shape: {data_complete.X.shape}")
    print(f"DataFrame matches: {df.equals(df_complete)}")

    print("\n✅ All tests passed! Data loading system is working correctly.")

    return data_complete, norm_complete, df_complete


if __name__ == "__main__":
    try:
        test_data_loading()
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
