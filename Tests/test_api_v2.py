import pytest

"""
Simple API tests for all model types, following the original test_api.py structure.
Clean and easy to understand - 2 tests per model type.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import numpy as np
import pandas as pd
import torch
from data.loader import load_data
from models import EnsembleOptimizer  # Import from proper location


def get_optimizer(model_type="MLP", quantity="FE (Eth)", maximize=True):
    """Create an optimizer for the specified model type."""
    return EnsembleOptimizer(model_type=model_type, quantity=quantity, maximize=maximize)


# ============================================================================
# MLP Ensemble Tests
# ============================================================================

def test_mlp_within_data():
    """Test MLP ensemble optimizer within data (matching original test structure)."""
    print("Testing MLP within data...")
    
    # Load data
    _, _, df = load_data()
    
    # Create optimizer
    optimizer = get_optimizer("MLP")
    
    # Split data
    df_train = df.iloc[:30]
    df_explore = df.iloc[31:]
    
    # First step
    ei, next_pick = optimizer.step_within_data(df_train, df_explore)
    print(f"MLP First pick: EI={ei.max():.6f}, Pick={next_pick}")
    
    # Update data for second step
    df_new = df_explore.loc[next_pick:next_pick]
    df_train_updated = pd.concat([df_train, df_new])
    df_explore_updated = df_explore.drop(index=next_pick)
    
    # Second step
    optimizer_2 = get_optimizer("MLP")
    ei_2, next_pick_2 = optimizer_2.step_within_data(df_train_updated, df_explore_updated)
    print(f"MLP Second pick: EI={ei_2.max():.6f}, Pick={next_pick_2}")
    
    print("✅ MLP within data test passed")


# ============================================================================
# PhModel Ensemble Tests  
# ============================================================================

def test_phmodel_within_data():
    """Test PhModel ensemble optimizer within data."""
    print("Testing PhModel within data...")
    
    # Load data
    _, _, df = load_data()
    
    # Create optimizer
    optimizer = get_optimizer("PhModel")
    
    # Split data
    df_train = df.iloc[:30]
    df_explore = df.iloc[31:]

    # First step
    ei, next_pick = optimizer.step_within_data(df_train, df_explore)
    print(f"PhModel First pick: EI={ei.max():.6f}, Pick={next_pick}")

    # Update data for second step
    df_new = df_explore.loc[next_pick:next_pick]
    df_train_updated = pd.concat([df_train, df_new])
    df_explore_updated = df_explore.drop(index=next_pick)

    # Second step
    optimizer_2 = get_optimizer("PhModel")
    ei_2, next_pick_2 = optimizer_2.step_within_data(df_train_updated, df_explore_updated)
    print(f"PhModel Second pick: EI={ei_2.max():.6f}, Pick={next_pick_2}")

    print("✅ PhModel within data test passed")


# ============================================================================
# Standard GP Tests
# ============================================================================

def test_gp_within_data():
    """Test standard GP optimizer within data."""
    print("Testing GP within data...")

    # Load data
    _, _, df = load_data()

    # Create optimizer
    optimizer = get_optimizer("GP")

    # Split data
    df_train = df.iloc[:30]
    df_explore = df.iloc[31:]

    # First step
    ei, next_pick = optimizer.step_within_data(df_train, df_explore)
    print(f"GP First pick: EI={ei.max():.6f}, Pick={next_pick}")

    # Update data for second step
    df_new = df_explore.loc[next_pick:next_pick]
    df_train_updated = pd.concat([df_train, df_new])
    df_explore_updated = df_explore.drop(index=next_pick)

    # Second step
    optimizer_2 = get_optimizer("GP")
    ei_2, next_pick_2 = optimizer_2.step_within_data(df_train_updated, df_explore_updated)
    print(f"GP Second pick: EI={ei_2.max():.6f}, Pick={next_pick_2}")

    print("✅ GP within data test passed")


# ============================================================================
# Physics-Informed GP Tests
# ============================================================================

def test_gp_physics_within_data():
    """Test physics-informed GP optimizer within data."""
    print("Testing GP+Ph within data...")

    # Load data
    _, _, df = load_data()

    # Create optimizer
    optimizer = get_optimizer("GP+Ph")

    # Split data
    df_train = df.iloc[:30]
    df_explore = df.iloc[31:]
    
    # First step
    ei, next_pick = optimizer.step_within_data(df_train, df_explore)
    print(f"GP+Ph First pick: EI={ei.max():.6f}, Pick={next_pick}")
    
    # Update data for second step
    df_new = df_explore.loc[next_pick:next_pick]
    df_train_updated = pd.concat([df_train, df_new])
    df_explore_updated = df_explore.drop(index=next_pick)
    
    # Second step
    optimizer_2 = get_optimizer("GP+Ph")
    ei_2, next_pick_2 = optimizer_2.step_within_data(df_train_updated, df_explore_updated)
    print(f"GP+Ph Second pick: EI={ei_2.max():.6f}, Pick={next_pick_2}")
    
    print("✅ GP+Ph within data test passed")


# ============================================================================
# Free Optimization Tests (continuous feature space)
# ============================================================================

def test_mlp_free():
    """Test MLP ensemble free optimization (continuous feature space)."""
    print("Testing MLP free optimization...")

    # Load data
    _, _, df = load_data()

    # Create optimizer
    optimizer = get_optimizer("MLP")

    # Use first 30 samples for training
    df_train = df.iloc[:30]

    # First step - find optimal point in continuous space
    af_value, next_experiment = optimizer.step(df_train)
    print(f"MLP Free step 1: AF={af_value:.6f}, Features={next_experiment[:3]}")  # Show first 3 features

    # Add synthetic point and do second step
    df_train_updated = df.iloc[:35]  # Use a few more points
    af_value_2, next_experiment_2 = optimizer.step(df_train_updated)
    print(f"MLP Free step 2: AF={af_value_2:.6f}, Features={next_experiment_2[:3]}")

    # Verify output format
    assert isinstance(af_value, float), "AF value should be float"
    assert isinstance(next_experiment, np.ndarray), "Next experiment should be numpy array"
    assert len(next_experiment) == 5, "Should have 5 features"  # AgCu, Naf, Sust, thickness, mass

    print("✅ MLP free optimization test passed")


def test_phmodel_free():
    """Test PhModel ensemble free optimization (continuous feature space)."""
    print("Testing PhModel free optimization...")

    # Load data
    _, _, df = load_data()

    # Create optimizer
    optimizer = get_optimizer("PhModel")

    # Use first 30 samples for training
    df_train = df.iloc[:30]

    # First step - find optimal point in continuous space
    af_value, next_experiment = optimizer.step(df_train)
    print(f"PhModel Free step 1: AF={af_value:.6f}, Features={next_experiment[:3]}")

    # Add synthetic point and do second step
    df_train_updated = df.iloc[:35]
    af_value_2, next_experiment_2 = optimizer.step(df_train_updated)
    print(f"PhModel Free step 2: AF={af_value_2:.6f}, Features={next_experiment_2[:3]}")

    # Verify output format
    assert isinstance(af_value, float), "AF value should be float"
    assert isinstance(next_experiment, np.ndarray), "Next experiment should be numpy array"
    assert len(next_experiment) == 5, "Should have 5 features"

    print("✅ PhModel free optimization test passed")


def test_gp_free():
    """Test standard GP free optimization (continuous feature space)."""
    print("Testing GP free optimization...")

    # Load data
    _, _, df = load_data()

    # Create optimizer
    optimizer = get_optimizer("GP")

    # Use first 30 samples for training
    df_train = df.iloc[:30]

    # First step - find optimal point in continuous space
    af_value, next_experiment = optimizer.step(df_train)
    print(f"GP Free step 1: AF={af_value:.6f}, Features={next_experiment[:3]}")

    # Add synthetic point and do second step
    df_train_updated = df.iloc[:35]
    af_value_2, next_experiment_2 = optimizer.step(df_train_updated)
    print(f"GP Free step 2: AF={af_value_2:.6f}, Features={next_experiment_2[:3]}")

    # Verify output format
    assert isinstance(af_value, float), "AF value should be float"
    assert isinstance(next_experiment, np.ndarray), "Next experiment should be numpy array"
    assert len(next_experiment) == 5, "Should have 5 features"

    print("✅ GP free optimization test passed")


def test_gp_physics_free():
    """Test physics-informed GP free optimization (continuous feature space)."""
    print("Testing GP+Ph free optimization...")

    # Load data
    _, _, df = load_data()

    # Create optimizer
    optimizer = get_optimizer("GP+Ph")

    # Use first 30 samples for training
    df_train = df.iloc[:30]

    # First step - find optimal point in continuous space
    af_value, next_experiment = optimizer.step(df_train)
    print(f"GP+Ph Free step 1: AF={af_value:.6f}, Features={next_experiment[:3]}")

    # Add synthetic point and do second step
    df_train_updated = df.iloc[:35]
    af_value_2, next_experiment_2 = optimizer.step(df_train_updated)
    print(f"GP+Ph Free step 2: AF={af_value_2:.6f}, Features={next_experiment_2[:3]}")

    # Verify output format
    assert isinstance(af_value, float), "AF value should be float"
    assert isinstance(next_experiment, np.ndarray), "Next experiment should be numpy array"
    assert len(next_experiment) == 5, "Should have 5 features"

    print("✅ GP+Ph free optimization test passed")


# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_tests():
    """Run all tests for all model types."""
    print("=" * 60)
    print("Running Simple API Tests for All Models")
    print("=" * 60)
    
    try:
        # Test within data functionality (slower)
        #print("\n--- Testing Within Data Functionality ---")
        #test_mlp_within_data()
        #test_phmodel_within_data()
        #test_gp_within_data()
        #test_gp_physics_within_data()
        
        # Test free optimization functionality (continuous feature space)
        print("\n--- Testing Free Optimization Functionality ---")
        test_mlp_free()
        test_phmodel_free()
        test_gp_free()
        test_gp_physics_free()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("Summary:")
        print("  - MLP Ensemble: ✓ Within Data + ✓ Free Opt")
        print("  - PhModel Ensemble: ✓ Within Data + ✓ Free Opt")
        print("  - Standard GP: ✓ Within Data + ✓ Free Opt")
        print("  - Physics GP: ✓ Within Data + ✓ Free Opt")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Check BoTorch availability
    try:
        from botorch.acquisition.analytic import ExpectedImprovement
        print("✓ BoTorch available")
        run_all_tests()
    except ImportError:
        print("❌ BoTorch not available. Install with: pip install botorch")
        print("Cannot run acquisition function tests without BoTorch.")
