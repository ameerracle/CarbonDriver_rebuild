"""
Test to verify triplet data handling matches the old implementation.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import numpy as np
import pandas as pd
import os
from data.loader import load_raw_data, load_data


def get_data_file_path(filename):
    """Return absolute path to a file in the data folder."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', filename))


def test_triplet_handling():
    """Test if triplet data structure is handled correctly."""
    print("Testing triplet data handling...")

    # Load raw data with our new loader
    print("\n1. Testing our new data loader...")
    df_new = load_raw_data()
    print(f"New loader data shape: {df_new.shape}")
    print(f"First few rows:\n{df_new.head()}")

    # Manually replicate old triplet logic to compare
    print("\n2. Manually replicating old triplet logic...")

    # Load raw Excel data (before triplet shuffling)
    raw_df = pd.read_excel(get_data_file_path('Characterization_data.xlsx'), skiprows=[1], index_col=0)
    raw_df = raw_df[['AgCu Ratio', 'Naf vol (ul)', 'Sust vol (ul)', 'Catalyst mass loading', 'FE (Eth)', 'FE (CO)']]
    raw_df = raw_df.sort_values(by=['AgCu Ratio', 'Naf vol (ul)'])
    raw_df = raw_df.dropna()

    # Convert FE percentages to fractions
    raw_df['FE (CO)'] = raw_df['FE (CO)'] / 100
    raw_df['FE (Eth)'] = raw_df['FE (Eth)'] / 100

    # Add thickness calculation
    dens_Ag = 10490
    dens_Cu = 8935
    dens_avg = (1 - raw_df['AgCu Ratio']) * dens_Cu + raw_df['AgCu Ratio'] * dens_Ag
    mass = raw_df['Catalyst mass loading'] * 1e-6
    area = 1.85**2
    A = area * 1e-4
    thickness = (mass / dens_avg) / A
    raw_df.insert(3, column='Zero_eps_thickness', value=thickness)

    print(f"Before triplet shuffling: {raw_df.shape}")
    print("Sample of grouped data (first 9 rows - should be 3 triplets):")
    print(raw_df.head(9)[['AgCu Ratio', 'Naf vol (ul)', 'Sust vol (ul)']])

    # Apply triplet shuffling (OLD METHOD)
    raw_df['triplet'] = np.arange(len(raw_df)) // 3
    gen = np.random.default_rng(2)  # Same seed as our loader
    order = gen.permutation(30)

    old_method_df = pd.DataFrame()
    for i in order:
        old_method_df = pd.concat([old_method_df, raw_df[raw_df['triplet'] == i]])
    old_method_df.reset_index(drop=True, inplace=True)
    old_method_df = old_method_df.drop(columns=['triplet'])

    print(f"\nAfter old method triplet shuffling: {old_method_df.shape}")
    print("First few rows after old method:")
    print(old_method_df.head(3)[['AgCu Ratio', 'Naf vol (ul)', 'Sust vol (ul)']])

    # Compare with our new loader
    print(f"\nFirst few rows from our new loader:")
    print(df_new.head(3)[['AgCu Ratio', 'Naf vol (ul)', 'Sust vol (ul)']])

    # Check if they match
    columns_to_compare = ['AgCu Ratio', 'Naf vol (ul)', 'Sust vol (ul)', 'Zero_eps_thickness', 'Catalyst mass loading', 'FE (Eth)', 'FE (CO)']
    match = old_method_df[columns_to_compare].equals(df_new[columns_to_compare])

    print(f"\n3. Comparison result:")
    print(f"Data matches old method: {match}")

    if match:
        print("✅ Triplet handling is correct!")
    else:
        print("❌ Triplet handling differs from old method")
        print("Checking differences...")
        for col in columns_to_compare:
            col_match = old_method_df[col].equals(df_new[col])
            print(f"  {col}: {col_match}")

    return match


if __name__ == "__main__":
    test_triplet_handling()
