"""
Clean data loading and preprocessing module.
Eliminates confusion between normalized/unnormalized data flows.
"""
from pathlib import Path
from typing import Tuple, Optional, NamedTuple
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
import os


class DataLoader:
    """Simple DataLoader class for compatibility with evaluation framework."""

    def __init__(self, excel_path: str = "Characterization_data.xlsx", random_seed: int = 42):
        # Resolve Excel path robustly, prefer Tests fixture, then project root, then provided path
        self.random_seed = random_seed
        self.excel_path = str(_resolve_excel_path(excel_path))

    def get_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load data and return X, y tensors."""
        df = load_raw_data(Path(self.excel_path))
        data_tensors, _ = prepare_tensors(df, normalize_features=True, normalize_targets=False)
        return data_tensors.X, data_tensors.y


@dataclass
class DataTensors:
    """Container for processed data tensors with clear naming."""
    X: torch.Tensor  # Input features
    y: torch.Tensor  # Target outputs
    feature_names: list[str]
    target_names: list[str]

    def __post_init__(self):
        """Validate tensor shapes on creation."""
        assert self.X.ndim == 2, f"X must be 2D, got shape {self.X.shape}"
        assert self.y.ndim == 2, f"y must be 2D, got shape {self.y.shape}"
        assert self.X.shape[0] == self.y.shape[0], f"Sample count mismatch: X={self.X.shape[0]}, y={self.y.shape[0]}"
        assert len(self.feature_names) == self.X.shape[1], f"Feature name count mismatch"
        assert len(self.target_names) == self.y.shape[1], f"Target name count mismatch"


@dataclass
class NormalizationParams:
    """Container for normalization parameters."""
    feature_means: torch.Tensor
    feature_stds: torch.Tensor
    target_means: Optional[torch.Tensor] = None
    target_stds: Optional[torch.Tensor] = None

    def normalize_features(self, X: torch.Tensor) -> torch.Tensor:
        """Apply feature normalization."""
        return (X - self.feature_means) / self.feature_stds

    def denormalize_features(self, X_norm: torch.Tensor) -> torch.Tensor:
        """Reverse feature normalization."""
        return X_norm * self.feature_stds + self.feature_means

    def normalize_targets(self, y: torch.Tensor) -> torch.Tensor:
        """Apply target normalization if parameters exist."""
        if self.target_means is not None and self.target_stds is not None:
            return (y - self.target_means) / self.target_stds
        return y

    def denormalize_targets(self, y_norm: torch.Tensor) -> torch.Tensor:
        """Reverse target normalization if parameters exist."""
        if self.target_means is not None and self.target_stds is not None:
            return y_norm * self.target_stds + self.target_means
        return y_norm


def _resolve_excel_path(default_name_or_path: str) -> Path:
    """
    Resolve the Excel path robustly across Windows/WSL and test runs.
    Preference order:
      1) If env CARBONDRIVER_DATA_PATH is set and exists, use it
      2) Tests/Characterization_data.xlsx if exists
      3) Project root Characterization_data.xlsx if exists
      4) The provided default_name_or_path if it exists (relative to CWD or absolute)
    """
    # 1) Environment override
    env_value = os.environ.get("CARBONDRIVER_DATA_PATH", "").strip()
    if env_value:
        env_path = Path(env_value).expanduser()
        # If a directory is provided, append default filename
        if env_path.is_dir():
            env_path = env_path / "Characterization_data.xlsx"
        # Only use if it points to a file
        if env_path.exists() and env_path.is_file():
            return env_path.resolve()

    # 2) Tests fixture
    tests_path = Path("Tests") / "Characterization_data.xlsx"
    if tests_path.exists():
        return tests_path.resolve()

    # 3) Project root
    root_path = Path("Characterization_data.xlsx")
    if root_path.exists():
        return root_path.resolve()

    # 4) Provided argument (may be a name or path)
    candidate = Path(default_name_or_path)
    if candidate.exists():
        return candidate.resolve()

    # If nothing found, return where we expect it by default (Tests) for helpful error
    return tests_path


def load_raw_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load raw experimental data from Excel file.

    Args:
        file_path: Path to Excel file. If None, uses default location.

    Returns:
        Clean DataFrame with processed features and targets.
    """
    # Resolve the path robustly
    if file_path is None:
        file_path = _resolve_excel_path("Characterization_data.xlsx")
    else:
        file_path = Path(file_path)
        # If a directory was mistakenly passed, append default filename
        if file_path.is_dir():
            file_path = file_path / "Characterization_data.xlsx"

    # Validate final path points to a readable file
    fp = Path(file_path)
    if not fp.exists() or not fp.is_file():
        raise FileNotFoundError(
            f"Characterization_data.xlsx not found. "
            f"Set CARBONDRIVER_DATA_PATH to an existing file or place Characterization_data.xlsx in Tests/ or project root."
        )

    # Load and clean data
    df = pd.read_excel(str(fp), skiprows=[1], index_col=0)
    df = df[['AgCu Ratio', 'Naf vol (ul)', 'Sust vol (ul)', 'Catalyst mass loading', 'FE (Eth)', 'FE (CO)']]
    df = df.sort_values(by=['AgCu Ratio', 'Naf vol (ul)'])
    df = df.dropna()

    # Convert FE percentages to fractions
    df['FE (CO)'] = df['FE (CO)'] / 100
    df['FE (Eth)'] = df['FE (Eth)'] / 100

    # Calculate thickness feature
    dens_Ag = 10490  # kg/m^3
    dens_Cu = 8935   # kg/m^3
    dens_avg = (1 - df['AgCu Ratio']) * dens_Cu + df['AgCu Ratio'] * dens_Ag
    mass = df['Catalyst mass loading'] * 1e-6  # kg
    area = 1.85**2  # cm^2
    A = area * 1e-4  # m^2
    thickness = (mass / dens_avg) / A  # m
    df.insert(3, column='Zero_eps_thickness', value=thickness)

    # Reshuffle triplets (reproducible randomization)
    df['triplet'] = np.arange(len(df)) // 3
    gen = np.random.default_rng(2)  # Fixed seed for reproducibility
    order = gen.permutation(30)
    new_df = pd.DataFrame()
    for i in order:
        new_df = pd.concat([new_df, df[df['triplet'] == i]])
    new_df.reset_index(drop=True, inplace=True)
    new_df = new_df.drop(columns=['triplet'])

    return new_df


def prepare_tensors(df: pd.DataFrame, normalize_features: bool = True,
                   normalize_targets: bool = False) -> Tuple[DataTensors, Optional[NormalizationParams]]:
    """
    Convert DataFrame to tensors with optional normalization.

    Args:
        df: Input DataFrame with features and targets
        normalize_features: Whether to normalize input features
        normalize_targets: Whether to normalize target outputs

    Returns:
        DataTensors: Container with X, y tensors and metadata
        NormalizationParams: Normalization parameters (None if no normalization)
    """
    # Split features and targets
    feature_cols = df.columns[:-2].tolist()  # All except last 2 columns
    target_cols = df.columns[-2:].tolist()   # Last 2 columns (FE outputs)

    X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    y = torch.tensor(df[target_cols].values, dtype=torch.float32)

    # Validate shapes
    assert X.shape[0] > 0, "No data samples found"
    assert X.shape[1] == len(feature_cols), "Feature dimension mismatch"
    assert y.shape[1] == len(target_cols), "Target dimension mismatch"

    norm_params = None

    if normalize_features or normalize_targets:
        # Calculate normalization parameters
        feature_means = X.mean(dim=0)
        feature_stds = X.std(dim=0, unbiased=False)

        # Avoid division by zero
        feature_stds = torch.where(feature_stds == 0, torch.ones_like(feature_stds), feature_stds)

        target_means = y.mean(dim=0) if normalize_targets else None
        target_stds = y.std(dim=0, unbiased=False) if normalize_targets else None
        if normalize_targets:
            target_stds = torch.where(target_stds == 0, torch.ones_like(target_stds), target_stds)

        norm_params = NormalizationParams(
            feature_means=feature_means,
            feature_stds=feature_stds,
            target_means=target_means,
            target_stds=target_stds
        )

        # Apply normalization
        if normalize_features:
            X = norm_params.normalize_features(X)
        if normalize_targets:
            y = norm_params.normalize_targets(y)

    data_tensors = DataTensors(
        X=X,
        y=y,
        feature_names=feature_cols,
        target_names=target_cols
    )

    return data_tensors, norm_params


def load_data(file_path: Optional[Path] = None, normalize_features: bool = True,
              normalize_targets: bool = False) -> Tuple[DataTensors, Optional[NormalizationParams], pd.DataFrame]:
    """
    Complete data loading pipeline.

    Args:
        file_path: Path to Excel file
        normalize_features: Whether to normalize input features (default True, matches old behavior)
        normalize_targets: Whether to normalize target outputs (default False, targets stay as 0-1 decimals)

    Returns:
        DataTensors: Processed tensor data
        NormalizationParams: Normalization parameters (None if no normalization)
        DataFrame: Original raw DataFrame for reference

    Note:
        This matches the original behavior where:
        - Input features are normalized (mean=0, std=1)
        - Output targets remain as decimal values (0-1) from percentage conversion
    """
    df = load_raw_data(file_path)
    data_tensors, norm_params = prepare_tensors(df, normalize_features, normalize_targets)

    return data_tensors, norm_params, df

# Absolute imports for project modules (if needed)
# Example: from models.physics_model import PhModel

