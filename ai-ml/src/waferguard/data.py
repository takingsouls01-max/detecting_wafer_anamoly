"""
Data loading and legacy probe station detection.

Implements Bug Fix #3: Proper handling of categorical firmware codes
from Probe Station PS-L (legacy).
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass


@dataclass
class DatasetInfo:
    """Dataset metadata for reproducibility."""
    n_samples: int
    n_features: int
    defect_count: int
    defect_rate: float
    legacy_count: int
    legacy_rate: float
    missing_rate: float
    timestamp_range: Tuple[str, str]

    def __str__(self):
        return (
            f"Dataset: {self.n_samples} samples x {self.n_features} features\n"
            f"Defects: {self.defect_count} ({self.defect_rate:.2%})\n"
            f"Legacy wafers: {self.legacy_count} ({self.legacy_rate:.2%})\n"
            f"Missing values: {self.missing_rate:.2%}\n"
            f"Time range: {self.timestamp_range[0]} to {self.timestamp_range[1]}"
        )


class WaferDataset:
    """
    Load and prepare wafer telemetry data with proper legacy station handling.

    Legacy Probe Station PS-L Business Rules:
    - sensor_0 == 0.0 (exact) identifies legacy wafers (sentinel value)
    - For legacy wafers, sensor_1 contains categorical firmware codes: 100.0, 200.0, 300.0
    - These codes MUST NOT enter continuous scaling
    - Modern wafers have continuous sensor_0 and sensor_1 values
    """

    LEGACY_SENTINEL = 0.0  # Exact value for sensor_0 indicating legacy station
    KNOWN_FIRMWARE_CODES = {100.0, 200.0, 300.0}  # Known categorical codes

    def __init__(self, filepath: str):
        """Load dataset and identify legacy wafers."""
        self.df = pd.read_csv(filepath)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])

        # Sort by timestamp for temporal ordering (critical for causality)
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)

        # Identify sensor columns
        self.sensor_cols = [c for c in self.df.columns if c.startswith('sensor_')]
        self.n_features = len(self.sensor_cols)

        # Detect legacy wafers using EXACT sentinel value
        self.legacy_mask = (self.df['sensor_0'] == self.LEGACY_SENTINEL).to_numpy()

        # Validate legacy station detection
        self._validate_legacy_detection()

        # Extract labels (1 = defective, 0 = pass)
        self.y = (self.df['label'] == 1).astype(np.int64).to_numpy()

        # Store timestamp for temporal splitting
        self.timestamps = self.df['timestamp'].to_numpy()

        self.info = self._compute_info()

    def _validate_legacy_detection(self):
        """Validate that legacy wafers have expected categorical codes in sensor_1."""
        if not self.legacy_mask.any():
            return  # No legacy wafers found

        legacy_sensor1 = self.df.loc[self.legacy_mask, 'sensor_1']
        unique_codes = set(legacy_sensor1.dropna().unique())

        # Check if codes are in expected set
        unexpected = unique_codes - self.KNOWN_FIRMWARE_CODES
        if unexpected:
            print(f"⚠️  Warning: Found unexpected firmware codes: {unexpected}")

        # Verify legacy wafers have sensor_0 == 0.0 exactly
        legacy_sensor0 = self.df.loc[self.legacy_mask, 'sensor_0']
        assert (legacy_sensor0 == self.LEGACY_SENTINEL).all(), \
            "Legacy mask inconsistent with sensor_0 values"

    def _compute_info(self) -> DatasetInfo:
        """Compute dataset statistics."""
        return DatasetInfo(
            n_samples=len(self.df),
            n_features=self.n_features,
            defect_count=self.y.sum(),
            defect_rate=self.y.mean(),
            legacy_count=self.legacy_mask.sum(),
            legacy_rate=self.legacy_mask.mean(),
            missing_rate=self.df[self.sensor_cols].isnull().sum().sum() /
                        (len(self.df) * self.n_features),
            timestamp_range=(
                str(self.timestamps.min()),
                str(self.timestamps.max())
            )
        )

    def get_features(self) -> np.ndarray:
        """
        Get raw feature matrix [N, F].

        Returns continuous sensor values as-is.
        Legacy sensor_1 codes remain as floats (100.0, 200.0, 300.0)
        but will be handled separately in preprocessing.
        """
        return self.df[self.sensor_cols].to_numpy(dtype=np.float64)

    def get_legacy_features(self) -> Dict[str, np.ndarray]:
        """
        Extract legacy-specific features for separate handling.

        Returns:
            dict with keys:
                'mask': boolean array indicating legacy wafers
                'sensor_1_codes': categorical codes for legacy wafers (with NaN for modern)
                'sensor_0_indicator': binary indicator (1=legacy, 0=modern)
        """
        # Create array for sensor_1 codes (NaN for modern wafers)
        sensor_1_codes = np.full(len(self.df), np.nan)
        sensor_1_codes[self.legacy_mask] = self.df.loc[self.legacy_mask, 'sensor_1'].to_numpy()

        return {
            'mask': self.legacy_mask,
            'sensor_1_codes': sensor_1_codes,
            'sensor_0_indicator': self.legacy_mask.astype(np.float32),
        }

    def temporal_split(
        self,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
        test_frac: float = 0.15
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data chronologically by timestamp.

        Critical for causality: ensures no future information leaks into past.

        Args:
            train_frac: Fraction for training (earliest)
            val_frac: Fraction for validation (middle)
            test_frac: Fraction for testing (latest)

        Returns:
            train_idx, val_idx, test_idx: indices for each split
        """
        assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, \
            "Split fractions must sum to 1.0"

        n = len(self.df)
        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))

        # Data is already sorted by timestamp
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(train_end, val_end)
        test_idx = np.arange(val_end, n)

        return train_idx, val_idx, test_idx
