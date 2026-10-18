"""
Causality-safe preprocessing pipeline.

Implements Bug Fix #1: Prevents data leakage by fitting only on training data.
Implements Bug Fix #3: Separates categorical and continuous features.
"""

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Dict, Tuple, Optional
import joblib


class WaferPreprocessor:
    """
    Preprocessing pipeline that respects temporal causality.

    Key fixes:
    1. Fit ONLY on training data (no test data leakage)
    2. Handle legacy categorical codes separately from continuous features
    3. Maintain explicit separation between modern and legacy sensor_1
    """

    def __init__(self):
        # Separate imputers for continuous and legacy features
        self.continuous_imputer = SimpleImputer(strategy='mean', keep_empty_features=True)
        self.legacy_imputer = SimpleImputer(strategy='most_frequent', keep_empty_features=True)

        # Scaler for continuous features only
        self.scaler = StandardScaler()

        # One-hot encoder for categorical firmware codes
        self.firmware_encoder = OneHotEncoder(
            categories=[[100.0, 200.0, 300.0]],  # Known codes
            handle_unknown='ignore',  # Handle unknown codes gracefully
            sparse_output=False
        )

        self.is_fitted = False
        self.n_continuous_features = None
        self.feature_names = None

    def fit(self, X: np.ndarray, legacy_info: Dict[str, np.ndarray]) -> 'WaferPreprocessor':
        """
        Fit preprocessing transformations ONLY on training data.

        Args:
            X: Raw feature matrix [N, F] (590 features)
            legacy_info: Dict with 'mask', 'sensor_1_codes', 'sensor_0_indicator'

        Returns:
            self (fitted preprocessor)
        """
        legacy_mask = legacy_info['mask']
        sensor_1_codes = legacy_info['sensor_1_codes']

        # Separate continuous and legacy features
        # For legacy wafers: sensor_0 is sentinel (0.0), sensor_1 is categorical
        # For modern wafers: both are continuous

        # Step 1: Prepare continuous features
        # - Exclude sensor_1 for legacy wafers (will be one-hot encoded)
        # - Keep sensor_0 as-is (it's the sentinel for legacy, continuous for modern)
        X_continuous = X.copy()
        X_continuous[legacy_mask, 1] = np.nan  # Mark legacy sensor_1 as missing

        # Fit imputer and scaler on continuous features
        self.continuous_imputer.fit(X_continuous)
        X_cont_imputed = self.continuous_imputer.transform(X_continuous)
        self.scaler.fit(X_cont_imputed)

        # Step 2: Fit categorical encoder on legacy sensor_1 codes
        # Extract valid (non-NaN) categorical codes
        valid_codes_mask = ~np.isnan(sensor_1_codes)
        if valid_codes_mask.any():
            codes_for_fitting = sensor_1_codes[valid_codes_mask].reshape(-1, 1)
            self.firmware_encoder.fit(codes_for_fitting)

        self.n_continuous_features = X.shape[1]
        self.feature_names = self._generate_feature_names()
        self.is_fitted = True

        return self

    def transform(self, X: np.ndarray, legacy_info: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Transform features using fitted preprocessors.

        Args:
            X: Raw feature matrix [N, F]
            legacy_info: Dict with legacy wafer information

        Returns:
            X_transformed: [N, F_out] where F_out = 590 + 3 + 1
                           = continuous_scaled + firmware_codes_one_hot + legacy_indicator
        """
        assert self.is_fitted, "Must call fit() before transform()"

        legacy_mask = legacy_info['mask']
        sensor_1_codes = legacy_info['sensor_1_codes']
        legacy_indicator = legacy_info['sensor_0_indicator']

        # Step 1: Process continuous features
        X_continuous = X.copy()
        X_continuous[legacy_mask, 1] = np.nan  # Exclude legacy sensor_1

        X_cont_imputed = self.continuous_imputer.transform(X_continuous)
        X_cont_scaled = self.scaler.transform(X_cont_imputed)

        # Step 2: Process categorical features (legacy sensor_1 codes)
        # Initialize one-hot matrix (N, 3) for three firmware codes
        X_firmware_onehot = np.zeros((X.shape[0], 3), dtype=np.float32)

        valid_codes_mask = ~np.isnan(sensor_1_codes)
        if valid_codes_mask.any():
            codes_to_encode = sensor_1_codes[valid_codes_mask].reshape(-1, 1)
            encoded = self.firmware_encoder.transform(codes_to_encode)
            X_firmware_onehot[valid_codes_mask] = encoded

        # Step 3: Add legacy indicator feature
        X_legacy_ind = legacy_indicator.reshape(-1, 1)

        # Concatenate all features
        X_transformed = np.concatenate([
            X_cont_scaled,           # 590 continuous features
            X_firmware_onehot,       # 3 one-hot encoded firmware codes
            X_legacy_ind             # 1 legacy indicator
        ], axis=1)

        return X_transformed

    def fit_transform(self, X: np.ndarray, legacy_info: Dict[str, np.ndarray]) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, legacy_info).transform(X, legacy_info)

    def _generate_feature_names(self):
        """Generate feature names for interpretability."""
        names = [f"sensor_{i}_scaled" for i in range(self.n_continuous_features)]
        names += ["firmware_100", "firmware_200", "firmware_300"]
        names += ["is_legacy"]
        return names

    def get_feature_count(self) -> int:
        """Get total number of output features."""
        return self.n_continuous_features + 4  # +3 firmware codes +1 legacy indicator

    def save(self, filepath: str):
        """Save fitted preprocessor to disk."""
        assert self.is_fitted, "Cannot save unfitted preprocessor"
        joblib.dump({
            'continuous_imputer': self.continuous_imputer,
            'legacy_imputer': self.legacy_imputer,
            'scaler': self.scaler,
            'firmware_encoder': self.firmware_encoder,
            'n_continuous_features': self.n_continuous_features,
            'feature_names': self.feature_names,
        }, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'WaferPreprocessor':
        """Load fitted preprocessor from disk."""
        data = joblib.load(filepath)
        processor = cls()
        processor.continuous_imputer = data['continuous_imputer']
        processor.legacy_imputer = data['legacy_imputer']
        processor.scaler = data['scaler']
        processor.firmware_encoder = data['firmware_encoder']
        processor.n_continuous_features = data['n_continuous_features']
        processor.feature_names = data['feature_names']
        processor.is_fitted = True
        return processor
