"""
Export feature schema documentation.
Explains all 594 output features, their transformations, and handling.
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from waferguard.data import WaferDataset
from waferguard.preprocessing import WaferPreprocessor


def generate_feature_schema():
    """Generate comprehensive feature schema CSV."""

    schema_rows = []

    # Original 590 continuous features
    for i in range(590):
        row = {
            'feature_index': i,
            'feature_name': f'sensor_{i}_scaled',
            'source': f'sensor_{i}',
            'transformation': 'mean_imputation -> z_score_scaling',
            'station_applicability': 'modern' if i == 1 else 'both',
            'special_handling': 'legacy_sensor_1_excluded_for_legacy_wafers' if i == 1 else 'none',
            'reconstruction_target': 'yes',
            'notes': 'Categorical codes (100,200,300) excluded from scaling for legacy sensor_1' if i == 1 else ''
        }
        schema_rows.append(row)

    # Firmware code one-hot features (3 features)
    for idx, code in enumerate([100.0, 200.0, 300.0]):
        row = {
            'feature_index': 590 + idx,
            'feature_name': f'firmware_{int(code)}',
            'source': 'sensor_1 (legacy wafers only)',
            'transformation': 'one_hot_encoding',
            'station_applicability': 'legacy_only',
            'special_handling': 'binary_indicator_for_categorical_code',
            'reconstruction_target': 'no',
            'notes': f'Indicator for firmware error code {int(code)}'
        }
        schema_rows.append(row)

    # Legacy indicator feature (1 feature)
    row = {
        'feature_index': 593,
        'feature_name': 'is_legacy',
        'source': 'sensor_0 == 0.0 (exact sentinel)',
        'transformation': 'binary_indicator',
        'station_applicability': 'both',
        'special_handling': 'station_type_indicator',
        'reconstruction_target': 'no',
        'notes': '1 = legacy station PS-L, 0 = modern station'
    }
    schema_rows.append(row)

    df = pd.DataFrame(schema_rows)
    return df


if __name__ == "__main__":
    import os
    os.makedirs('reports', exist_ok=True)

    schema = generate_feature_schema()
    schema.to_csv('reports/feature_schema.csv', index=False)

    print("Feature Schema Generated")
    print("=" * 60)
    print(f"Total features: {len(schema)}")
    print(f"\nBreakdown:")
    print(f"  Continuous (scaled): 590")
    print(f"  Firmware codes (one-hot): 3")
    print(f"  Legacy indicator: 1")
    print(f"  Total: 594")
    print(f"\nSaved to: reports/feature_schema.csv")

    # Show key rows
    print(f"\nKey features:")
    print(schema[schema['feature_index'].isin([0, 1, 590, 591, 592, 593])][
        ['feature_index', 'feature_name', 'source', 'transformation']
    ].to_string(index=False))
