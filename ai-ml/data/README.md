# Data Directory

This directory contains the dataset for the wafer anomaly detector challenge.

## Files (Not Included in Git)

The following files are **not pushed to GitHub** due to size:

- `secom_telemetry_modified.csv` (5.5 MB) - Main dataset
- `secom_legacy_mask_EVALUATOR_ONLY.csv` (4.4 KB) - Evaluator reference

## How to Obtain Data

**For reviewers/reproducers:**

1. Download from the GDG challenge materials:
   - Google Drive: https://drive.google.com/drive/folders/1bdLOpmJkJ1Ck3esjAscStXcz0Sj3C9RS

2. Or use `gdown`:
   ```bash
   pip install gdown
   cd ai-ml/data
   gdown --folder https://drive.google.com/drive/folders/1bdLOpmJkJ1Ck3esjAscStXcz0Sj3C9RS
   ```

## Files Included

- `bnypi.ipynb` - Original broken notebook (for reference)

## Data Description

**Dataset:** Modified SECOM semiconductor manufacturing data  
**Samples:** 1,567 wafers  
**Features:** 590 sensor measurements  
**Labels:** -1 (pass), +1 (fail)  
**Defect Rate:** 6.64% (104 fails / 1,463 pass)  
**Legacy Wafers:** 146 (9.32%) from Probe Station PS-L

## SHA-256 Checksums

Verify downloaded files:

```
1128d0b5b3a8bfdfe65348447b7479583d36b04bd79e51d1b9359e40c71495f3  bnypi.ipynb
bc56e9a69d890995db7ddaffa6b967558c8f6bdda2aa75ad9025500081f1ff47  secom_legacy_mask_EVALUATOR_ONLY.csv
443edf718380a62feab8449401a9033bfffd2886ef8ee1d25b21a8f91eda6778  secom_telemetry_modified.csv
```
