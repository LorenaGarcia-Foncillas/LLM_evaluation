# Name Anonymization Guide

This guide explains how to use the `anonymize_names.py` script to de-identify personal names in medical reports using a trained spaCy NER model and the Faker library.

## Overview

The script:
1. Loads a trained spaCy NER model that detects PER (person name) entities
2. Reads `.txt` files from an input directory
3. Identifies all personal names in the text
4. Replaces each name with a randomly selected fake name
5. Saves the anonymized files to an output directory

## Prerequisites

### 1. Python Environment

Requires Python 3.9+ (tested with Python 3.12).

```bash
# Create virtual environment
python3.12 -m venv venv

# Activate it
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install spacy faker
```

### 3. Trained Model

The script expects trained spaCy NER models in the following structure:

```
PER_CV5_Model/
├── cv0/
│   └── model-best/    # Best performing model (F1=1.0)
├── cv1/
│   └── model-best/
├── cv2/
│   └── model-best/
├── cv3/
│   └── model-best/
└── cv4/
    └── model-best/
```

By default, the script uses `cv0` (the best performing fold with perfect F1 score on test data).

## Usage

### Basic Usage

```bash
python anonymize_names.py
```

This will:
- Read files from `sample_data/`
- Use the cv0 model
- Output to `sample_data_anonymized/`

### Command Line Options

```bash
python anonymize_names.py [OPTIONS]

Options:
  -i, --input PATH       Input directory with .txt files (default: sample_data)
  -o, --output PATH      Output directory for results (default: sample_data_anonymized)
  -f, --fold {0,1,2,3,4} CV fold to use (default: 0)
  -s, --seed INT         Random seed for reproducibility (optional)
  --model-base PATH      Base directory for models (default: PER_CV5_Model)
```

### Examples

```bash
# Process custom input directory
python anonymize_names.py --input my_reports/ --output my_reports_anonymized/

# Use a different model fold
python anonymize_names.py --fold 3

# Reproducible results with seed
python anonymize_names.py --seed 42

# Full example with all options
python anonymize_names.py \
    --input data/raw_reports/ \
    --output data/anonymized_reports/ \
    --fold 0 \
    --seed 123
```

## How It Works

### 1. Fake Name Pool Generation

The script generates 50 unique fake names using the Faker library:

```python
from faker import Faker
fake = Faker()
names = [fake.name() for _ in range(50)]
# e.g., ["John Smith", "Maria Garcia", "James Wilson", ...]
```

### 2. Entity Detection

The spaCy model processes each document and identifies PER entities:

```python
doc = nlp(text)
for ent in doc.ents:
    if ent.label_ == "PER":
        # Found a person name at positions ent.start_char to ent.end_char
```

### 3. Name Replacement

Names are replaced in reverse order (end to start) to preserve character offsets:

```python
# Original: "Reported by John Smith and Jane Doe"
# After:    "Reported by Maria Garcia and Bob Wilson"
```

Each detected name is replaced with a randomly selected name from the pool. The same original name may get different fake names in different occurrences.

## Output

### Console Output

```
Loading model from: PER_CV5_Model/cv0/model-best
Generated pool of 50 fake names
Found 5 files to process

Processed: report_1550.txt
  - Names found: 2
  - Original names: Danielle Buchanan, William Brown
  - Saved to: sample_data_anonymized/report_1550.txt

...

==================================================
SUMMARY
==================================================
Files processed: 5
Total names replaced: 7
Unique names found: 7
Output directory: sample_data_anonymized
```

### Output Files

Anonymized files are saved with the same filename in the output directory:

```
sample_data_anonymized/
├── report_1550.txt
├── report_1786.txt
├── report_2644.txt
├── report_2645.txt
└── report_13366.txt
```

## Example

**Input** (`sample_data/report_1550.txt`):
```
report: 1550
Patient: 6a9f2b74bf (Hashed Hospital Number (HN))
...
Reported by Danielle Buchanan and William Brown on 03/10/2014
```

**Output** (`sample_data_anonymized/report_1550.txt`):
```
report: 1550
Patient: 6a9f2b74bf (Hashed Hospital Number (HN))
...
Reported by Ryan Munoz and Holly Wood on 03/10/2014
```

## Model Performance

The PER NER models were trained with 5-fold cross-validation. Test set performance:

| Fold | Precision | Recall | F1-Score |
|------|-----------|--------|----------|
| cv0  | 1.000     | 1.000  | 1.000    |
| cv1  | 0.992     | 0.992  | 0.992    |
| cv2  | 0.984     | 0.972  | 0.978    |
| cv3  | 0.992     | 1.000  | 0.996    |
| cv4  | 1.000     | 0.996  | 0.998    |

**Recommendation**: Use cv0 (default) for best performance.

## Troubleshooting

### spaCy Version Warning

```
UserWarning: [W095] Model 'en_pipeline' (0.0.0) was trained with spaCy v3.7.5
and may not be 100% compatible with the current version (3.8.11).
```

This warning can be safely ignored - the model works correctly. To eliminate it, install the exact spaCy version used for training:

```bash
pip install spacy==3.7.5
```

### Model Not Found

```
Error: Model not found at 'PER_CV5_Model/cv0/model-best'
```

Ensure the model directory exists and contains the trained model files. The structure should include:
- `config.cfg`
- `meta.json`
- `ner/` directory
- `tok2vec/` directory
- `vocab/` directory

### No Files Found

```
No .txt files found in 'sample_data'
```

Ensure your input directory contains `.txt` files and the path is correct.

## Customization

### Using a Different Fake Name Locale

Edit `anonymize_names.py` to use locale-specific names:

```python
from faker import Faker
fake = Faker('en_GB')  # British names
fake = Faker('es_ES')  # Spanish names
fake = Faker(['en_US', 'en_GB'])  # Mixed
```

### Changing Pool Size

Modify the `generate_fake_names()` call in `main()`:

```python
fake_names = generate_fake_names(n=100, seed=args.seed)  # 100 names instead of 50
```

### Consistent Name Mapping

To ensure the same original name always maps to the same fake name, modify `anonymize_text()`:

```python
name_mapping = {}
for start, end, original in sorted_entities:
    if original not in name_mapping:
        name_mapping[original] = random.choice(fake_names)
    fake_name = name_mapping[original]
    # ... rest of replacement logic
```
