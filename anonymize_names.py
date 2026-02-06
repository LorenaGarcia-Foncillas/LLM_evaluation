"""
Anonymize personal names in medical reports using spaCy NER and Faker.

This script:
1. Loads a trained spaCy NER model (from cross-validation folds)
2. Reads .txt files from an input directory
3. Detects PER (person name) entities
4. Replaces detected names with random fake names from a pool of 50
5. Saves anonymized files to an output directory
"""

import argparse
import random
from pathlib import Path

import spacy
from faker import Faker


def generate_fake_names(n: int = 50, seed: int = None) -> list[str]:
    """Generate a pool of n unique fake full names.

    Args:
        n: Number of fake names to generate (default: 50)
        seed: Random seed for reproducibility (optional)

    Returns:
        List of unique fake full names
    """
    fake = Faker()
    if seed is not None:
        Faker.seed(seed)
        random.seed(seed)

    names = set()
    while len(names) < n:
        names.add(fake.name())

    return list(names)


def load_model(model_dir: Path) -> spacy.Language:
    """Load a trained spaCy model from disk.

    Args:
        model_dir: Path to the model directory

    Returns:
        Loaded spaCy Language object
    """
    print(f"Loading model from: {model_dir}")
    return spacy.load(model_dir)


def extract_per_entities(nlp: spacy.Language, text: str) -> list[tuple[int, int, str]]:
    """Extract PER entities from text using the NER model.

    Args:
        nlp: Loaded spaCy model
        text: Input text to process

    Returns:
        List of (start, end, text) tuples for each PER entity
    """
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        if ent.label_ == "PER":
            entities.append((ent.start_char, ent.end_char, ent.text))
    return entities


def anonymize_text(text: str, entities: list[tuple[int, int, str]],
                   fake_names: list[str]) -> tuple[str, int]:
    """Replace detected entities with random fake names.

    Processes entities in reverse order to preserve character offsets.
    Cycles through the fake names pool if needed.

    Args:
        text: Original text
        entities: List of (start, end, original_text) tuples
        fake_names: Pool of fake names to use

    Returns:
        Tuple of (anonymized text, count of replacements made)
    """
    if not entities:
        return text, 0

    # Sort entities by start position in reverse order
    sorted_entities = sorted(entities, key=lambda x: x[0], reverse=True)

    anonymized = text
    replacement_count = 0

    for start, end, original in sorted_entities:
        # Pick a random fake name (cycling through pool)
        fake_name = random.choice(fake_names)
        anonymized = anonymized[:start] + fake_name + anonymized[end:]
        replacement_count += 1

    return anonymized, replacement_count


def process_file(file_path: Path, nlp: spacy.Language,
                 fake_names: list[str]) -> tuple[str, int, list[str]]:
    """Process a single file: read, detect entities, anonymize.

    Args:
        file_path: Path to input file
        nlp: Loaded spaCy model
        fake_names: Pool of fake names

    Returns:
        Tuple of (anonymized text, replacement count, list of original names found)
    """
    text = file_path.read_text(encoding="utf-8")
    entities = extract_per_entities(nlp, text)
    original_names = [ent[2] for ent in entities]
    anonymized_text, count = anonymize_text(text, entities, fake_names)
    return anonymized_text, count, original_names


def main():
    parser = argparse.ArgumentParser(
        description="Anonymize personal names in medical reports using spaCy NER"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("sample_data"),
        help="Input directory containing .txt files (default: sample_data)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("sample_data_anonymized"),
        help="Output directory for anonymized files (default: sample_data_anonymized)"
    )
    parser.add_argument(
        "--fold", "-f",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4],
        help="Cross-validation fold to use (default: 0, best performer)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        help="Random seed for reproducibility (optional)"
    )
    parser.add_argument(
        "--model-base",
        type=Path,
        default=Path("PER_CV5_Model"),
        help="Base directory for CV models (default: PER_CV5_Model)"
    )

    args = parser.parse_args()

    # Validate input directory
    if not args.input.exists():
        print(f"Error: Input directory '{args.input}' does not exist")
        return 1

    # Create output directory if needed
    args.output.mkdir(parents=True, exist_ok=True)

    # Build model path
    model_path = args.model_base / f"cv{args.fold}" / "model-best"
    if not model_path.exists():
        print(f"Error: Model not found at '{model_path}'")
        return 1

    # Load model
    nlp = load_model(model_path)

    # Generate fake names pool
    fake_names = generate_fake_names(n=50, seed=args.seed)
    print(f"Generated pool of {len(fake_names)} fake names")

    # Find all .txt files
    txt_files = list(args.input.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in '{args.input}'")
        return 1

    print(f"Found {len(txt_files)} files to process\n")

    # Process each file
    total_replacements = 0
    all_original_names = []

    for file_path in sorted(txt_files):
        anonymized_text, count, original_names = process_file(file_path, nlp, fake_names)

        # Save anonymized file
        output_path = args.output / file_path.name
        output_path.write_text(anonymized_text, encoding="utf-8")

        total_replacements += count
        all_original_names.extend(original_names)

        print(f"Processed: {file_path.name}")
        print(f"  - Names found: {count}")
        if original_names:
            print(f"  - Original names: {', '.join(original_names)}")
        print(f"  - Saved to: {output_path}")
        print()

    # Summary
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Files processed: {len(txt_files)}")
    print(f"Total names replaced: {total_replacements}")
    print(f"Unique names found: {len(set(all_original_names))}")
    print(f"Output directory: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
