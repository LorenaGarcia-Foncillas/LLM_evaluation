import spacy
from spacy.training import Example, offsets_to_biluo_tags
import json
import pickle
from spacy.tokens import DocBin
from sklearn.model_selection import KFold
import numpy as np
import os


def create_doc_bin(indices, files, nlp):
    """
    Create a DocBin object from specified indices and files.
    
    Parameters:
    - indices: A list of indices pointing to specific items in the 'files' list.
    - files: A list of file dictionaries containing text data and annotations.
    - nlp: A SpaCy NLP object used to process the text data.
    
    Returns:
    - doc_bin: A SpaCy DocBin object containing processed documents with annotated entities.
    """
    # Initialize an empty DocBin object to store processed documents
    doc_bin = DocBin()

    # Iterate over each index provided in the 'indices' list
    for idx in indices:
        # Retrieve the corresponding item from the 'files' list using the current index
        item = files[idx]
        # Extract the text data from the item
        text = item['data']['text']
        # Create a SpaCy Doc object from the text
        doc = nlp.make_doc(text)
        
        # Initialize lists to hold entity information and SpaCy Span objects
        entities = []
        ents_list = []
        
        # Iterate over each annotation associated with the current item
        for annotation in item['annotations']:
            # Process each result within the annotation
            for result_item in annotation['result']:
                # Check if the label is either "PER" (Person) or "PRESENT_TUMOUR_SIZE"
                # if result_item['value']['labels'][0] == "PER" or result_item['value']['labels'][0] == "PRESENT_TUMOUR_SIZE":
                # if result_item['value']['labels'][0] == "PER":
                if result_item['value']['labels'][0] == "PRESENT_TUMOUR_SIZE":

                    # Extract the start and end positions of the entity in the text
                    start = result_item['value']['start']
                    end = result_item['value']['end']
                    # Extract the entity label
                    label = result_item['value']['labels'][0]
                    # Append the entity information as a tuple (start, end, label) to the 'entities' list
                    entities.append((start, end, label))
                    # Create a SpaCy Span object from the entity information
                    span = doc.char_span(start, end, label)
                    # If the span is valid (not None), append it to the 'ents_list'
                    if span is not None:
                        ents_list.append(span)
        
        # Convert the entity offsets to BILUO tags using SpaCy's utility function
        labels = offsets_to_biluo_tags(nlp.make_doc(text), entities)
        
        # Check if the labels list does not contain any invalid tags ('-')
        if '-' not in labels:
            # If there are valid entity spans, set them as the document's entities
            if len(ents_list) > 0:
                doc.ents = ents_list
            # Add the processed document to the DocBin object
            doc_bin.add(doc)
    
    # Return the populated DocBin object containing all processed documents
    return doc_bin





# CROSS-VALIDATION
# Load SpaCy model
nlp = spacy.load('en_core_web_lg')

# Load indices from a pickled file
with open('indices_all_labels.pkl', 'rb') as f:
    train_indices, val_indices, test_indices = pickle.load(f)

# Combine indices for cross-validation
combined_indices = np.concatenate([train_indices, val_indices])

# Define directories and filenames
output_dir = "MResProject/PRESENT_TUMOUR_CV5_Data" # "MResProject/PER_PRESENT_TUMOUR_CV5_Data" | "MResProject/PER_CV5_Data" | "MResProject/PRESENT_TUMOUR_CV5_Data"     #'SpacyData_PER_Tumour_Present_Size/'
os.makedirs(output_dir, exist_ok=True)

# Load data from JSON file
file_path = '400_reports_labelled_25July2024.json'
with open(file_path, 'r') as file:
    files = json.load(file)

# Set up KFold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Iterate through folds
for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(combined_indices)):
    print(f"Processing fold {fold_idx + 1}/{kfold.get_n_splits()}...")
    
    # Create DocBin for training and validation data
    train_doc_bin = create_doc_bin(train_indices, files, nlp)
    val_doc_bin = create_doc_bin(val_indices, files, nlp)
    
    # Save DocBin to disk
    train_data_path = os.path.join(output_dir, f'train_data_fold_{fold_idx}.spacy')
    val_data_path = os.path.join(output_dir, f'val_data_fold_{fold_idx}.spacy')
    train_doc_bin.to_disk(train_data_path)
    val_doc_bin.to_disk(val_data_path)
    
    print(f"Saved {len(train_doc_bin)} training examples to {train_data_path}")
    print(f"Saved {len(val_doc_bin)} validation examples to {val_data_path}")

print("Cross-validation data preparation complete.")

# TEST SET
# test_indices = [0,1]
output_dir = "MResProject/PRESENT_TUMOUR_CV5_Data" # "MResProject/PER_PRESENT_TUMOUR_CV5_Data" | "MResProject/PER_CV5_Data" | "MResProject/PRESENT_TUMOUR_CV5_Data"      #'SpacyAbstractData'
test_doc_bin = create_doc_bin(test_indices, files, nlp)
test_data_path = os.path.join(output_dir, 'test_data.spacy')
test_doc_bin.to_disk(test_data_path)
# Print confirmation message
print(f"Saved {len(test_doc_bin)} testing examples to {test_data_path}")


"""
python -m spacy train config.cfg --output Spacy_models_PER_Tumour_Present_Size/cv0/ > Spacy_models_PER_Tumour_Present_Size/cv0/spacy_output_cv0.txt


python -m spacy train config.cfg --output MResProject/PER_PRESENT_TUMOUR_CV5_Model/cv0/ > MResProject/PER_PRESENT_TUMOUR_CV5_Model/cv0/spacy_output_cv0.txt
"""
