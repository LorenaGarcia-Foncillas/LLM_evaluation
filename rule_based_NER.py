import json
import spacy
from spacy.tokens import DocBin
from spacy.training import Example
import pandas as pd
import numpy as np
import pickle
from utils_train import evaluate
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# Load the spaCy model
nlp = spacy.load('en_core_web_trf')

# Load the exported JSON data
with open('/home/lorena/Documents/MRes_project/TumourSize/400_annotated_reports_with_names.json', 'r') as file:
    files = json.load(file)

# Load indices from a pickled file
with open('/home/lorena/Documents/MRes_project/TumourSize/indices_all_labels.pkl', 'rb') as f:
    train_indices, val_indices, test_indices = pickle.load(f)


# Read xlsx file with related data to reports
table = pd.read_excel("/home/lorena/Documents/meningioma_data 2.xlsx", index_col=0)

# Extract the first names and last names, convert to lowercase
table['client_firstname'] = table['client_firstname'].str.lower()
table['client_lastname'] = table['client_lastname'].str.lower()
# Drop duplicates to ensure unique pairs
unique_pairs_df = table[['client_firstname', 'client_lastname']].drop_duplicates()
# Convert to list of lists
unique_pairs = unique_pairs_df.values.tolist()
print(len(unique_pairs))


# Prepare the data - Step 1
data = []
# Initialise variable to store the file number
file_numbers = []
# Loop through each entry of the JSON file (each file)
for number, item in enumerate(files):
    if number in test_indices:
        # print(number)
        # Get the Textual Observation
        text = item['data']['text']#.lower()

        # Get the file name which is also the index into the report number
        file = item['file_upload'].split("_")[1].split(".")[0]
        file_numbers.append(int(file))

        # Obtain the labels
        entities = []
        # Loop through the list of annotations
        for annotation in item['annotations']:
            # Loop through the list of results 
            for result_item in annotation['result']:
                if result_item['value']['labels'][0] == "PER":
                    # Get start of label
                    start = result_item['value']['start']
                    # Get end of label
                    end = result_item['value']['end']
                    # Get label - extracting the first element of a list since the word could have had multiple labels
                    label = result_item['value']['labels'][0]
                            
                    # Add tuple to list
                    entities.append((start, end, label))
        # Add text and dict of entities to list
        data.append((text, {"entities": entities}))    

# OBTAIN TRUE LABEL LIST IN BILUO FORMAT
# Load the spaCy model
nlp = spacy.load('en_core_web_trf')
# Create a DocBin which stores the text and annotations
db = DocBin()
# Create a list to store Example objects
examples_list = []
# Loop through our train data
for text, annotations in data:
    # Turn text to doc object
    doc = nlp.make_doc(text)
    # Create an Example object with doc and entities dict
    example = Example.from_dict(doc, annotations)
    # Add Example object to list
    examples_list.append(example)


# Store the true labels
true_labels = []
# Initialise list to store labels
rule_based_labels = []

# Additional name prefix
prefixes = ["DR", "Dr", "Dr.", "dr", "dr.", "MR", "Mr.", "mr", "mr.", "MRS", "Mrs", "mrs"]

# Loop through examples
for num, example in enumerate(examples_list):
    labels = example.get_aligned_ner()
    if None not in labels:
            
        # Get list of BILUO values representing the tokens for NER annotation
        true_labels.append(labels)

        # Get the doc object with text
        text = example.text

        # Extract patient name, surname and dob from pandas dataframe
        patient_name = table['client_firstname'][file_numbers[num]]
        patient_surname = table['client_lastname'][file_numbers[num]]
        patient_dob = table['client_dob'][file_numbers[num]]

        # Pass text through nlp 
        doc = nlp(text)
        
        # Initialise list of length of words to store labels of this entry
        labels = ["O"] * len(doc)

        # Loop through words 
        for number, token in enumerate(doc):
            # Consultant: + NAME
            if number > 1 and ("Consultant" in doc[number-2].text and ":" in doc[number-1].text):
                future_number = number
                while doc[future_number].text != ".":
                    if doc[future_number].text != " ":
                        labels[future_number] = "PER"
                    future_number += 1
            # Requested By: + NAME
            elif number > 2 and ("Requested" in doc[number-3].text and "By" in doc[number-2].text and ":" in doc[number-1].text):
                future_number = number
                while doc[future_number].text != ".":
                    if doc[future_number].text != " ":
                        labels[future_number] = "PER"
                    future_number += 1
            # Reported by + NAME + on + DATE
            elif number > 1 and (("Reported" in doc[number-2].text and "by" in doc[number-1].text) or ("Reportedby" in doc[number-1].text)):
                future_number = number
                while future_number < len(doc) and doc[future_number].text != "on":
                    if doc[future_number].text != " " and doc[future_number].text != "and":
                        labels[future_number] = "PER"
                    future_number += 1
            # ADDENDUM START by + NAME + DATE
            elif number > 2 and (("ADDENDUM" in doc[number-3].text and "START" in doc[number-2].text and "by" in doc[number-1].text)):
                future_number = number
                while not any(char.isdigit() for char in doc[future_number].text):
                    if doc[future_number].text != " ":
                        labels[future_number] = "PER"
                    future_number += 1
            # (DR / DR. / Dr / Dr.) or (MR. / Mr /Mr.) or () + Surname
            elif number > 1 and (any([doc[number-1].text == prefix for prefix in prefixes])):
                labels[number - 1] = "PER"
                labels[number] = "PER"

            # Check if patient's name and surname is inside the report -> label it as PER
            elif len(patient_name) >= 3 and doc[number].text.lower() == patient_name:
                labels[number] = "PER"
            elif len(patient_surname) >= 3 and doc[number].text.lower() == patient_surname:
                labels[number] = "PER"

            
        rule_based_labels.append(labels)

# Changing the format of the spacy true labels (BILUO -> "O" or "PER")
new_list = []
for sublist in true_labels:
    new_sublist = []
    for label in sublist:
        if "PER" in label:
            new_sublist.append("PER")
        else:
            new_sublist.append(label)
    new_list.append(new_sublist)

# Obtaining the number of PERFECT MATCH reports
total_number = 0
same_results = 0
for true_sublist, pred_sublist in zip(new_list, rule_based_labels):
    if true_sublist == pred_sublist:
        same_results += 1
    total_number += 1

print(total_number, same_results)
print(f"Perfectly accurate labelled predictions at a report level: {round(same_results/total_number, 3)*100}%")

# Obtain the number of SOFT MATCH of reports (does it contain all the true labels and FPs)
some_match = 0
# Iterate over each pair of true and predicted sublists
for true_sublist, pred_sublist in zip(new_list, rule_based_labels):
    # Count the number of labeled tokens (i.e., non-zero elements) in the true sublist
    labelled_tokens = len([label for label in true_sublist if label != 'O'])
    same_results = 0
            
    # Check if each labeled token in the true sublist is in the predicted sublist
    for true_label, pred_label in zip(true_sublist, pred_sublist):
        # print(true_label, pred_label)
        if true_label != 'O' and true_label == pred_label:
            same_results += 1
                    
    # If all labeled tokens are correctly predicted, increment the match counter
    if same_results == labelled_tokens:
        some_match += 1

print("Total number of reports:", total_number, "Some match results:", some_match)
print(f"Reports containing the GT labels (soft evaluation): {round(some_match/total_number, 3)*100}%")

# Process the lists of true and predicted labels to binary
true_labels = np.array([1 if "PER" in label else 0 for sublist in true_labels for label in sublist])
rule_based_labels = np.array([1 if "PER" == label else 0  for list in rule_based_labels for label in list])

# Obtain dictionary of labels and numbers
unique_labels = ["O", "PER"]
label2id = {tag: id for id, tag in enumerate(unique_labels)}
print("label2id:", label2id)
id2label = {id: tag for tag, id in label2id.items()}
print("id2label:", id2label)

# Calculate accuracy
accuracy = accuracy_score(true_labels, rule_based_labels)
print(f"Accuracy: {accuracy:.4f}")

# Classification report
label_names = [id2label[i] for i in range(len(unique_labels))]
classification_rep = classification_report(true_labels, rule_based_labels, target_names=label_names, zero_division=0)
print("Classification Report:\n", classification_rep)

# Confusion matrix
conf_matrix = confusion_matrix(true_labels, rule_based_labels)
print("Confusion Matrix:\n", conf_matrix)

# Calculate metrics for each class
tn_list, fp_list, fn_list, tp_list = [], [], [], []
for i in range(len(unique_labels)):
    tn, fp, fn, tp = confusion_matrix(true_labels == i, rule_based_labels == i).ravel()
    tn_list.append(tn)
    fp_list.append(fp)
    fn_list.append(fn)
    tp_list.append(tp)

# Compute specificity, sensitivity, precision, recall, and f1-score for each class
specificity = [tn / (tn + fp + 1e-5) for tn, fp in zip(tn_list, fp_list)]
sensitivity = [tp / (tp + fn + 1e-5) for tp, fn in zip(tp_list, fn_list)]
precision = [tp / (tp + fp + 1e-5) for tp, fp in zip(tp_list, fp_list)]
recall = [tp / (tp + fn + 1e-5) for tp, fn in zip(tp_list, fn_list)]
f1score = [(2 * tp) / (2 * tp + fp + fn + 1e-5) for tp, fp, fn in zip(tp_list, fp_list, fn_list)]

# Round each value to 3 decimal points
specificity = [round(val, 3) for val in specificity]
sensitivity = [round(val, 3) for val in sensitivity]
precision = [round(val, 3) for val in precision]
recall = [round(val, 3) for val in recall]
f1score = [round(val, 3) for val in f1score]

print(f"SPECIFICITY: {specificity}")
print(f"SENSITIVITY: {sensitivity}")
print(f"PRECISION: {precision}")
print(f"RECALL: {recall}")
print(f"F1-SCORE: {f1score}")

# Average metrics across all classes
average_specificity = sum(specificity) / len(specificity)
average_sensitivity = sum(sensitivity) / len(sensitivity)
average_precision = sum(precision) / len(precision)
average_recall = sum(recall) / len(recall)
average_f1score = sum(f1score) / len(f1score)

print(f"Average SPECIFICITY: {round(average_specificity, 3)}")
print(f"Average SENSITIVITY: {round(average_sensitivity, 3)}")
print(f"Average PRECISION: {round(average_precision, 3)}")
print(f"Average RECALL: {round(average_recall, 3)}")
print(f"Average F1-SCORE: {round(average_f1score, 3)}")