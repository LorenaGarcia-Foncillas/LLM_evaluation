import spacy
from spacy.tokens import DocBin
from spacy.training import Example, offsets_to_biluo_tags
import warnings
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, classification_report


def load_data(file_path: str, nlp):
    doc_bin = DocBin().from_disk(file_path)
    samples, entities_count = [], 0
    for doc in doc_bin.get_docs(nlp.vocab):
        sample = {
            "text": doc.text,
            "entities": []
        }
        if len(doc.ents) > 0:
            entities = [(e.start_char, e.end_char, e.label_) for e in doc.ents]
            sample["entities"] = entities
            entities_count += len(entities)
        else:
            warnings.warn("Sample without entities!")
        samples.append(sample)
    return samples, entities_count

def get_cleaned_label(label: str):
    if "-" in label:
        return label.split("-")[1]
    else:
        return label

def create_total_target_vector(nlp, samples):
    target_vector = []
    for sample in samples:
        doc = nlp.make_doc(sample["text"])
        ents = sample["entities"]
        bilou_ents = offsets_to_biluo_tags(doc, ents)
        vec = [get_cleaned_label(label) for label in bilou_ents]
        # target_vector.extend(vec)
        target_vector.append(vec)
    return target_vector


def get_all_ner_predictions(nlp, text):
    doc = nlp(text)
    entities = [(e.start_char, e.end_char, e.label_) for e in doc.ents]
    bilou_entities = offsets_to_biluo_tags(doc, entities)
    return bilou_entities

def create_prediction_vector(nlp, text):
    return [get_cleaned_label(prediction) for prediction in get_all_ner_predictions(nlp, text)]


def create_total_prediction_vector(nlp, samples):
    prediction_vector = []
    for sample in samples:
        # prediction_vector.extend(create_prediction_vector(nlp, sample["text"]))
        prediction_vector.append(create_prediction_vector(nlp, sample["text"]))
    return prediction_vector

number = 4
print(f"Fold: {number}")
# Load the trained model from the output directory
model_dir = f'MResProject/PER_PRESENT_TUMOUR_CV5_Model/cv{number}/model-best' #f'Spacy_models_PER_Tumour_Present_Size/cv{number}/model-best' # also available model-last
nlp = spacy.load(model_dir)

# Load the test data
val_data_path = f"MResProject/PER_PRESENT_TUMOUR_CV5_Data/val_data_fold_{number}.spacy" #f"SpacyData_PER_Tumour_Present_Size/val_data_fold_{number}.spacy"

samples, entities_count = load_data(val_data_path, nlp)
y_true = create_total_target_vector(nlp, samples)
y_pred = create_total_prediction_vector(nlp, samples)

# Define the labels
labels = ["O", "PER", "PRESENT_TUMOUR_SIZE"] # ["O", "PER"] | ["O", "PRESENT_TUMOUR_SIZE"] | ["O", "PER", "PRESENT_TUMOUR_SIZE"]

# Convert y_true and y_pred to multilabel format
def convert_to_multilabel(y, labels):
    multilabel = []
    for sample in y:
        sample_labels = [0] * len(labels)
        for entity in sample:
            if entity in labels:
                sample_labels[labels.index(entity)] = 1
        multilabel.append(sample_labels)
    return multilabel
# (Interesting) Flatten lists
y_true = sum(y_true, [])
y_pred = sum(y_pred, [])

# Convert list to multilabel (for each entry there is a vector of size of label and so it is hot-top)
y_true_multilabel = convert_to_multilabel(y_true, labels)
y_pred_multilabel = convert_to_multilabel(y_pred, labels)


print("VALIDATION")
# Obtain classification report
print("CLASSIFICATION REPORT:")
print(classification_report(y_true, y_pred, target_names=labels))

# Calculate multilabel confusion matrix
ml_conf_matrix = multilabel_confusion_matrix(y_true, y_pred)
print("Confusion Matrix for each label:")
print(ml_conf_matrix)
print()

all_accuracy = []
all_specificity = []
all_sensitivity = []
all_precision = []
all_recall = []
all_f1score = []

for idx, label in enumerate(labels):
    tn, fp, fn, tp = ml_conf_matrix[idx,:,:].ravel()
    print(f"Label: {label}")
    print("TN:", tn, "FP:", fp, "FN:", fn, "TP:", tp)

    # Calculate evaluation metrics for each label
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    specificity = tn / (tn + fp + 1e-5)
    sensitivity = tp / (tp + fn + 1e-5)
    precision = tp / (tp + fp + 1e-5)
    recall = tp / (tp + fn + 1e-5)
    f1score = (2*tp) / (2*tp + fp + fn + 1e-5)

    print(f"Sensitivity: {round(sensitivity, 3)}")
    print(f"Accuracy: {round(accuracy, 3)}")
    print(f"Specificity: {round(specificity, 3)}")
    print(f"Precision: {round(precision, 3)}")
    print(f"Recall: {round(recall, 3)}")
    print(f"F1-Score: {round(f1score, 3)}")


    all_sensitivity.append(round(sensitivity, 3))
    all_accuracy.append(round(accuracy, 3))
    all_specificity.append(round(specificity, 3))
    all_precision.append(round(precision, 3))
    all_recall.append(round(recall, 3))
    all_f1score.append(round(f1score, 3))

    print()

print(f"Macro avg Sensitivity: {round(sum(all_sensitivity)/len(all_sensitivity), 3)}")
print(f"Macro avg Accuracy: {round(sum(all_accuracy)/len(all_accuracy), 3)}")
print(f"Macro avg Specificity: {round(sum(all_specificity)/len(all_specificity), 3)}")
print(f"Macro avg Precision: {round(sum(all_precision)/len(all_precision), 3)}")
print(f"Macro avg Recall: {round(sum(all_recall)/len(all_recall), 3)}")
print(f"Macro avg F1-Score: {round(sum(all_f1score)/len(all_f1score), 3)}")

########################################################################################

# Load the test data
test_data_path = "MResProject/PER_PRESENT_TUMOUR_CV5_Data/test_data.spacy" #"SpacyData_PER_Tumour_Present_Size/test_data.spacy"

samples, entities_count = load_data(test_data_path, nlp)
y_true = create_total_target_vector(nlp, samples)
y_pred = create_total_prediction_vector(nlp, samples)

# PERFECT MATCH OF REPORTS
total_number = 0
same_results = 0
for true_sublist, pred_sublist in zip(y_true, y_pred):
    if true_sublist == pred_sublist:
        same_results += 1
        # print(true_sublist, pred_sublist)
    total_number += 1

# print(total_number, same_results)
print(f"\nPerfectly accurate labelled predictions at a report level: {round(same_results/total_number, 3)*100}%\n")

some_match = 0
# Iterate over each pair of true and predicted sublists
for true_sublist, pred_sublist in zip(y_true, y_pred):
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

# # EVALUATING MODELS AS USUAL WITH TP, TN, FP, FN
# label = "PRESENT_TUMOUR_SIZE"
# y_true = [1 if label == element else 0 for element in y_true]
# y_pred = [1 if label == element else 0 for element in y_pred]

# print("TESTING")
# print("CLASSIFICATION REPORT:")
# print(classification_report(y_true, y_pred))
# # Calculate confusion matrix values
# tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
# print("TN:", tn, "FP:", fp, "FN:", fn, "TP:", tp)
# # Calculate evaluation metrics
# accuracy = (tp + tn) / (tp + tn + fp +fn)
# specificity = tn / (tn + fp + 1e-5)
# sensitivity = tp / (tp + fn + 1e-5)
# precision = tp / (tp + fp + 1e-5)
# recall = tp / (tp + fn + 1e-5)
# f1score = (2*tp) / (2*tp + fp + fn + 1e-5)
# # Print evaluation metrics
# print("SENSITIVITY:", round(sensitivity, 3))
# print("SPECIFITY:", round(specificity, 3))
# print("ACCURACY:", round(accuracy, 3))
# print("PRECISION:", round(precision, 3))
# print("RECALL:", round(recall, 3))
# print("F1-SCORE:", round(f1score, 3))

# (Interesting) Flatten lists
y_true = sum(y_true, [])
y_pred = sum(y_pred, [])

# Evaluate the testing set
print("TESTING")
# Get a classification report
print("CLASSIFICATION REPORT:")
print(classification_report(y_true, y_pred, target_names=labels))
ml_conf_matrix = multilabel_confusion_matrix(y_true, y_pred)
# Calculate multilabel confusion matrix
print("Confusion Matrix for each label:")
print(ml_conf_matrix)
print()

all_accuracy = []
all_specificity = []
all_sensitivity = []
all_precision = []
all_recall = []
all_f1score = []

for idx, label in enumerate(labels):
    tn, fp, fn, tp = ml_conf_matrix[idx,:,:].ravel()
    print(f"Label: {label}")
    print("TN:", tn, "FP:", fp, "FN:", fn, "TP:", tp)

    # Calculate evaluation metrics for each label
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    specificity = tn / (tn + fp + 1e-5)
    sensitivity = tp / (tp + fn + 1e-5)
    precision = tp / (tp + fp + 1e-5)
    recall = tp / (tp + fn + 1e-5)
    f1score = (2*tp) / (2*tp + fp + fn + 1e-5)

    print(f"Sensitivity: {round(sensitivity, 3)}")
    print(f"Accuracy: {round(accuracy, 3)}")
    print(f"Specificity: {round(specificity, 3)}")
    print(f"Precision: {round(precision, 3)}")
    print(f"Recall: {round(recall, 3)}")
    print(f"F1-Score: {round(f1score, 3)}")
    
    all_sensitivity.append(round(sensitivity, 3))
    all_accuracy.append(round(accuracy, 3))
    all_specificity.append(round(specificity, 3))
    all_precision.append(round(precision, 3))
    all_recall.append(round(recall, 3))
    all_f1score.append(round(f1score, 3))

    print()

print(f"Macro avg Sensitivity: {round(sum(all_sensitivity)/len(all_sensitivity), 3)}")
print(f"Macro avg Accuracy: {round(sum(all_accuracy)/len(all_accuracy), 3)}")
print(f"Macro avg Specificity: {round(sum(all_specificity)/len(all_specificity), 3)}")
print(f"Macro avg Precision: {round(sum(all_precision)/len(all_precision), 3)}")
print(f"Macro avg Recall: {round(sum(all_recall)/len(all_recall), 3)}")
print(f"Macro avg F1-Score: {round(sum(all_f1score)/len(all_f1score), 3)}")