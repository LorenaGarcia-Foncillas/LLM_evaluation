from transformers import AutoTokenizer,AutoModelForTokenClassification,AutoModelForMaskedLM
from LoRA_DoRA import LinearWithLoRA, LinearWithDoRA, LinearWithLoRAMerged
import torch
import numpy as np
from collections import Counter
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def choose_model(model_type):
    # BERT
    if model_type == 0:
        # Load the tokenizer and model
        model_type = "google-bert/bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_type)
        model = AutoModelForTokenClassification.from_pretrained(model_type)
        batch_size = 8
        model_name = "BERT"
        print(f'CHOSEN MODEL: {model_type}')

    # DistilBERT
    elif model_type == 1:
        model_type = "distilbert/distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_type)
        model = AutoModelForTokenClassification.from_pretrained(model_type)
        batch_size = 8    
        model_name = "DistilBERT"
        print(f'CHOSEN MODEL: {model_type}')

    # BioBERT
    elif model_type == 2:
        model_type = "dmis-lab/biobert-base-cased-v1.2"
        tokenizer = AutoTokenizer.from_pretrained(model_type)
        model = AutoModelForTokenClassification.from_pretrained(model_type)
        batch_size = 8
        model_name = "BioBERT"
        print(f'CHOSEN MODEL: {model_type}')

    # DistilBioBERT
    elif model_type == 3:
        model_type = "nlpie/distil-biobert"
        tokenizer = AutoTokenizer.from_pretrained(model_type)
        model = AutoModelForTokenClassification.from_pretrained(model_type)
        batch_size = 8
        model_name = "DistilBioBERT"
        print(f'CHOSEN MODEL: {model_type}')

    # ClinicalBioBERT
    elif model_type == 4:
        model_type = "emilyalsentzer/Bio_ClinicalBERT"
        tokenizer = AutoTokenizer.from_pretrained(model_type)
        model = AutoModelForTokenClassification.from_pretrained(model_type)
        batch_size = 8
        model_name = "ClinicalBERT"
        print(f'CHOSEN MODEL: {model_type}')

    # Distil Clinical BERT
    elif model_type == 5:
        model_type = "nlpie/distil-clinicalbert"
        tokenizer = AutoTokenizer.from_pretrained(model_type)
        model = AutoModelForTokenClassification.from_pretrained(model_type)
        batch_size = 8
        model_name = "DistilClinicalBERT"
        print(f'CHOSEN MODEL: {model_type}')
    
    return model, tokenizer, batch_size, model_name, model_type

def apply_lora_to_transformer(model, rank, alpha, model_name):
    """
    Args:
    - model (nn.Module): Transformer model instance.
    - rank (int): Rank of the low-rank matrix approximation.
    - alpha (float): Scaling factor for the LoRA operation.

    Returns:
    - nn.Module: Updated model with LoRA applied to specified layers.
    """

    # modules_to_modify = []
    # for name, module in model.named_modules():
    #     for _, param in module.named_parameters(recurse=False):
    #         if ("key" in name or "query" in name) or ("k_lin" in name or "q_lin" in name):
    #             modules_to_modify.append((name, module))
    #             param.requires_grad = False
    #         elif "classifier" in name:
    #             pass
    #         else:
    #             param.requires_grad = False
            
    # for name, module in modules_to_modify:
    #     lora_layer = LinearWithLoRAMerged(module, rank, alpha) # LinearWithLoRA | LinearWithLoRAMerged(module, rank, alpha)
    #     setattr(model, name, lora_layer)
    
    # # Re-enable gradients for vocab_projector (distilBERT) and predictions.decoder (BERT) parameters specifically
    # for name, module in model.named_modules():
    #     if "vocab_projector" in name or "cls.predictions.decoder" in name:
    #         for param_name, param in module.named_parameters(recurse=False):
    #             param.requires_grad = True

    # return model
    if model_name == "DistilBERT":
        # Initialise LoRA layers for Query and Key layers within attention module
        for layer in model.distilbert.transformer.layer:

            layer.attention.q_lin = LinearWithLoRA(layer.attention.q_lin, rank=rank, alpha=alpha) #  LinearWithLoRAMerged
            layer.attention.k_lin = LinearWithLoRA(layer.attention.k_lin, rank=rank, alpha=alpha) # LinearWithLoRAMerged

        # First, set requires_grad to False for all parameters
        for param in model.parameters():
            param.requires_grad = False

        # Then, set requires_grad to True for specific layers
        for layer in model.distilbert.transformer.layer:
            # Set requires_grad for LoRA parameters
            if hasattr(layer.attention.q_lin, 'lora'):
                lora_query = layer.attention.q_lin.lora
                lora_key = layer.attention.k_lin.lora
                
                lora_query.A.requires_grad = True
                lora_query.B.requires_grad = True
                lora_key.A.requires_grad = True
                lora_key.B.requires_grad = True
        
        # Set requires_grad for classifier parameters
        if hasattr(model, 'classifier'):
            classifier = model.classifier
            classifier.weight.requires_grad = True
            classifier.bias.requires_grad = True
        
        return model

    elif model_name == "DistilBioBERT" or model_name == "DistilClinicalBERT":
        # Initialise LoRA layers for Query and Key layers within attention module
        for layer in model.bert.encoder.layer:

            layer.attention.self.query = LinearWithLoRA(layer.attention.self.query, rank=rank, alpha=alpha) #  LinearWithLoRAMerged
            layer.attention.self.key = LinearWithLoRA(layer.attention.self.key, rank=rank, alpha=alpha) # LinearWithLoRAMerged

        # First, set requires_grad to False for all parameters
        for param in model.parameters():
            param.requires_grad = False

        # Then, set requires_grad to True for specific layers
        for layer in model.bert.encoder.layer:
            # Set requires_grad for LoRA parameters
            if hasattr(layer.attention.self.query, 'lora'):
                lora_query = layer.attention.self.query.lora
                lora_key = layer.attention.self.key.lora
                
                lora_query.A.requires_grad = True
                lora_query.B.requires_grad = True
                lora_key.A.requires_grad = True
                lora_key.B.requires_grad = True
        
        # Set requires_grad for classifier parameters
        if hasattr(model, 'classifier'):
            classifier = model.classifier
            classifier.weight.requires_grad = True
            classifier.bias.requires_grad = True
        
        return model


def apply_dora_to_transformer(model, rank, alpha, model_name):
    """
    Applies DoRA (Weight Decomposition Low-Rank Adaptation) to specified linear layers in the transformer model
    and freezes other layers, excluding the classifier layer.

    Args:
    - model (nn.Module): Transformer model instance.
    - rank (int): Rank of the low-rank matrix approximation.
    - alpha (float): Scaling factor for the LoRA operation.

    Returns:
    - nn.Module: Updated model with LoRA applied to specified layers.
    """

    if model_name == "DistilBERT":
        # Initialise LoRA layers for Query and Key layers within attention module
        for layer in model.distilbert.transformer.layer:

            layer.attention.q_lin = LinearWithDoRA(layer.attention.q_lin, rank=rank, alpha=alpha) #  LinearWithLoRAMerged
            layer.attention.k_lin = LinearWithDoRA(layer.attention.k_lin, rank=rank, alpha=alpha) # LinearWithLoRAMerged

        # First, set requires_grad to False for all parameters
        for param in model.parameters():
            param.requires_grad = False

        # Then, set requires_grad to True for specific layers
        for layer in model.distilbert.transformer.layer:
            # Set requires_grad for LoRA parameters
            if hasattr(layer.attention.q_lin, 'lora'):
                query = layer.attention.q_lin
                key = layer.attention.k_lin
                
                query.lora.A.requires_grad = True
                query.lora.B.requires_grad = True
                key.lora.A.requires_grad = True
                key.lora.B.requires_grad = True
                query.m.requires_grad = True
                key.m.requires_grad = True
        
        # Set requires_grad for classifier parameters
        if hasattr(model, 'classifier'):
            classifier = model.classifier
            classifier.weight.requires_grad = True
            classifier.bias.requires_grad = True
        
        return model

    elif model_name == "DistilBioBERT" or model_name == "DistilClinicalBERT":
        # Initialise LoRA layers for Query and Key layers within attention module
        for layer in model.bert.encoder.layer:

            layer.attention.self.query = LinearWithDoRA(layer.attention.self.query, rank=rank, alpha=alpha) #  LinearWithLoRAMerged
            layer.attention.self.key = LinearWithDoRA(layer.attention.self.key, rank=rank, alpha=alpha) # LinearWithLoRAMerged

        # First, set requires_grad to False for all parameters
        for param in model.parameters():
            param.requires_grad = False

        # Then, set requires_grad to True for specific layers
        for layer in model.bert.encoder.layer:
            # Set requires_grad for LoRA parameters
            if hasattr(layer.attention.self.query, 'lora'):
                query = layer.attention.self.query
                key = layer.attention.self.key
                
                query.lora.A.requires_grad = True
                query.lora.B.requires_grad = True
                key.lora.A.requires_grad = True
                key.lora.B.requires_grad = True
                query.m.requires_grad = True
                key.m.requires_grad = True
        
        # Set requires_grad for classifier parameters
        if hasattr(model, 'classifier'):
            classifier = model.classifier
            classifier.weight.requires_grad = True
            classifier.bias.requires_grad = True
        
        return model

def check_requires_grad(model):
    """
    Check and print whether each parameter in the model requires gradients.

    Args:
    - model (nn.Module): PyTorch model instance.

    Prints:
    - For each parameter in each module, prints the module name and whether
      the parameter requires gradients (True/False).
    """
    for name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            print(f"Module: {name}, Parameter: {param_name}, Requires gradient: {param.requires_grad}")

def get_class_weights(train_dataloader):
    # Calculate the (normalised) inverse frequency of classes

    # Initialize a counter for the classes
    class_counter = Counter()

    # Iterate through the training data and update the counter
    for batch in train_dataloader:
        # Flatten the labels and convert to list
        labels = batch['labels'][batch['labels'] != -100].view(-1).tolist()  
        class_counter.update(labels)

    # Convert the counter to class weights
    # Get the total number of samples
    total_samples = sum(class_counter.values())

    # Calculate the class weights (inverse of frequency)
    class_weights = {cls: total_samples / count for cls, count in class_counter.items()}

    # Normalize the weights to sum to 1 (optional)
    total_weight = sum(class_weights.values())
    class_weights = {cls: weight / total_weight for cls, weight in class_weights.items()}

    # Convert class weights to a tensor
    num_labels = len(class_weights)
    weights = torch.zeros(num_labels)
    for cls, weight in class_weights.items():
        weights[cls] = weight

    print(f"Class Weights: {weights}")

    return weights

def evaluate(test_true_labels, test_predictions, num_labels, unique_labels, id2label):
    # if num_labels == 2:
    #     accuracy = accuracy_score(test_true_labels, test_predictions)
    #     # label_names = [id2label[i] for i in range(len(unique_labels))]
    #     # classification_rep = classification_report(test_true_labels, test_predictions)#, target_names=label_names)

    #     print(f"Accuracy: {round(accuracy, 3)}")
    #     # print("Classification Report:\n", classification_rep)

    #     # # Confusion matrix
    #     # tn, fp, fn, tp = confusion_matrix(test_true_labels, test_predictions).ravel()
    #     # print("TN:", tn, "FP:", fp, "FN:", fn, "TP:", tp)

    #     # specificity = tn / (tn + fp + 1e-5)
    #     # sensitivity = tp / (tp + fn + 1e-5)
    #     # precision = tp / (tp + fp + 1e-5)
    #     # recall = tp / (tp + fn + 1e-5)
    #     # f1score = (2*tp) / (2*tp + fp + fn + 1e-5)
    #     # print("SENSITIVITY:", round(sensitivity, 3))
    #     # print("SPECIFITY:", round(specificity, 3))
    #     # print("PRECISION:", round(precision, 3))
    #     # print("RECALL:", round(recall, 3))
    #     # print("F1-SCORE:", round(f1score, 3))

    #     # Calculate metrics for each class
    #     tn_list, fp_list, fn_list, tp_list = [], [], [], []
    #     for i in range(len(unique_labels)):
    #         tn, fp, fn, tp = confusion_matrix(np.array(test_true_labels) == i, np.array(test_predictions) == i).ravel()
    #         tn_list.append(tn)
    #         fp_list.append(fp)
    #         fn_list.append(fn)
    #         tp_list.append(tp)

    #     # Compute specificity, sensitivity, precision, recall, and f1-score for each class
    #     specificity = [tn / (tn + fp + 1e-5) for tn, fp in zip(tn_list, fp_list)]
    #     sensitivity = [tp / (tp + fn + 1e-5) for tp, fn in zip(tp_list, fn_list)]
    #     precision = [tp / (tp + fp + 1e-5) for tp, fp in zip(tp_list, fp_list)]
    #     recall = [tp / (tp + fn + 1e-5) for tp, fn in zip(tp_list, fn_list)]
    #     f1score = [(2 * tp) / (2 * tp + fp + fn + 1e-5) for tp, fp, fn in zip(tp_list, fp_list, fn_list)]

    #     # Round each value to 3 decimal points
    #     specificity = [round(val, 3) for val in specificity]
    #     sensitivity = [round(val, 3) for val in sensitivity]
    #     precision = [round(val, 3) for val in precision]
    #     recall = [round(val, 3) for val in recall]
    #     f1score = [round(val, 3) for val in f1score]

    #     print(f"SPECIFICITY: {specificity}")
    #     print(f"SENSITIVITY: {sensitivity}")
    #     print(f"PRECISION: {precision}")
    #     print(f"RECALL: {recall}")
    #     print(f"F1-SCORE: {f1score}")

    #     # Average metrics across all classes
    #     average_specificity = sum(specificity) / len(specificity)
    #     average_sensitivity = sum(sensitivity) / len(sensitivity)
    #     average_precision = sum(precision) / len(precision)
    #     average_recall = sum(recall) / len(recall)
    #     average_f1score = sum(f1score) / len(f1score)

    #     print(f"Average SPECIFICITY: {round(average_specificity, 3)}")
    #     print(f"Average SENSITIVITY: {round(average_sensitivity, 3)}")
    #     print(f"Average PRECISION: {round(average_precision, 3)}")
    #     print(f"Average RECALL: {round(average_recall, 3)}")
    #     print(f"Average F1-SCORE: {round(average_f1score, 3)}")

    #     return (precision, recall, f1score, average_precision, average_recall, average_f1score)

    # else:
        # Calculate accuracy
        accuracy = accuracy_score(test_true_labels, test_predictions)
        print(f"Accuracy: {accuracy:.4f}")

        # Classification report
        label_names = [id2label[i] for i in range(len(unique_labels))]
        classification_rep = classification_report(test_true_labels, test_predictions, target_names=label_names, zero_division=0)
        print("Classification Report:\n", classification_rep)

        # Confusion matrix
        conf_matrix = confusion_matrix(test_true_labels, test_predictions)
        print("Confusion Matrix:\n", conf_matrix)

        # Calculate metrics for each class
        tn_list, fp_list, fn_list, tp_list = [], [], [], []
        for i in range(len(unique_labels)):
            tn, fp, fn, tp = confusion_matrix(np.array(test_true_labels) == i, np.array(test_predictions) == i).ravel()
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
    
        return (precision, recall, f1score, average_precision, average_recall, average_f1score)
