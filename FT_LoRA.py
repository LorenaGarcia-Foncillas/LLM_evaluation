from transformers import AutoModelForTokenClassification
from utils.load_data import load_data, create_NER_dataset, get_labels
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import pickle
from utils_train import *
from sklearn.model_selection import KFold

# Define paths - data folder containing txt/ and ann/ subfolders
data_folder = 'Data/'

# LOAD DATA - input: path to data folder & output: dict
# dictionary follows: examples = {'guid': [], 'text': [], 'ann': [[]]}
data_dict = load_data(data_folder)

# Only obtain the present_tumour_size labels
new_ann = []
# Loop through list
for sublist in data_dict["ann"]:
    # Initialise list
    new_sublist = []
    # Loop through sublist
    for element in sublist:
        # Check if label is of right type
        if element.entity_type == "PRESENT_TUMOUR_SIZE" or element.entity_type == "PER":
            # print(element.entity_type)
            new_sublist.append(element)
    # Append new sublist to new ann list
    new_ann.append(new_sublist)
# Assign new ann list to data dict
data_dict["ann"] = new_ann

print(data_dict.keys(), len(data_dict["guid"]), len(data_dict["text"]), len(data_dict["ann"]))

# Load indices (initially 70-15-15)
with open('indices_all_labels.pkl', 'rb') as f:
    train_indices, val_indices, test_indices = pickle.load(f)

# Applying cross-validation of 5 
num_folds = 5

# Combine indices for cross-validation
combined_indices = np.concatenate([train_indices, val_indices])

# Define K-fold cross-validation splitter
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Iterate through folds
for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(combined_indices)):
    print("#"*150)
    print(f"Fold {fold_idx + 1}/{num_folds}")

    # Specify model type: 0 - BERT, 1 - DistilBERT, 2 - BioBERT, 3 - DistilBioBERT, 4 - ClinicalBERT, 5 - DistilClinicalBERT
    model_type = 1
    model, tokenizer, batch_size, model_name, model_type = choose_model(model_type)

    # Access the state dictionary
    state_dict = model.state_dict()
    num_labels = 3
    # Modify the classifier to have 5 output features (4 annotation labels, 1 for other)
    model.classifier = nn.Linear(in_features=model.config.hidden_size, out_features=num_labels)
    # print("Modified Model Architecture:\n", model, "\n\n")

    # Apply LoRA to the query and key linear layers
    rank = 4
    alpha = 16
    model = apply_lora_to_transformer(model, rank, alpha, model_name)

    # Check requires_grad for each parameter in the model
    # check_requires_grad(model)

    # Checking number of parameters and trainable training
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    # Separate training-validation and testing indices
    train_indices = combined_indices[train_indices]
    val_indices = combined_indices[val_indices]

    # Split data into training, validation, and testing sets
    train_data = {key: [value[i] for i in train_indices] for key, value in data_dict.items()}
    val_data = {key: [value[i] for i in val_indices] for key, value in data_dict.items()}
    test_data = {key: [value[i] for i in test_indices] for key, value in data_dict.items()}

    # Create UNIQUE LABELS FOLLOWING PHI
    unique_labels = get_labels(data_dict['ann'])
    label2id = {tag: id for id, tag in enumerate(unique_labels)}
    print("label2id:", label2id)
    id2label = {id: tag for tag, id in label2id.items()}
    print("id2label:", id2label)

    # Create ner_datasets
    train_dataset = create_NER_dataset(train_data, tokenizer, label2id=label2id)
    val_dataset = create_NER_dataset(val_data, tokenizer, label2id=label2id)
    test_dataset = create_NER_dataset(test_data, tokenizer, label2id=label2id)
    print("Train dataset size:", len(train_dataset))
    print("Validation dataset size:", len(val_dataset))
    print("Test dataset size:", len(test_dataset))

    # Define dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Determine device 
    device = torch.device('cpu')  # Use 'cuda' if you have a GPU

    # Pass model to device and set to training mode
    model.to(device)

    num_epochs = 50

    # Define optimizer and scheduler
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)

    # Set the loss function
    weights = get_class_weights(train_dataloader)
    weights = weights.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights).to(device)
    # loss_fn = nn.CrossEntropyLoss().to(device)

    # Initialize variables for early stopping
    best_val_loss = float('inf')
    avg_train_loss = []
    avg_val_loss = []

    # ITERATE THROUGH LOOPS
    for epoch in range(num_epochs):
        # Set the model to training mode
        model.train()
        # Initialise training loss
        train_loss = 0.0
        # TRAINING LOOP
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Get the inputs for the models and pass them to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Clear gradients in optimizer
            optimizer.zero_grad()
            
            # Pass inputs to model
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits']

            # Flatten logits and labels, ignoring padding tokens
            masked_predictions = logits[labels != -100]
            masked_labels = labels[labels != -100]

            # Compute the loss
            loss = loss_fn(masked_predictions, masked_labels)
            train_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()
            # scheduler.step()

        # VALIDATION
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_true_labels = []
        with torch.no_grad():
            # VALIDATION LOOP 
            for batch in val_dataloader:
                # Get the inputs for the models and pass them to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                # Pass inputs to model
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs['logits']

                # Flatten logits and labels, ignoring padding tokens
                masked_predictions = logits[labels != -100]
                masked_labels = labels[labels != -100]

                # Compute the loss
                loss = loss_fn(masked_predictions, masked_labels)
                val_loss += loss.item()

                # Apply softmax to logits to get probabilities
                probabilities = F.softmax(logits, dim=-1)
                predicted_classes = torch.argmax(probabilities, dim=-1)

                # Gather predictions and true labels
                val_predictions.extend(predicted_classes[labels != -100].cpu().numpy())
                val_true_labels.extend(labels[labels != -100].cpu().numpy())   

        # Convert lists to numpy arrays for evaluation metrics
        val_predictions = np.array(val_predictions)
        val_true_labels = np.array(val_true_labels)

        # Calculate average losses
        train_loss /= len(train_dataloader)
        val_loss /= len(val_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Store losses
        avg_train_loss.append(train_loss)
        avg_val_loss.append(val_loss)
        

        # Evaluate
        print("\nVALIDATION RESULTS\n")
        # print(classification_report(val_true_labels, val_predictions))
        precision, recall, f1score, average_precision, average_recall, average_f1score = evaluate(val_true_labels, val_predictions, num_labels, unique_labels, id2label)

        # Check for early stopping - patience
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the model, optimiser and epoch
            checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
            torch.save(checkpoint, f'models/{model_name}/LoRA/best_checkpoint_LoRA_{num_labels}_labels_cv{fold_idx}.pth')
            # Save losses arrays
            average_losses_array = np.array(avg_train_loss)
            np.save(f'models/{model_name}/LoRA/best_LoRA_{num_labels}_labels_cv{fold_idx}_average_train_losses.npy', average_losses_array)
            average_losses_array = np.array(avg_val_loss)
            np.save(f'models/{model_name}/LoRA/best_LoRA_{num_labels}_labels_cv{fold_idx}_average_val_losses.npy', average_losses_array)

            # Save the evaluation results
            precision = np.array(precision)
            np.save(f'models/{model_name}/LoRA/best_LoRA_{num_labels}_labels_cv{fold_idx}_precision.npy', precision)
            recall = np.array(recall)
            np.save(f'models/{model_name}/LoRA/best_LoRA_{num_labels}_labels_cv{fold_idx}_recall.npy', recall)
            f1score = np.array(f1score)
            np.save(f'models/{model_name}/LoRA/best_LoRA_{num_labels}_labels_cv{fold_idx}_f1score.npy', f1score)
            average_precision = np.array(average_precision)
            np.save(f'models/{model_name}/LoRA/best_LoRA_{num_labels}_labels_cv{fold_idx}_average_precision.npy', average_precision)
            average_recall = np.array(average_recall)
            np.save(f'models/{model_name}/LoRA/best_LoRA_{num_labels}_labels_cv{fold_idx}_average_recall.npy', average_recall)
            average_f1score = np.array(average_f1score)
            np.save(f'models/{model_name}/LoRA/best_LoRA_{num_labels}_labels_cv{fold_idx}_average_f1score.npy', average_f1score)

        # Save the model, optimiser and epoch
        checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
        torch.save(checkpoint, f'models/{model_name}/LoRA/last_checkpoint_LoRA_{num_labels}_labels_cv{fold_idx}.pth')
        # Save losses arrays
        average_losses_array = np.array(avg_train_loss)
        np.save(f'models/{model_name}/LoRA/last_LoRA_{num_labels}_labels_cv{fold_idx}_average_train_losses.npy', average_losses_array)
        average_losses_array = np.array(avg_val_loss)
        np.save(f'models/{model_name}/LoRA/last_LoRA_{num_labels}_labels_cv{fold_idx}_average_val_losses.npy', average_losses_array)


    # Load the model and tokenizer
    model = AutoModelForTokenClassification.from_pretrained(model_type, num_labels=num_labels)

    # Determine the device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the checkpoint based on the device
    checkpoint_path = f'models/{model_name}/LoRA/best_checkpoint_LoRA_{num_labels}_labels_cv{fold_idx}.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load the model state dictionary
    state_dict = checkpoint['model_state_dict']

    # Add the LoRA parameters
    model = apply_lora_to_transformer(model, rank, alpha, model_name)

    # Load the filtered state dictionary into the model
    model.load_state_dict(state_dict)

    # Ensuring we are testing on the right model
    print("TESTING MODEL")
    print(model)
    for name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                    if "classifier" in name or "lora" in name:
                        print(f"Module: {name}, Parameter: {param_name}, Requires gradient: {param}")


    # Check requires_grad for each parameter in the model
    # check_requires_grad(model)

    # Move the model to the appropriate device
    model.to(device)

    # TESTING
    model.eval()
    test_predictions = []
    test_true_labels = []
    json_data = []
    with torch.no_grad():
        for batch in test_dataloader:
            # Pass variables to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits']

            # Apply softmax to logits to get probabilities
            probabilities = F.softmax(logits, dim=-1)
            predicted_classes = torch.argmax(probabilities, dim=-1)

            # Gather predictions and true labels
            test_predictions.extend(predicted_classes[labels != -100].cpu().numpy())
            test_true_labels.extend(labels[labels != -100].cpu().numpy())

    # Convert lists to numpy arrays for evaluation metrics
    test_predictions = np.array(test_predictions)
    test_true_labels = np.array(test_true_labels)

    # Evaluate
    print("\nTESTING RESULTS\n")
    precision, recall, f1score, average_precision, average_recall, average_f1score = evaluate(test_true_labels, test_predictions, num_labels, unique_labels, id2label)

    # Save the evaluation results
    precision = np.array(precision)
    np.save(f'models/{model_name}/LoRA/test_LoRA_{num_labels}_labels_cv{fold_idx}_precision.npy', precision)
    recall = np.array(recall)
    np.save(f'models/{model_name}/LoRA/test_LoRA_{num_labels}_labels_cv{fold_idx}_recall.npy', recall)
    f1score = np.array(f1score)
    np.save(f'models/{model_name}/LoRA/test_LoRA_{num_labels}_labels_cv{fold_idx}_f1score.npy', f1score)
    average_precision = np.array(average_precision)
    np.save(f'models/{model_name}/LoRA/test_LoRA_{num_labels}_labels_cv{fold_idx}_average_precision.npy', average_precision)
    average_recall = np.array(average_recall)
    np.save(f'models/{model_name}/LoRA/test_LoRA_{num_labels}_labels_cv{fold_idx}_average_recall.npy', average_recall)
    average_f1score = np.array(average_f1score)
    np.save(f'models/{model_name}/LoRA/test_LoRA_{num_labels}_labels_cv{fold_idx}_average_f1score.npy', average_f1score)
