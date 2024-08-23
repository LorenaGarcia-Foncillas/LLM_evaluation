import random
import pickle
from utils.load_data import load_data

# Define paths - data folder containing txt/ and ann/ subfolders
data_folder = 'Data/'

# LOAD DATA - input: path to data folder & output: dict
# dictionary follows: examples = {'guid': [], 'text': [], 'ann': [[]]}
data_dict = load_data(data_folder)

# STORE INDICES WHERE THERE IS A LABEL FOR PRESENT_TUMOUR_SIZE
present_tumour_indices = []
other_indices = []
# Iterate through annotations
for idx, sublist in enumerate(data_dict['ann']):
    found_present_tumour_size = False
    # Loop through sublist
    for label in sublist:
        # Check if present label is of type PRESENT_TUMOUR_SIZE
        if label.entity_type == "PRESENT_TUMOUR_SIZE":
            # print(idx)
            present_tumour_indices.append(idx)
            found_present_tumour_size = True
            break
    # Check if there was no label for PRESENT_TUMOUR_SIZE
    if not found_present_tumour_size:
        other_indices.append(idx)

print("Length of dataset:", len(data_dict['ann']))
print("Reports with PRESENT_TUMOUR_SIZE:", len(present_tumour_indices))
print("Reports with not such label:", len(other_indices))

# Shuffle indices
random.shuffle(present_tumour_indices)
random.shuffle(other_indices)

train_split = 0.7
val_split = 0.15
test_split = 0.15

# Function to split indices
def split_indices(indices, train_split, val_split):
    train_size = int(train_split * len(indices))
    val_size = int(val_split * len(indices))
    test_size = len(indices) - train_size - val_size
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    return train_indices, val_indices, test_indices

# Split present_tumour_indices
train_indices1, val_indices1, test_indices1 = split_indices(present_tumour_indices, train_split, val_split)

# Split other_indices
train_indices2, val_indices2, test_indices2 = split_indices(other_indices, train_split, val_split)

# Combine indices
train = train_indices1 + train_indices2
val = val_indices1 + val_indices2
test = test_indices1 + test_indices2

# Shuffle the combined indices
random.shuffle(train)
random.shuffle(val)
random.shuffle(test)

print(f"Train set total number of reports: {len(train)}, total number of reports with PRESENT TUMOUR SIZE: {len(train_indices1)}")
print(f"Validation set total number of reports: {len(val)}, total number of reports with PRESENT TUMOUR SIZE: {len(val_indices1)}")
print(f"Test set total number of reports: {len(test)}, total number of reports with PRESENT TUMOUR SIZE: {len(test_indices1)}")

# # Save the lists to a file using pickle
# with open('indices_all_labels.pkl', 'wb') as f:
#     pickle.dump((train, val, test), f)
