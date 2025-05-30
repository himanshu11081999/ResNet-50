from IPython import get_ipython
from IPython.display import display
# %%
# Install necessary libraries if not already present
!pip install torch torchvision scikit-learn kagglehub

# %%
!pip install medmnist
# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, random_split
import os
import shutil # Import shutil for removing directories
import tarfile # Import tarfile for extracting .tar.gz files
import kagglehub # Import kagglehub for dataset download
import medmnist # Import medmnist
from medmnist.dataset import PneumoniaMNIST # Import the specific dataset class


# Dataset: PneumoniaMNIST
# Using kagglehub to download the dataset

# Define the target data directory where you want the data to reside
data_dir = 'data/pneumoniamnist'
dataset_file_name = 'pneumoniamnist.npz' # Expecting .npz file now
downloaded_npz_path = os.path.join(data_dir, dataset_file_name) # Define the path for the .npz file

# Check if the data directory already exists and contains data
# Check for the existence of the .npz file within the target data_dir
# Note: medmnist expects the .npz file in the root directory *before* processing.
# It then extracts/loads from this .npz into memory or potentially other files
# depending on the dataset.
if not os.path.exists(downloaded_npz_path):
    print(f"Dataset not found in {data_dir}. Downloading using kagglehub...")

    try:
        # Download the dataset using kagglehub.
        # This downloads the dataset files to a cache location.
        # The path returned is the root directory of the downloaded files in the cache.
        kaggle_download_root = kagglehub.dataset_download("rijulshr/pneumoniamnist")
        print(f"Kaggle dataset downloaded to cache: {kaggle_download_root}")

        # Locate the expected .npz file within the downloaded directory.
        downloaded_source_npz_path = os.path.join(kaggle_download_root, dataset_file_name)


        # Check if the expected .npz file exists in the downloaded path
        if not os.path.exists(downloaded_source_npz_path):
            print(f"Error: Expected file {dataset_file_name} not found in the Kaggle download path: {kaggle_download_root}")
            print("Please check the contents of the downloaded dataset on Kaggle or inspect the downloaded path.")
            # Optionally, list files in the downloaded directory for inspection
            print("Files found in downloaded directory:")
            for root, dirs, files in os.walk(kaggle_download_root):
                level = root.replace(kaggle_download_root, '').count(os.sep)
                indent = ' ' * 4 * (level)
                print(f'{indent}{os.path.basename(root)}/')
                subindent = ' ' * 4 * (level + 1)
                for f in files:
                    print(f'{subindent}{f}')
            raise FileNotFoundError(f"{dataset_file_name} not found in {kaggle_download_root}")

        print(f"Found dataset npz file at: {downloaded_source_npz_path}")

        # Create the target data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

        # **Changed:** Use shutil.copy2 instead of shutil.move
        # shutil.move attempts to delete the source file, which is not allowed in read-only /kaggle/input
        # shutil.copy2 copies the file metadata as well, which is generally preferred over shutil.copy
        print(f"Copying {downloaded_source_npz_path} to {downloaded_npz_path}")
        shutil.copy2(downloaded_source_npz_path, downloaded_npz_path)
        print("Copy complete.")


    except Exception as e:
        print(f"An error occurred during kagglehub download or processing: {e}")
        raise # Re-raise the exception

else:
    print(f"Dataset already found in {downloaded_npz_path}. Skipping download.")


# Transforms
# MedMNIST datasets are typically grayscale (1 channel). Normalize accordingly.
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Converts PIL Image to Tensor (H x W x C) to (C x H x W) and scales pixels to [0, 1]
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x), # Convert grayscale to 3 channels if needed by the model
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalization
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x), # Convert grayscale to 3 channels if needed by the model
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalization
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x), # Convert grayscale to 3 channels if needed by the model
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalization
    ])
}

# Load the datasets using medmnist
# PneumoniaMNIST is dataset index 5 in medmnist v2
# DataClass = medmnist.INFO['pneumoniamnist']['python_class'] # This returns the class NAME as a string
DataClass = PneumoniaMNIST # Assign the imported class to DataClass

info = medmnist.INFO['pneumoniamnist']
num_classes = len(info['label']) # Get the number of classes from the medmnist info

# **FIX:** Create a dummy instance to trigger medmnist's internal loading/processing
# from the .npz file we've placed in data_dir.
# This only needs to be done once. Subsequent dataset instances will find the processed data.
# We pass a basic transform as the actual transforms will be applied later.
# Set download=True to trigger the loading/processing logic if the processed data isn't found.
print("Checking and processing dataset files with medmnist (if necessary)...")
# Use a simple transform for this initial load if transforms are required by the constructor
# otherwise, None might suffice depending on the medmnist version/dataset
initial_transform = transforms.Compose([transforms.ToTensor()]) # Use a minimal transform
_ = DataClass(split='train', transform=initial_transform, download=True, root=data_dir)
print("Medmnist processing check complete.")


# Now load the actual datasets for training, validation, and testing
# Set download=False as the data should now be processed and available in data_dir
train_dataset = DataClass(split='train', transform=data_transforms['train'], download=False, root=data_dir)
val_dataset = DataClass(split='val', transform=data_transforms['val'], download=False, root=data_dir)
test_dataset = DataClass(split='test', transform=data_transforms['test'], download=False, root=data_dir)


# Print dataset sizes and class names for verification
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")
# MedMNIST datasets provide label information in the INFO dictionary
print(f"Classes: {list(info['label'].values())}")
print(f"Class to index mapping: {info['label']}")


# Data Loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) # Add a loader for the test set

# Load pretrained ResNet50
# Using weights=ResNet50_Weights.DEFAULT is the modern way to load default weights
# You need to import ResNet50_Weights
from torchvision.models import ResNet50_Weights
# ResNet50 expects 3 input channels, we've handled this in the transforms
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

# Freeze all layers except the final fully connected layer
for param in model.parameters():
    param.requires_grad = False

# Replace the final fully connected layer
# The number of output features should match the number of classes in your dataset.
# num_classes is already defined from medmnist info
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")

# Fine-tuning
criterion = nn.CrossEntropyLoss()
# Only optimize the parameters of the newly added final layer
optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)

# Training Loop
num_epochs = 1 # Define the number of epochs
for epoch in range(num_epochs):
    # Ensure the cell defining 'model' has been run before this cell
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        # Move data to device (GPU or CPU)
        inputs, labels = inputs.to(device), labels.to(device)

        # Squeeze the labels tensor to remove the extra dimension if it exists
        # CrossEntropyLoss expects target shape (batch_size,)
        if labels.ndim > 1 and labels.size(-1) == 1:
            labels = labels.squeeze(-1)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Calculate training accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()


    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct_train / total_train
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.4f}")

# Collect probabilities for the positive class
# For PneumoniaMNIST, typically label 1 corresponds to 'Pneumonia'
if num_classes > 1:
     # Use the index for 'pneumonia' label as the positive class index
     # MedMNIST labels are usually 0 for Normal, 1 for Pneumonia
     positive_class_label_name = 'pneumonia'
     # Find the index corresponding to 'pneumonia'
     positive_class_idx = -1
     for idx, name in info['label'].items():
         if name.lower() == positive_class_label_name:
             # Convert the string index to an integer
             positive_class_idx = int(idx)
             break

     if positive_class_idx != -1:
         y_prob_epoch.extend(probs_val[:, positive_class_idx].cpu().numpy())
     else:
         # Fallback or error if 'pneumonia' label is not found
         print(f"Warning: Label '{positive_class_label_name}' not found in dataset info.")
         # You might need to inspect the dataset labels to determine the positive class index
         # For PneumoniaMNIST, it's typically 1.
         y_prob_epoch.extend(probs_val[:, 1].cpu().numpy()) # Assuming 1 is the positive class index
else:
     # Handle case with only one class if necessary, though unlikely for this problem
     y_prob_epoch.extend(probs_val[:, 0].cpu().numpy()) # Or handle differently


epoch_val_acc = running_correct_val / running_total_val
print(f"Validation Accuracy: {epoch_val_acc:.4f}")

# Calculate validation F1 and AUC if there are at least two classes
if num_classes > 1 and len(set(y_true_epoch)) > 1:
    try:
        # Use the index for 'pneumonia' label as the positive class index for binary metrics
        positive_class_label_name = 'pneumonia'
        positive_class_idx = -1
        for idx, name in info['label'].items():
            if name.lower() == positive_class_label_name:
                # Convert the string index to an integer
                positive_class_idx = int(idx)
                break

        if positive_class_idx != -1:
             val_f1 = f1_score(y_true_epoch, y_pred_epoch, average='binary', pos_label=positive_class_idx)
             # roc_auc_score requires scores/probabilities for the positive class
             val_auc = roc_auc_score(y_true_epoch, y_prob_epoch)
             print(f"Validation F1 Score: {val_f1:.4f}, Validation AUC: {val_auc:.4f}")
        else:
             print(f"Could not calculate validation metrics: Label '{positive_class_label_name}' not found in dataset info.")


    except ValueError as e:
         print(f"Could not calculate validation metrics: {e}. This can happen if only one class is present in the validation batch for binary metrics.")


# Evaluation on Validation Set (Final)
print("\nFinal Validation Results:")
model.eval()
y_true_val_final, y_pred_val_final, y_prob_val_final = [], [], []
with torch.no_grad():
    # Indent the following lines
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        # Squeeze validation labels as well for final evaluation
        if labels.ndim > 1 and labels.size(-1) == 1:
            labels = labels.squeeze(-1)

        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        y_true_val_final.extend(labels.cpu().numpy())
        y_pred_val_final.extend(preds.cpu().numpy())
        if num_classes > 1:
            positive_class_label_name = 'pneumonia'
            positive_class_idx = -1
            for idx, name in info['label'].items():
                if name.lower() == positive_class_label_name:
                    # Convert the string index to an integer
                    positive_class_idx = int(idx)
                    break
            if positive_class_idx != -1:
                 y_prob_val_final.extend(probs[:, positive_class_idx].cpu().numpy())


# Calculate final validation metrics
if num_classes > 1 and len(set(y_true_val_final)) > 1:
    try:
        positive_class_label_name = 'pneumonia'
        positive_class_idx = -1
        for idx, name in info['label'].items():
             if name.lower() == positive_class_label_name:
                 # Convert the string index to an integer
                 positive_class_idx = int(idx)
                 break

        if positive_class_idx != -1:
             print("F1 Score:", f1_score(y_true_val_final, y_pred_val_final, average='binary', pos_label=positive_class_idx))
             print("AUC:", roc_auc_score(y_true_val_final, y_prob_val_final))
        else:
             print(f"Could not calculate final validation metrics: Label '{positive_class_label_name}' not found in dataset info.")
    except ValueError as e:
         print(f"Could not calculate final validation metrics: {e}")
else:
    print("Cannot calculate F1 and AUC for validation set (less than 2 classes or only one class in true labels).")


# Evaluation on Test Set
print("\nTest Results:")
model.eval()
y_true_test, y_pred_test, y_prob_test = [], [], []
with torch.no_grad():
    # Indent the following lines
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        # Squeeze test labels as well
        if labels.ndim > 1 and labels.size(-1) == 1:
            labels = labels.squeeze(-1)

        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        y_true_test.extend(labels.cpu().numpy())
        y_pred_test.extend(preds.cpu().numpy())
        if num_classes > 1:
            positive_class_label_name = 'pneumonia'
            positive_class_idx = -1
            for idx, name in info['label'].items():
                if name.lower() == positive_class_label_name:
                    # Convert the string index to an integer
                    positive_class_idx = int(idx)
                    break

            if positive_class_idx != -1:
                y_prob_test.extend(probs[:, positive_class_idx].cpu().numpy())


# Calculate test metrics
if num_classes > 1 and len(set(y_true_test)) > 1:
    try:
        positive_class_label_name = 'pneumonia'
        positive_class_idx = -1
        for idx, name in info['label'].items():
            if name.lower() == positive_class_label_name:
                # Convert the string index to an integer
                positive_class_idx = int(idx)
                break

        if positive_class_idx != -1:
             print("F1 Score:", f1_score(y_true_test, y_pred_test, average='binary', pos_label=positive_class_idx))
             print("AUC:", roc_auc_score(y_true_test, y_prob_test))
        else:
             print(f"Could not calculate test metrics: Label '{positive_class_label_name}' not found in dataset info.")

    except ValueError as e:
         print(f"Could not calculate test metrics: {e}")
else:
     print("Cannot calculate F1 and AUC for test set (less than 2 classes or only one class in true labels).")
