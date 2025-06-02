# ResNet-50
# Pneumonia Detection using ResNet-50 on PneumoniaMNIST

This project fine-tunes a ResNet-50 model to classify chest X-ray images into pneumonia or normal using the PneumoniaMNIST dataset.

## 📦 Dataset
- Dataset: [PneumoniaMNIST](https://www.kaggle.com/datasets/rijulshr/pneumoniamnist/data)
- Format: `.npz`, processed using the `medmnist` library.

## 🚀 How to Run

```bash
# set path to your directory 
source myenv/bin/activate

# Clone the repo and navigate into the directory
pip install -r requirements.txt

# Run the training script (inside a Jupyter notebook or Python script)
python ResNet50.py
```

## 📊 Evaluation Metrics
- **F1 Score**: Reflects balance between sensitivity (recall) and precision.
- **AUC-ROC**: Clinical relevance in identifying true disease cases vs. false positives.
- **Accuracy**: General performance indicator.

## 🧠 Model
- **Backbone**: ResNet-50 pre-trained on ImageNet
- Final layer modified to output 2 classes (pneumonia, normal)
- Only the final fully connected layer was fine-tuned

## 🧪 Hyperparameters
- Learning Rate: `1e-4`
- Batch Size: `32`
- Epochs: `1` (adjustable)
- Optimizer: `Adam`
- Loss Function: `CrossEntropyLoss`

## 🔐 Class Imbalance Handling
- Dataset is balanced (checked using label counts in MedMNIST)
- Oversampling / data augmentation (e.g., horizontal flip) was applied

## 🛡️ Overfitting Mitigation
- Data augmentation (flip, resize)
- Early stopping can be added in extended training
