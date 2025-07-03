# 🩺 Pneumonia Detection from Chest X-Rays

A deep learning-based system to automatically detect **pneumonia** in chest X-ray images.

We built and compared multiple models — including a **custom CNN** and **transfer learning architectures** like **VGG16** and **ResNet50** — to classify chest X-rays into:
- `NORMAL`
- `PNEUMONIA`

This system aims to assist radiologists by providing a fast, reliable, and automated diagnostic tool.

## 📌 Table of Contents

- [🩺 Project Title & Overview](#-pneumonia-detection-from-chest-x-rays)
- [📁 Dataset](#-dataset)
- [🧠 Models Used](#-models-used)
- [📂 Folder Structure](#-folder-structure)
- [⚙️ Installation & Setup](#️-installation--setup)
- [📓 Notebooks Explained](#-notebooks-explained)
- [📊 Evaluation Metrics](#-evaluation-metrics)
- [📌 Key Observations](#-key-observations)
- [🚀 Future Work / Improvements](#-future-work--improvements)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)
- [🙋‍♀️ About Me / Credits](#-about-me--credits)

## 📁 Dataset

We used the **Chest X-Ray Images (Pneumonia)** dataset by **Kermany et al.**, available on [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

### 📊 Dataset Structure

The dataset is organized into three folders:

- `train/` – 5,216 images  
- `val/` – 16 images (used for validation during training)  
- `test/` – 624 images  

Each of these contains two classes:

- `NORMAL/` — Chest X-rays without pneumonia  
- `PNEUMONIA/` — Chest X-rays showing signs of pneumonia

### 🛠️ Preprocessing

- Images were resized to `(224, 224)` for consistency across models.
- Applied **data augmentation** (rotation, shift, zoom, horizontal flip) on the training set to improve generalization.
- Computed and applied **class weights** to handle class imbalance.
- Used a **very small validation set**, so techniques like:
  - `EarlyStopping`
  - `ReduceLROnPlateau`
  - Proper learning rate tuning  
  were used to avoid overfitting.

## 🧠 Models Used

We implemented and compared the performance of three different models:

### 1. 🧪 Custom CNN
- A baseline **Convolutional Neural Network** built from scratch.
- Contains Conv2D → MaxPooling → Dropout → Dense layers.
- Good starting point for comparison with more complex models.

### 2. 🔍 Transfer Learning — VGG16
- Leveraged the **VGG16** architecture pre-trained on ImageNet.
- Used only the convolutional base and added custom classification head.
- **Fine-tuned** top layers to adapt to pneumonia X-ray features.
- Achieved the **highest test accuracy** (~90.71%).

### 3. 🔁 Transfer Learning — ResNet50
- Used **ResNet50** as a feature extractor.
- Added global average pooling and dense classification layers.
- Fine-tuning was performed with care, using learning rate scheduling and early stopping.
- Performance was **slightly lower than VGG16** in our case.

### 📈 Performance Summary

| Model        | Test Accuracy | Precision | Recall | F1-Score |
|--------------|---------------|-----------|--------|----------|
| Custom CNN   | ~82.37%       | Moderate  | High   | Good     |
| VGG16        | **~90.71%**   | High      | High   | **Best** |
| ResNet50     | ~84.46%       | High      | Moderate | Good   |

> Note: Final scores are based on test set evaluation. VGG16 was selected as the best-performing model.

## 📁 Project Structure

```
pneumonia-detection/
├── datasets/                 # Directory to store training, validation, test images
├── notebooks/
│   ├── baseline_cnn_model.ipynb         # Custom CNN implementation
│   ├── transfer_learning_vgg16.ipynb    # VGG16 Transfer Learning
│   ├── transfer_learning_resnet50.ipynb # ResNet50 Transfer Learning
├── models/
│   ├── baseline/             # Saved CNN model
│   ├── vgg16/                # Saved VGG16 model
│   ├── resnet50/             # Saved ResNet50 model
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── .gitignore                # Files to ignore during commit
```
> 📁 Each notebook saves its best and final model into the corresponding folder inside `models/`.

