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
- [📂 Project Structure](#-project-structure)
- [⚙️ Installation & Setup](#️-installation--setup)
- [📓 Notebooks Explained](#-notebooks-explained)
- [📊 Evaluation Metrics](#-evaluation-metrics)
- [📌 Key Observations](#-key-observations)
- [🚀 Future Work / Improvements](#-future-work--improvements)
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
| Custom CNN   | ~84.46%       | High  | Moderate   | Good     |
| VGG16        | **~90.71%**   | Very High | Very High | **Best** |
| ResNet50     | ~84.46%       | High      | High | Good   |

> Note: Final scores are based on test set evaluation. VGG16 was selected as the best-performing model.

## 📁 Project Structure

```
pneumonia-detection/
├── datasets/                     # Training, validation, and test images (not tracked in Git)
├── notebooks/
│   ├── baseline_cnn_model.ipynb         # Custom CNN implementation
│   ├── transfer_learning_vgg16.ipynb    # VGG16 Transfer Learning
│   ├── transfer_learning_resnet50.ipynb # ResNet50 Transfer Learning
│   ├── visualize_model_predictions.ipynb # Visualize predictions from all models
├── models/
│   ├── baseline/                # Saved CNN model
│   ├── vgg16/                   # Saved VGG16 model
│   ├── resnet50/                # Saved ResNet50 model
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── .gitignore                   # Files to ignore during commit
```

> 🖼️ `visualize_model_predictions.ipynb` helps in **visual inspection** of model predictions on test samples with predicted vs actual labels.

> 📁 Each notebook saves its best and final model into the corresponding folder inside `models/`.

## ⚙️ Installation & Setup

Follow these steps to set up and run the project locally:

### 1. 🛠️ Clone the Repository

```bash
git clone https://github.com/poojithadolai05/pneumonia-detection.git
cd pneumonia-detection
```
### 2. 🐍 Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```
### 3. 📦 Install Required Dependencies
```bash
pip install -r requirements.txt
```
> Make sure you have `pip` and `Python 3.7+` installed.

### 4. 📂 Prepare Dataset

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
2. Extract the ZIP file.
3. Place the extracted `chest_xray/` folder inside a directory named `data/` so your structure looks like:
```bash
pneumonia-detection/
└── data/
        ├── train/
        ├── val/
        └── test/
```
### 5. ▶️ Run the Notebooks
Launch Jupyter Lab or Notebook:
```bash
jupyter notebook
```

## 📓 Notebooks Explained

This project contains four Jupyter notebooks, each dedicated to training and evaluating a specific model or visualizing results:

### 1. `baseline_cnn_model.ipynb`
- Implements a custom Convolutional Neural Network from scratch.
- Includes steps for preprocessing, training, validation, and saving the best model.
- Provides accuracy and loss plots, confusion matrix, and final evaluation.

### 2. `transfer_learning_vgg16.ipynb`
- Applies **Transfer Learning** using pre-trained **VGG16** model (with `include_top=False`).
- Uses feature extraction with a custom classifier on top.
- Includes model saving, callbacks (early stopping, LR reduction), and evaluation.

### 3. `transfer_learning_resnet50.ipynb`
- Applies **ResNet50** for transfer learning.
- Similar structure to VGG16 but optimized for deeper network support.
- Performs training, evaluation, and confusion matrix plotting.

### 4. `visualize_model_predictions.ipynb`
- Loads saved models and performs **predictions on test set**.
- Displays sample chest X-ray images with:
  - **True labels**
  - **Predicted labels**
- Helps in analyzing model performance visually.

### 📊 Evaluation Metrics

We evaluated our models using the following metrics:

- **Accuracy**: Overall correctness of predictions.
- **Precision**: How many predicted positives are actually positive.
- **Recall (Sensitivity)**: How many actual positives were correctly predicted.
- **F1-Score**: Harmonic mean of precision and recall.
- **Confusion Matrix**: Breakdown of true/false positives and negatives.

Each model was evaluated on the test set using these metrics for a fair comparison.

### 📌 Key Observations

- **VGG16** outperformed all other models with the highest accuracy and F1-score.
- **ResNet50** was more stable during training but slightly underperformed compared to VGG16.
- The **custom CNN** showed decent results but struggled with complex features due to its simplicity.
- **Class imbalance** had a noticeable impact; using **class weights** helped improve performance.
- Visualization of predictions revealed that even the best models occasionally misclassify borderline cases.
- Models performed better on **PNEUMONIA** class than **NORMAL**, likely due to more distinct visual features.

### 🚀 Future Work / Improvements

- 📈 **Larger Validation Set**: Use a more representative validation set to improve generalization during training.
- 🧪 **Try More Architectures**: Experiment with other deep learning models like EfficientNet, DenseNet, or MobileNet.
- 🔍 **Explainability**: Integrate tools like Grad-CAM to visualize which parts of the image the model focuses on.
- 🏥 **Clinical Testing**: Evaluate the models on real-world hospital datasets for practical reliability.
- 📉 **Post-Processing**: Apply confidence thresholding or ensemble techniques to reduce false positives.
- ☁️ **Web App Deployment**: Create a frontend interface or deploy the model using tools like Streamlit or Flask.

### 📜 License

This project is licensed under the [MIT License](LICENSE).  
Feel free to use, modify, and distribute it as per the terms of the license.

### 🙋‍♀️ About Me / Credits 

Hi! I'm **Poojitha**, a passionate CSE student with a strong interest in **Deep Learning**, **Machine Learning**, and **AI for Social Good**.

This project was developed as part of my deep learning portfolio to apply medical AI in real-world contexts.  
I’m always eager to learn, build impactful tools, and collaborate on innovative ideas!

📫 Connect with me on:  
- GitHub: [@poojithadolai05](https://github.com/poojithadolai05)
- LinkedIn: [Poojitha Dolai](https://www.linkedin.com/in/poojitha-dolai/)
