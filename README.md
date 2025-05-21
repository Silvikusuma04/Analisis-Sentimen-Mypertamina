<<<<<<< HEAD
## Project Overview

This project demonstrates the implementation of a deep learning model for **food image classification** using transfer learning with **EfficientNetV2L**. It is developed as part of the **Submission 2** requirement for the Dicoding course: *Belajar Pengembangan Machine Learning*.The model is trained on the **Food11 Image Dataset**, which contains 16,643 images across 11 categories of common food items. This project covers the complete pipeline: data preprocessing, model training, performance evaluation, and model conversion for deployment on web and mobile platforms.

## Model 
**Trained model and exported formats**, the trained model along with its exported formats (**SavedModel**, **TFLite**, and **TFJS**) is available for 
download [here](https://drive.google.com/drive/folders/1TvFO4QMHQRdQuFNe-OMm8fBUWfNU7Y-l?usp=sharing).

## Objective

The main objective of this project is to build a robust and efficient image classification model that can accurately classify food images into predefined categories. It also aims to demonstrate skills in using pretrained models, fine-tuning, model evaluation, and exporting for cross-platform deployment.

## Model Architecture

The model is built using **EfficientNetV2L** as the base feature extractor with the top classification layers removed (`include_top=False`). Only the last 30 layers are set to be trainable to allow fine-tuning on the food dataset.

Custom layers added after the base model include:
- `Conv2D` layers with ReLU activation and `BatchNormalization`
- `MaxPooling2D` and `GlobalAveragePooling2D`
- `Dropout` for regularization
- Fully connected `Dense` layers
- `Softmax` output layer matching the number of classes

## Dataset

- **Source**: [Kaggle - Food11 Image Dataset](https://www.kaggle.com/datasets/trolukovich/food11-image-dataset)
- **Total images**: 16,643
- **Classes**:
  - Bread, Dairy product, Dessert, Egg, Fried food, Meat, Noodles-Pasta, Rice, Seafood, Soup, Vegetable-Fruit
- **Splits**:
  - `training/`
  - `validation/`
  - `evaluation/`

## Preprocessing Steps

- Resize images to 384x384 pixels
- Normalize pixel values using `preprocess_input` from EfficientNet
- Load datasets using TensorFlow's `image_dataset_from_directory`

## How to Run

1. Open `notebook-clasifikasi-makanan.ipynb` in Google Colab
2. Upload the dataset or mount it from Kaggle
3. Run each cell sequentially
4. The model will be saved and ready for inference or deployment

## Project Structure

- `notebook-clasifikasi-makanan.ipynb`: Main Jupyter notebook
- `saved_model/`: Exported TensorFlow model
- `tflite/`: TFLite model and labels
- `tfjs_model/`: TensorFlow.js version
- `inference/`: Sample script for testing predictions
=======
# Sentiment Analysis on MyPertamina App Reviews

This repository contains a sentiment analysis project conducted on user reviews of the **MyPertamina** application, scraped directly from the **Google Play Store**. The project was developed to fulfill the submission requirement for the **"Belajar Pengembangan Machine Learning"** course by Dicoding and received a 5 star rating.

## Project Overview

* **Domain:** Customer feedback analysis
* **Objective:** Classify user reviews into Positive, Neutral, and Negative sentiments
* **Total Dataset:** 20,000 reviews scraped from Google Play Store
* **Language:** Bahasa Indonesia

### Data Collection

* Reviews were scraped using a custom script targeting the MyPertamina app on Google Play Store.
* A total of 20,000 review samples were collected and cleaned for analysis.

### Data Preprocessing and Feature Engineering

* Text Cleaning: Lowercasing, punctuation removal, stopword filtering
* Feature Extraction:
  * TF-IDF (Term Frequency–Inverse Document Frequency)
  * Embedding (for deep learning models)

### Sentiment Labeling
* Reviews were labeled using a lexicon-based approach with three categories:

  * Positive
  * Neutral
  * Negative

## Models Used

The project tested three different machine learning models with varied feature extraction methods and train-test splits:

| Model | Feature Extraction | Data Split | Training Accuracy | Testing Accuracy |
| ----- | ------------------ | ---------- | ----------------- | ---------------- |
| ANN   | TF-IDF             | 80:20      | 97%             | 89%            |
| SVM   | TF-IDF             | 70:30      | 99%             | 88%            |
| LSTM  | Embedding          | 90:10      | 97%             | 92%            |

>>>>>>> eaba145c29e25c784bd75732d964b3a35b7c1899
