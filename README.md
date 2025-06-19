# Sentiment Analysis on MyPertamina App Reviews

This repository contains a sentiment analysis project conducted on user reviews of the **MyPertamina** application, scraped directly from the **Google Play Store**. The project was developed to fulfill the submission requirement for the **"Belajar Pengembangan Machine Learning"** course by Dicoding and received a 5-star rating.

## Project Overview

* **Domain:** Customer feedback analysis
* **Objective:** Classify user reviews into Positive, Neutral, and Negative sentiments
* **Total Dataset:** 20,000 reviews scraped from Google Play Store
* **Language:** Bahasa Indonesia

## Data Collection

* Reviews were scraped using a custom script targeting the MyPertamina app on Google Play Store.
* A total of 20,000 review samples were collected and cleaned for analysis.

## Data Preprocessing and Feature Engineering

* **Text Cleaning:** Lowercasing, punctuation removal, stopword filtering
* **Feature Extraction:**

  * TF-IDF (Term Frequency–Inverse Document Frequency)
  * Embedding (for deep learning models)

## Sentiment Labeling

Reviews were labeled using a lexicon-based approach with three categories:

* Positive
* Neutral
* Negative

## Models Used

The project tested three different machine learning models with varied feature extraction methods and train-test splits:

| Model | Feature Extraction | Data Split | Training Accuracy | Testing Accuracy |
| ----- | ------------------ | ---------- | ----------------- | ---------------- |
| ANN   | TF-IDF             | 80:20      | 97%               | 89%              |
| SVM   | TF-IDF             | 70:30      | 99%               | 88%              |
| LSTM  | Embedding          | 90:10      | 97%               | 92%              |

## Deployment

The final model was deployed using **Google Cloud Run** and automated via **GitHub Actions** for CI/CD.
You can access the live sentiment analysis API at:

[Sentiment Analysis API (GCP Cloud Run)](https://sentimen-analisis-pertamina-325126223708.us-central1.run.app)

## Tech Stack

* Python
* TensorFlow / scikit-learn
* Google Cloud Platform (GCP)
* GitHub Actions
* Flask (for deployment)

