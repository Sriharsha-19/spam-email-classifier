# Spam Email Classifier

## 📌 Project Description

This project is a Machine Learning based Spam Email Classifier. It checks an email/message and predicts whether it is **SPAM** or **HAM (Not Spam)**.

## 🎯 Objective

The main objective of this project is to automatically identify unwanted or spam messages using Machine Learning.

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Machine Learning

## 📂 Project Files

- `app.py` – Streamlit web application
- `train_model.py` – Code used to train the Machine Learning model
- `spam.csv` – Dataset used for training
- `spam_model.pkl` – Trained Machine Learning model
- `requirements.txt` – Required Python libraries

## ⚙️ How It Works

1. The dataset is loaded from `spam.csv`.
2. The text messages are converted into numerical features.
3. A Machine Learning model is trained using the dataset.
4. The trained model is saved as `spam_model.pkl`.
5. The Streamlit application loads the trained model.
6. The user enters a message.
7. The application predicts whether the message is **SPAM** or **HAM**.

## ▶️ How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
