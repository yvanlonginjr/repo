# HW2 – Stroke Prediction Classification Pipeline

## 📋 Summary
This project builds a simple, reproducible end-to-end machine learning pipeline to predict stroke occurrences based on patient data.  
The focus is on demonstrating proper data preprocessing, model training, evaluation, and submission generation for a Kaggle-style classification task.

## ⚙️ Modeling Pipeline
1. **Data Cleaning**
   - Handled missing values using `SimpleImputer` (median for numerical, most frequent for categorical).
   - Converted string-based numeric columns like `bmi` into floats.
   - Dropped duplicates based on the unique ID column.

2. **Feature Engineering**
   - Identified numerical and categorical variables.
   - Applied standard scaling to numeric features and one-hot encoding to categorical features.

3. **Modeling**
   - Three classifiers were tested inside a unified `Pipeline`:
     - Logistic Regression (L2 penalty)
     - Logistic Regression (L1 penalty)
     - K-Nearest Neighbors (KNN)
   - Each model was evaluated using F1, Precision, Recall, and AUC on a stratified 80/20 train-validation split.

4. **Model Selection**
   - The model with the highest **F1-score** on the validation set was chosen as the final model.
   - This model was then refitted on the full training set and used to generate binary predictions (0/1) for the test set.

5. **Submission**
   - Predictions were saved in `Output/submission4.csv` with columns `[ID, TARGET]`.

## 📊 Evaluation Metrics (Validation Set)
| Model | F1 | Precision | Recall | AUC |
|-------|----|------------|--------|-----|
| Logistic Regression (L2) | *0.XXX* | *0.XXX* | *0.XXX* | *0.XXX* |
| Logistic Regression (L1) | *0.XXX* | *0.XXX* | *0.XXX* | *0.XXX* |
| KNN | *0.XXX* | *0.XXX* | *0.XXX* | *0.XXX* |
| **Best Model:** Logistic Regression (L2)** |

*(Replace the X’s with your actual printed scores.)*

## 🏆 Kaggle Leaderboard
A screenshot of the Kaggle leaderboard showing the ranking is included under:

