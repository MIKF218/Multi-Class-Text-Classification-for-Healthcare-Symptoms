# Healthcare Symptom → Disease Classification
Multi-Class NLP Classification with TF-IDF, SimpleNN, RNN, and LSTM

This project builds and compares multiple machine learning and deep learning models to predict diseases or disease groups from short symptom descriptions. It includes three complete data-labeling scenarios and four model architectures.

## Project Overview
Predicting a disease from symptoms is challenging due to:
- Overlapping symptoms across diseases
- Very short symptom descriptions
- Noisy or inconsistent labels

This project evaluates strategies to handle label noise and improve prediction accuracy.

## Three Scenarios (Label Strategies)
### Scenario A — Raw Diseases
Uses raw disease labels.  
Result: Low accuracy due to label noise.

### Scenario B — Cleaned Diseases (Canonical Labels)
Groups rows by identical symptom text, assigns disease if ≥80% agree.  
Result: Strong performance improvement.

### Scenario C — Semantic Disease Clustering
1. Concatenate symptoms per disease  
2. Vectorize with TF-IDF  
3. Cluster diseases into 5 groups using KMeans  
4. Assign each sample its disease’s cluster  
Result: Balanced clusters and strong predictive performance.

## Models Implemented
- TF-IDF + Logistic Regression
- Simple Neural Network (Embedding → Dense)
- Recurrent Neural Network (RNN)
- Long Short-Term Memory (LSTM)

## Evaluation Metrics
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix


## Key Findings
- Raw disease labels produce poor results.
- Cleaned labels and semantic clusters significantly improve performance.
- TF-IDF + Logistic Regression is a strong baseline.
- LSTM is the best-performing deep model.

