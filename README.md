# Anomaly Detection in Transactions

## Overview
This project focuses on detecting anomalies in transaction data to identify potential fraudulent activities. The dataset consists of various transactional attributes such as `Transaction_Amount`, `Account_Type`, `Age`, and many more. By utilizing machine learning techniques like **Isolation Forest** and **Local Outlier Factor (LOF)**, I tried to classify transactions as either normal or anomalous. Additionally, I performed Exploratory Data Analysis (EDA) to gain insights into customer behavior, spending patterns, and potential anomalies.

## EDA (Exploratory Data Analysis)
### Transaction Amount Distribution: To identify the normal spending range and potential outliers

![Transaction Amount Distribution](output/Transaction_Amount_Distribution.png)

### Transaction Amount vs Account Type: To compare transaction amounts across `Savings` and `Current` accounts.

![Transaction Amount vs Account Type](output/Transaction_amount_vs_Account_type.png)

### Average Transaction Amount by Age: To analyze the relationship between `Age` and `Average_Transaction_Amount`.

![Average Transaction Amount by Age](/output/Average_Transaction_amount_by_Age.png)

### Number of Transactions by Day of the Week: To analyze the transaction patterns across days of the week.

![Number of Transactions by Day of the Week](/output/Number_of_transactions_by_Days.png)

### Correlation Heatmap: To understand the relationships between numeric features in the dataset.

![Correlation Heatmap](/output/correlation_heatmap.png)

### Customer Segmentation by Age and Income: To segment customers based on `Age` and `Income`.

![Customer Segmentation by Age and Income](/output/Customer_Segmentation_by_Age_and_Income.png)

### Gender and Transaction Amount: To analyze transaction behavior across gender.

![Gender vs Transaction Amount](/output/Gender_vs_Transaction_Amount.png)

## Machine Learning Models
### Isolation Forest
I used the **Isolation Forest** algorithm to detect anomalies in the transaction data. The model was trained on features like `Transaction_Amount`, `Average_Transaction_Amount`, and `Frequency_of_Transactions`. The results showed that a small percentage of transactions (around 2%) were flagged as anomalies.

#### Evaluation
The model's performance was evaluated using precision, recall, and F1-score, which were calculated using the classification report.

```plaintext
              precision    recall  f1-score   support

      Normal       1.00      1.00      1.00       196
     Anomaly       1.00      1.00      1.00         4

    accuracy                           1.00       200
   macro avg       1.00      1.00      1.00       200
weighted avg       1.00      1.00      1.00       200
```

### Local Outlier Factor (LOF)

The Local Outlier Factor was used to further explore local density-based anomalies. As per the standards, the contamination rate was set to 2%, and the model detected a similar percentage of anomalies.

#### Evaluation
The LOF model's performance was also evaluated using a classification report.

```plaintext
              precision    recall  f1-score   support

      Normal       1.00      1.00      1.00       196
     Anomaly       1.00      1.00      1.00         4

    accuracy                           1.00       200
   macro avg       1.00      1.00      1.00       200
weighted avg       1.00      1.00      1.00       200
```

## Conclusion
- EDA revealed several interesting insights about transaction behavior across different customer attributes, including Account_Type, Age, and Income using these several outliers were identified  in transaction amounts, which could indicate potential fraudulent activities.
- Isolation Forest and Local Outlier Factor (LOF) were both effective in identifying anomalies in the dataset, though Isolation Forest showed slightly better precision and recall.
- This project builds on the foundational work provided in The Clever Programmer's Anomaly Detection in Transactions using Python by expanding the analysis to provide deeper insights into various other aspects of the anomaly detection.

## References:
1. [Isolation Forest](https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.IsolationForest.html)
2. [Anomaly Detection using Isolation Forest](https://www.digitalocean.com/community/tutorials/anomaly-detection-isolation-forest)
3. [Local Outlier Factor](https://scikit-learn.org/dev/modules/generated/sklearn.neighbors.LocalOutlierFactor.html)
4. [Anomaly Detection using LOF](https://medium.com/@ilyurek/anomaly-detection-with-local-outlier-factor-lof-b1b82227c15e)
5. [Project Reference](https://thecleverprogrammer.com/2023/08/21/anomaly-detection-in-transactions-using-python/)
6. [Dataset](https://statso.io/anomaly-detection-case-study/)
