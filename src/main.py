#Import required libraries
import pandas as pd
import plotly.express as px

#Loading the data
data = pd.read_csv("../data/transaction_anomalies_dataset.csv")
print(data.head())

#Descriptives
print(data.info())

print(data.isnull().sum())

print(data.describe())

#EDA

#Transaction Amount Distribution

fig_amount = px.histogram(data,
                          x = 'Transaction_Amount',
                          nbins=20,
                          title="Transaction Amount Distribution")

fig_amount.show()

fig_amount.write_image("../output/Transaction_Amount_Distribution.png")

#COMMENTS:
# 
# 
# 