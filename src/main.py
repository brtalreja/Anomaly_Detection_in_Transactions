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
# The plot shows a highly right-skewed distribution where the majority of transactions are concentrated between 850 and 1050.
# There are a few outliers visible in the higher transaction range, specifically in the 2500–3000+ range which could represent anomalies and in turn indicate potential anomalies like fraud.
# As the range of transactions is in between 850 and 1100, it can be a typical spending pattern or a standard transactional limit.

#Transaction Amount vs Account Type

fig_box_amount = px.box(data,
                        x = 'Account_Type',
                        y = 'Transaction_Amount',
                        title = 'Transaction Amount vs Account Type')

fig_box_amount.show()

fig_box_amount.write_image("../output/Transaction_amount_vs_Account_type.png")

#COMMENTS:
# The median transaction amount for both Savings and Current account types appears to be around 1000.
# Also, both the account types exhibit a similar interquartile range (IQR), meaning that the majority of transactions fall within the range 950 to 1050 for both Savings and Current accounts.
# Both Savings and Current accounts have a notable number of outliers with transaction amounts exceeding 2500. These high-value transactions are infrequent but can be seen in both types of accounts.
# These outliers may indicate high-spending customers or potentially anomalous behavior that warrants further investigation. 