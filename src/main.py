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

# Average Transaction Amount by Age

fig_scatter_amount = px.scatter(data,
                                x = "Age",
                                y = "Average_Transaction_Amount",
                                color = "Account_Type",
                                title = "Average Transaction Amount by Age",
                                trendline="ols")

fig_scatter_amount.show()

fig_scatter_amount.write_image("../output/Average_Transaction_amount_by_Age.png")

#COMMENTS:
# There is no significant increase or decrease in the average transaction amount as age progresses, indicating that age may not be a primary determinant of how much individuals are transacting on average.
# Both savings and current accounts display a similar range of average transaction amounts, suggesting that account type does not drastically affect average spending behavior.
# There are a few outliers at the lower end of the transaction amounts for older individuals (age 60+) in both account types, indicating occasional lower transaction averages for this age group.
# The trendline for both account types is relatively flat, thus concreting the observation that the average transaction amount does not show significant variation with age.

# Number of Transactions by the days of the week

day_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

fig_bar_transactions = px.bar(data,
                              x = "Day_of_Week",
                              title = "Number of Transactions by the Day of the Week",
                              category_orders={"Day_of_Week": day_order})

fig_bar_transactions.show()

fig_bar_transactions.write_image("../output/Number_of_transactions_by_Days.png")

#COMMENTS:
# Tuesday records the highest number of transactions, suggesting a possible trend in user behavior where more people engage in transactions early in the work week.
# Both Monday and Saturday show a slightly lower number of transactions compared to the mid-week, which could indicate that customers engage less in financial transactions at the start of the week or during the weekend.
# There is no significant drop-off or peak towards the end of the work week (Wednesday to Friday), which indicates a balanced distribution of transactional activity.

# Correlation Map

numeric_data = data.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_data.corr()
fig_corr_map = px.imshow(correlation_matrix, title = "Correlation Heatmap")
fig_corr_map.show()

fig_corr_map.write_image("../output/correlation_heatmap.png")

#COMMENTS:
# No interesting results.

# Visualizing the anomalies

mean_amount = data['Transaction_Amount'].mean()
std_amount = data['Transaction_Amount'].std()

anomaly_thereshold = mean_amount + 2 * std_amount

data['Is_Anomaly'] = data['Transaction_Amount'] > anomaly_thereshold

fig_anomalies = px.scatter(data,
                           x = "Transaction_Amount",
                           y = "Average_Transaction_Amount",
                           color = "Is_Anomaly",
                           title = "Anomalies in the transaction data")

fig_anomalies.update_traces(marker = dict(size=12),
                            selector = dict(mode = 'markers', marker_size = 1))

fig_anomalies.show()

fig_anomalies.write_image("../output/Anomalies_Transaction_Amount.png")

#COMMENTS:
# Regular transactions tend to cluster around a 'Transaction_Amount' range of 900 to 1100, with corresponding 'Average_Transaction_Amount' values between 940 and 1080.
# Anomalous transactions appear to have much higher 'Transaction_Amount' values, primarily ranging from 2000 to over 3000.
# All detected anomalies exhibit significantly higher transaction amounts compared to the regular transactions.

# Anomaly ratio

num_anomalies = data['Is_Anomaly'].sum()

total_instances = data.shape[0]

anomaly_ratio = num_anomalies/total_instances
print(anomaly_ratio)

# Customer Segmentation by Age and Income

fig_age_income = px.scatter(data,
                            x = "Age",
                            y = "Transaction_Amount",
                            color = "Income",
                            title = "Customer Segmentation by Age and Income")

fig_age_income.show()

fig_age_income.write_image("../output/Customer_Segmentation_by_Age_and_Income.png")