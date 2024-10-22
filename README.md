# Predicting Customer Churn

## Table of Contents

- [Project Overview](#project-overview)
- [Data Sources](#data-sources)
- [Tools](#tools)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Evaluation Summary](#model-evaluation-summary)
- [Model Training Code](#model-training-code)

### Project Overview

In this day and age there seems to be a countless number of companies offering subscription-based services to customers, and many of these companies are competing with other companies to sell people on the exact same type of subscription-based service. This is why it is becoming increasingly important for these companies to not only sell new customers on their subscriptions, but retain their existing customers as well. For this reason, it is crucial for companies to determine what factors play a significant role in keeping their consumers happy and willing to keep their subscriptions going. The goal of this model is to predict which customers are likely to end their subscriptions and determine the factors that may lead them to do so. This would allow companies to take action in an attempt to retain customers that may be on the brink of leaving by potentially offering new deals, increasing their catalogue of services, and more.

### Data Sources

Customer Usage Data: The primary dataset used for this analysis is the "customer_usage.csv" file, which contains customer usage data for a subscription service.

### Tools

- Python: Data Exploration, Data Visualization, Data Modeling

### Exploratory Data Analysis

In order to determine which variables could be useful to incorporate in my model for predicting customer churn, I used chi-square. Using this method I was able to identify more than 20 “significant” features. However, I did not use all 20 of these features in my model. Instead I experimented with various different feature combinations and compared their metrics with one another to determine which was best. Upon first glance, the following numeric variables from the CustomerChurn.csv data set were found to be the most helpful with creating my model:

Table 1: Statistically Significant Numeric Variables from the customer_usage.csv Dataset

![image](https://github.com/user-attachments/assets/c79275b0-26a2-407a-af76-68053dafe8d5)

The numeric variables listed in Table 1 were among the most statistically significant with respect to their chi-scores. Based on the averages of these features, among customers who either did or did not end up churning, some observations can be made. As expected, customers who use their subscriptions more often appear to be less likely to churn than those who do not. Customers who decide to keep their subscriptions active have more viewing hours per week, longer viewing durations, and they download more content per month on average. It is also important to note that customers who pay less appear to be more likely to continue with their subscriptions. To further demonstrate some of the differences between customers who did and did not churn, with respect to these features, visualizations such as box plots and bar charts can be used.

![image](https://github.com/user-attachments/assets/3fe585f9-9e57-4a37-98a6-269030093532)
![image](https://github.com/user-attachments/assets/b9b8fbff-d1ef-4882-b9a8-94403b3f23ea)
![image](https://github.com/user-attachments/assets/2dafe1b1-2a1e-46c8-95d5-08742d1c046e)
![image](https://github.com/user-attachments/assets/e6688a62-dce3-4759-9dd1-4aaee415cdd5)

By visualizing some of the significant variables impacting customer churn, I was able to reinforce some of the assumptions and findings I had from the Customer Churn data set. Figure 1 is a box plot comparing monthly charges for customers who churned and those who did not. Although the minimum and maximum monthly charges are the same for all customers, those who churned had a higher upper quartile, median, and lower quartile than those who did not churn. This further proves that churned customers were paying more for their subscriptions than those who stayed. Figure 2 is a box plot displaying the number of content downloads per month by churned and retained customers. This plot shows us that customers who did not churn were consistently downloading more content than those who did, which is signaled by the higher upper quartile, median, and lower quartile values by those who did not churn. Figure 3 and Figure 4 illustrate once again that customers who spend more time using the services they pay for are more likely to keep paying for them. On average, customers who churn are spending approximately 3.7 less hours per week and 19.5 less minutes per session watching content than those who continue with their subscriptions. Figure 5 and Figure 6 are box plots illustrating the range of total charges and account age respectively for customers who did and did not churn. It appears that these two variables may be negatively correlated with each other because customers who did not churn have been charged more than customers who did churn. This is likely because the customers who did not churn have been subscribed for longer, as seen by their longer account ages, allowing for them to be charged more in total. Figure 7 is a bar chart that shows us that customers who churned filed approximately 0.63 more support tickets per month than those who did not. This could be telling us that customers who had more problems with their subscription service were more likely to leave as a result. Lastly, Figure 8 compares the average watchlist size for customers who did and did not churn. Based on this bar chart, it appears that customers who churned had roughly 0.43 more shows/movies to watch than those who remained subscribed. This is unexpected because one would expect that people who unsubscribed did not have enough content that they were looking forward to watching, but the opposite appears to be true. Perhaps the customers who churned felt overwhelmed by the amount of content there was to watch, and just couldn’t find the time to watch it. Overall, these statistics and visualizations provide us with additional context for the Customer Churn data set to help us see how they may impact customer churn.

### Model Evaluation Summary

Table 2: Comparing Statistics from Different Models

![image](https://github.com/user-attachments/assets/03205937-4c91-4abd-ba58-67910f3bc0dc)

In order to determine what the best model was for predicting customer churn, I compared the average accuracy, precision, recall, and F1 values for three of my best models. Model 3 returned the worst values for each of average accuracy, precision, recall, and F1, so the main comparison to be made is between Models 1 and 2. For Model 1 I used eleven significant features, with respect to chi-score, that I found using the customer churn dataset. Of the three models that I compared, Model 1 had the highest scores for each of average accuracy, precision, recall, and F1. Based on these values, I concluded that Model 1 was the best for predicting customer churn. However, Model 2 is not very far behind. Not only does Model 2 have competitive values for average accuracy, precision, recall, and F1, it also has slightly smaller standard deviation values for accuracy and precision. This means that Model 2’s performance metrics may be more consistent than those of Model 1. It should also be noted that Model 2 only requires seven features to function, whereas Model 1 requires eleven. Although using less features reduces the accuracy, precision, recall, and F1 of the model, one can argue that it is a fair trade-off for the reduction in complexity seen in the model because it makes it more practical to use. Overall, I think Model 1 is the best out of the three because it returns the best statistical scores for each of accuracy, precision, recall, and F1. Model 2’s relatively high scores combined with its simplicity makes it an intriguing option to use, but it’s difficult to say for a fact that this simplicity makes up for the lower statistical scores.

### Model Training Code

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn import metrics
import pickle

# Import data into a DataFrame.
PATH = "C:/Python/DataSets/"
FILE = "customer_usage.csv"
df = pd.read_csv(PATH + FILE)

# Any data prep, imputing, binning, dummy variable creation etc goes here.
# Start imputing
def imputeNullValues(colName, df):
    # Create two new column names based on original column name.
    indicatorColName = 'm_'   + colName # Tracks whether imputed.
    imputedColName   = 'imp_' + colName # Stores original & imputed data.

    # Get mean or median depending on preference.
    imputedValue = df[colName].mean()

    # Populate new columns with data.
    imputedColumn  = []
    indictorColumn = []
    for i in range(len(df)):
        isImputed = False

        # mi_OriginalName column stores imputed & original data.
        if(np.isnan(df.loc[i][colName])):
            isImputed = True
            imputedColumn.append(imputedValue)
        else:
            imputedColumn.append(df.loc[i][colName])

        # mi_OriginalName column tracks if is imputed (1) or not (0).
        if(isImputed):
            indictorColumn.append(1)
        else:
            indictorColumn.append(0)

    # Append new columns to dataframe but always keep original column.
    df[indicatorColName] = indictorColumn
    df[imputedColName]   = imputedColumn
    del df[colName]     # Drop column with null values.
    return df

# Use imputeNullValues function
df = imputeNullValues('AccountAge', df)
df = imputeNullValues('ViewingHoursPerWeek', df)
df = imputeNullValues('AverageViewingDuration', df)


X = df.copy() # Create separate copy to prevent unwanted tampering of data.

del X['CustomerID'] # Delete unique identifier which is completely random.

X['AccountAgeBin'] = pd.cut(x=df['imp_AccountAge'], bins=[1, 40, 80, 120])

# Get Dummy variables
X = pd.get_dummies(X, columns=['SubscriptionType', 'PaymentMethod', 'PaperlessBilling', 'ContentType',
                               'MultiDeviceAccess', 'DeviceRegistered', 'GenrePreference', 'Gender',
                               'ParentalControl', 'SubtitlesEnabled', 'AccountAgeBin'], dtype=int)


from sklearn.linear_model import LogisticRegression

# Re-assign X with significant columns based on chi score.
X = X[['MonthlyCharges', 'TotalCharges', 'ContentDownloadsPerMonth',
       'SupportTicketsPerMonth', 'WatchlistSize', 'imp_AccountAge',
       'imp_ViewingHoursPerWeek', 'imp_AverageViewingDuration',
       'GenrePreference_Sci-Fi', 'AccountAgeBin_(1, 40]', 'AccountAgeBin_(80, 120]']]

# Load pre-trained model.
file = open("bestModel.pkl",'rb')
loadedModel = pickle.load(file)
file.close()

# Make predictions.
predictions = loadedModel.predict(X)

# Store predictions in a dataframe
dfPredictions = pd.DataFrame()
dfPredictions['Churn'] = predictions

# Save predictions to a CSV file
dfPredictions.to_csv('CustomerChurn_Predictions.csv', index=False)
```
