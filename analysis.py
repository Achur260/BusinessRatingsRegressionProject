import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



#Reading the Json files into different DataFrames
business = pd.read_json('yelp_business.json', lines=True)
checkIn = pd.read_json('yelp_checkin.json', lines=True)
photo = pd.read_json('yelp_photo.json', lines=True)
review = pd.read_json('yelp_review.json', lines=True)
tip = pd.read_json('yelp_tip.json', lines=True)
user = pd.read_json('yelp_user.json', lines=True)


#Merging all the dataFrames in order to analyze all of the data.
df = pd.merge(business, checkIn, how='left', on='business_id')
df = pd.merge(df, photo, how='left', on='business_id')
df = pd.merge(df, review, how='left', on='business_id')
df = pd.merge(df, tip, how='left', on='business_id')
df = pd.merge(df, user, how='left', on='business_id')



#Dropping Irrelevant Columns (These columns are irrelevant for our prediction of Stars(rating))
df.drop(columns=['address', 'attributes', 'categories', 'city', 'hours', 'name', 'neighborhood', 'postal_code', 'state',
                 'time', 'latitude', 'longitude','business_id','is_open'],axis=1,inplace=True)

pd.options.display.max_columns = len(df.columns)
pd.options.display.max_colwidth = 500


#Fill all null/NaN values with zeroes these are numerical variables, so a missing value would indicate a zero
df.fillna({'weekday_checkins':True, 'weekend_checkins':True, 'average_caption_length':True,
           'number_pics':True, 'average_tip_length':True, 'number_tips':True}, inplace=True)


#Dropping our target variable in order to create a DataFrame of only our feature variables.
dfInput = df.drop(columns=['stars'])


#Creating a correlation matrix to check for collinearity amongst feature variables. The Pearson Correlation Coefficient (for Linear Regression) is used.
corr_matrix = dfInput.corr(method='pearson')


#Created another correlation matrix. This matrix also includes our target variable. This allows for the selection of the features with the most correlation with our target
corr_matrix2 = df.corr(method='pearson')


#Printing out the feature variable-pairs that are highly correlated, so that we can eliminate one (out of each pair) from our feature list. This reduces dimensionality and allows for a simpler model.
for i in range(len(corr_matrix.columns)):
    for j in range(i):

        if abs(corr_matrix.iloc[i,j]) >= .7:
            print(corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i,j])



#Printing out the correlation coefficients of individual features with respect to the target variable.
for i in range(len(corr_matrix2.columns)):
    if corr_matrix2.columns[i] == 'stars':
        print(corr_matrix2.iloc[i,])


#Creating our feature list(X) and target variable list(y)
X = df[['average_review_sentiment','average_review_length','average_review_age','price_range','average_review_count','average_number_years_elite','has_bike_parking']]
y = df['stars']


#Splitting the data into two portions. One portion is for training the linear regression model and the other is for testing it.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


#Initializing our Linear Regression model and fitting it to the data.
lr = LinearRegression()
lr.fit(X_train, y_train)


#Printing out the correlation of our linear model with respect to our training data.
print(lr.score(X_train, y_train))
#Printing out the correlation of our linear model with respect to our testing data.
print(lr.score(X_test, y_test))
#Printing out the coefficients of our linear model.
print(lr.coef_)

plt.scatter(pd.DataFrame(X_test)['average_review_sentiment'], y_test)
plt.title("Ratings vs. Average Review Sentiment (Testing Data)")
plt.xlabel("Average Review Sentiment")
plt.ylabel("Rating (out of 5)")
plt.show()
plt.clf()

plt.scatter(pd.DataFrame(X_test)['average_review_length'], y_test)
plt.title("Ratings vs. Average Review Length (Testing Data")
plt.xlabel("Average Review Length")
plt.ylabel("Rating (out of 5)")
plt.show()
plt.clf()

plt.scatter(pd.DataFrame(X_test)['average_review_age'], y_test)
plt.title("Ratings vs. Average Review Age (Testing Data)")
plt.xlabel("Average Review Age")
plt.ylabel("Rating (out of 5)")
plt.show()
plt.clf()


