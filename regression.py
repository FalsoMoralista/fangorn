import numpy as np 
import matplotlib as mat 
import pandas as pd 

dataframe = pd.read_csv('resources/airbnb_rio.csv')
# Drop null values
dataframe = dataframe.dropna()

# Drops not used at all attributes
dataframe = dataframe.drop('require_guest_phone_verification',axis=1)
dataframe = dataframe.drop('require_guest_profile_picture',axis=1)
dataframe = dataframe.drop('review_scores_location',axis=1)
dataframe = dataframe.drop('review_scores_communication',axis=1)
dataframe = dataframe.drop('extra_people',axis=1)

# Drops host related attributes
dataframe = dataframe.drop('host_id',axis=1)
dataframe = dataframe.drop('host_response_time',axis=1)
dataframe = dataframe.drop('host_response_rate',axis=1)
dataframe = dataframe.drop('host_is_superhost',axis=1)
dataframe = dataframe.drop('host_identity_verified',axis=1)
dataframe = dataframe.drop('host_has_profile_pic',axis=1)
dataframe = dataframe.drop('host_listings_count',axis=1)
# Drop possbily usable review related attributes
dataframe = dataframe.drop('reviews_per_month',axis=1)
dataframe = dataframe.drop('review_scores_value',axis=1)
dataframe = dataframe.drop('review_scores_checkin',axis=1)
dataframe = dataframe.drop('review_scores_cleanliness',axis=1)
dataframe = dataframe.drop('review_scores_accuracy',axis=1)
dataframe = dataframe.drop('review_scores_rating',axis=1)
dataframe = dataframe.drop('number_of_reviews',axis=1)
# Drop room related attributes
dataframe = dataframe.drop('is_business_travel_ready',axis=1)
dataframe = dataframe.drop('cancellation_policy',axis=1)
dataframe = dataframe.drop('amenities',axis=1)
dataframe = dataframe.drop('instant_bookable',axis=1)
dataframe = dataframe.drop('room_type',axis=1)
dataframe = dataframe.drop('bed_type',axis=1)
# Drops property related attributes
dataframe = dataframe.drop('property_type',axis=1)
dataframe = dataframe.drop('neighbourhood',axis=1)
dataframe = dataframe.drop('is_location_exact',axis=1)
dataframe = dataframe.drop('longitude',axis=1)
dataframe = dataframe.drop('latitude',axis=1)
dataframe = dataframe.drop('maximum_nights',axis=1)


# Fill null values
columns = dataframe.columns

def fillMean(columns):  
    for i in columns:
        dataframe[i].fillna(dataframe[i].mean(),inplace=True)

def fillMode(columns):  
    for i in columns:
        dataframe[i].fillna(dataframe[i].mode()[0],inplace=True)

fillMean(columns=columns)
fillMode(columns=columns)

dataframe = dataframe[dataframe.price < 2000]
prices = dataframe.iloc[:,8:9]
bins = [0,500,1000,2000]
labels = ['medio','medio_alto','alto']
dataframe.price = pd.cut(x=dataframe.price,bins=bins,labels=labels)
print(teste.value_counts())

dataframe = dataframe[dataframe.price < 500]
# BASELINE
dataframe = dataframe[dataframe.price < 700]
dataframe = dataframe[dataframe.security_deposit < 1000]
dataframe = dataframe[dataframe.price > 15]
dataframe = dataframe[dataframe.accommodates < 5]
# BASELINE
dataframe = dataframe[dataframe.bathrooms < 3]
dataframe = dataframe[dataframe.bedrooms < 4]
dataframe = dataframe[dataframe.beds< 10]
dataframe = dataframe[dataframe.cleaning_fee< 500]
dataframe = dataframe[dataframe.guests_included< 6]
dataframe = dataframe[dataframe.minimum_nights < 10]
print(dataframe['price'].value_counts())
# Discretizar:
# cleaning_fee
# security_deposit
#

#dataframe = dataframe.fillna()

# Splits the dataframe predictors & targets
prices = dataframe['price'].values
predictors = dataframe.iloc[:,0:8].values
dataframe = dataframe.drop('price',axis=1)

#teste = dataframe['price'].value_counts() # nein

# Create our & train our model 
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0, min_samples_split=300, criterion='mae')
# Evaluate our model through cross validation passing our predictor attributes and the expected price values
# cv = amount of cross validation iterations
# n_jobs = The number of CPUs to use to do the computation.
regressor.fit(X=predictors,y=prices)
scores = cross_val_score(estimator=regressor,X=predictors,y=prices,cv=10,n_jobs=-1,scoring='neg_mean_absolute_error')
# Calculates the mean and standard deviation
mean = scores.mean()
sd = scores.std()
# Now let's visualize

from sklearn.tree import export_graphviz  
# export the decision tree to a tree.dot file 
# for visualizing the plot easily anywhere 
export_graphviz(regressor, out_file ='export/tree.dot', feature_names=dataframe.columns, leaves_parallel=True)


# test

import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

dataframe = dataframe[dataframe.price < 2000]
sns.distplot(dataframe['price'])
plot = sns.distplot(dataframe['accommodates'])
fig = plot.get_figure()
fig.savefig("accomodates")
plot = sns.distplot(dataframe['bathrooms']).get_figure().savefig("bathrooms")
sns.distplot(dataframe['bedrooms']).get_figure().savefig("bedrooms")
sns.distplot(dataframe['beds']).get_figure().savefig("beds")
sns.distplot(dataframe['cleaning_fee']).get_figure().savefig("cleaning_fee")
sns.distplot(dataframe['guests_included']).get_figure().savefig("guests_included")
sns.distplot(dataframe['minimum_nights']).get_figure().savefig("minimum_nights")
sns.distplot(dataframe['maximum_nights']).get_figure().savefig("maximum_nights")
sns.distplot(dataframe['price']).get_figure().savefig("price")



