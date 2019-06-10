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
dataframe = dataframe.drop('is_location_exact',axis=1)
dataframe = dataframe.drop('longitude',axis=1)
dataframe = dataframe.drop('latitude',axis=1)
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
print(dataframe.columns)


