import numpy as np 
import matplotlib as mat 
import pandas as pd 

dataframe = pd.read_csv('resources/airbnb_rio.csv')
# Drop not used attributes
dataframe = dataframe.drop('host_id',axis=1)
dataframe = dataframe.drop('host_response_time',axis=1)
dataframe = dataframe.drop('host_response_rate',axis=1)
dataframe = dataframe.drop('require_guest_phone_verification',axis=1)
dataframe = dataframe.drop('require_guest_profile_picture',axis=1)
dataframe = dataframe.drop('review_scores_location',axis=1)
dataframe = dataframe.drop('review_scores_communication',axis=1)
dataframe = dataframe.drop('is_location_exact',axis=1)
dataframe = dataframe.drop('longitude',axis=1)
dataframe = dataframe.drop('latitude',axis=1)
dataframe = dataframe.drop('host_identity_verified',axis=1)
dataframe = dataframe.drop('host_has_profile_pic',axis=1)
dataframe = dataframe.drop('host_listings_count',axis=1)


print(dataframe.columns)

dataframe = dataframe.dropna()
