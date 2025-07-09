#!/usr/bin/env python
# coding: utf-8

# # Import statements

# In[1]:


import numpy as np 
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import re
from time import sleep
import pickle
import googlemaps
import os
from tqdm import tqdm
from scipy.optimize import minimize,Bounds

import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import RadiusNeighborsRegressor

from xgboost import XGBRegressor


# In[2]:


tqdm.pandas()


# In[3]:


sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (10,6)
plt.rcParams['font.size'] = 10


# In[4]:


R = 6071.088 #Radius of the earth in km
def simplified_haversine(lat1,lng1,lat2,lng2): #Function to get the distance between two latlngs in km
    lat1,lng1,lat2,lng2 = np.pi*lat1/180,np.pi*lng1/180,np.pi*lat2/180,np.pi*lng2/180
    dlat = lat2 - lat1
    dlng = np.cos(lat1)*(lng2 - lng1)
    return R*np.sqrt(dlat**2 + dlng**2)


# Click [here](#Checkpoint1-Load)  to go to Checkpoint1.\
# Click [here](#Checkpoint2-Load)  to go to Checkpoint2.\
# Click [here](#Checkpoint3-Load)  to go to Checkpoint3.\
# Click [here](#Checkpoint4-Load)  to go to Checkpoint4.\
# Click [here](#Checkpoint5-Load)  to go to Checkpoint5.\
# Click [here](#Checkpoint6-Load)  to go to Checkpoint6.\
# Click [here](#Checkpoint7-Load)  to go to Checkpoint7.

# # Load the data

# In[ ]:


df = pd.read_csv('train.csv',index_col = ['ID'])
dist_from_city_centre = pd.read_csv('dist_from_city_centre.csv',index_col = 'location')
avg_rent = pd.read_csv('avg_rent.csv',index_col='location')
test = pd.read_csv('test.csv',index_col = ['ID'])


# In[ ]:


df.head()


# In[ ]:


avg_rent.head()


# In[ ]:


dist_from_city_centre.head()


# In[ ]:


test.head()


# # Data inspection and cleaning

# ## location
# 
# We have a column indicating location in each of the datasets we have been give. But because the place name can have multiple strings associated with it (e.g RR Nagar = Rajarajeshwari Nagar), we should get the (latitude, longitude) for each location. Let's use google geocoding API to get latlong for each location in all the tables.

# In[ ]:


gmaps = googlemaps.Client(key=os.environ['GOOGLE_MAPS_API_KEY'])


# In[ ]:


def get_latlong(area_name):
    try:
        area_name = re.sub(r'[^a-zA-Z0-9\s]', '', area_name) #Remove all non-alphanumeric data
        geocode_result = gmaps.geocode(area_name+', Bengaluru')
        sleep(0.02) #To not trigger the rate limit
        return geocode_result[0]['geometry']['location']
    except:
        return {'lat':np.nan,'lng':np.nan}


# In[ ]:


train_latlngs = pd.DataFrame({x:get_latlong(x) for x in df['location'].unique()}).T
df[['lat','lng']] = train_latlngs.reindex(df['location']).reset_index(drop=True).round(6)


# Now do the same for avg_rent, dist_from_city_centre and the test data

# In[ ]:


avg_rent_latlngs = pd.DataFrame({x:get_latlong(x) for x in avg_rent.index}).T
avg_rent[['lat','lng']] = avg_rent_latlngs.round(6)


# In[ ]:


dist_from_city_centre_latlngs = pd.DataFrame({x:get_latlong(x) for x in dist_from_city_centre.index}).T
dist_from_city_centre[['lat','lng']] = dist_from_city_centre_latlngs.round(6)


# In[ ]:


test_latlngs = pd.DataFrame({x:get_latlong(x) for x in test['location'].unique()}).T
test[['lat','lng']] = test_latlngs.reindex(test['location']).reset_index(drop=True).round(6)


# ### Checkpoint1 Save

# In[ ]:


pickle.dump({'df':df,'avg_rent':avg_rent,'dist_from_city_centre':dist_from_city_centre,'test':test},open('Checkpoint1.pickle','wb'))


# ### Checkpoint1 Load

# In[ ]:


tmp = pickle.load(open('Checkpoint1.pickle','rb'))
df = tmp['df']
avg_rent = tmp['avg_rent']
dist_from_city_centre = tmp['dist_from_city_centre']
test = tmp['test']
del tmp


# Now, to get area from lat,lng reverse_geocode

# In[ ]:


gmaps = googlemaps.Client(key=os.environ['GOOGLE_MAPS_API_KEY'])


# In[ ]:


def reverse_geocode(lat,lng):
    "reverse geocode to get sublocality upto level 3"
    try:
        sleep(0.02) #To not trigger rate limit...
        result = gmaps.reverse_geocode((lat,lng),result_type = 'locality|sublocality|sublocality_level_1|sublocality_level_2|sublocality_level_3',
                                       location_type = 'APPROXIMATE')
        assert len(result)>0
    except:
        return pd.Series(np.nan,index = ['locality','sublocality','sublocality_level_1','sublocality_level_2','sublocality_level_3'],dtype = object)
    
    ans = pd.DataFrame('',index = pd.RangeIndex(len(result)),columns = ['locality','sublocality','sublocality_level_1','sublocality_level_2','sublocality_level_3'])
    for ctr,i in enumerate(result):
        ans.loc[ctr,[x for x in ans.columns if x in i['types']]] = i['formatted_address']
    
    return ans.apply(lambda x:max(x,key=len)).replace('',np.nan)


# In[ ]:


unique_latlngs = pd.concat((df[['lat','lng']],avg_rent[['lat','lng']],dist_from_city_centre[['lat','lng']],test[['lat','lng']])).drop_duplicates().dropna(
    ignore_index=True)
unique_latlngs.shape


# So there are 1152 locations which we want to reverse geocode:

# In[ ]:


unique_latlngs[['locality','sublocality','sublocality_level_1','sublocality_level_2','sublocality_level_3']] = unique_latlngs.progress_apply(
    lambda x:reverse_geocode(x['lat'],x['lng']),axis=1)


# In[ ]:


df = df.merge(unique_latlngs,on = ['lat','lng'],how = 'left')
avg_rent = avg_rent.reset_index().merge(unique_latlngs,on = ['lat','lng'],how = 'left')
dist_from_city_centre = dist_from_city_centre.reset_index().merge(unique_latlngs,on = ['lat','lng'],how = 'left')
test = test.merge(unique_latlngs,on = ['lat','lng'],how = 'left')


# In[ ]:


df['locality'] = df['locality'].replace('Bengaluru, Karnataka, India',np.nan)
test['locality'] = test['locality'].replace('Bengaluru, Karnataka, India',np.nan)


# ### Checkpoint2 Save

# In[ ]:


pickle.dump({'df':df,'avg_rent':avg_rent,'dist_from_city_centre':dist_from_city_centre,'test':test},open('Checkpoint2.pickle','wb'))


# ### Checkpoint2 Load

# In[ ]:


tmp = pickle.load(open('Checkpoint2.pickle','rb'))
df = tmp['df']
avg_rent = tmp['avg_rent']
dist_from_city_centre = tmp['dist_from_city_centre']
test = tmp['test']
del tmp


# To get the average 2bhk rent in a locality, first compare by the lowest granularlity (sublocality_level_3), and so on up the levels.

# In[ ]:


def get_avg_rent(row):
    if not (sub_df:=avg_rent.query(f'sublocality_level_3=="{row["sublocality_level_3"]}"')).empty:
        return round(sub_df['avg_2bhk_rent'].mean(),2)
    if not (sub_df:=avg_rent.query(f'sublocality_level_2=="{row["sublocality_level_2"]}"')).empty:
        return round(sub_df['avg_2bhk_rent'].mean(),2)
    if not (sub_df:=avg_rent.query(f'sublocality_level_1=="{row["sublocality_level_1"]}"')).empty:
        return round(sub_df['avg_2bhk_rent'].mean(),2)
    if not (sub_df:=avg_rent.query(f'sublocality=="{row["sublocality"]}"')).empty:
        return round(sub_df['avg_2bhk_rent'].mean(),2)
    if not (sub_df:=avg_rent.query(f'locality=="{row["locality"]}"')).empty:
        return round(sub_df['avg_2bhk_rent'].mean(),2)
    return np.nan


# In[ ]:


df['avg_2bhk_rent'] = df.progress_apply(get_avg_rent,axis=1)
test['avg_2bhk_rent'] = test.progress_apply(get_avg_rent,axis=1)


# In[ ]:


df['avg_2bhk_rent'].isna().mean()


# ~25% of the records have not got a match using the google geocoding - reverse_geocoding.

# In[ ]:


def get_dist_from_city_centre(row):
    if not (sub_df:=dist_from_city_centre.query(f'sublocality_level_3=="{row["sublocality_level_3"]}"')).empty:
        return round(sub_df['dist_from_city'].mean(),1)
    if not (sub_df:=dist_from_city_centre.query(f'sublocality_level_2=="{row["sublocality_level_2"]}"')).empty:
        return round(sub_df['dist_from_city'].mean(),1)
    if not (sub_df:=dist_from_city_centre.query(f'sublocality_level_1=="{row["sublocality_level_1"]}"')).empty:
        return round(sub_df['dist_from_city'].mean(),1)
    if not (sub_df:=dist_from_city_centre.query(f'sublocality=="{row["sublocality"]}"')).empty:
        return round(sub_df['dist_from_city'].mean(),1)
    if not (sub_df:=dist_from_city_centre.query(f'locality=="{row["locality"]}"')).empty:
        return round(sub_df['dist_from_city'].mean(),1)
    return np.nan


# In[ ]:


df['dist_from_city'] = df.progress_apply(get_dist_from_city_centre,axis=1)
test['dist_from_city'] = test.progress_apply(get_dist_from_city_centre,axis=1)


# In[ ]:


df['dist_from_city'].isna().mean()


# ~6% of the records don't have the distance from city feature using geocoding - reverse_geocoding.
# 
# We have extracted all we can from the column 'location' - remove it.

# In[ ]:


del df['location']
del test['location']
del avg_rent['location']
del dist_from_city_centre['location']


# Let's create a new standardized 'location' category for df and test: 
# 
# * We give first preference to sublocality_level_1, then 2, then 3. 
# 
# If all are null, then go for sublocality, and then locality.

# In[ ]:


def coalesce(*args):    
    # Start with the first series
    result = args[0]
    
    # Combine each subsequent series using combine_first
    for series in args[1:]:
        result = result.combine_first(series)
    
    return result


# In[ ]:


df['location'] = coalesce(df['sublocality_level_1'],df['sublocality_level_2'],df['sublocality_level_3'],df['sublocality'],df['locality'])
test['location'] = coalesce(test['sublocality_level_1'],test['sublocality_level_2'],test['sublocality_level_3'],test['sublocality'],test['locality'])


# In[ ]:


df['location'].nunique()


# Now, remove all the sublocality, locality columns which we got from google

# In[ ]:


df.drop(['locality','sublocality','sublocality_level_1','sublocality_level_2','sublocality_level_3'],axis=1,inplace=True)
test.drop(['locality','sublocality','sublocality_level_1','sublocality_level_2','sublocality_level_3'],axis=1,inplace=True)


# ### Checkpoint3 Save

# In[ ]:


pickle.dump({'df':df,'avg_rent':avg_rent,'dist_from_city_centre':dist_from_city_centre,'test':test},open('Checkpoint3.pickle','wb'))


# ### Checkpoint3 Load

# In[ ]:


tmp = pickle.load(open('Checkpoint3.pickle','rb'))
df = tmp['df']
avg_rent = tmp['avg_rent']
dist_from_city_centre = tmp['dist_from_city_centre']
test = tmp['test']
del tmp


# ## area_type

# In[ ]:


df['area_type'].value_counts(dropna=False)


# In[ ]:


test['area_type'].value_counts(dropna=False)


# Clean up spaces...

# In[ ]:


df['area_type'] = df['area_type'].apply(lambda x:x.replace('  ',' '))
test['area_type'] = test['area_type'].apply(lambda x:x.replace('  ',' '))


# No null values are here, and there are only four values. Convert to categorical, with 'Carpet Area as first'

# In[ ]:


area_type = pd.CategoricalDtype(['Carpet Area','Built-up Area','Super built-up Area','Plot Area'])
df['area_type'] = df['area_type'].astype(area_type)
test['area_type'] = test['area_type'].astype(area_type)


# In[ ]:


df['area_type'].value_counts(dropna=False).sort_index()


# In[ ]:


test['area_type'].value_counts(dropna=False).sort_index()


# ## availability

# In[ ]:


df['availability'].value_counts().head(30)


# In[ ]:


test['availability'].value_counts()


# Again no null values, but we can take Ready To Move/Immediate Posession as null and convert it into a date. No year(s) has been given, so we will assume that it's a single year (2025) for the purpose of calculation.

# In[ ]:


df['availability_dt'] = df['availability'].apply(lambda x:pd.NaT if x in ('Ready To Move','Immediate Possession') else datetime.strptime(x + '-2025', '%d-%b-%Y'))
df['availability_month'] = df['availability_dt'].dt.strftime('%b')


# In[ ]:


test['availability_dt'] = test['availability'].apply(lambda x:pd.NaT if x in ('Ready To Move','Immediate Possession') else datetime.strptime(x + '-2025', '%d-%b-%Y'))
test['availability_month'] = test['availability_dt'].dt.strftime('%b')


# Plot availability per month

# In[ ]:


sns.countplot(df,x = 'availability_month',order=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
plt.title('Availability per month')
plt.show()


# The assumption that this is a single year can't be relied upon. The 'Dec' category may combine data from multiple years. Some buildings may be available only in 2026/2027.
# 
# So the only reliable information we can get from this column is 'day of year'.

# In[ ]:


df['availability_doy'] = df['availability_dt'].dt.dayofyear
test['availability_doy'] = test['availability_dt'].dt.dayofyear


# Create a ready_to_move flag:

# In[ ]:


df['ready_to_move_flag'] = df['availability'].isin(['Ready To Move','Immediate Possession']).astype('b')
test['ready_to_move_flag'] = test['availability'].isin(['Ready To Move','Immediate Possession']).astype('b')


# Remove the availability columns from which we have extracted all the data

# In[ ]:


df.drop(['availability','availability_dt','availability_month'],axis=1,inplace=True)
test.drop(['availability','availability_dt','availability_month'],axis=1,inplace=True)


# ## size

# In[ ]:


df['size'].value_counts(dropna=False)


# In[ ]:


test['size'].value_counts(dropna=False)


# There are a few null values we need to deal with later. Apart from 13 records with 1RK, all the houses are of type (N)BHK or (N) Bedroom which we assume is the same. So we can extract the number of bedrooms as a numeric variable, that will contain all the information we need. We will take 1 RK = 0.5.

# In[ ]:


def get_num_bedrooms(size):
    if not type(size)==str:
        return np.nan
    elif size=='1 RK':
        return 0.5
    else:
        return int(size.split(' ')[0])


# In[ ]:


df['num_bedrooms'] = df['size'].apply(get_num_bedrooms)
test['num_bedrooms'] = test['size'].apply(get_num_bedrooms)


# Now, let's look at the distribution:

# In[ ]:


df['num_bedrooms'].value_counts(dropna=False).sort_index()


# Now  that we have extracted all the information from this column, we can remove it

# In[ ]:


del df['size']
del test['size']


# ## society

# In[ ]:


df['society'].nunique(),len(df)


# There are many unique values here, let's see if they can be combined.

# In[ ]:


df['society'].value_counts(normalize=True).cumsum().reset_index(drop=True).plot(kind = 'line',title ='Cumulative sum of society frequency')
plt.xlabel('Society')
plt.ylabel('Cumulative frequency')
plt.show()


# Combining categories is not so easy
# 
# Create an indicator that the flat is not a society:

# In[ ]:


df['not_society'] = df['society'].isna().astype('b')
test['not_society'] = test['society'].isna().astype('b')


# Now, the column 'society' is no longer needed.

# In[ ]:


del df['society']
del test['society']


# ## total_sqft

# In[ ]:


df['total_sqft'].value_counts()


# In[ ]:


test['total_sqft'].value_counts()


# There are some records with other units like Sq. Meter, and there are those where a range is given, not a single value. Let's look at them more closely:

# In[ ]:


def check_non_numeric(string):
    try:
        _ = float(string)
        return False
    except ValueError:
        return True


# In[ ]:


non_numeric = pd.Series(filter(check_non_numeric,pd.concat((df['total_sqft'],test['total_sqft']))))
non_numeric.head(30)


# Since there is a range of lower-upper in some records, we will get two numeric columns out of it - the mean and the range. 
# 
# For records with exact value, mean is the value and range is zero.
# 
# Let's list the records which contain measures other than sqft:

# In[ ]:


alphabetic = non_numeric.loc[non_numeric.apply(lambda x:not bool(re.findall(r'^\d+\.?\d* - \d+\.?\d*$',x)))]
alphabetic.head(30)


# So we have these other measures of area. Let's convert them into sqft:

# In[ ]:


sqft_converter = pd.Series({'Sq. Yards':9,'Sq. Meter':10.7639,'Perch':272.25,'Cents':435.6,'Guntha':1089,'Grounds':2400,'Acres':43560})
sqft_converter


# Number of records usinf these units:

# In[ ]:


pd.Series(sqft_converter.index,index = sqft_converter.index).apply(lambda x:alphabetic.apply(lambda y:y.endswith(x)).sum())


# Now, write a function that will handle all cases:

# In[ ]:


def clean_sqft(string):
    mul_factor = 1
    
    if any(chk := [string.endswith(x) for x in sqft_converter.index]): #When the data has been given in other units
        mul_factor = sqft_converter[chk].squeeze()
        string = string.replace(sqft_converter.index[chk.index(True)],'')
    
    #Check if it's a single number, if not split by ' - '
    try:
        ans = float(string)
        return mul_factor*ans,0
    except ValueError:
        low,high = string.split(' - ')
        low,high = mul_factor*float(low), mul_factor*float(high)
        return (low+high)/2,high - low


# In[ ]:


df[['sqft_avg','sqft_range']] = np.array(df['total_sqft'].apply(clean_sqft).tolist())
test[['sqft_avg','sqft_range']] = np.array(test['total_sqft'].apply(clean_sqft).tolist())


# Let's also extract the unit in which the area was given as a categorical variable, this may contain some useful information:

# In[ ]:


area_units = pd.CategoricalDtype(['Sq. Feet','Sq. Yards','Sq. Meter','Perch','Cents','Guntha','Grounds','Acres'])


# Write a function that will extract units from the 'total_sqft' column:

# In[ ]:


def get_units(string):
    check_unit = [x in string for x in sqft_converter.index]
    if any(check_unit):
        return sqft_converter.index[check_unit.index(True)]
    else:
        return 'Sq. Feet'


# In[ ]:


df['area_units'] = df['total_sqft'].apply(get_units).astype(area_units)
test['area_units'] = test['total_sqft'].apply(get_units).astype(area_units)


# In[ ]:


df['area_units'].value_counts(dropna=False).sort_index()


# In[ ]:


test['area_units'].value_counts(dropna=False).sort_index()


# Now remove the column 'total_sqft'

# In[ ]:


del df['total_sqft']
del test['total_sqft']


# ## bath

# In[ ]:


df['bath'].value_counts(dropna=False).sort_index()


# In[ ]:


test['bath'].value_counts(dropna=False).sort_index()


# These are the number of bathrooms with null values specifying lack of data.

# ## balcony

# In[ ]:


df['balcony'].value_counts(dropna=False).sort_index()


# In[ ]:


test['balcony'].value_counts(dropna=False).sort_index()


# Here null values may specify lack of data - zero has it's own category.

# ### Checkpoint4 Save

# In[ ]:


pickle.dump({'df':df,'avg_rent':avg_rent,'dist_from_city_centre':dist_from_city_centre,'test':test},open('Checkpoint4.pickle','wb'))


# ### Checkpoint4 Load

# In[ ]:


tmp = pickle.load(open('Checkpoint4.pickle','rb'))
df = tmp['df']
avg_rent = tmp['avg_rent']
dist_from_city_centre = tmp['dist_from_city_centre']
test = tmp['test']
del tmp


# # Handle null values

# In[ ]:


null_df = pd.DataFrame({'train # nulls':df.isna().sum(),'train % nulls':(100*df.isna().sum()/len(df)).round(2),'test # nulls':test.isna().sum(),
                       'test % nulls':(100*test.isna().sum()/len(test)).round(2)})
null_df


# Note that lat, lng come from location which has one null in the training data.  Since it has no nulls in the test data, we can simply remove this record.

# In[ ]:


df.dropna(subset = ['lat','lng'],inplace=True)


# For distance from city_centre, first find the location of the city center. Suppose the location is center_point, we need to minimize:

# In[ ]:


def fun(centre_point):
    centre_lat,centre_lng = centre_point
    d = simplified_haversine(centre_lat,centre_lng,dist_from_city_centre['lat'],dist_from_city_centre['lng'])
    return ((d - dist_from_city_centre['dist_from_city'])**2).mean()


# In[ ]:


ans = minimize(fun,x0 = dist_from_city_centre.loc[dist_from_city_centre['dist_from_city'].idxmin(),['lat','lng']].tolist())
ans


# In[ ]:


centre_lat,centre_lng = ans.x


# From here we can fill in distance from city centre for the null values, we will also keep a flag indicating nulls which the model can learn from:

# In[ ]:


df['dist_from_city_null'] = df['dist_from_city'].isna().astype('b')
test['dist_from_city_null'] = test['dist_from_city'].isna().astype('b')


# In[ ]:


df.loc[df['dist_from_city'].isna(),'dist_from_city'] = df.loc[df['dist_from_city'].isna()].apply(lambda x:simplified_haversine(
    centre_lat,centre_lng,x['lat'],x['lng']),axis=1)
test.loc[test['dist_from_city'].isna(),'dist_from_city'] = test.loc[test['dist_from_city'].isna()].apply(lambda x:simplified_haversine(
    centre_lat,centre_lng,x['lat'],x['lng']),axis=1)


# Now, we can fill in avg_2bhk_rent using distance from city centre, since the two are related:

# In[ ]:


sns.regplot(df,x = 'dist_from_city',y = 'avg_2bhk_rent')
plt.title('Avg 2BHK rent vs distance from city centre');


# Build a linear regression model of avg_2bhk_rent vs dist_from_city

# In[ ]:


tmp = pd.concat((df[['lat','lng','dist_from_city','avg_2bhk_rent']],test[['lat','lng','dist_from_city','avg_2bhk_rent']])).dropna(ignore_index=True)
linear_model = LinearRegression().fit(X = tmp[['dist_from_city']],y = tmp['avg_2bhk_rent'])
mean_absolute_percentage_error(tmp['avg_2bhk_rent'],linear_model.predict(tmp[['dist_from_city']]))


# Build a RNR (RadiusNeighborsRegressor) for this same data

# In[ ]:


rnr_model = RadiusNeighborsRegressor(radius=2.0, metric = lambda arr1,arr2: simplified_haversine(arr1[0],arr1[1],arr2[0],arr2[1]))
_ = rnr_model.fit(tmp[['lat','lng']],tmp['avg_2bhk_rent'])
mean_absolute_percentage_error(tmp['avg_2bhk_rent'],rnr_model.predict(tmp[['lat','lng']]))


# Make a prediction from both and average them to get the 2bhk rent for the unknown locations. But also keep a column for these nulls:

# In[ ]:


df['avg_2bhk_rent_null'] = df['avg_2bhk_rent'].isna().astype('b')
test['avg_2bhk_rent_null'] = test['avg_2bhk_rent'].isna().astype('b')


# In[ ]:


linear_prediction = linear_model.predict(df.loc[df['avg_2bhk_rent'].isna(),['dist_from_city']])
rnr_prediction = rnr_model.predict(df.loc[df['avg_2bhk_rent'].isna(),['lat','lng']])
df.loc[df['avg_2bhk_rent'].isna(),'avg_2bhk_rent'] = np.nanmean(np.stack((linear_prediction, rnr_prediction)), axis=0)


# In[ ]:


linear_prediction = linear_model.predict(test.loc[test['avg_2bhk_rent'].isna(),['dist_from_city']])
rnr_prediction = rnr_model.predict(test.loc[test['avg_2bhk_rent'].isna(),['lat','lng']])
test.loc[test['avg_2bhk_rent'].isna(),'avg_2bhk_rent'] = np.nanmean(np.stack((linear_prediction, rnr_prediction)), axis=0)


# Now, let's look at the nulls again:

# In[ ]:


null_df = pd.DataFrame({'train # nulls':df.isna().sum(),'train % nulls':(100*df.isna().sum()/len(df)).round(2),'test # nulls':test.isna().sum(),
                       'test % nulls':(100*test.isna().sum()/len(test)).round(2)})
null_df


# For availability_doy, the nulls have a specific meaning - they represent when the house is available immediately. We can fill the nulls with some value less than the minimum for availability_doy:

# In[ ]:


df['availability_doy'] = df['availability_doy'].fillna(-1)
test['availability_doy'] = test['availability_doy'].fillna(-1)


# Now, for 'num_bedrooms','bath' and 'balcony' - let's predict them in terms of other variables:

# In[ ]:


def symmetric_mean_absolute_percentage_error(y_true,y_pred):
    return np.mean((y_true - y_pred).abs()/((y_true + y_pred)/2))


# In[ ]:


all_cols = ['area_type','sqft_avg','balcony','bath','num_bedrooms','avg_2bhk_rent','avg_2bhk_rent_null','not_society']
for col in ['balcony','bath','num_bedrooms']:
    predictors = all_cols.copy()
    predictors.remove(col)
    
    X = pd.get_dummies(df.dropna(subset = all_cols)[predictors],drop_first=True,dtype = 'b')
    y = df.dropna(subset = all_cols)[col]
    X_test = pd.get_dummies(test.dropna(subset = all_cols)[predictors],drop_first=True,dtype = 'b')
    y_test = test.dropna(subset = all_cols)[col]
    
    model = RandomForestRegressor().fit(X,y)
    y_pred = model.predict(X_test)
    print(round(100*symmetric_mean_absolute_percentage_error(y_test,y_pred),2))


# So a random forest regressor gives a SMAPE of 41% for 'balcony' - better than nothing, and for the others it's even better.
# 
# Let's first solve for num_bedrooms, then bath and then balcony, using the results of each for the next. But first keep a null indicator for balcony:

# In[ ]:


df['balcony_null'] = df['balcony'].isna().astype('b')
test['balcony_null'] = test['balcony'].isna().astype('b')


# In[ ]:


slct_cols = all_cols.copy()
slct_cols.remove('num_bedrooms')
slct_cols.remove('bath')
slct_cols[slct_cols.index('balcony')] = 'balcony_null'

X = pd.concat((pd.get_dummies(df.dropna(subset = ['num_bedrooms'])[slct_cols],drop_first=True,dtype = 'b'),
               pd.get_dummies(test.dropna(subset = ['num_bedrooms'])[slct_cols],drop_first=True,dtype = 'b')))
y = pd.concat((df['num_bedrooms'].dropna(),test['num_bedrooms'].dropna()))
model = RandomForestRegressor().fit(X,y)
df.loc[df['num_bedrooms'].isna(),'num_bedrooms'] = model.predict(pd.get_dummies(df.loc[df['num_bedrooms'].isna(),slct_cols],drop_first=True,dtype = 'b'))
test.loc[test['num_bedrooms'].isna(),'num_bedrooms'] = model.predict(pd.get_dummies(test.loc[test['num_bedrooms'].isna(),slct_cols],drop_first=True,dtype = 'b'))


# In[ ]:


slct_cols.append('num_bedrooms')

X = pd.concat((pd.get_dummies(df.dropna(subset = ['bath'])[slct_cols],drop_first=True,dtype = 'b'),
               pd.get_dummies(test.dropna(subset = ['bath'])[slct_cols],drop_first=True,dtype = 'b')))
y = pd.concat((df['bath'].dropna(),test['bath'].dropna()))
model = RandomForestRegressor().fit(X,y)
df.loc[df['bath'].isna(),'bath'] = model.predict(pd.get_dummies(df.loc[df['bath'].isna(),slct_cols],drop_first=True,dtype = 'b'))
test.loc[test['bath'].isna(),'bath'] = model.predict(pd.get_dummies(test.loc[test['bath'].isna(),slct_cols],drop_first=True,dtype = 'b'))


# In[ ]:


slct_cols.append('bath')

X = pd.concat((pd.get_dummies(df.dropna(subset = ['balcony'])[slct_cols],drop_first=True,dtype = 'b'),
               pd.get_dummies(test.dropna(subset = ['balcony'])[slct_cols],drop_first=True,dtype = 'b')))
y = pd.concat((df['balcony'].dropna(),test['balcony'].dropna()))
model = RandomForestRegressor().fit(X,y)
df.loc[df['balcony'].isna(),'balcony'] = model.predict(pd.get_dummies(df.loc[df['balcony'].isna(),slct_cols],drop_first=True,dtype = 'b'))
test.loc[test['balcony'].isna(),'balcony'] = model.predict(pd.get_dummies(test.loc[test['balcony'].isna(),slct_cols],drop_first=True,dtype = 'b'))


# In[ ]:


null_df = pd.DataFrame({'train # nulls':df.isna().sum(),'train % nulls':(100*df.isna().sum()/len(df)).round(2),'test # nulls':test.isna().sum(),
                       'test % nulls':(100*test.isna().sum()/len(test)).round(2)})
null_df


# Now, the null values are there for the 'location' column, which is the standardized location. We can simply put it as a separate category, but also have a null indicator:

# In[ ]:


df['location_null'] = df['location'].isna().astype('b')
test['location_null'] = test['location'].isna().astype('b')


# In[ ]:


df['location'] = df['location'].fillna('NA')
test['location'] = test['location'].fillna('NA')


# In[ ]:


df.isna().any().any()


# In[ ]:


test.isna().any().any()


# ### Checkpoint5 Save

# In[ ]:


pickle.dump({'df':df,'test':test},open('Checkpoint5.pickle','wb'))


# ### Checkpoint5 Load

# In[ ]:


tmp = pickle.load(open('Checkpoint5.pickle','rb'))
df = tmp['df']
test = tmp['test']
del tmp


# # XGBoost model
# 
# When we do train_test_split, we want to stratify on 'location'. But there are some locations with only one record. Set them to NA:

# In[ ]:


stratifier = df['location'].replace(df['location'].value_counts().pipe(lambda x:x.index[x==1]),'NA')
train,validation = train_test_split(df,test_size=0.3,random_state=0,stratify=stratifier)


# Let's calculate the average price at each location, and use it for our estimates:

# In[ ]:


location_avg_price_mapper = train.groupby('location')['price'].mean()
train['location_avg_price'] = location_avg_price_mapper.reindex(train['location'],fill_value = location_avg_price_mapper['NA']).to_numpy()
validation['location_avg_price'] = location_avg_price_mapper.reindex(validation['location'],fill_value = location_avg_price_mapper['NA']).to_numpy()
test['location_avg_price'] = location_avg_price_mapper.reindex(test['location'],fill_value = location_avg_price_mapper['NA']).to_numpy()


# Now, drop location and some other useless columns (which  we learn by trial and error) and convert area_type to dummies and we are good to go...

# In[ ]:


X_train = pd.get_dummies(train.drop(['location','price','lat','lng','sqft_range','area_units','location_null'],axis=1),drop_first=True,dtype = float)
y_train = train['price']


# In[ ]:


X_validation = pd.get_dummies(validation.drop(['location','price','lat','lng','sqft_range','area_units','location_null'],axis=1),drop_first=True,dtype = float)
y_validation = validation['price']


# In[ ]:


X_test = pd.get_dummies(test.drop(['location','lat','lng','sqft_range','area_units','location_null'],axis=1),drop_first=True,dtype = float)


# In[ ]:


assert (X_train.columns==X_validation.columns).all()


# In[ ]:


xgb_model = XGBRegressor(objective='reg:squarederror',eval_metric='rmse',early_stopping_rounds=10,random_state=0)
_ = xgb_model.fit(X_train, y_train, eval_set=[(X_validation, y_validation)],verbose=True)


# In[ ]:


y_pred_validation_xgb = xgb_model.predict(X_validation)
mean_squared_error(y_validation,y_pred_validation_xgb)**0.5


# So the model does best with around 30 iterations. Let's use it now for the full data

# In[ ]:


X = pd.concat((X_train,X_validation)).sort_index()
y = pd.concat((y_train,y_validation)).sort_index()


# Now, do location_avg_price mapping for the whole data, and update it and test data:

# In[ ]:


final_location_avg_price_mapper = df.groupby('location')['price'].mean()
X['location_avg_price'] = final_location_avg_price_mapper.reindex(df['location'],fill_value = final_location_avg_price_mapper['NA']).to_numpy()
X_test['location_avg_price'] = final_location_avg_price_mapper.reindex(test['location'],fill_value = final_location_avg_price_mapper['NA']).to_numpy()


# In[ ]:


assert (X.columns==X_test.columns).all()


# In[ ]:


final_xgb_model = XGBRegressor(objective='reg:squarederror',eval_metric='rmse',n_estimators=xgb_model.best_iteration)
_ = final_xgb_model.fit(X,y)


# Now, predict for the test data

# In[ ]:


ans = pd.Series(final_xgb_model.predict(X_test).round(2),index = pd.RangeIndex(len(X_test),name = 'ID'),name = 'price')
ans.to_csv('submission_24.csv')


# # Random Forest Model

# In[ ]:


rf_model = RandomForestRegressor(random_state=0)
_ = rf_model.fit(X_train,y_train)


# In[ ]:


y_pred_validation_rf = rf_model.predict(X_validation)
mean_squared_error(y_validation,y_pred_validation_rf)**0.5


# In[ ]:


final_rf_model = rf_model.fit(X,y)


# In[ ]:


ans = pd.Series(final_rf_model.predict(X_test).round(2),index = pd.RangeIndex(len(X_test),name = 'ID'),name = 'price')
ans.to_csv('submission_25.csv')


# # Gradient Boosting Model

# In[ ]:


gb_model = GradientBoostingRegressor(random_state=0)
_ = gb_model.fit(X_train,y_train)


# In[ ]:


y_pred_validation_gb = gb_model.predict(X_validation)
mean_squared_error(y_validation,y_pred_validation_gb)**0.5


# In[ ]:


final_gb_model = gb_model.fit(X,y)


# In[ ]:


ans = pd.Series(final_gb_model.predict(X_test).round(2),index = pd.RangeIndex(len(X_test),name = 'ID'),name = 'price')
ans.to_csv('submission_26.csv')


# ### Checkpoint6 Save

# In[ ]:


tmp = {'X':X,'y':y,'X_train':X_train,'y_train':y_train,'X_validation':X_validation,'y_validation':y_validation,'X_test':X_test,
      'y_pred_validation_xgb':y_pred_validation_xgb,'y_pred_validation_rf':y_pred_validation_rf,'y_pred_validation_gb':y_pred_validation_gb}
pickle.dump(tmp,open('Checkpoint6.pickle','wb'))


# ### Checkpoint6 Load

# In[ ]:


tmp = pickle.load(open('Checkpoint6.pickle','rb'))
X = tmp['X']
y = tmp['y']
X_train = tmp['X_train']
y_train = tmp['y_train']
X_validation = tmp['X_validation']
y_validation = tmp['y_validation']
X_test = tmp['X_test']
y_pred_validation_xgb = tmp['y_pred_validation_xgb']
y_pred_validation_rf = tmp['y_pred_validation_rf']
y_pred_validation_gb = tmp['y_pred_validation_gb']
del tmp


# # Data Transformations

# In[ ]:


X_train.describe()


# In[ ]:


non_flag_columns = ['bath','balcony','avg_2bhk_rent','dist_from_city','availability_doy','num_bedrooms','sqft_avg','location_avg_price']


# ## avg_2bhk_rent

# In[ ]:


X_train['avg_2bhk_rent'].plot.hist(bins=20,title = 'avg_2bhk_rent');


# Let's normalize this:

# In[ ]:


mean,std = X_train['avg_2bhk_rent'].mean(),X_train['avg_2bhk_rent'].std()
X_train['avg_2bhk_rent'] = (X_train['avg_2bhk_rent'] - mean)/std
X_validation['avg_2bhk_rent'] = (X_validation['avg_2bhk_rent'] - mean)/std


# In[ ]:


X_train['avg_2bhk_rent'].plot.hist(bins=20,title = 'avg_2bhk_rent');


# Now apply to the whole:

# In[ ]:


mean,std = X['avg_2bhk_rent'].mean(),X['avg_2bhk_rent'].std()
X['avg_2bhk_rent'] = (X['avg_2bhk_rent'] - mean)/std
X_test['avg_2bhk_rent'] = (X_test['avg_2bhk_rent'] - mean)/std


# Now that this is in the standard normal distributon, we can clip the outliers to 3$\sigma$:

# In[ ]:


X_train['avg_2bhk_rent'] = np.clip(X_train['avg_2bhk_rent'],a_min = -3, a_max = 3)
X_validation['avg_2bhk_rent'] = np.clip(X_validation['avg_2bhk_rent'],a_min = -3, a_max = 3)
X['avg_2bhk_rent'] = np.clip(X['avg_2bhk_rent'],a_min = -3, a_max = 3)
X_test['avg_2bhk_rent'] = np.clip(X_test['avg_2bhk_rent'],a_min = -3, a_max = 3)


# In[ ]:


X_train['avg_2bhk_rent'].plot.hist(bins=20,title = 'avg_2bhk_rent');


# ## dist_from_city

# In[ ]:


X_train['dist_from_city'].plot.hist(bins=20, title = 'dist_from_city');


# There is a slight positive skew. Let's take square root and see:

# In[ ]:


X_train['dist_from_city'].apply('sqrt').plot.hist(bins=20, title = 'dist_from_city');


# The skew reduces. Now apply square root transformation:

# In[ ]:


X_train['dist_from_city'] = X_train['dist_from_city'].apply('sqrt')
X_validation['dist_from_city'] = X_validation['dist_from_city'].apply('sqrt')
X['dist_from_city'] = X['dist_from_city'].apply('sqrt')
X_test['dist_from_city'] = X_test['dist_from_city'].apply('sqrt')


# Clip outliers:

# In[ ]:


low,high = X_train['dist_from_city'].mean() - 3*X_train['dist_from_city'].std(),X_train['dist_from_city'].mean() + 3*X_train['dist_from_city'].std()
X_train['dist_from_city'] = np.clip(X_train['dist_from_city'],a_min = low,a_max = high)
X_validation['dist_from_city'] = np.clip(X_validation['dist_from_city'],a_min = low,a_max = high)


# In[ ]:


X_train['dist_from_city'].plot.hist(bins=20, title = 'dist_from_city');


# In[ ]:


low,high = X['dist_from_city'].mean() - 3*X['dist_from_city'].std(),X['dist_from_city'].mean() + 3*X['dist_from_city'].std()
X['dist_from_city'] = np.clip(X['dist_from_city'],a_min = low,a_max = high)
X_test['dist_from_city'] = np.clip(X_test['dist_from_city'],a_min = low,a_max = high)


# ## availability_doy

# In[ ]:


X_train['availability_doy'].replace(-1,np.nan).plot.hist(title = 'availability_doy');


# Do [0-1] scaling

# In[ ]:


X_train['availability_doy'] = (X_train['availability_doy'].replace(-1,np.nan)/365).fillna(-1)
X_validation['availability_doy'] = (X_validation['availability_doy'].replace(-1,np.nan)/365).fillna(-1)
X['availability_doy'] = (X['availability_doy'].replace(-1,np.nan)/365).fillna(-1)
X_test['availability_doy'] = (X_test['availability_doy'].replace(-1,np.nan)/365).fillna(-1)


# In[ ]:


X_train['availability_doy'].replace(-1,np.nan).plot.hist(title = 'availability_doy');


# ## sqft_avg

# In[ ]:


X_train['sqft_avg'].plot.hist(bins = 20);


# Do a log transformation:

# In[ ]:


X_train['sqft_avg'] = X_train['sqft_avg'].apply('log')
X_validation['sqft_avg'] = X_validation['sqft_avg'].apply('log')
X['sqft_avg'] = X['sqft_avg'].apply('log')
X_test['sqft_avg'] = X_test['sqft_avg'].apply('log')


# In[ ]:


X_train['sqft_avg'].plot.hist(bins = 30);


# This data looks much more 'gaussian'.
# 
# Clip outliers

# In[ ]:


low,high = X['sqft_avg'].mean() - 3*X['sqft_avg'].std(),X['sqft_avg'].mean() + 3*X['sqft_avg'].std()
X_train['sqft_avg'] = np.clip(X_train['sqft_avg'],a_min = low,a_max = high)
X_validation['sqft_avg'] = np.clip(X_validation['sqft_avg'],a_min = low,a_max = high)


# In[ ]:


X_train['sqft_avg'].plot.hist(bins = 20);


# In[ ]:


low,high = X['sqft_avg'].mean() - 3*X['sqft_avg'].std(),X['sqft_avg'].mean() + 3*X['sqft_avg'].std()
X['sqft_avg'] = np.clip(X['sqft_avg'],a_min = low,a_max = high)
X_test['sqft_avg'] = np.clip(X_test['sqft_avg'],a_min = low,a_max = high)


# ## location_price

# In[ ]:


X_train['location_avg_price'].plot.hist(title = 'location_avg_price');


# Log transformation:

# In[ ]:


X_train['location_avg_price'].apply('log').plot.hist(title = 'location_avg_price');


# Apply log transformation:

# In[ ]:


X_train['location_avg_price'] = X_train['location_avg_price'].apply('log')
X_validation['location_avg_price'] = X_validation['location_avg_price'].apply('log')
X['location_avg_price'] = X['location_avg_price'].apply('log')
X_test['location_avg_price'] = X_test['location_avg_price'].apply('log')


# Clip outliers

# In[ ]:


low,high = X_train['location_avg_price'].mean() - 3*X_train['location_avg_price'].std(),X_train['location_avg_price'].mean() + 3*X_train['location_avg_price'].std()
X_train['location_avg_price'] = np.clip(X_train['location_avg_price'],a_min = low,a_max = high)
X_validation['location_avg_price'] = np.clip(X_validation['location_avg_price'],a_min = low,a_max = high)


# In[ ]:


low,high = X['location_avg_price'].mean() - 3*X['location_avg_price'].std(),X['location_avg_price'].mean() + 3*X['location_avg_price'].std()
X['location_avg_price'] = np.clip(X['location_avg_price'],a_min = low,a_max = high)
X_test['location_avg_price'] = np.clip(X_test['location_avg_price'],a_min = low,a_max = high)


# ## bath

# In[ ]:


X_train['bath'].plot.hist(title = 'bath');


# Log (1+x) transformation:

# In[ ]:


X_train['bath'].apply('log1p').plot.hist(title = 'bath');


# In[ ]:


X_train['bath'] = X_train['bath'].apply('log1p')
X_validation['bath'] = X_validation['bath'].apply('log1p')
X['bath'] = X['bath'].apply('log1p')
X_test['bath'] = X_test['bath'].apply('log1p')


# Clip outliers

# In[ ]:


low,high = X_train['bath'].mean() - 3*X_train['bath'].std(),X_train['bath'].mean() + 3*X_train['bath'].std()
X_train['bath'] = np.clip(X_train['bath'],a_min = low,a_max = high)
X_validation['bath'] = np.clip(X_validation['bath'],a_min = low,a_max = high)


# In[ ]:


X_train['bath'].plot.hist(title = 'bath');


# In[ ]:


low,high = X['bath'].mean() - 3*X['bath'].std(),X['bath'].mean() + 3*X['bath'].std()
X['bath'] = np.clip(X['bath'],a_min = low,a_max = high)
X_test['bath'] = np.clip(X_test['bath'],a_min = low,a_max = high)


# ## balcony

# In[ ]:


X_train['balcony'].plot.hist(title = 'balcony');


# Keep as-is.

# ## num_bedrooms

# In[ ]:


X_train['num_bedrooms'].plot.hist(title = 'num_bedrooms');


# Log (1+x) transformation

# In[ ]:


X_train['num_bedrooms'].apply('log1p').plot.hist();


# In[ ]:


X_train['num_bedrooms'] = X_train['num_bedrooms'].apply('log1p')
X_validation['num_bedrooms'] = X_validation['num_bedrooms'].apply('log1p')
X['num_bedrooms'] = X['num_bedrooms'].apply('log1p')
X_test['num_bedrooms'] = X_test['num_bedrooms'].apply('log1p')


# Clip outliers

# In[ ]:


low,high = X_train['num_bedrooms'].mean() - 3*X_train['num_bedrooms'].std(),X_train['num_bedrooms'].mean() + 3*X_train['num_bedrooms'].std()
X_train['num_bedrooms'] = np.clip(X_train['num_bedrooms'],a_min = low,a_max = high)
X_validation['num_bedrooms'] = np.clip(X_validation['num_bedrooms'],a_min = low,a_max = high)


# In[ ]:


X_train['num_bedrooms'].plot.hist(title = 'num_bedrooms');


# In[ ]:


low,high = X['num_bedrooms'].mean() - 3*X['num_bedrooms'].std(),X['num_bedrooms'].mean() + 3*X['num_bedrooms'].std()
X['num_bedrooms'] = np.clip(X['num_bedrooms'],a_min = low,a_max = high)
X_test['num_bedrooms'] = np.clip(X_test['num_bedrooms'],a_min = low,a_max = high)


# In[ ]:


X.describe()


# In[ ]:


assert (y_train.index==X_train.index).all()
assert (y_validation.index==X_validation.index).all()
assert (y.index==X.index).all()


# ### Checkpoint7 Save

# In[ ]:


tmp = {'X':X,'y':y,'X_train':X_train,'y_train':y_train,'X_validation':X_validation,'y_validation':y_validation,'X_test':X_test,
      'y_pred_validation_xgb':y_pred_validation_xgb,'y_pred_validation_rf':y_pred_validation_rf,'y_pred_validation_gb':y_pred_validation_gb}
pickle.dump(tmp,open('Checkpoint7.pickle','wb'))


# ### Checkpoint7 Load

# In[5]:


tmp = pickle.load(open('Checkpoint7.pickle','rb'))
X = tmp['X']
y = tmp['y']
X_train = tmp['X_train']
y_train = tmp['y_train']
X_validation = tmp['X_validation']
y_validation = tmp['y_validation']
X_test = tmp['X_test']
y_pred_validation_xgb = tmp['y_pred_validation_xgb']
y_pred_validation_rf = tmp['y_pred_validation_rf']
y_pred_validation_gb = tmp['y_pred_validation_gb']
del tmp


# # OLS

# In[6]:


ols_model = sm.OLS(y_train.apply('log'),sm.add_constant(X_train)).fit()
ols_model.summary()


# In[7]:


y_pred_validation_ols = np.exp(ols_model.predict(sm.add_constant(X_validation)))
mean_squared_error(y_validation,y_pred_validation_ols)**0.5


# In[8]:


final_ols_model = sm.OLS(y.apply('log'),sm.add_constant(X)).fit()
final_ols_model.summary()


# In[ ]:


ans = pd.Series(np.exp(final_ols_model.predict(sm.add_constant(X_test))).round(2),index = pd.RangeIndex(len(X_test),name = 'ID'),name = 'price')
ans.to_csv('submission_28.csv')


# # Mixing the results

# In[ ]:


def mix_fun(r):
    mix_result = y_pred_validation_xgb*r[0] + y_pred_validation_rf*r[1] + y_pred_validation_gb*r[2] + y_pred_validation_ols*r[3]
    return mean_squared_error(y_validation,mix_result)**0.5


# In[ ]:


cons = {'type': 'eq', 'fun': lambda x:x.sum()-1}


# In[ ]:


result = minimize(mix_fun, x0 = np.array([0.25,0.25,0.25,0.25]), method='SLSQP', constraints=cons,bounds = Bounds(0,1))
ratios = result.x


# In[ ]:


xgb_best_result = pd.read_csv('submission_16_xgb_best.csv',index_col = 'ID').squeeze()
rf_best_result = pd.read_csv('submission_25_rf_best.csv',index_col = 'ID').squeeze()
gb_best_result = pd.read_csv('submission_26_gb_current_best.csv',index_col = 'ID').squeeze()
ols_best_result = pd.read_csv('submission_28_ols_best.csv',index_col = 'ID').squeeze()


# In[ ]:


mix_result = ratios[0]*xgb_best_result + ratios[1]*rf_best_result + ratios[2]*gb_best_result + ratios[3]*ols_best_result
mix_result.round(2).to_csv('submission_29.csv')


# In[ ]:




