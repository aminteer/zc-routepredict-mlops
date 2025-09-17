#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip freeze | grep scikit-learn')


# In[2]:


get_ipython().system('python -V')


# In[3]:


import pickle
import pandas as pd


# In[4]:


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[5]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[15]:


# parametized by year and month
year = 2023
month = 3


# In[ ]:


df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')


# In[10]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# In[13]:


print(y_pred)

st_dev = y_pred.std()
print(f'Standard Deviation of predictions: {st_dev}')


# In[16]:


df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


# In[17]:


output_file = f'results_{year:04d}-{month:02d}.parquet'

df['prediction'] = y_pred

df_result = df[['ride_id', 'prediction']]

df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)

