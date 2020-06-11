# MachineLearning
Machine Learning
In [1]:
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
In [ ]:
 
In [9]:
df = pd.read_csv('train.csv')
In [10]:
df.head()
Out[10]:
id	vendor_id	pickup_datetime	dropoff_datetime	passenger_count	pickup_longitude	pickup_latitude	dropoff_longitude	dropoff_latitude	store_and_fwd_flag	trip_duration
0	id2875421	2	2016-03-14 17:24:55	2016-03-14 17:32:30	1	-73.982155	40.767937	-73.964630	40.765602	N	455
1	id2377394	1	2016-06-12 00:43:35	2016-06-12 00:54:38	1	-73.980415	40.738564	-73.999481	40.731152	N	663
2	id3858529	2	2016-01-19 11:35:24	2016-01-19 12:10:48	1	-73.979027	40.763939	-74.005333	40.710087	N	2124
3	id3504673	2	2016-04-06 19:32:31	2016-04-06 19:39:40	1	-74.010040	40.719971	-74.012268	40.706718	N	429
4	id2181028	2	2016-03-26 13:30:55	2016-03-26 13:38:10	1	-73.973053	40.793209	-73.972923	40.782520	N	435
In [11]:
df.describe()
Out[11]:
vendor_id	passenger_count	pickup_longitude	pickup_latitude	dropoff_longitude	dropoff_latitude	trip_duration
count	1.458644e+06	1.458644e+06	1.458644e+06	1.458644e+06	1.458644e+06	1.458644e+06	1.458644e+06
mean	1.534950e+00	1.664530e+00	-7.397349e+01	4.075092e+01	-7.397342e+01	4.075180e+01	9.594923e+02
std	4.987772e-01	1.314242e+00	7.090186e-02	3.288119e-02	7.064327e-02	3.589056e-02	5.237432e+03
min	1.000000e+00	0.000000e+00	-1.219333e+02	3.435970e+01	-1.219333e+02	3.218114e+01	1.000000e+00
25%	1.000000e+00	1.000000e+00	-7.399187e+01	4.073735e+01	-7.399133e+01	4.073588e+01	3.970000e+02
50%	2.000000e+00	1.000000e+00	-7.398174e+01	4.075410e+01	-7.397975e+01	4.075452e+01	6.620000e+02
75%	2.000000e+00	2.000000e+00	-7.396733e+01	4.076836e+01	-7.396301e+01	4.076981e+01	1.075000e+03
max	2.000000e+00	9.000000e+00	-6.133553e+01	5.188108e+01	-6.133553e+01	4.392103e+01	3.526282e+06
In [12]:
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1458644 entries, 0 to 1458643
Data columns (total 11 columns):
 #   Column              Non-Null Count    Dtype  
---  ------              --------------    -----  
 0   id                  1458644 non-null  object 
 1   vendor_id           1458644 non-null  int64  
 2   pickup_datetime     1458644 non-null  object 
 3   dropoff_datetime    1458644 non-null  object 
 4   passenger_count     1458644 non-null  int64  
 5   pickup_longitude    1458644 non-null  float64
 6   pickup_latitude     1458644 non-null  float64
 7   dropoff_longitude   1458644 non-null  float64
 8   dropoff_latitude    1458644 non-null  float64
 9   store_and_fwd_flag  1458644 non-null  object 
 10  trip_duration       1458644 non-null  int64  
dtypes: float64(4), int64(3), object(4)
memory usage: 122.4+ MB
In [13]:
print('id is of type:\t',type(df['id'][0]))
print('vendor_id is of type:\t',type(df['vendor_id'][0]))
print('pickup_datetime is of type:\t',type(df['pickup_datetime'][0]))
print('dropoff_datetime is of type:\t',type(df['dropoff_datetime'][0]))
print('passenger_count is of type:\t',type(df['passenger_count'][0]))
print('pickup_longitude is of type:\t',type(df['pickup_longitude'][0]))
print('pickup_latitude is of type:\t',type(df['pickup_latitude'][0]))
print('dropoff_longitude is of type:\t',type(df['dropoff_longitude'][0]))
print('dropoff_latitude is of type:\t',type(df['dropoff_latitude'][0]))
print('store_and_fwd_flag is of type:\t',type(df['store_and_fwd_flag'][0]))
print('trip_duration is of type:\t',type(df['trip_duration'][0]))
id is of type:	 <class 'str'>
vendor_id is of type:	 <class 'numpy.int64'>
pickup_datetime is of type:	 <class 'str'>
dropoff_datetime is of type:	 <class 'str'>
passenger_count is of type:	 <class 'numpy.int64'>
pickup_longitude is of type:	 <class 'numpy.float64'>
pickup_latitude is of type:	 <class 'numpy.float64'>
dropoff_longitude is of type:	 <class 'numpy.float64'>
dropoff_latitude is of type:	 <class 'numpy.float64'>
store_and_fwd_flag is of type:	 <class 'str'>
trip_duration is of type:	 <class 'numpy.int64'>
In [14]:
print('Mean Pickup Latitude: ', df['pickup_latitude'].mean())
print('Mean Dropoff Latitude: ',df['dropoff_latitude'].mean())
Mean Pickup Latitude:  40.750920908391734
Mean Dropoff Latitude:  40.7517995149002
In [15]:
print('Mean Pickup Longitude: ', df['pickup_longitude'].mean())
print('Mean Dropoff Longitude: ',df['dropoff_longitude'].mean())
Mean Pickup Longitude:  -73.97348630489282
Mean Dropoff Longitude:  -73.9734159469458
In [16]:
print('CORR(Vendor ID, Pickup Latitude): ',df['vendor_id'].corr(df['pickup_latitude']))
print('CORR(Vendor ID, Dropoff Latitude): ',df['vendor_id'].corr(df['dropoff_latitude']))
CORR(Vendor ID, Pickup Latitude):  0.001741587726983292
CORR(Vendor ID, Dropoff Latitude):  0.004496034679383403
In [17]:
print('CORR(Vendor ID, Pickup Longitude): ',df['vendor_id'].corr(df['pickup_longitude']))
print('CORR(Vendor ID, Dropoff Longitude): ',df['vendor_id'].corr(df['dropoff_longitude']))
CORR(Vendor ID, Pickup Longitude):  0.007820251202659058
CORR(Vendor ID, Dropoff Longitude):  0.0015284524154382624
In [18]:
fig, ax = plt.subplots(2,2,figsize=(20, 10), sharex=False, sharey = False)
sns.despine(left=True)
sns.distplot(df['pickup_latitude'].values, label = 'pickup_latitude',color="g",bins = 100, ax=ax[0,0])
sns.distplot(df['dropoff_latitude'].values, label = 'dropoff_latitude',color="r",bins = 100, ax=ax[0,1])
sns.distplot(df['pickup_longitude'].values, label = 'pickup_longitude',color="g",bins = 100, ax=ax[1,0])
sns.distplot(df['dropoff_longitude'].values, label = 'dropoff_longitude',color="r",bins = 100, ax=ax[1,1])
Out[18]:
<matplotlib.axes._subplots.AxesSubplot at 0x23f4716bf08>

In [19]:
df_coord = df.loc[(df.pickup_latitude > 40.6) & (df.pickup_latitude < 40.9)]
df_coord = df_coord.loc[(df.dropoff_latitude > 40.6) & (df.dropoff_latitude < 40.9)]
df_coord = df_coord.loc[(df.pickup_longitude > -74.05) & (df.pickup_longitude < -73.7)]
df_coord = df_coord.loc[(df.dropoff_longitude > -74.05) & (df.dropoff_longitude < -73.7)]
In [20]:
fig, ax = plt.subplots(2,2,figsize=(20, 10), sharex=False, sharey = False)
sns.despine(left=True)
sns.distplot(df_coord['pickup_latitude'].values, label = 'pickup_latitude',color="g",bins = 100, ax=ax[0,0])
sns.distplot(df_coord['dropoff_latitude'].values, label = 'dropoff_latitude',color="r",bins = 100, ax=ax[0,1])
sns.distplot(df_coord['pickup_longitude'].values, label = 'pickup_longitude',color="g",bins = 100, ax=ax[1,0])
sns.distplot(df_coord['dropoff_longitude'].values, label = 'dropoff_longitude',color="r",bins = 100, ax=ax[1,1])
plt.setp(ax, yticks=[])
plt.tight_layout()

In [21]:
plt.figure(figsize=(20,10))
Out[21]:
<Figure size 1440x720 with 0 Axes>
<Figure size 1440x720 with 0 Axes>
In [22]:
plt.figure(figsize=(20,10))

longitudes = list(df_coord.pickup_longitude) + list(df_coord.dropoff_longitude)
latitudes = list(df_coord.pickup_latitude) + list(df_coord.dropoff_latitude)

plt.plot(longitudes, latitudes, '.', alpha=0.6, markersize=0.05, color='purple')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('City Map of NYC Trips using Latitudes and Longitudes')
Out[22]:
Text(0.5, 1.0, 'City Map of NYC Trips using Latitudes and Longitudes')

In [23]:
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharey=True, figsize=(20,30))

df_coord.plot(kind='scatter', x='pickup_longitude', y='pickup_latitude', color='green', s=0.02, alpha=0.6, subplots=True, ax=ax1)
df_coord.plot(kind='scatter', x='dropoff_longitude', y='dropoff_latitude', color='blue', s=0.02, alpha=0.6, subplots=True, ax=ax2)
ax1.set_title('Taxi Trip Pickups (Long,Lat)')
ax2.set_title('Taxi Trip Dropoffs (Long,Lat)')
Out[23]:
Text(0.5, 1.0, 'Taxi Trip Dropoffs (Long,Lat)')

In [1]:
from IPython.display import Image
Image('nyc_img.png')
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
~\anaconda3\lib\site-packages\IPython\core\display.py in _data_and_metadata(self, always_both)
   1271         try:
-> 1272             b64_data = b2a_base64(self.data).decode('ascii')
   1273         except TypeError:

TypeError: a bytes-like object is required, not 'str'

During handling of the above exception, another exception occurred:

FileNotFoundError                         Traceback (most recent call last)
~\anaconda3\lib\site-packages\IPython\core\formatters.py in __call__(self, obj, include, exclude)
    968 
    969             if method is not None:
--> 970                 return method(include=include, exclude=exclude)
    971             return None
    972         else:

~\anaconda3\lib\site-packages\IPython\core\display.py in _repr_mimebundle_(self, include, exclude)
   1260         if self.embed:
   1261             mimetype = self._mimetype
-> 1262             data, metadata = self._data_and_metadata(always_both=True)
   1263             if metadata:
   1264                 metadata = {mimetype: metadata}

~\anaconda3\lib\site-packages\IPython\core\display.py in _data_and_metadata(self, always_both)
   1273         except TypeError:
   1274             raise FileNotFoundError(
-> 1275                 "No such file or directory: '%s'" % (self.data))
   1276         md = {}
   1277         if self.metadata:

FileNotFoundError: No such file or directory: 'nyc_img.png'
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
~\anaconda3\lib\site-packages\IPython\core\display.py in _data_and_metadata(self, always_both)
   1271         try:
-> 1272             b64_data = b2a_base64(self.data).decode('ascii')
   1273         except TypeError:

TypeError: a bytes-like object is required, not 'str'

During handling of the above exception, another exception occurred:

FileNotFoundError                         Traceback (most recent call last)
~\anaconda3\lib\site-packages\IPython\core\formatters.py in __call__(self, obj)
    343             method = get_real_method(obj, self.print_method)
    344             if method is not None:
--> 345                 return method()
    346             return None
    347         else:

~\anaconda3\lib\site-packages\IPython\core\display.py in _repr_png_(self)
   1290     def _repr_png_(self):
   1291         if self.embed and self.format == self._FMT_PNG:
-> 1292             return self._data_and_metadata()
   1293 
   1294     def _repr_jpeg_(self):

~\anaconda3\lib\site-packages\IPython\core\display.py in _data_and_metadata(self, always_both)
   1273         except TypeError:
   1274             raise FileNotFoundError(
-> 1275                 "No such file or directory: '%s'" % (self.data))
   1276         md = {}
   1277         if self.metadata:

FileNotFoundError: No such file or directory: 'nyc_img.png'
Out[1]:
<IPython.core.display.Image object>
In [2]:
data_string = os.environ['INTRINIO_USER'] + ":" + os.environ['INTRINIO_PASSWORD']

data_bytes = data_string.encode("utf-8")

base64.b64encode(data_bytes)
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-2-68ceb62ee89f> in <module>
----> 1 data_string = os.environ['INTRINIO_USER'] + ":" + os.environ['INTRINIO_PASSWORD']
      2 
      3 data_bytes = data_string.encode("utf-8")
      4 
      5 base64.b64encode(data_bytes)

NameError: name 'os' is not defined
In [3]:
email['html'] = base64.b64encode(email.get('html'))
change to:
email['html'] = base64.b64encode(email.get('html').encode('utf-8')).decode('utf-8')
  File "<ipython-input-3-df3df950584e>", line 2
    change to:
            ^
SyntaxError: invalid syntax
In [ ]:
 
