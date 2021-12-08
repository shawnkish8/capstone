#!/usr/bin/env python
# coding: utf-8

# ### Imports for the whole project

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import qgrid
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# ### Data Import
# 
# - Once data is loaded in, convert the timestamp object column into 3 individual columns to be used by the model
#   and for searching in the interactive query.
# - Once the data has been loaded, the X/y sets are created for our model.
# - After the sets are created, we run the Random Forest Regressor model on the training data.
# - Finally, run the model on the test data and confirm accuracy score.

# In[2]:


data_set = pd.read_csv("bike_data.csv")

data_set.timestamp = pd.to_datetime(data_set.timestamp)

data_set['Month'] = data_set.timestamp.dt.month
data_set['Day'] = data_set.timestamp.dt.day
data_set['Hour'] = data_set.timestamp.dt.hour


# In[3]:


np.random.seed(42)

X = data_set.drop(["timestamp","cnt"], axis = 1)
y = data_set["cnt"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
percentage = "{:.0%}".format(score)
print("The ensemble model is " + percentage + " accurate.")


# ### This is a correlation chart that is comparing all the various data points within the data set.

# In[4]:


plt.figure(figsize=(10,5), dpi=150)
sns.heatmap(data_set.corr(),annot=True)


# ## This is the interactive query piece of the project

# In[5]:


qgrid_widget = qgrid.show_grid(data_set, show_toolbar=True)
qgrid_widget


# In[6]:


prediction = model.predict(X_test)

mock_table = X_test


# Below is the chart after the test data is used and the predicted counts of rentals.

# In[7]:


mock_table["Prediction Count"] = prediction.astype(int)
mock_table


# In[8]:


plt.hist(data_set)
plt.show


# In[9]:


get_ipython().run_line_magic('load_ext', 'watermark')


# In[ ]:




