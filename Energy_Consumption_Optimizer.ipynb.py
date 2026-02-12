#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import zipfile

# Path ya zip file yako
zip_path = r"C:\Users\user\Desktop\ML_PRROJECT\individual+household+electric+power+consumption.zip"

# Fungua zip na tazama file ndani
with zipfile.ZipFile(zip_path) as z:
    print(z.namelist())  # hii itakuonyesha jina la file ndani, kawaida: 'household_power_consumption.txt'
    
    # Soma file moja kwa moja kutoka zip
    with z.open('household_power_consumption.txt') as f:
        df = pd.read_csv(f, sep=';', na_values='?')

# Unganisha Date + Time
df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)

# Convert numeric columns
numeric_cols = ['Global_active_power','Global_reactive_power','Voltage','Sub_metering_1','Sub_metering_2','Sub_metering_3']
df[numeric_cols] = df[numeric_cols].astype(float)

# Angalia data
print(df.head())


# In[6]:


# Jumla ya rows/columns
print(df.shape)

# Check missing values
print(df.isnull().sum())

# Basic statistics
print(df.describe())

# Plot energy usage kwa saa moja kwa mfano
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot(df['datetime'][:1000], df['Global_active_power'][:1000])
plt.xlabel("Datetime")
plt.ylabel("Global Active Power (kW)")
plt.title("Energy Usage Sample")
plt.show()


# In[8]:


# Old line (warning)
# df['rolling_avg_3h'] = df['Global_active_power'].rolling(window=3).mean().fillna(method='bfill')

# Fixed line
df['rolling_avg_3h'] = df['Global_active_power'].rolling(window=3).mean().bfill()


# In[9]:


# Features: hour, day_of_week, month, rolling_avg_3h
X = df[['hour','day_of_week','month','rolling_avg_3h']]
y = df['Global_active_power']


# In[10]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# In[12]:


# Remove rows where target or features have NaN
df_clean = df.dropna(subset=['Global_active_power', 'rolling_avg_3h', 'hour', 'day_of_week', 'month'])

X = df_clean[['hour','day_of_week','month','rolling_avg_3h']]
y = df_clean['Global_active_power']


# In[13]:


y = df['Global_active_power'].fillna(df['Global_active_power'].mean())


# In[14]:


print(X.isnull().sum())  # Should all be 0
print(y.isnull().sum())  # Should be 0


# In[16]:


# Step 1: Create rolling average
df['rolling_avg_3h'] = df['Global_active_power'].rolling(window=3).mean().bfill()

# Step 2: Drop any rows with NaN in features or target
df_clean = df.dropna(subset=['Global_active_power','rolling_avg_3h'])

# Step 3: Prepare X and y
X = df_clean[['hour','day_of_week','month','rolling_avg_3h']]
y = df_clean['Global_active_power']

# Confirm lengths
print("X length:", len(X))
print("y length:", len(y))


# In[17]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.4f}")


# In[ ]:


# ===============================
# Energy Consumption Optimizer
# ===============================

# Step 0: Import libraries
import pandas as pd
import numpy as np
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Step 1: Load dataset directly from zip
zip_path = r"C:\Users\user\Desktop\ML_PRROJECT\individual+household+electric+power+consumption.zip"

with zipfile.ZipFile(zip_path) as z:
    print("Files inside zip:", z.namelist())
    # Replace with actual filename inside zip
    file_name = z.namelist()[0]  
    with z.open(file_name) as f:
        df = pd.read_csv(f, sep=';', na_values='?')

# Step 2: Combine Date + Time into datetime
df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)

# Step 3: Convert numeric columns
numeric_cols = ['Global_active_power','Global_reactive_power','Voltage',
                'Sub_metering_1','Sub_metering_2','Sub_metering_3']
df[numeric_cols] = df[numeric_cols].astype(float)

# Step 4: Feature engineering
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['month'] = df['datetime'].dt.month

# Rolling average past 3 hours
df['rolling_avg_3h'] = df['Global_active_power'].rolling(window=3).mean().bfill()

# Step 5: Drop any rows with NaN in features or target
df_clean = df.dropna(subset=['Global_active_power','rolling_avg_3h','hour','day_of_week','month'])

# Step 6: Prepare X and y
X = df_clean[['hour','day_of_week','month','rolling_avg_3h']]
y = df_clean['Global_active_power']

# Confirm lengths
print("X length:", len(X))
print("y length:", len(y))

# Step 7: Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 8: Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 9: Predict
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.4f}")

# Step 10: Detect High Energy Usage
threshold = y_train.mean() + y_train.std()
high_usage_hours = X_test[y_pred > threshold]
print("⚡ High energy usage predicted at these times:")
print(high_usage_hours.head())

# Step 11: Generate Recommendations
def recommend_action(pred_value, threshold):
    if pred_value > threshold:
        return "⚡ High usage predicted! Turn off non-essential appliances or delay heavy usage."
    else:
        return "✅ Energy usage normal."

# Sample recommendations for last 10 predictions
for pred in y_pred[-10:]:
    print(recommend_action(pred, threshold))

# Step 12: Optional - Plot a sample
plt.figure(figsize=(12,6))
plt.plot(df_clean['datetime'][:1000], df_clean['Global_active_power'][:1000])
plt.xlabel("Datetime")
plt.ylabel("Global Active Power (kW)")
plt.title("Energy Usage Sample")
plt.show()


# In[ ]:


import pandas as pd
import numpy as np
import zipfile
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# In[ ]:


zip_path = r"C:\Users\user\Desktop\ML_PRROJECT\individual+household+electric+power+consumption.zip"

with zipfile.ZipFile(zip_path) as z:
    print("Files inside zip:", z.namelist())
    file_name = z.namelist()[0]  
    with z.open(file_name) as f:
        df = pd.read_csv(f, sep=';', na_values='?')


# In[ ]:


df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
numeric_cols = ['Global_active_power','Global_reactive_power','Voltage',
                'Sub_metering_1','Sub_metering_2','Sub_metering_3']
df[numeric_cols] = df[numeric_cols].astype(float)


# In[ ]:


df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['month'] = df['datetime'].dt.month
df['rolling_avg_3h'] = df['Global_active_power'].rolling(window=3).mean().bfill()
df_clean = df.dropna(subset=['Global_active_power','rolling_avg_3h','hour','day_of_week','month'])


# In[ ]:


X = df_clean[['hour','day_of_week','month','rolling_avg_3h']]
y = df_clean['Global_active_power']
print("X length:", len(X))
print("y length:", len(y))


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.4f}")


# In[ ]:


threshold = y_train.mean() + y_train.std()
high_usage_hours = X_test[y_pred > threshold]
print("⚡ High energy usage predicted at these times:")
print(high_usage_hours.head())

def recommend_action(pred_value, threshold):
    if pred_value > threshold:
        return "⚡ High usage predicted! Turn off non-essential appliances or delay heavy usage."
    else:
        return "✅ Energy usage normal."

for pred in y_pred[-10:]:
    print(recommend_action(pred, threshold))


# In[ ]:


plt.figure(figsize=(12,6))
plt.plot(df_clean['datetime'][:1000], df_clean['Global_active_power'][:1000])
plt.xlabel("Datetime")
plt.ylabel("Global Active Power (kW)")
plt.title("Energy Usage Sample")
plt.show()


# In[ ]:




