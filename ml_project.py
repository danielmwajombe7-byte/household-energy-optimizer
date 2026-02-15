#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib


# In[8]:


import pandas as pd

df = pd.read_csv(
    r"C:\Users\user\Desktop\ML_PRROJECT\tanzania_power_data.csv",
    sep=";",              # acha hii kama CSV yako inatumia ;
    engine="python"
)

# Unda Datetime
df["Datetime"] = pd.to_datetime(
    df["Date"].astype(str) + " " + df["Time"].astype(str),
    dayfirst=True,
    errors="coerce"
)

# Ondoa Date na Time
df.drop(columns=["Date", "Time"], inplace=True)

# Badilisha columns zote (isipokuwa Datetime) ziwe numeric
for col in df.columns:
    if col != "Datetime":
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Ondoa rows zenye NaN
df.dropna(inplace=True)

# Hakiki
df.info()
df.head()


# In[9]:


target = "Global_active_power"
X = df.drop(columns=[target, "Datetime"])
y = df[target]


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# In[11]:


lr = LinearRegression()
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2 = r2_score(y_test, lr_pred)


# In[12]:


dt = DecisionTreeRegressor(max_depth=10, random_state=42)
dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)
dt_rmse = np.sqrt(mean_squared_error(y_test, dt_pred))
dt_r2 = r2_score(y_test, dt_pred)


# In[13]:


results = pd.DataFrame({
    "Model": ["Linear Regression", "Decision Tree"],
    "RMSE": [lr_rmse, dt_rmse],
    "R2 Score": [lr_r2, dt_r2]
})

results


# In[14]:


results.plot(x="Model", y="RMSE", kind="bar", title="Model Comparison")
plt.show()


# In[15]:


best_model = dt if dt_rmse < lr_rmse else lr


# In[16]:


joblib.dump(best_model, "model.pkl")


# In[ ]:




