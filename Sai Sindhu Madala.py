#!/usr/bin/env python
# coding: utf-8

# ## Labor force

# In[1]:


# https://data.worldbank.org/indicator/SL.TLF.TOTL.IN


# In[1]:


import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def worldbank_data(path):
    """
    Function to import World Bank data

    Returns:
    - df_country: Original dataframe.
    - df_years: Transposed dataframe.
    """    
    df_country = pd.read_csv(path, skiprows=4)
    df_years = pd.read_csv(path, skiprows=4).set_index(['Country Name']).T
    
    return df_country, df_years

df_country, df_years = worldbank_data('API_SL.TLF.TOTL.IN_DS2_en_csv_v2_6299589.csv')
growth = df_country[['Country Name', 'Indicator Name'] + list(map(str, range(2000, 2021)))].dropna()


# In[3]:


growth.head()


# In[4]:


subset = growth[["Country Name", "2000"]].copy()
subset.head()


# In[5]:


subset["Growth (%)"] = 100.0 * (growth["2020"] - growth["2000"]) / (growth["2000"])
subset = subset.dropna()
subset.describe()


# In[6]:


plt.figure(figsize=(7, 5))
scatter_plot = plt.scatter(subset["2000"], subset["Growth (%)"], 5, label="Labour Force")
plt.xlabel("Labour Force in 2000")
plt.ylabel("Labour Force Growth from 1960 to 2022 (%)")
plt.title("Labour Force in 2000 vs. Labour Force Growth (%)")
plt.legend()
plt.show()


# In[7]:


import sklearn.preprocessing as pp

def scale_and_transform(data, features=["2000", "Growth (%)"], scaler_type=pp.RobustScaler()):
    """
    Scale and transform the specified features using a given scaler.

    Returns:
    - DataFrame: A DataFrame with the specified features scaled and transformed.
    """
    x = data[features].copy()
    scaler = scaler_type
    scaler.fit(x)
    x_norm = scaler.transform(x)

    return x_norm

x_norm = scale_and_transform(subset)

plt.figure(figsize=(7, 5))
plt.scatter(x_norm[:, 0], x_norm[:, 1], 5, label="Labour Force")
plt.xlabel("Normalized Labour Force in 2000")
plt.ylabel("Normalized Labour Force Growth from 1960 to 2022 (%)")
plt.title("Normalized Data")
plt.show()


# In[25]:


def silhouette_score(xy, n):
    """Calculates silhouette score for n clusters"""
    kmeans = KMeans(n_clusters=n, n_init=20)
    kmeans.fit(xy)
    labels = kmeans.labels_
    score = skmet.silhouette_score(xy, labels)
    return score


# In[29]:


from sklearn.cluster import KMeans
import sklearn.metrics as skmet

for i in range(2, 10):
    score = silhouette_score(x_norm, i)
    print(f"The silhouette score for {i: 3d} is {score: 7.4f}")


# In[33]:


kmeans = KMeans(n_clusters=3, n_init=20)
kmeans.fit(x_norm)
labels = kmeans.labels_
cen = kmeans.cluster_centers_
cen = scaler.inverse_transform(cen)
xkmeans = cen[:, 0]
ykmeans = cen[:, 1]


# In[34]:


plt.figure(figsize=(7, 5))
plt.scatter(subset["2000"], subset["Growth (%)"], c=labels, marker="o", s=10)
plt.scatter(xkmeans, ykmeans, marker="^", c="black", s=50, label="Cluster Centroids")
plt.xlabel("Labour Force in 2000")
plt.ylabel("Labour Force Growth from 1960 to 2022 (%)")
plt.title("Labour Force in 2000 vs. Labour Force Growth (%)")
plt.legend()
plt.show()


# In[36]:


ind_df = df_years.loc['2000':'2020', ['India']].reset_index().rename(columns={'index': 'Year', 'India': 'Labour Force'})
ind_df = ind_df.apply(pd.to_numeric, errors='coerce')
ind_df.describe()


# In[37]:


plt.figure(figsize=(7, 5))
sns.lineplot(data=ind_df, x='Year', y='Labour Force')
plt.xlabel('Year')
plt.ylabel('Labour Force')
plt.title('Labour Force in India (2000-2020)')
plt.show()


# In[38]:


def poly(x, a, b, c, d, e):
    """ Calulates polynominal"""
    x = x - 2001
    f = a + b*x + c*x**2 + d*x**3 + e*x**4
    return f


# In[45]:


import scipy.optimize as opt
import errors

param, covar = opt.curve_fit(poly, ind_df["Year"], ind_df["Labour Force"])
sigma = np.sqrt(np.diag(covar))
year = np.arange(2000, 2031)
forecast = poly(year, *param)
sigma = errors.error_prop(year, poly, param, covar)
low = forecast - sigma
up = forecast + sigma
ind_df["fit"] = poly(ind_df["Year"], *param)

plt.figure(figsize=(7, 5))
sns.lineplot(data=ind_df, x="Year", y="Labour Force", label="Labour Force")
sns.lineplot(data=ind_df, x="Year", y="fit", label="Poly Fit")
plt.xlabel("Year")
plt.ylabel('Labour Force')
plt.title('Labour Force in India between 1960-2022')
plt.legend()
plt.show()


# In[47]:


plt.figure(figsize=(7, 5))
plt.plot(ind_df["Year"], ind_df["Labour Force"], label="Labour Force")
plt.plot(year, forecast, label="forecast")
plt.fill_between(year, low, up, color="yellow", alpha=0.7)
plt.title("Labour Force in India - Predictions")
plt.xlabel("Year")
plt.ylabel("Labour Force")
plt.legend()
plt.show()


# In[ ]:




