# Customer Personality Analysis

This analysis takes a look at a large dataset, containing the shopping behaviour and personal information of customers. The goal of the analysis is to find patterns between different types of customers, either by age, income, etc. and general habits of customers included in the dataset.

The first part of the project is an exploratory data analysis, followed by a clustering ML-model to further identify customer groups.

## The Data

The source of the data used for the analysis is from [Kaggle.](https://www.kaggle.com/imakash3011/customer-personality-analysis)

## The Analysis

A compact overview of the [notebook.](https://github.com/zangerls/Customer-Personality-Analysis/blob/main/main.ipynb)

### Libraries

```python
# Standard
import sys
import math

# Processing & Cleaning
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
#import rpy2.robjects.lib.ggplot2 as gp

# Machine Learning
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Elbow method
from scipy.spatial.distance import cdist
```

### Cleaning

```python
# Mean Imputation for null values
df.isna().sum().sort_values(ascending=False)
mean_income = round(df['Income'].mean(),2)
df['Income'] = df['Income'].fillna(mean_income)
```

```python
# Formatting data types
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])
```

```python
# dropping ambiguous values
df.nunique().sort_values(ascending=True)
df.drop(['Z_Revenue','Z_CostContact'], axis=1, inplace=True)
df['Age'] = (2022 - df['Year_Birth'])
df.drop('Year_Birth', axis=1, inplace=True)
```

```python
# removing outliers
df.drop(df[df.Age > 100].index, inplace=True)
df.drop(df[df.Income > 500_000].index, inplace=True)
```

```python
# cleaning columns
df['Education'] = df['Education'].replace('Basic', 'Undergraduate')
df['Education'] = df['Education'].replace(['Graduation','PhD','Master','2n Cycle'], 'Postgraduate')

df['Marital_Status'] = df['Marital_Status'].replace(['Together','Married'],'Relationship')
df['Marital_Status'] = df['Marital_Status'].replace(['Single','Divorced','Widow','Alone','Absurd','YOLO'], 'Single')

df['Children'] = df['Kidhome'] + df['Teenhome']
df.drop(['Kidhome','Teenhome'], axis=1, inplace=True)

df['MntTotal'] = df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']
df['PurchasesTotal'] = df['NumDealsPurchases'] + df['NumWebPurchases'] + df['NumCatalogPurchases'] + df['NumStorePurchases']
df['Accepted_Any_Cmp'] = (df['AcceptedCmp1'] == 1) | (df['AcceptedCmp2'] == 1) | (df['AcceptedCmp3'] == 1) | (df['AcceptedCmp4'] == 1) | (df['AcceptedCmp5'] == 1)
df['Accepted_Any_Cmp'] = df['Accepted_Any_Cmp'].replace([True, False],[1,0])
```

### EDA

See in [notebook.](https://github.com/zangerls/Customer-Personality-Analysis/blob/main/main.ipynb)

## Clustering (ML)

```python
# drop unnecessary columns
df.drop(['ID','Last_Purchase_Ago','Deals_Purchases','Web_Purchases','Catalog_Purchases','Store_Purchases',
'Accepted_Cmp_1','Accepted_Cmp_2','Accepted_Cmp_3','Accepted_Cmp_4','Accepted_Cmp_5','Did_Complain','Accepted_Any_Cmp'], 1, inplace=True)
```

```python
# Label Encoding non-numerical data
encoder.fit(df['Education'])
df['Education'] = encoder.transform(df['Education'])
df['Education'].head()

encoder.fit(df['Marital_Status'])
df['Marital_Status'] = encoder.transform(df['Marital_Status'])
df['Marital_Status'].head()

encoder.fit(df['Enrollment_Date'])
df['Enrollment_Date'] = encoder.transform(df['Enrollment_Date'])
df['Enrollment_Date'].head()
```

```python
# Scaling the data
df_scaled = df.copy()

scaler.fit(df_scaled)
df_scaled = pd.DataFrame(scaler.transform(df_scaled), columns=df_scaled.columns)
df_scaled
```

```python
# Dimensionality Reduction
pca = PCA(n_components=3)
pca.fit(df_scaled)

df_pca = pd.DataFrame(pca.transform(df_scaled), columns=(['x','y','z']))
df_pca.describe()
```

```python
# Elbow Method for optimal clusters
distortions = []

for k in range(1,10):
    km = KMeans(n_clusters=k).fit(df)
    km.fit(df)
    distortions.append(sum(np.min(cdist(df, km.cluster_centers_, 'euclidean'), axis=1)) / df.shape[0])
plt.plot(range(1,10), distortions, 'bx-')
plt.show()
```

```python
# KMeans Clustering
km = KMeans(
    n_clusters=4, init='random', n_init=50, random_state=44
)

y_pred = km.fit_predict(df_pca)
y_pred

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_pca['x'],df_pca['y'],df_pca['z'], c=y_pred)
plt.show()
```
