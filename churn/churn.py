import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df1 = pd.read_excel("/content/drive/MyDrive/Telco_customer_churn.xlsx")

df1.describe()
df1.info()
df1.columns
#data preprocessing
# drop the unwanted columns
df = df1.drop(['Customer ID', 'Count', 'Quarter', 'Referred a Friend','Offer', 'Phone Service', 'Multiple Lines','Internet Service', 'Internet Type',
       'Online Security', 'Online Backup', 'Device Protection Plan','Premium Tech Support', 'Streaming TV', 'Streaming Movies',
       'Streaming Music', 'Unlimited Data', 'Contract', 'Paperless Billing','Payment Method'], axis=1)


# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
Final = norm_func(df)
Final.describe()

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i,)
    kmeans.fit(Final)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-')
plt.xlabel("No_of_Clusters")
plt.ylabel("total_within_SS")
plt.show()
# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 6)
model.fit(Final)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
df1['clust'] = mb # creating a  new column and assigning it to new column 

#df1.head()
#df2.head()

#Univ = Univ.iloc[:,[7,0,1,2,3,4,5,6]]
#Univ.head()

df1.iloc[:, 2:8].groupby(df1.clust).mean()

df1.to_csv("churn.csv", encoding = "utf-8")

import os
os.getcwd()
