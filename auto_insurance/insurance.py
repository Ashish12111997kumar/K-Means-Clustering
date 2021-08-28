import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df1 = pd.read_csv("/content/drive/MyDrive/AutoInsurance.csv")
df1.head()
df1.info()
# removing unwanted columns
df=df1.copy()
df2=df.drop(['Marital Status','Customer','State','Education','Renew Offer Type','Policy','Policy Type'],axis=1)
df_cat=[col for col in df2.columns if df2[col].dtype=="O"]
df_cat=df2[df_cat].drop(['Effective To Date'],axis=1)
# encoding
df_cat=pd.get_dummies(df_cat,drop_first=True)
num=df._get_numeric_data()
#scaling using Normalization
from sklearn.preprocessing import MinMaxScaler
MS=MinMaxScaler()
num=pd.DataFrame(MS.fit_transform(num),columns=num.columns)

Final=pd.concat([df_cat,num],axis=1)

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(Final)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-')
plt.xlabel("No_of_Clusters")
plt.ylabel("total_within_SS")
plt.show()
# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 4)
model.fit(Final)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
df1['clust'] = mb # creating a  new column and assigning it to new column 

#df1.head()
#df2.head()

#Univ = Univ.iloc[:,[7,0,1,2,3,4,5,6]]
#Univ.head()

df1.iloc[:, 2:8].groupby(df1.clust).mean()

df1.to_csv("auto_insurance.csv", encoding = "utf-8")

import os
os.getcwd()
