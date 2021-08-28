import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
df=pd.ExcelFile('/content/drive/MyDrive/EastWestAirlines.xlsx')
df1=pd.read_excel(df,'data')
df2=df1.drop(['ID#','Award?'],axis=1) # This Data is Not important. Because these columns are not putting any effect on our dataset.
#df2.info() # there is no null value

#  Normailzation to scale values between 0 to 1.
norm=MinMaxScaler()
Final=pd.DataFrame(norm.fit_transform(df2),columns=df2.columns)
#  standardscaler to scale values.

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
model = KMeans(n_clusters = 5)
model.fit(Final)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
df1['clust'] = mb # creating a  new column and assigning it to new column 

#df1.head()
#df2.head()

#Univ = Univ.iloc[:,[7,0,1,2,3,4,5,6]]
#Univ.head()

df1.iloc[:, 2:8].groupby(df1.clust).mean()

df1.to_csv("airline.csv", encoding = "utf-8")

import os
os.getcwd()
