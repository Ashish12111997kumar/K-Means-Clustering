import pandas as pd
import numpy as np
df=pd.read_csv('/content/drive/MyDrive/Insurance Dataset.csv')
df1=df.copy()
df1=df1.drop(['Age','Days to Renew'],axis=1)
from sklearn.preprocessing import MinMaxScaler
MS=MinMaxScaler()
data=pd.DataFrame(MS.fit_transform(df1),columns=df1.columns)

from sklearn.cluster import KMeans
twss=[]
k=list(range(2,9))
for i in (k):
  km=KMeans(n_clusters=i,init='k-means++',random_state=10)
  km1=km.fit(data)
  twss.append(km.inertia_)
plt.plot(k,twss,'ro-')
plt.show()

km=KMeans(n_clusters=5,init='k-means++',random_state=10)
pred=km.fit_predict(data)
#kmeans.labels_  
from sklearn.metrics import silhouette_score
sc=silhouette_score(data,pred)
print(sc)
df['clust']=pd.Series(km.labels_)  
df.groupby(df.clust).mean()
