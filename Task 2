#!/usr/bin/env python
# coding: utf-8

%config InlineBackend.figure_format = 'retina'
%matplotlib inline    
import seaborn as sns 
sns.set(font= "Kaiti",style="ticks",font_scale=1.4)
import matplotlib
matplotlib.rcParams['axes.unicode_minus']=False # Solve the display problem of the negative sign of the coordinate axis
## Import the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno 
import plotly.express as px
from sklearn.cluster import *
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.manifold import TSNE

# ## Task 2
# Under the condition that the pollutant discharge situation remains unchanged, when the meteorological conditions in a certain area are conducive to 
# the diffusion or settlement of pollutants, the AQI of the area will decrease, and vice versa. Use the data in Annex 1 to reasonably classify meteorological 
# conditions according to the degree of influence on the concentration of pollutants, and describe the characteristics of various meteorological conditions.

## Use the clustering algorithm to perform cluster analysis on the data. The data used for clustering can use the AQI data calculated by each pollutant,
## The data can be regarded as data in the same dimension, and there is no need to standardize the data
## (Although even if the data is in the same dimension, there are still differences in the data value range between variables, whether to continue 
## standardization does not affect the results of some clusters)
## Therefore, I compare the clustering differences before and after standardization
## It is also feasible to use the original monitoring data. When using the original monitoring data, pay attention to the standardization of the data.
clustdf = IAQIpdf.iloc[:,0:6]
clustdf.head()

## At first, Try K-means clustering
## Use the elbow method to search for the appropriate number of clusters
kmax = 10
K = np.arange(1,kmax)
iner = [] ## Sum of squared errors within a class
for ii in K:
    kmean = KMeans(n_clusters=ii,random_state=1)
    kmean.fit(clustdf)
    ## Calculate the sum of squared errors within the class
    iner.append(kmean.inertia_) 

## Visualize changes in the sum of squares of errors within a class
plt.figure(figsize=(10,6))
plt.plot(K,iner,"r-o")
plt.xlabel("Number of clusters")
plt.ylabel("Sum of squared errors within a class")
plt.title("K-means clustering")
plt.grid()
plt.show()

## Standardized preprocessing of feature data
clustdfs = StandardScaler().fit_transform(np.array(clustdf))
clustdfs.shape

## K-means clustering is not a good way to judge clustering into several clusters
## Using TSNE for data will later visualize the distribution of data

## Use TSNE algorithm to reduce dimensionality and visualize it
## TSNE performs dimensionality reduction of the data, reducing the dimensionality to a 3-dimensional space
tsne = TSNE(n_components = 3,perplexity = 100,
            early_exaggeration = 2,random_state=123) 

## Get the data after dimensionality reduction
tsne_usedata_x = tsne.fit_transform(clustdf)
print(tsne_usedata_x.shape)
## Visualize the spatial distribution of popular data after dimensionality reduction in 3D space
fig = plt.figure(figsize=(15,10))
## Set the coordinate system to 3D
ax1 = fig.add_subplot(111, projection="3d")
ax1.scatter(tsne_usedata_x[:,0],tsne_usedata_x[:,1],tsne_usedata_x[:,2],s = 40)
ax1.set_xlabel("TSNE1",rotation=-20)
ax1.set_ylabel("TSNE2",rotation=45)
ax1.set_zlabel("TSNE3",rotation=90)
ax1.set_title("TSNE feature space visualization")
plt.show()

## Try to use K-means clustering to divide it into 3 clusters to see the effect
kmean = KMeans(n_clusters=3,random_state=1)
k_pre = kmean.fit_predict(clustdf)
print("Number of samples contained in each cluster:",np.unique(k_pre,return_counts = True))
print("The cluster center of each cluster is:\n",kmean.cluster_centers_)

## Use contour coefficient to evaluate clustering effect
## Calculate the overall average profile coefficient, K-means
sil_score = silhouette_score(clustdf,k_pre)
## Calculate the silhouette value of each sample, K mean
sil_samp_val = silhouette_samples(clustdf,k_pre)

## Visual cluster analysis contour map, K-means
plt.figure(figsize=(10,6))
y_lower = 10
n_clu = len(np.unique(k_pre))
for ii in np.arange(n_clu):  ## Cluster into 3 categories
    ## Put the silhouette values of Category ii together to sort
    iiclu_sil_samp_sort = np.sort(sil_samp_val[k_pre == ii])
    ## Calculate the number of Category ii
    iisize = len(iiclu_sil_samp_sort)
    y_upper = y_lower + iisize
    ## Set the color of class ii images
    color = plt.cm.Spectral(ii / n_clu)
    plt.fill_betweenx(np.arange(y_lower,y_upper),0,iiclu_sil_samp_sort,
                      facecolor = color,alpha = 0.7)
    # Add a label in the middle of the y-axis corresponding to the cluster
    plt.text(-0.08,y_lower + 0.5*iisize,"Cluster"+str(ii+1)) 
    ## Update y_lower
    y_lower = y_upper + 5
## Add average profile coefficient score line
plt.axvline(x=sil_score,color="red",label = "mean:"+str(np.round(sil_score,3)))
plt.xlim([-0.1,1])   
plt.yticks([])
plt.legend(loc = 1)
plt.xlabel("Contour coefficient score")
plt.ylabel("Cluster label")
plt.title("K-means clustering contour map")
plt.show()
## After this, Try to use K-means clustering to divide it into 4 or more clusters to see the effect, and replicate the process above

