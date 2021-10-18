# Air-quality-forecast-secondary-modeling
Establishing an air quality forecast model, knowing the process of air pollution that may occur in advance and taking corresponding control measures

# For Task 1: 
There is no difficulty. According to the given method, a reasonable method can be used to perform the corresponding calculation. It should be 
noted that the filling method of missing values may affect the final calculation result. There are many ways to fill missing values. However, for this kind of 
time series monitoring data, the missing value is greatly affected by the previous moment and the later moment. The missing value can use the mean value of 
the previous moment, the value of the previous moment, and the mean value of several moments before and after. (For the abnormal situation of the data, in 
addition to the missing values, you should also pay attention to the influence of accidental factors) It is recommended to use visual methods such as time series 
line graphs to observe the data globally and locally.

Missing value filling method: use the previous filling, use the previous and the following means to fill, use the mean (median) to fill, consider the filling method 
of the influence of multiple variables, such as KNN.

Data visualization methods: line chart, box plot, etc.

# For Task 2 (Reasonable classification): 
Since there are no fixed categories, it should be analyzed according to the distribution of the data. The intuitive way is cluster analysis (there are many 
clustering algorithms that can be used), according to the aggregation of data The situation is classified, and then the meteorological condition characteristics 
of each type of data (the simplest representation of this characteristic can use the center of the cluster), and the influence of these meteorological 
characteristics on the diffusion or sedimentation of pollutants, and then on the AQI.

Clustering methods: K-means clustering, K-median clustering, density clustering, hierarchical clustering, etc. The visualization of clustering results can be used 
for dimensionality reduction visualization, contour coefficient map, etc. with the help of TSNE and other methods.

# For Task 3: 
Use data sets from three locations to establish a secondary forecast mathematical model. There can be many methods for the specific method used by the model. 
How to evaluate the quality of the prediction effect, through the relative error of the AQI and the accuracy of the primary pollutant prediction (both of these
indicators can be calculated using the given data, so it can be considered that there are two monitoring objectives for the model, and these two There is a 
certain relationship.) The value of AQI can be regarded as a regression type model, the accuracy of prediction for the primary pollutant can be regarded as a 
classification problem, and the two loss functions can also be merged into one. In view of the time sequence of the data, the impression of time sequence cannot 
be ignored when modeling. For example, a combined modeling method that combines multiple and multiple models, and deep learning algorithms related to time series 
LSTM, can be used.

Machine learning algorithms: support vector machines, neural networks, random forests, multiple linear regression, Ridge regression, ARIMA, ARIMAX, Prophet, 
LSTM, RNN, etc.

# For Task 4: Compared to Task 3, this task needs to take a closer look at the impact of orientation and distance on air quality. The evaluation indicators are 
the same as in Question 3, so corresponding considerations can be made on the basis of the modeling results of Question 3, for example: adding new features. 
Or consider factors such as the flow of pollutants based on meteorological indicators such as wind direction.

