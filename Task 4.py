# ## Task 4
# Concentrations of pollutants in adjacent areas often have a certain degree of correlation, and regional coordinated forecasting may improve the accuracy 
# of air quality forecasts. As shown in Figure 4, there are monitoring points A1, A2, and A3 in the area adjacent to monitoring point A. Use the data in 
# Annexes 1 and 3 to establish a collaborative forecast model that includes four monitoring points A, A1, A2, and A3. The maximum relative error of the AQI 
# prediction value in the model prediction results should be as small as possible, and the accuracy of the primary pollutant prediction should be as high as 
# possible. Use this model to predict the single-day concentration values of 6 conventional pollutants at monitoring points A, A1, A2, and A3 from July 13 to 
# July 15, 2021, and calculate the corresponding AQI and primary pollutants.

## Task 4 has the same way as Task 3

## Readin data
df = IAQIpdf.iloc[:,[6,1]]
df.columns = ["ds","y"]
## Define the data type of time data
df["ds"] = pd.to_datetime(df["ds"])
print(df.head())

IAQIpdf.head()

## Quwestion 4 has the same way as Question 3
## Divide the data into training set and test set
train = df[0:800]
test = df[800:]

## Model building
model = Prophet(growth = "linear",   # Linear growth trend
                yearly_seasonality = True, # Annual cycle trend
                weekly_seasonality = False,# Weekly trend
                daily_seasonality = False,  # Trend in days
                seasonality_mode = "multiplicative", # Seasonal pattern
                seasonality_prior_scale = 12, # Seasonal Periodic Length
               )
model.fit(train)
## Use the model to make predictions on the test set
forecast = model.predict(test)
## Output partial prediction results
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
print("The absolute value prediction error on the test set is:",mean_absolute_error(test.y,forecast.yhat))
## Visualize raw data and forecast data for comparison
fig, ax = plt.subplots()
train.plot(x = "ds",y = "y",figsize=(14,7),label="Training data",ax = ax)
test.plot(x = "ds",y = "y",figsize=(14,7),label="Test Data",ax = ax)
forecast.plot(x = "ds",y = "yhat",style = "g--o",label="Forecast data",ax = ax)
## Visualize the confidence interval
ax.fill_between(test["ds"].values, forecast["yhat_lower"], 
                forecast["yhat_upper"],color='k',alpha=.2,
                label = "95% CI")
plt.grid()
plt.xlabel("Time")
plt.ylabel("Numerical value")
plt.title("Prophet Model")
plt.legend(loc=2)
plt.show()
