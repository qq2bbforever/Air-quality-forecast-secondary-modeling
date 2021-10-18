# ## Task 3
# Using the data in Annexes 1 and 2, establish a secondary forecasting mathematical model that is applicable to three monitoring points A, B, and C 
# at the same time (the distance between the two monitoring points is more than 100km, ignoring the mutual influence) to predict the future three The 
# single-day concentration values of 6 conventional pollutants in a day require that the maximum relative error of the AQI forecast value in the prediction 
# results of the secondary forecast model should be as small as possible, and the accuracy of the primary pollutant forecast should be as high as possible. 
# And use the model to predict the single-day concentration values of 6 conventional pollutants at monitoring points A, B, and C from July 13 to July 15, 2021, 
# and calculate the corresponding AQI and primary pollutants.

## For the change of a single pollutant with strong time series, the Prophet model can be used to predict, which is more accurate for a single sequence.
from fbprophet import Prophet

## Read data
df = usedf.iloc[:,[0,3]]
df.columns = ["ds","y"]
## Define the data type of time data
df["ds"] = pd.to_datetime(df["ds"])
print(df.head())

## Divide data into training set and test set
train = df[0:780]
test = df[780:]

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

