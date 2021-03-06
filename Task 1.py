#!/usr/bin/env python
# coding: utf-8

# ## Air quality forecast secondary modeling
# 
# Atmospheric pollution refers to the fact that certain substances enter the atmosphere due to human activities or natural processes, present a sufficient 
# concentration, reach a sufficient time, and therefore endanger the comfort, health and welfare of the human body or endanger the ecological environment. 
# Pollution prevention and control practices have shown that establishing an air quality forecast model, knowing the process of air pollution that may occur 
# in advance and taking corresponding control measures are one of the effective ways to reduce the harm caused by air pollution to human health and the 
# environment, and to improve ambient air quality.

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

# ## Task 1
# Using the data in Annex 1, calculate the AQI and primary pollutants measured daily at monitoring point A from August 25 to August 28, 2020 according to 
# the method in the appendix

## Read the data to be calculated
usedf = pd.read_excel("Data for Question 1.xlsx")
usedf.head()

## Delete the last three rows of data in the data first
usedf.tail()

usedf = usedf.iloc[0:-3,:]
usedf.tail()

## Data preprocessing for missing values

## Use visualization methods to view the distribution of missing values in the data
msno.matrix(usedf,figsize=(14, 7),width_ratios=(13, 2),color=(0.25, 0.25, 0.5))
plt.show()

## The missing values of some samples are out of time and place, and the rest of the monitoring features are all missing

## For these data that have time series characteristics and rarely change abruptly, I simply use the average value of the previous day and the next day for 
## the missing value data of each column, and round up the data of some variables.

## Note: Many missing values are caused by missing values at certain moments of the day, not all days are missing values???
## It is also possible to estimate or fill in the missing value data throughout the day when there are no missing values. 
## In short, there are many ways to deal with missing values.

## Here, I chose the simpler one

Sx = usedf.O3
Sx
Sxb = Sx.fillna(method="bfill")
Sxf = Sx.fillna(method="ffill")

Sxbf = (Sxb + Sxf) / 2
sum(Sxbf.isna())

## Define a function for missing value processing for each variable
def myfillna1(Sx):
    Sxb = Sx.fillna(method="bfill")
    Sxf = Sx.fillna(method="ffill")
    Sxbf = (Sxb + Sxf) / 2
    return Sxbf

## Fill in missing values
usedf.iloc[:,2:8] = usedf.iloc[:,2:8].apply(func = myfillna1,axis = 0)
usedf.head(20)

## Use visualization methods to view the distribution of missing values in the data
msno.matrix(usedf,figsize=(14, 7),width_ratios=(13, 2),color=(0.25, 0.25, 0.5))
plt.show()

## Visualize fluctuations after data filling
fig = px.line(usedf, x='Date', y="SO2",title="SO2",
              width=1000,height=500)
fig.show()

## Visualize fluctuations after data filling
fig = px.line(usedf, x='Date', y="SO2",title="SO2",
              width=1000,height=500)
fig.show()
## There is a certain periodicity

## Define a position index function to find the interval
def which(self):
    try:
        self = list(iter(self))
    except TypeError as e:
        raise Exception("""'which' method can only be applied to iterables.
        {}""".format(str(e)))
    indices = [i for i, x in enumerate(self) if bool(x) == True]
    return(indices)
  
  
## Next, calculate the AQI by the hour, taking O3 as an example
CP = 185  ## ?????????P??????????????????
BPHi = [100,160,215,265,800]
BPLo = [0,100,160,215,265]
IAQIHi = [50,100,150,200,300,400,500]
IAQILo = [0,50,100,150,200,300,400]

## Calculate the interval where CP is located
index = which(np.array(BPLo) <= CP)[-1]
IAQIp = (IAQIHi[index]-IAQILo[index])/(BPHi[index]-BPLo[index])*(CP-BPLo[index])+IAQILo[index]
np.ceil(IAQIp)

def IAQIp(CP,BPHi,BPLo,IAQIHi,IAQILo,O3 = False):
    IAQIpval = 0
    if O3 == True:
        if (CP > 800):
            IAQIpval = "NaN"
        else:
            ## Calculate the interval where CP is located
            index = which(np.array(BPLo) <= CP)[-1]
            IAQIpval = (IAQIHi[index]-IAQILo[index])/(BPHi[index]-BPLo[index])*(CP-BPLo[index])+IAQILo[index]
            IAQIpval = np.ceil(IAQIpval)
    else:
        index = which(np.array(BPLo) <= CP)[-1]
        IAQIpval = (IAQIHi[index]-IAQILo[index])/(BPHi[index]-BPLo[index])*(CP-BPLo[index])+IAQILo[index]
        IAQIpval = np.ceil(IAQIpval)

    return(IAQIpval)

CP = 100  ## The mass concentration value of pollutant P
BPHi = [100,160,215,265,800]
BPLo = [0,100,160,215,265]
IAQIHi = [50,100,150,200,300,400,500]
IAQILo = [0,50,100,150,200,300,400]
A = IAQIp(CP,BPHi,BPLo,IAQIHi,IAQILo,O3 = False)
## ??????PM10
BPHi = [50,150,250,350,420,500,600]
BPLo = [0,50,150,250,350,420,500]
A = IAQIp(CP,BPHi,BPLo,IAQIHi,IAQILo,O3 = False)
A

## Call the IAQIp function for each variable to calculate the IAQIp of each pollutant per day
IAQIpdf = usedf.iloc[:,2:8]
IAQIHi = [50,100,150,200,300,400,500]
IAQILo = [0,50,100,150,200,300,400]

## SO2
BPHi = [50,150,475,800,1600,2100,2620]
BPLo = [0,50,150,475,800,1600,2100]
IAQIpdf.SO2 = IAQIpdf.SO2.apply(func = IAQIp,BPHi = BPHi,BPLo=BPLo,IAQIHi=IAQIHi,
                                IAQILo=IAQILo,O3 = False)

## NO2
BPHi = [40,80,180,280,565,750,940]
BPLo = [0,40,80,180,280,565,750]
IAQIpdf.NO2 = IAQIpdf.NO2.apply(func = IAQIp,BPHi = BPHi,BPLo=BPLo,IAQIHi=IAQIHi,
                                IAQILo=IAQILo,O3 = False)
## PM10
BPHi = [50,150,250,350,420,500,600]
BPLo = [0,50,150,250,350,420,500]
IAQIpdf["PM10"] = IAQIpdf.PM10.apply(func = IAQIp,BPHi = BPHi,BPLo=BPLo,IAQIHi=IAQIHi,
                                     IAQILo=IAQILo,O3 = False)
## PM2.5
BPHi = [35,75,115,150,250,350,500]
BPLo = [0,35,75,115,150,250,350]
IAQIpdf["PM2.5"] = IAQIpdf["PM2.5"].apply(func = IAQIp,BPHi = BPHi,BPLo=BPLo,IAQIHi=IAQIHi,
                                          IAQILo=IAQILo,O3 = False)
## O3
BPHi = [100,160,215,265,800]
BPLo = [0,100,160,215,265]
IAQIpdf["O3"] = IAQIpdf["O3"].apply(func = IAQIp,BPHi = BPHi,BPLo=BPLo,IAQIHi=IAQIHi,
                                    IAQILo=IAQILo,O3 = True)
## CO
BPHi = [2,4,14,24,36,48,60]
BPLo = [0,2,4,14,24,36,48]
IAQIpdf["CO"] = IAQIpdf["CO"].apply(func = IAQIp,BPHi = BPHi,BPLo=BPLo,IAQIHi=IAQIHi,
                                    IAQILo=IAQILo,O3 = False)

IAQIpdf["Date"] = usedf.Date
IAQIpdf.head(20)

## Calculate daily AQI and primary pollutants
IAQIpdf.isna().sum()
## Calculate whether IAQI>=500
np.sum(IAQIpdf.iloc[:,0:6] >= 500)

AQIdf = pd.DataFrame({"Data" : IAQIpdf["Date"],
                      "AQI" : IAQIpdf.iloc[:,0:6].apply(max,axis = 1)})
## The status of the primary pollutant
shouyao = []
name = np.array(['SO2', 'NO2', 'PM10', 'PM2.5', 'O3', 'CO'])
for ii in AQIdf.index:
    val = which(IAQIpdf.iloc[ii,0:6] == AQIdf.AQI[ii])
    shouyao.append(name[val])
AQIdf["shouyao"] = shouyao
AQIdf.head(10)

