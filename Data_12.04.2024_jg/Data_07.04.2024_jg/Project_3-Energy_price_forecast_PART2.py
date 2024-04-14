# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 17:46:43 2024

@author: julia
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import  metrics
import pickle



##                          ENERGY SERVICES - PROJECT #3

#                         Energy Price Forecast For Greece.
#              Loading the models and Forecasting energy price for 2020


#CSV reading: 2020 Data to forecast
data_to_forecast = pd.read_csv('data_to_forecast.csv') 
data_to_forecast['Date']=pd.to_datetime(data_to_forecast['Date'])
data_to_forecast=data_to_forecast.set_index(['Date'],drop=True)


'Testing the Model'
with open('RF_model.pkl','rb') as file:
    RF_model=pickle.load(file)
    
#Create matrix from data frame
Z=data_to_forecast .values
#Identify output Y
Y=Z[:,0]
#Identify input Y
X=Z[:,[1,2,3,4,5,6,7,8,9]]    





#Random Forest 
Y_RF = RF_model.predict(X)

plt.figure()
plt.plot(Y, label='Real')
plt.plot(Y_RF,label='RF_Forecast')
plt.title('RF_Forecast_2020')
plt.legend()
plt.show()

plt.figure()
plt.title('RF_Forecast_2020')
plt.scatter(Y,Y_RF)


MAE_RF_forecast=metrics.mean_absolute_error(Y,Y_RF)
MBE_RF_forecast=np.mean(Y-Y_RF)
MSE_RF_forecast=metrics.mean_squared_error(Y,Y_RF)  
RMSE_RF_forecast= np.sqrt(metrics.mean_squared_error(Y,Y_RF))
cvRMSE_RF_forecast = RMSE_RF_forecast / np.mean(Y)
NMBE_RF_forecast=MBE_RF_forecast/np.mean(Y)
print('---------------Random Forest-------------------------------------------------------------------')
print('          MAE             MBE                 MSE               RMSE             cvRMSE               NMBE')
print(MAE_RF_forecast,MBE_RF_forecast,MSE_RF_forecast,RMSE_RF_forecast,cvRMSE_RF_forecast,NMBE_RF_forecast)


header_RF= ["Criteria","ASHRAE 14","IMPMVP", "RF Model"]
criteria_RF=["NMBE","+/- 10%","+/- 5%", "{:.4%}".format(NMBE_RF_forecast)]
criteria_RF2=["CV(RMSE","30%","20%","{:.4%}".format(cvRMSE_RF_forecast)]


RF_Results_forecast = [header_RF, criteria_RF, criteria_RF2]

RF_results_forecast = pd.DataFrame(RF_Results_forecast [1:], columns=RF_Results_forecast [0])
RF_results_forecast.set_index('Criteria', inplace=True)



#Linear Regression
with open('LR_model.pkl','rb') as file:
    LR_model=pickle.load(file)
Y_LR = LR_model.predict(X)


plt.figure()
plt.plot(Y, label='Real')
plt.plot(Y_LR,label='LR_Forecast')
plt.title('LR_Forecast_2020')
plt.legend()
plt.show()

plt.figure()
plt.title('LR_Forecast_2020')
plt.scatter(Y,Y_RF)


MAE_LR_forecast=metrics.mean_absolute_error(Y,Y_LR)
MBE_LR_forecast=np.mean(Y-Y_LR)
MSE_LR_forecast=metrics.mean_squared_error(Y,Y_LR)  
RMSE_LR_forecast= np.sqrt(metrics.mean_squared_error(Y,Y_LR))
cvRMSE_LR_forecast = RMSE_LR_forecast / np.mean(Y)
NMBE_LR_forecast=MBE_LR_forecast/np.mean(Y)
print('------------------Linear Regression-------------------------------------------------------')
print('          MAE             MBE                 MSE               RMSE             cvRMSE               NMBE')
print(MAE_LR_forecast,MBE_LR_forecast,MSE_LR_forecast,RMSE_LR_forecast,cvRMSE_LR_forecast,NMBE_LR_forecast)






criteria_LR="{:.4%}".format(NMBE_LR_forecast)
criteria_LR2="{:.4%}".format(cvRMSE_LR_forecast)
LR_Results_forecast = [criteria_LR, criteria_LR2]

RF_results_forecast['LR Model'] = LR_Results_forecast


#Decision Tree


with open('DT_model.pkl','rb') as file:
    DT_model=pickle.load(file)
Y_DT = DT_model.predict(X)


plt.figure()
plt.plot(Y, label='Real')
plt.plot(Y_DT,label='DT_Forecast')
plt.title('DT_Forecast_2020')
plt.legend()
plt.show()

plt.figure()
plt.title('DT_Forecast_2020')
plt.scatter(Y,Y_DT)


MAE_DT_forecast=metrics.mean_absolute_error(Y,Y_DT)
MBE_DT_forecast=np.mean(Y-Y_DT)
MSE_DT_forecast=metrics.mean_squared_error(Y,Y_DT)  
RMSE_DT_forecast= np.sqrt(metrics.mean_squared_error(Y,Y_DT))
cvRMSE_DT_forecast = RMSE_DT_forecast / np.mean(Y)
NMBE_DT_forecast=MBE_DT_forecast/np.mean(Y)
print('-------------------Decision Tree----------------------------------------------------')
print('          MAE             MBE                 MSE               RMSE             cvRMSE               NMBE')
print(MAE_DT_forecast,MBE_DT_forecast,MSE_DT_forecast,RMSE_DT_forecast,cvRMSE_DT_forecast,NMBE_DT_forecast)




criteria_DT="{:.4%}".format(NMBE_DT_forecast)
criteria_DT2="{:.4%}".format(cvRMSE_DT_forecast)
DT_Results_forecast = [criteria_DT, criteria_DT2]

RF_results_forecast['DT Model'] = DT_Results_forecast


#Graduent Boosting 
with open('GB_model.pkl','rb') as file:
    GB_model=pickle.load(file)
Y_GB = GB_model.predict(X)


plt.figure()
plt.plot(Y, label='Real')
plt.plot(Y_GB,label='GB_Forecast')
plt.title('GB_Forecast_2020')
plt.legend()
plt.show()

plt.figure()
plt.title('GB_Forecast_2020')
plt.scatter(Y,Y_GB)

MAE_GB_forecast=metrics.mean_absolute_error(Y,Y_GB)
MBE_GB_forecast=np.mean(Y-Y_GB)
MSE_GB_forecast=metrics.mean_squared_error(Y,Y_GB)  
RMSE_GB_forecast= np.sqrt(metrics.mean_squared_error(Y,Y_GB))
cvRMSE_GB_forecast = RMSE_GB_forecast / np.mean(Y)
NMBE_GB_forecast=MBE_GB_forecast/np.mean(Y)
print('-------------------Gradient Boosting----------------------------------------------------')
print('          MAE             MBE                 MSE               RMSE             cvRMSE               NMBE')
print(MAE_GB_forecast,MBE_GB_forecast,MSE_GB_forecast,RMSE_GB_forecast,cvRMSE_GB_forecast,NMBE_GB_forecast)



criteria_GB="{:.4%}".format(NMBE_GB_forecast)
criteria_GB2="{:.4%}".format(cvRMSE_GB_forecast)
GB_Results_forecast = [criteria_GB, criteria_GB2]

RF_results_forecast['GB Model'] = GB_Results_forecast

#Auto Regressive
#with open('AR_model.pkl','rb') as file:
#    AR_model=pickle.load(file)
#Y_AR = AR_model.predict(X)


#plt.figure()
#plt.plot(Y, label='Real')
#plt.plot(Y_AR,label='GB_Forecast')
#plt.title('AR_Forecast_2020')
#plt.legend()
#plt.show()

#plt.figure()
#plt.title('AR_Forecast_2020')
#plt.scatter(Y,Y_AR)

data_comparisson = {
    'Random Forest': [MAE_RF_forecast, MBE_RF_forecast, MSE_RF_forecast, RMSE_RF_forecast, cvRMSE_RF_forecast, NMBE_RF_forecast],
    'Linear Regression': [MAE_LR_forecast, MBE_LR_forecast, MSE_LR_forecast, RMSE_LR_forecast, cvRMSE_LR_forecast, NMBE_LR_forecast],
    'Decision Trees': [MAE_DT_forecast, MBE_DT_forecast, MSE_DT_forecast, RMSE_DT_forecast, cvRMSE_DT_forecast, NMBE_DT_forecast],
    'Gradient Boosting': [MAE_GB_forecast, MBE_GB_forecast, MSE_GB_forecast, RMSE_GB_forecast, cvRMSE_GB_forecast, NMBE_GB_forecast],
}

# create a pandas dataframe from the dictionary
df_model_comparisson = pd.DataFrame(data_comparisson)

# set the index to the evaluation metrics
df_model_comparisson.index = ['MAE', 'MBE', 'MSE', 'RMSE', 'cvRMSE', 'NMBE']

# convert cvRMSE and NMBE to percentage format
df_model_comparisson.loc[['cvRMSE', 'NMBE']] = df_model_comparisson.loc[['cvRMSE', 'NMBE']] * 100
df_model_comparisson.loc[['cvRMSE', 'NMBE']] = df_model_comparisson.loc[['cvRMSE', 'NMBE']].applymap("{:.2f}%".format)

# print the dataframe with the rows and columns switched
print('')
print('----------------Model Comparisson-----------------------------------------')
print(df_model_comparisson.transpose())


print('')
print('--------------STANDARDS EVALUATION----------------------------------------')
print(RF_results_forecast)



