# **Predictive Analysis: Optimizing Pressure Control through Machine Learning**

By    : Muhammad Fatih Idlan (faiti.alfaqar@gmail.com)

# Project Domain
This project was done to fulfil the *Machine Learning Terapan* 1st assignment submission on Dicoding. The domain used in this project is manufacturing control process, especially pressure control.

# Background
Pressure control is a fundamental aspect of many industrial processes, particularly in chemical engineering, where maintaining optimal pressure levels can significantly enhance efficiency, safety, and product quality. However, real-time fluctuations due to varying input conditions, system disturbances, and equipment aging pose challenges to traditional control methods. Traditional pressure control methods rely on Proportional-Integral-Derivative (PID) controllers, which require manual tuning and often struggle with dynamic system behaviors or process disturbances. Moreover, increasing feedback noise making PID performs poorly comparing to neural network model [[ 1 ]](https://www.semanticscholar.org/paper/A-comparison-between-a-traditional-PID-controller-a-Conradt/efb1c57c0dbc3b88cd35085f677869104fce5474). The existence of feedback noise is inevitably present in real world setting. Thus making machine learning based model is more flexible in real-time fluctuations.

# Business Understanding
## Problem Statement
Starting with explanation from the background above, core problems that this project aims to solve are:

* What are the variables that hugely affect target i.e. source pressure for developing predictive models that dynamically adjust it?
* How are the variables those hugely affect the source pressure is related?
* How the performance of each model to predict the source pressure that has been build?

## Objectives
According to problem statement above, this project has several objectives too, that are:

* Knowing the most influencial variables toward source pressure in the system
* Learn the relation between influencial variables to source pressure
* Determining high performance models

## Solution
To achive the objectives, we need to perform several things such as:

* Implementing correlation heatmap for each variables to identify influencial variables
* Using Linear Regression, K-Nearest Neighbour, and Dense Neural Network to selecting high performance corresponding to evaluation metrics (MSE)

# Data Understanding
![Data Understanding](/Assets/Kaggle.png "Data Understanding")
The dataset that used in this project is Smart Pressure Control Prediction, which can be accessed through kaggle [[ 2 ]](https://www.kaggle.com/datasets/guanlintao/smart-pressure-control-prediction). This dataset consist of 2 csv files, train and test, which in total has 4320 rows with 32 column. The explanation for each column can be seen below:

*   DEGC1PV = Equipment temperature in zone 1
*   DEGC2PV = Equipment temperature in zone 2
*   DEGC3PV = Equipment temperature in zone 3
*   DEGC4PV = Equipment temperature in zone 4
*   DEGC5PV = Equipment temperature in zone 5
*   DEGC6PV = Equipment temperature in zone 6
*   DEGC1SV = Desired equipment temperature in zone 1
*   DEGC2SV = Desired equipment temperature in zone 2
*   DEGC3SV = Desired equipment temperature in zone 3
*   DEGC4SV = Desired equipment temperature in zone 4
*   DEGC5SV = Desired equipment temperature in zone 5
*   DEGC6SV = Desired equipment temperature in zone 6
*   NM3/H.1PV = Air flowrate in zone 1
*   NM3/H.2PV = Air flowrate in zone 2
*   NM3/H.3PV = Air flowrate in zone 3
*   NM3/H.4PV = Air flowrate in zone 4
*   NM3/H.5PV = Air flowrate in zone 5
*   NM3/H.6PV = Air flowrate in zone 6
*   NM3/H.1SV = Desired air flowrate in zone 1
*   NM3/H.2SV = Desired air flowrate in zone 2
*   NM3/H.3SV = Desired air flowrate in zone 3
*   NM3/H.4SV = Desired air flowrate in zone 4
*   NM3/H.5SV = Desired air flowrate in zone 5
*   NM3/H.6SV = Desired air flowrate in zone 6
*   TEMP = Air temperature
*   FC1 = Control valve opening degree in zone 1
*   FC2 = Control valve opening degree in zone 2
*   FC3 = Control valve opening degree in zone 3
*   FC4 = Control valve opening degree in zone 4
*   FC5 = Control valve opening degree in zone 5
*   FC6 = Control valve opening degree in zone 6
*   mmH2O = Source input pressure

# Exploratory Data Analysis (EDA)
## Statistical Properties
json
'''
[{"index":"count","DEGC1PV":"3840.0","DEGC2PV":"3840.0","DEGC3PV":"3840.0","DEGC4PV":"3840.0","DEGC5PV":"3840.0","DEGC6PV":"3840.0","DEGC1SV":"3840.0","DEGC2SV":"3840.0","DEGC3SV":"3840.0","DEGC4SV":"3840.0","DEGC5SV":"3840.0","DEGC6SV":"3840.0","NM3/H.1PV":"3840.0","NM3/H.2PV":"3840.0","NM3/H.3PV":"3840.0","NM3/H.4PV":"3840.0","NM3/H.5PV":"3840.0","NM3/H.6PV":"3840.0","NM3/H.1SV":"3840.0","NM3/H.2SV":"3840.0"},{"index":"mean","DEGC1PV":"939.2851041666668","DEGC2PV":"952.687421875","DEGC3PV":"1112.672109375","DEGC4PV":"1089.26828125","DEGC5PV":"1064.5071875","DEGC6PV":"1044.1438802083333","DEGC1SV":"1010.6500000000001","DEGC2SV":"1070.0","DEGC3SV":"1214.9348958333333","DEGC4SV":"1201.3463541666667","DEGC5SV":"1147.6432291666667","DEGC6SV":"1182.91015625","NM3/H.1PV":"2644.1302083333335","NM3/H.2PV":"3062.5533854166665","NM3/H.3PV":"3509.8098958333335","NM3/H.4PV":"7790.544270833333","NM3/H.5PV":"2049.137760416667","NM3/H.6PV":"5026.225","NM3/H.1SV":"2609.134375","NM3/H.2SV":"3013.773958333333"},{"index":"std","DEGC1PV":"135.45595304145127","DEGC2PV":"154.76821993607717","DEGC3PV":"178.98477503964477","DEGC4PV":"173.77740241218478","DEGC5PV":"158.93625657797813","DEGC6PV":"177.66221850870477","DEGC1SV":"14.950441706365089","DEGC2SV":"0.0","DEGC3SV":"10.169618062337365","DEGC4SV":"10.608466161522355","DEGC5SV":"25.667546218462046","DEGC6SV":"61.20395788273972","NM3/H.1PV":"1179.3317961947675","NM3/H.2PV":"1464.637077204743","NM3/H.3PV":"1600.8417586831647","NM3/H.4PV":"4059.337145401351","NM3/H.5PV":"841.6380473776278","NM3/H.6PV":"1323.413675538872","NM3/H.1SV":"1198.357081977947","NM3/H.2SV":"1487.8200100610572"},{"index":"min","DEGC1PV":"474.1","DEGC2PV":"455.9","DEGC3PV":"551.6","DEGC4PV":"493.0","DEGC5PV":"525.7","DEGC6PV":"496.1","DEGC1SV":"1005.0","DEGC2SV":"1070.0","DEGC3SV":"1180.0","DEGC4SV":"1180.0","DEGC5SV":"1100.0","DEGC6SV":"1100.0","NM3/H.1PV":"0.0","NM3/H.2PV":"0.0","NM3/H.3PV":"0.0","NM3/H.4PV":"0.0","NM3/H.5PV":"0.0","NM3/H.6PV":"0.0","NM3/H.1SV":"977.0","NM3/H.2SV":"1488.0"},{"index":"25%","DEGC1PV":"860.6500000000001","DEGC2PV":"868.2249999999999","DEGC3PV":"993.875","DEGC4PV":"983.275","DEGC5PV":"1010.95","DEGC6PV":"965.775","DEGC1SV":"1005.0","DEGC2SV":"1070.0","DEGC3SV":"1210.0","DEGC4SV":"1200.0","DEGC5SV":"1140.0","DEGC6SV":"1130.0","NM3/H.1PV":"2090.0","NM3/H.2PV":"2296.75","NM3/H.3PV":"2536.0","NM3/H.4PV":"4227.75","NM3/H.5PV":"1538.0","NM3/H.6PV":"4268.75","NM3/H.1SV":"2139.0","NM3/H.2SV":"2214.0"},{"index":"50%","DEGC1PV":"995.9","DEGC2PV":"1029.7","DEGC3PV":"1215.2","DEGC4PV":"1188.2","DEGC5PV":"1142.6","DEGC6PV":"1136.4","DEGC1SV":"1005.0","DEGC2SV":"1070.0","DEGC3SV":"1215.0","DEGC4SV":"1200.0","DEGC5SV":"1140.0","DEGC6SV":"1180.0","NM3/H.1PV":"2447.5","NM3/H.2PV":"2882.0","NM3/H.3PV":"2982.5","NM3/H.4PV":"6931.5","NM3/H.5PV":"1897.0","NM3/H.6PV":"5017.5","NM3/H.1SV":"2495.0","NM3/H.2SV":"2897.0"},{"index":"75%","DEGC1PV":"1026.8","DEGC2PV":"1058.4","DEGC3PV":"1226.725","DEGC4PV":"1203.7","DEGC5PV":"1160.4","DEGC6PV":"1162.725","DEGC1SV":"1005.0","DEGC2SV":"1070.0","DEGC3SV":"1220.0","DEGC4SV":"1210.0","DEGC5SV":"1150.0","DEGC6SV":"1245.0","NM3/H.1PV":"2998.5","NM3/H.2PV":"3470.0","NM3/H.3PV":"4058.0","NM3/H.4PV":"11146.75","NM3/H.5PV":"2245.25","NM3/H.6PV":"6170.5","NM3/H.1SV":"2981.5","NM3/H.2SV":"3450.5"},{"index":"max","DEGC1PV":"1156.0","DEGC2PV":"1164.9","DEGC3PV":"1314.6","DEGC4PV":"1260.4","DEGC5PV":"1264.7","DEGC6PV":"1287.4","DEGC1SV":"1050.2","DEGC2SV":"1070.0","DEGC3SV":"1265.0","DEGC4SV":"1240.0","DEGC5SV":"1260.0","DEGC6SV":"1245.0","NM3/H.1PV":"12590.0","NM3/H.2PV":"15302.0","NM3/H.3PV":"12955.0","NM3/H.4PV":"15630.0","NM3/H.5PV":"6536.0","NM3/H.6PV":"9406.0","NM3/H.1SV":"12114.0","NM3/H.2SV":"14855.0"}]
'''
