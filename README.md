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
|index|DEGC1PV|DEGC2PV|DEGC3PV|DEGC4PV|DEGC5PV|DEGC6PV|DEGC1SV|DEGC2SV|DEGC3SV|DEGC4SV|DEGC5SV|DEGC6SV|NM3/H\.1PV|NM3/H\.2PV|NM3/H\.3PV|NM3/H\.4PV|NM3/H\.5PV|NM3/H\.6PV|NM3/H\.1SV|NM3/H\.2SV|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|count|3840\.0|3840\.0|3840\.0|3840\.0|3840\.0|3840\.0|3840\.0|3840\.0|3840\.0|3840\.0|3840\.0|3840\.0|3840\.0|3840\.0|3840\.0|3840\.0|3840\.0|3840\.0|3840\.0|3840\.0|
|mean|939\.2851041666668|952\.687421875|1112\.672109375|1089\.26828125|1064\.5071875|1044\.1438802083333|1010\.6500000000001|1070\.0|1214\.9348958333333|1201\.3463541666667|1147\.6432291666667|1182\.91015625|2644\.1302083333335|3062\.5533854166665|3509\.8098958333335|7790\.544270833333|2049\.137760416667|5026\.225|2609\.134375|3013\.773958333333|
|std|135\.45595304145127|154\.76821993607717|178\.98477503964477|173\.77740241218478|158\.93625657797813|177\.66221850870477|14\.950441706365089|0\.0|10\.169618062337365|10\.608466161522355|25\.667546218462046|61\.20395788273972|1179\.3317961947675|1464\.637077204743|1600\.8417586831647|4059\.337145401351|841\.6380473776278|1323\.413675538872|1198\.357081977947|1487\.8200100610572|
|min|474\.1|455\.9|551\.6|493\.0|525\.7|496\.1|1005\.0|1070\.0|1180\.0|1180\.0|1100\.0|1100\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|977\.0|1488\.0|
|25%|860\.6500000000001|868\.2249999999999|993\.875|983\.275|1010\.95|965\.775|1005\.0|1070\.0|1210\.0|1200\.0|1140\.0|1130\.0|2090\.0|2296\.75|2536\.0|4227\.75|1538\.0|4268\.75|2139\.0|2214\.0|
|50%|995\.9|1029\.7|1215\.2|1188\.2|1142\.6|1136\.4|1005\.0|1070\.0|1215\.0|1200\.0|1140\.0|1180\.0|2447\.5|2882\.0|2982\.5|6931\.5|1897\.0|5017\.5|2495\.0|2897\.0|
|75%|1026\.8|1058\.4|1226\.725|1203\.7|1160\.4|1162\.725|1005\.0|1070\.0|1220\.0|1210\.0|1150\.0|1245\.0|2998\.5|3470\.0|4058\.0|11146\.75|2245\.25|6170\.5|2981\.5|3450\.5|
|max|1156\.0|1164\.9|1314\.6|1260\.4|1264\.7|1287\.4|1050\.2|1070\.0|1265\.0|1240\.0|1260\.0|1245\.0|12590\.0|15302\.0|12955\.0|15630\.0|6536\.0|9406\.0|12114\.0|14855\.0|
