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
