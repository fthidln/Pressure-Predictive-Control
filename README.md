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

## Multivariate Analysis
### Correlation Matrix
![Correlation Matrix](Assets/CorrMat.png "Correlation Matrix")

### Important Key Points from EDA
*   All DEGC2SV variable values are stagnant at 1070, so they have no impact on the target
*   Each variable has quite a lot of outlier values, but it is still retained because it can represent noise in real time
*   From correlation matrix above, we can conclude that NM3/H.1PV, NM3/H.2PV, NM3/H.1SV, and NM3/H.2SV is the most influencial variables to source input pressure, so we can drop the other unnacessary variables

## Principal Component Analysis
This step is important, Principal Component Analysis (PCA) helps to eliminate redundancy by transforming the original features into a smaller set of uncorrelated variables (principal components), making the data easier to analyze by the model. Turns out that the most influencial principal component variance is 0.978, followed by 0.012 and 0.009. We can ignore the last two dimension because it has a very small variance corresponding to the first one [[ 3 ]](https://www.sciencedirect.com/science/article/pii/S1877050919321507). Thus simplify the problem that the models try to solve [[ 4 ]](https://royalsocietypublishing.org/doi/10.1098/rsta.2015.0202). 

# Spliting Dataset into Train and Test Set
To initiate the model development, splitting the data into train and test set is necessary. Moreover, this project using supervised learning. The train set serve as learning agent while test set serve as evaluating agent.

## Standardization
In order to scaling the dataset value, we can use standardization method. It transform the dataset in such a way to have a mean of 0 and standard deviation of 1. Moreover, standardization method is the superior scaling technique for medium and large dataset [[ 5 ]](https://ieeexplore.ieee.org/document/10681438).

# Model Development
In this step, the algorithm used for model developments are K-Nearest Neighbour, Linear Regression, and Dense Neural Network.

* K-Nearest Neighbour = KNN is a simple, instance-based learning algorithm. It classifies a new data point based on the majority class of its K-nearest neighbors in the feature space.
 * Pros
   * Simple to understand and implement
   * No explicit training phase (lazy learning)
 * Cons
   * Computationally expensive for large datasets (due to distance calculations)
   * Sensitive to irrelevant or unscaled features
   * Performance depends on the choice of K and distance metric
* Linear Regression = Linear regression models the relationship between a dependent variable (target) and one or more independent variables (features) by fitting a linear equation to the data.
 * Pros
   * Simple, interpretable model
   * Works well when there is a linear relationship between features and the target
 * Cons
   * Limited to linear relationships
   * Sensitive to outliers
   * Assumes no multicollinearity between features (when using multiple features)
* Dense Neural Network = A dense neural network (DNN) consists of layers of neurons where each neuron in one layer is connected to every neuron in the next layer (hence the term "fully connected").
 * Pros
   * Can model highly complex relationships between input and output
   * Scalable to large datasets and tasks like image recognition, natural language processing, etc
 * Cons
   * Requires a large amount of data and computational resources to train effectively
   * Prone to overfitting, especially with small datasets
   * Difficult to interpret compared to simpler models like linear regression

# Model Evaluation
The matrix evaluation used for this step is Mean Squared Error
\begin{align*}
$$text{MSE}(y, \hat{y}) = \frac{\sum_{i=0}^{N - 1} (y_i - x_i)^2}{N}$$
\end{align*}

Where:

* N = Amount of the data
* i = Index of the data
* y = Actual value
* x = Predicted value

MSE is a metric used to measure the average squared difference between the predicted values and the actual values in the dataset. It is calculated by taking the average of the squared residuals, where the residual is the difference between predicted value and the actual value for each data point [[ 6 ]](https://www.geeksforgeeks.org/mean-squared-error/). A lower MSE indicates that the model's predictions are closer to the actual values signifying better accuracy. While, a higher MSE suggests that the model's predictions deviate further from true values indicating the poorer performance.

### Performance of Each Machine Learning Algorithm
![Perfomance of Each Algorithm](Assets/HistPerform.png "Perfomance of Each Algorithm")

|index|train|test|
|---|---|---|
|KNN|1731\.4725651041665|4193\.04046875|
|Linear Regression|5030\.204432958909|5404\.623603896129|
|ANN|2773\.7929275104184|4416\.747012630459|

From metric evaluation table above, we can conclude that K-Nearest Neighbour algorithm is the most desired algortihm because has the lowest MSE value in train and test set, followed by Dense Neural Network, and the last is Linear Regression.

## Model Prediction
This step is carried out to see how each machine learning algorithm predicting the target data (source pressure).

|index|y\_true|dimension|LR|KNN|ANN|
|---|---|---|---|---|---|
|770|601|0\.25246995242719095|580\.21152119611|590\.6|577\.8556518554688|

![Prediction Scatter](Assets/RealPredScatter.png "Prediction Scatter")

From the figure above, we can compare how prediction data and real data from each machine learning algorithm (K-Nearest Neighbour, Linear Regression, Dense Neural Network). Clearly, Linear Regression generated data point in a straight line. K-Nearest Neighbour generated data points that gather in one area. Then Dense Neural Network seems to struggle with its predictions forming a smoother but lower curve that doesn't capture the wide spread of real data.

# Reference

*   [ 1 ] J. Conradt, “A comparison between a traditional PID controller and an Artificial Neural Network controller in manipulating a robotic arm,” 2019. Accessed: Oct. 22, 2024. [Online]. Available: https://www.semanticscholar.org/paper/A-comparison-between-a-traditional-PID-controller-a-Conradt/efb1c57c0dbc3b88cd35085f677869104fce5474

*   [ 2 ] “Smart Pressure Control Prediction.” Accessed: Oct. 23, 2024. [Online]. Available: https://www.kaggle.com/datasets/guanlintao/smart-pressure-control-prediction

*   [ 3 ] N. Salem and S. Hussein, “Data dimensional reduction and principal components analysis,” Procedia Computer Science, vol. 163, pp. 292–299, Jan. 2019, doi: 10.1016/j.procs.2019.12.111.

*   [ 4 ] I. T. Jolliffe and J. Cadima, “Principal component analysis: a review and recent developments,” Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences, vol. 374, no. 2065, p. 20150202, Apr. 2016, doi: 10.1098/rsta.2015.0202.

*   [ 5 ] K. Mahmud Sujon, R. Binti Hassan, Z. Tusnia Towshi, M. A. Othman, M. Abdus Samad, and K. Choi, “When to Use Standardization and Normalization: Empirical Evidence From Machine Learning Models and XAI,” IEEE Access, vol. 12, pp. 135300–135314, 2024, doi: 10.1109/ACCESS.2024.3462434.

*   [ 6 ] “Mean Squared Error | Definition, Formula, Interpretation and Examples,” GeeksforGeeks. Accessed: Oct. 23, 2024. [Online]. Available: https://www.geeksforgeeks.org/mean-squared-error/
