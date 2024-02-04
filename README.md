# California House Price Prediction

## Overview
    This project focuses on predicting median house prices in California using a linear regression model, Lasso(L1), Ridge(L2) and using Stochastic Gradient descent(SGD).
The dataset used contains various features such as median income, median age, total rooms, total )bedrooms, population, households, latitude, longitude, and distances to coast and major cities in California.
![image](https://github.com/SaadElDine/PRODIGY_ML_01/assets/113860522/442dce12-0c12-424c-8351-46667b8e63dc)

--

## Project Structure

The project is structured as follows:

- `data/`: Contains the dataset `California_Houses.csv`.
- `notebooks/`: Jupyter notebooks for data exploration, preprocessing, modeling, and evaluation.
- `src/`: Python scripts for data preprocessing, modeling, and evaluation.
- `README.md`: Overview of the project and instructions for running the code.

--

## Dataset
The dataset California_Houses.csv contains the following columns:
- Median_House_Value: Target variable (median house value in dollars).
- Median_Income: Median income in tens of thousands of dollars.
- Median_Age: Median age of the population.
- Tot_Rooms: Total number of rooms.
- Tot_Bedrooms: Total number of bedrooms.
- Population: Total population.
- Households: Total number of households.
- Latitude, Longitude: Latitude and longitude coordinates.
- Distance_to_coast, Distance_to_LA, Distance_to_SanDiego, Distance_to_SanJose, Distance_to_SanFrancisco: Distances to major cities and the coast in miles.

--

## Linear Regression
    Linear regression is a simple and widely used model for predicting a continuous target variable based on one or more input features. It assumes a linear relationship between the input features and the target variable.
    In this project, we implemented linear regression using the LinearRegression class from the sklearn.linear_model module. We trained the model on the training data and evaluated its performance on the validation and test sets using metrics such as mean squared error (MSE), mean absolute error (MAE), and R-squared score.

--

## Data Exploration
The data_exploration.ipynb notebook explores the dataset by visualizing the relationships between each feature and the target variable using scatter plots. This helps in understanding the potential impact of each feature on the target variable.

--

## Data Preprocessing
- The data_preprocessing.ipynb notebook preprocesses the dataset by:
- Identifying and handling skewed features using log transformation.
- Removing outliers using the Interquartile Range (IQR) method.
- Checking for and handling missing values (no missing values found in this dataset).
- Splitting the data into training (70%), validation (15%), and test (15%) sets.

--

## Lasso Regression
    Lasso (Least Absolute Shrinkage and Selection Operator) regression is a linear regression technique that uses L1 regularization to penalize the magnitude of the coefficients, leading to feature selection by shrinking some coefficients to zero.
    We implemented Lasso regression using the Lasso class from the sklearn.linear_model module. We used grid search with cross-validation to find the best hyperparameters and evaluated the model's performance on the validation set.

--

## Ridge Regression
    Ridge regression is another linear regression technique that uses L2 regularization to penalize the squared magnitude of the coefficients, which can help reduce overfitting by shrinking the coefficients towards zero.
    We implemented Ridge regression using the Ridge class from the sklearn.linear_model module. Similar to Lasso regression, we used grid search with cross-validation to find the best hyperparameters and evaluated the model's performance on the validation set.

--

## Stochastic Gradient Descent (SGD) Regression
    Stochastic Gradient Descent (SGD) is an optimization algorithm that iteratively updates the model parameters to minimize a loss function. In the context of regression, SGD can be used to train linear regression models.
    We implemented SGD regression using the SGDRegressor class from the sklearn.linear_model module. We trained the model using SGD with a maximum number of iterations and a tolerance value for convergence. We then evaluated the model's performance on the validation and test sets.

--

## Evaluation
    The performance of each model is evaluated using mean squared error (MSE), mean absolute error (MAE), and R-squared score on both the validation and test sets. The results are compared to select the best-performing model

## Requirements

To run the code, you need the following Python libraries:

- numpy
- pandas
- seaborn
- matplotlib
- scipy
- scikit-learn
  
--

## Conclusion
    In this project, we explored different regression techniques for predicting median house prices in California. We implemented and evaluated linear regression, Lasso regression, Ridge regression, and Stochastic Gradient Descent (SGD) regression models. The results and performance metrics can be found in the respective Jupyter notebooks.
