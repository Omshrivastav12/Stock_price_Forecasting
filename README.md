# Stock Price Prediction

## Project Overview

This project aims to predict the future closing prices of stock data using various time series forecasting models. The goal is to utilize both classical statistical methods and advanced machine learning techniques to achieve accurate predictions.

## Table of Contents

1. [Introduction](#introduction)
2. [Data Preprocessing](#data-preprocessing)
3. [Models](#models)
   - [AR (AutoRegressive) Model](#ar-autoregressive-model)
   - [MA (Moving Average) Model](#ma-moving-average-model)
   - [ARMA (AutoRegressive Moving Average) Model](#arma-autoregressive-moving-average-model)
   - [ARIMA (AutoRegressive Integrated Moving Average) Model](#arima-autoregressive-integrated-moving-average-model)
   - [SARIMA (Seasonal ARIMA) Model](#sarima-seasonal-arima-model)
   - [LSTM (Long Short-Term Memory) Model](#lstm-long-short-term-memory-model)
4. [Usage](#usage)
5. [Dependencies](#dependencies)
6. [Acknowledgements](#acknowledgements)

## Introduction

This project focuses on forecasting stock prices using various time series models. The primary models used include AR, MA, ARMA, ARIMA, SARIMA, and LSTM. Each model is evaluated based on its performance in predicting future stock prices.

## Data Preprocessing

1. **Data Collection**: Gather historical stock price data, which typically includes columns such as Date, Open, High, Low, Close, and Volume.

2. **Data Cleaning**:
   - Handle missing values.
   - Remove any anomalies or outliers.
   - Ensure data consistency.

3. **Feature Engineering**:
   - Create additional features if needed, such as moving averages or lag features.

4. **Data Splitting**:
   - Split the data into training and testing sets.
   - Use the training set to train the models and the testing set to evaluate performance.

5. **Normalization**:
   - Normalize the data if required, especially for LSTM models, to ensure better performance.

## Models

### AR (AutoRegressive) Model

The AR model predicts future values based on the weighted sum of past values. It is useful when the data shows a clear dependency on its previous values.

### MA (Moving Average) Model

The MA model uses past forecast errors to predict future values. It helps in smoothing the data and capturing patterns.

### ARMA (AutoRegressive Moving Average) Model

Combines AR and MA models to account for both the dependency on past values and past forecast errors. It is suitable for stationary time series data.

### ARIMA (AutoRegressive Integrated Moving Average) Model

An extension of ARMA that includes differencing to make non-stationary data stationary. It is useful for datasets with trends and seasonality.

### SARIMA (Seasonal ARIMA) Model

An extension of ARIMA that includes seasonal components. It is ideal for data with strong seasonal patterns.

### LSTM (Long Short-Term Memory) Model

An advanced deep learning model that can capture long-term dependencies in time series data. It uses a network of LSTM cells to learn patterns over time, making it suitable for complex forecasting tasks.

## Usage

1. **Install Dependencies**: Ensure you have all the required libraries installed.
   
2. **Load Data**: Load the historical stock price data into your project.

3. **Preprocess Data**: Follow the data preprocessing steps to prepare your data for modeling.

4. **Train Models**: Fit AR, MA, ARMA, ARIMA, SARIMA, and LSTM models on the training data.

5. **Evaluate Models**: Use the testing data to evaluate model performance and compare results.

6. **Make Predictions**: Use the trained models to forecast future stock prices.

## Dependencies

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- statsmodels
- tensorflow (for LSTM)

## Acknowledgements

- [yfinance](Library) for providing the historical data used in this project.

