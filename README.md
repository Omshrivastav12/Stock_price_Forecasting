# 📈 Stock Price Prediction

## Project Overview

This project aims to forecast **future closing prices** of stocks by leveraging a variety of **time series forecasting models**. By using a blend of statistical methods and machine learning, we strive to achieve precise stock price predictions.

---

## 📚 Table of Contents

1. [📖 Introduction](#-introduction)
2. [🔄 Data Preprocessing](#-data-preprocessing)
3. [📊 Models](#-models)
   - [AR (AutoRegressive) Model](#ar-autoregressive-model)
   - [MA (Moving Average) Model](#ma-moving-average-model)
   - [ARMA (AutoRegressive Moving Average) Model](#arma-autoregressive-moving-average-model)
   - [ARIMA (AutoRegressive Integrated Moving Average) Model](#arima-autoregressive-integrated-moving-average-model)
   - [SARIMA (Seasonal ARIMA) Model](#sarima-seasonal-arima-model)
   - [LSTM (Long Short-Term Memory) Model](#lstm-long-short-term-memory-model)
4. [⚙️ Usage](#%EF%B8%8F-usage)
5. [📦 Dependencies](#-dependencies)
6. [💬 Acknowledgements](#-acknowledgements)
7. [📫 Contact](#-contact)

---

## 📖 Introduction

The goal of this project is to predict stock prices using time series forecasting models like **AR, MA, ARMA, ARIMA, SARIMA, and LSTM**. By comparing each model's performance, we aim to identify the most effective approach for accurately forecasting future prices.

---

## 🔄 Data Preprocessing

1. **Data Collection**: Obtain historical stock price data, including fields like Date, Open, High, Low, Close, and Volume.
2. **Data Cleaning**:
   - Handle missing values and outliers.
   - Ensure data consistency across records.
3. **Feature Engineering**:
   - Add new features as needed, such as moving averages or lag features.
4. **Data Splitting**:
   - Split the data into training and testing sets to evaluate model performance effectively.
5. **Normalization**:
   - Normalize data for certain models (e.g., LSTM) to enhance model performance and stability.

---

## 📊 Models

### AR (AutoRegressive) Model

The **AR model** uses past values to predict future stock prices by relying on a weighted sum of historical data. It’s effective for data with time-based dependencies.

### MA (Moving Average) Model

The **MA model** smooths data by averaging past forecast errors, helping identify trends and reduce noise.

### ARMA (AutoRegressive Moving Average) Model

**ARMA** combines AR and MA methods, making it suitable for stationary data where both dependencies and errors influence future values.

### ARIMA (AutoRegressive Integrated Moving Average) Model

An enhanced version of ARMA, **ARIMA** accounts for non-stationary data by incorporating differencing, making it ideal for time series with trends or seasonality.

### SARIMA (Seasonal ARIMA) Model

**SARIMA** expands ARIMA by including seasonal elements, perfect for time series with repeating patterns over intervals.

### LSTM (Long Short-Term Memory) Model

A powerful deep learning model, **LSTM** excels at capturing long-term dependencies in data, making it suitable for complex stock price forecasting tasks.

---

## ⚙️ Usage

1. **Install Dependencies**: Start by installing all necessary libraries.
   
2. **Load Data**: Import historical stock price data into your project environment.

3. **Preprocess Data**: Follow preprocessing steps to prepare data for modeling.

4. **Train Models**: Fit each model (AR, MA, ARMA, ARIMA, SARIMA, and LSTM) on the training set.

5. **Evaluate Models**: Use the testing set to measure model accuracy and performance.

6. **Make Predictions**: Apply the trained models to forecast future prices.

---

## 📦 Dependencies

Make sure you have these libraries installed:

```plaintext
numpy
pandas
matplotlib
seaborn
scikit-learn
statsmodels
tensorflow (for LSTM)
yfinance (for obtaining stock data)
```

---

## 💬 Acknowledgements

Special thanks to **[yfinance](https://pypi.org/project/yfinance/)** for providing the historical stock data used in this project. 📅

---

## 📫 Contact

Feel free to reach out for questions or support:

- **Name**: Om Subhash Shrivastav
- **Email**: [omshrivastav1005@gmail.com](mailto:omshrivastav1005@gmail.com)
- **GitHub**: [Omshrivastav12](https://github.com/Omshrivastav12)

Happy recommending! 🌟
