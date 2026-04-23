# SkyCity Restaurants & Bars — Predictive Modeling and Profit Optimization

A machine learning project for revenue prediction and profit optimization across multi-channel restaurant operations, built with Python and Streamlit. The app enables business stakeholders to explore sales forecasts, channel-wise performance, and data-driven pricing strategies through an interactive dashboard.

## Overview

This project applies end-to-end ML techniques to real-world restaurant data from SkyCity Auckland's Restaurants & Bars portfolio. It benchmarks multiple regression models to forecast revenue across dine-in, delivery, and takeaway channels, and surfaces profit optimization insights via an interactive Streamlit dashboard.

## Features

- **Multi-Model Benchmarking** — Trains and compares Linear Regression, Decision Tree, Random Forest, and Gradient Boosting regressors
- **Feature Engineering** — Derives revenue signals from channel type, time periods, pricing, and customer volume features
- **Cross-Validation & Evaluation** — Models evaluated using RMSE, MAE, and R² with k-fold cross-validation to prevent overfitting
- **Profit Optimization** — Simulates channel-mix and pricing scenarios to identify maximum-margin configurations
- **Interactive Dashboard** — Streamlit app with Plotly visualizations for exploring predictions and business scenarios in real time

## Project Structure

```
sky-city-restaurants/
├── SkyCity Auckland Restaurants & Bars - SkyCity Auckland Restaurants & Bars.csv  # Dataset
├── app.py               # Streamlit dashboard application
├── requirements.txt     # Python dependencies
└── README.md
```

## Streamlit Dashboard
https://nhm9rrocjxvuzewrz4ngpm.streamlit.app/

## Dataset

`SkyCity Auckland Restaurants & Bars.csv` — Real-world operational data from SkyCity Auckland's food & beverage outlets, including revenue, channel, pricing, and outlet-level attributes.

**Key features used:**
- `Channel` — Dine-in, Delivery, Takeaway
- `Outlet` — Restaurant/bar name
- `Revenue` — Target variable
- `Price`, `CustomerVolume`, `TimePeriod`, and more

## Setup & Usage

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/Rohan473/sky-city-restaurants.git
cd sky-city-restaurants
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

## Dependencies

| Package | Purpose |
|---------|---------|
| streamlit | Interactive dashboard |
| scikit-learn | ML models and evaluation |
| pandas | Data manipulation |
| numpy | Numerical computing |
| matplotlib / seaborn | Data visualization |
| plotly | Interactive charts |

## Results

The project benchmarks models on held-out test data and selects the best performer by R² and RMSE. Feature importance analysis highlights the most influential drivers of revenue, enabling targeted business decisions around pricing and channel allocation.

## Relevance

The techniques used here — structured data preprocessing, multi-model comparison, cross-validation, and feature importance analysis — are directly transferable to risk scoring, credit modeling, and fraud detection workflows in fintech.

## Author

**Rohan M Patil**  
B.Tech CSE, Indian Institute of Information Technology Guwahati  
[GitHub](https://github.com/Rohan473) | [Email](mailto:rohan.patil22b@iiitg.ac.in)
