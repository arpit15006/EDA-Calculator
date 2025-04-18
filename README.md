# Bank Loan Default Prediction

This project provides a comprehensive solution for predicting bank loan defaults based on client application data. It includes data exploration, preprocessing, model building, and a web application for visualizing results and making predictions.

## Problem Statement

The goal of this project is to predict whether a client will default on a loan based on their application information. This helps banks make better lending decisions and manage risk effectively.

## Dataset

The dataset consists of three main files:
- `application_data.csv`: Contains information about loan applications
- `previous_application.csv`: Contains information about previous loan applications
- `columns_description.csv`: Contains descriptions of the columns in the datasets

## Features

- **Data Exploration**: Visualize and understand the dataset
- **Data Preprocessing**: Handle missing values, encode categorical variables, and create new features
- **Model Building**: Train and evaluate multiple machine learning models
- **Web Application**: Interactive dashboard for visualizing results and making predictions

## Project Structure

```
├── app.py                  # Streamlit web application
├── README.md               # Project documentation
├── models/                 # Saved models
├── src/
│   ├── data/
│   │   ├── data_loader.py  # Functions for loading data
│   │   └── preprocessor.py # Functions for preprocessing data
│   ├── models/
│   │   └── model_trainer.py # Functions for training and evaluating models
│   ├── visualization/
│   │   └── eda.py          # Functions for exploratory data analysis
│   └── utils/              # Utility functions
```

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```
   python3 -m venv venv
   ```
3. Activate the virtual environment:
   ```
   source venv/bin/activate
   ```
4. Install the required packages:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn streamlit plotly
   ```

## Usage

1. Run the Streamlit application:
   ```
   streamlit run app.py
   ```
2. Open your browser and navigate to the URL shown in the terminal (usually http://localhost:8501)
3. Use the sidebar to navigate between different pages:
   - **Home**: Overview of the project
   - **Data Exploration**: Visualize and understand the dataset
   - **Model Performance**: Evaluate the performance of different machine learning models
   - **Make Prediction**: Predict loan default risk for new clients

## Models

The project uses the following machine learning models:
- Logistic Regression
- Random Forest
- Gradient Boosting

## Results

The models are evaluated using the following metrics:
- ROC AUC
- Precision
- Recall
- F1 Score

## License

This project is licensed under the MIT License - see the LICENSE file for details.
