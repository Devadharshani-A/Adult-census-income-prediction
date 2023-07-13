# Adult-census-income-prediction
This model predicts/classifies a person's income based on some input features. A GUI is also created using the Gradio library.

This project aims to predict the salary level of individuals based on various demographic and socioeconomic factors using machine learning techniques. The dataset used in this project is the Adult Census dataset from kaggle.

## Dataset
The dataset contains information about individuals such as age, education level, marital status, occupation, race, sex, and other attributes. The target variable is the salary level, which is classified as either ">50K" or "<=50K".

The python file, dataset and video of the user inteface is uploaded. 

## Data preprocessing
The dataset undergoes several preprocessing steps, including handling missing values, encoding categorical variables, and scaling numerical features. Exploratory data analysis (EDA) is performed to gain insights into the data and visualize the relationships between different variables.

## Model Training
Several classification models are implemented, including Random Forest, Gaussian Naive Bayes, Decision Tree, AdaBoost, XGBoost, and MLP. The models are trained using the training dataset and evaluated using cross-validation. The best-performing model based on accuracy is selected for further evaluation.

## Model Evaluation
The selected model is evaluated on the test dataset to assess its performance. Metrics such as accuracy, confusion matrix, and classification report are provided to evaluate the model's predictive power.

## Predicting Salary
The trained model can be used to predict the salary level of individuals based on their demographic and socioeconomic information. A user-friendly GUI (Graphical User Interface) is provided to interactively input the required data and obtain the salary prediction.

Contributions to this project are welcome!
