# Disaster Response Pipeline Project

## Project Background.
This project creates an NLP ML model to classify disaster response messages.
It uses a repository of labelled messages provided by figure eight. 


## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Screenshots of the App

### Loading screen

### Classification of data

## Modules

### ETL Pipeline:
#### Code: 
- data/process_data.py
#### Key steps:
- Load and join data from csv.
- Process Data. 
- Clean for duplicates.
- Write to a SQL database.

### NLP Pipeline 
#### Code:
- model/train_classifier.py
#### key Steps:
- Load data from SQL database.
- Build the ML Model using pipelines. 
- Train the model and evaludate performance.
- Save the model.

### App
#### Code
- app/run.py
#### Key Steps:
- Load the data and the model.
- Visualize the training data.
- Predict the categories based on user input.


