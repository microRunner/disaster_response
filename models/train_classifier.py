import sys
import pickle
from sqlalchemy import create_engine
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
import string
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
import string
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.multioutput import MultiOutputClassifier


def load_data(database_filepath):
    """
    Loads data from the SQL Database
    Args:
        database_filepath: The path to SQL databse
    Returns:
        X: Features
        y: Targets    
        category_names: The names of the categories
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages', engine)
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    category_names = y.columns.values
    return X, y, category_names


def tokenize(text):
    """
    Tokenize a sentence into words. 
        Convert to lower case
        Remove punctuations. 
        Remove stop words
        Lemmatize
     Args:
        text: Raw Sentence.    
     Returns:
        words: Tokenized output.    
    """    
    # lower case
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    #Tokenize
    words = word_tokenize(text)
    
    #Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    #Lemmatize
    wnl = WordNetLemmatizer()
    words = [wnl.lemmatize(word) for word in words]
    return words


def build_model():
    """
    Builds the ML Model. 
        Create pipeline
        Define Parameters
        Define GridSearchCV
     Returns:
        cv: GridSearch Object.    
    """
    
    pipeline = Pipeline([
    ('count_vect', CountVectorizer(tokenizer=tokenize)),
    ('tfid', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])
    parameters = {
        'clf__estimator__n_estimators':[50, 100]
    }

    cv = GridSearchCV(pipeline, param_grid = parameters)
    
    return cv 
    

def evaluate_model(model, X_test, y_test, category_names):
    """
    Get classification report.
        Print classification for each category. 
        Print mean of f1 score for all categories.
    Input:
        Y_test - the actual labels for our test set
        Y_pred - the predicted labels for our test set
    """
    y_pred = pd.DataFrame(model.predict(X_test), columns = category_names)
    f1_score_dict = {}
    for col in y_test.columns.values:
        print('{} {}'.format(col, classification_report(y_test[col], y_pred[col])))
        f1_score_dict[col] = f1_score(y_test[col], y_pred[col], average = 'weighted')
    
    mean_f1_score = np.mean(list(f1_score_dict.values()))    
    print('Mean F1 score is {}'.format(mean_f1_score))


def save_model(model, model_filepath):
    """
    Save the ML model. 
    Args:
        model: The trained ML model.
        model_filepath: The path where it should be saved
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()