import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads the data from CSVs
    Args:
        messages_filepath: Filepath to main data. 
        categories_filepath: Filepath to the meta data. 
    Returns:
        df: combined dataframe.     
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = messages.merge(categories, how = 'left', on = 'id')
    return df
    


def clean_data(df):
    """
    Clean the raw_data.
        Clean the category information
            Convert text to get the column names. 
            Separate category 'class'
        Remove duplicates.
    Args:
        df: Raw Dataframe
     Return:
        df: clean dataframe. 
    """
    categories_separated = df['categories'].str.split(';', expand = True)
    # select the first row of the categories dataframe
    row = categories_separated.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])

    # rename the columns of `categories`
    categories_separated.columns = category_colnames
    for column in categories_separated:
    # set each value to be the last character of the string
        categories_separated[column] = categories_separated[column].apply(lambda x: x[-1:])

        # convert column from string to numeric
        categories_separated[column] = categories_separated[column].apply(lambda x: int(x))
    df = df.drop('categories', axis = 1)
    df = pd.concat([df, categories_separated], axis = 1)
    
    df.drop_duplicates(inplace = True)
    
    return df



def save_data(df, database_filename):
    """
    Save the data to a SQL database.
    Args:
        df: Cleaned Dataframe.
        database_filename: Path where data should be saved to.
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()