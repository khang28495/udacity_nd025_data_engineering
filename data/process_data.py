import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load and combine message and category data from csv files.
    
    Args:
        messages_filepath: path of message CSV file
        categories_filepath: path of category CSV file
    Return:
        a dataframe which is merged the messages and categories datasets
    '''
    
    message_df = pd.read_csv(messages_filepath)
    category_df = pd.read_csv(categories_filepath)
    
    return pd.merge(message_df, category_df, on='id')


def clean_data(df):
    '''
    Clean dataset.
    
    Args:
        df: a dataframe need to be cleaned
    Return:
        cleaned dataframe
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True) 
    
    # extract a list of new column names for categories.
    row = categories.loc[0, :]
    categories.columns = [col.split('-')[0] for col in row]
    
    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1].replace('2', '1')
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # Replace categories column in df with new category columns
    output_df = df.drop(columns=['categories'])
    output_df = pd.concat([output_df, categories], axis=1)
    
    #dropping related observations
    output_df = df.drop(columns=['related'])
    
    # drop duplicates
    output_df.drop_duplicates(inplace=True)   
    return output_df


def save_data(df, database_filename):
    '''
    Save data into sqlite database.
    
    Args:
        df: a dataframe need to be saved
        database_filename: database file
    Return:
        None
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages_categories', engine, index=False, if_exists='replace')  

    #check data
    data = pd.read_sql('select count(*) from messages_categories', engine)
    print(data)  


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