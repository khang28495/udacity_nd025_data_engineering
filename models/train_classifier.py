import sys
# import libraries
import pickle
import pandas as pd
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from sqlalchemy import create_engine


import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def load_data(database_filepath):
    '''
    This function loads data from database.
    
    Args:
        database_filepath: path of database file
    Return:
        X: messages data
        Y: categories data
        category_names: list of category names
    '''
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages_categories', engine)
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    
    category_names = Y.columns
    
    return X, Y, category_names

def tokenize(text):
    '''
    A tokenization function using nltk to case normalize, lemmatize, and tokenize text.
    
    Args:
        text: text needed to process
    Return:
        tokens
    '''
    # Normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Lemmatize and remove stop words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stopwords.words('english')]
    return tokens


def build_model():
    '''
    This function builds a NLP pipeline.
    
    Args: None
    Return:
        pipeline: NLP pipeline
    '''
    #create pipeline
    pipeline = Pipeline(steps=[     
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=10))),
    ])

    # specify parameters for grid search
    parameters = {
        #'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__n_estimators': [5, 10]
        #'clf__estimator__min_samples_split': [2, 3, 4]
    }

    # create grid search object
    cv = GridSearchCV(pipeline, parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Show the accuracy, precision, and recall of the tuned model for each categories.
    
    Args:
        model: model to evaluate
        X_test: X test dataset
        Y_test: Y test dataset
        category_names: list of category names
        
    Return: None
    '''
    
    scores = dict()
    Y_predict = model.predict(X_test)
    
    for i, col in enumerate(Y_test):
        print(col)
        print(classification_report(Y_test[col], Y_predict[:, i]))


def save_model(model, model_filepath):
    '''
    Save the model as a pickle    file.
      Args:
        model: model to save
        model_filepath: file path to save
        
    Return: None
    '''
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


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