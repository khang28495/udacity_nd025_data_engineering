import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/messages_response.db')
df = pd.read_sql_table('messages_categories', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # extract data needed for message categories bar chart - second graph
    category_names = sorted(df.columns[-36:], key=lambda category : df[category].sum(), reverse=True)
    category_counts = [round(df[category].sum()/df.shape[0] * 100 , 2) for category in category_names] #calculate the percentage of each categories
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    # create visuals
    graphs = [
        # FIRST GRAPH
        {
            'data': [
                Bar(
                    y=genre_names, 
                    x=genre_counts, 
                    orientation='h'
                )
            ],

            'layout': {
                'title': 'The Number Of Messages By Genres'
            }
        },
        
        # SECOND GRAPH
        {
            'data': [
                Bar(
                    x=[" ".join([word.capitalize() for word in category.split("_")]) for category in category_names], #format category names
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'The Percentage Of Occurrence Of Message Categories',
                'yaxis': {
                    'title': "Percent (%)"
                }
            }
        },
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()