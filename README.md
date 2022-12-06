# Disaster Response Project

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#project_motivation)
3. [How to run](#how_run)
4. [File Descriptions](#file_descriptions)
5. [Licensing, Authors, and Acknowledgements](#licensing)
   
## 1. Installation <a name='installation'></a>
This project requires:
* Python (>= 3.6.3)
* Pandas (>= 0.23.3)
* NLTK (>= 3.2.5)
* Scikit-learn (>= 0.19.1)
* SQLAlchemy (>= 1.2.19)
* Flask framework (>= 0.12.5)
* Plotly (>= 2.9.0)
* Bootstrap (>= 3.3.7)


## 2. Project Motivation <a name='project_motivation'></a>

In this project, I'm going to analyze real disaster data from Appen to build a model for an API that classifies disaster messages.

This project will include  a machine learning pipeline to categorize these events, and a web app where an emergency worker can input a new message and get classification results in several categories as well as display visualizations of the data.


## 3. How to run <a name='how_run'></a>
1. Run the following commands in the project's root directory to set up your database and model.

- To run ETL pipeline that cleans data and stores in database:

```
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/messages_response.db
```

 - To run ML pipeline that trains classifier and saves:

```
python models/train_classifier.py data/messages_response.db models/classifier.pkl
```

2. Go to app directory: 
```
cd app
```

3. Run your web app:   
```
python run.py
```

4. Go to http://0.0.0.0:3000/


## 4. File Descriptions <a name='file_descriptions'></a>
The coding for this project can be completed using the Project Workspace IDE provided. Here's the file structure of the project:
```
- app
|   - template
|   |   - master.html  # main page of web app
|   |   - go.html  # classification result page of web app
|   - run.py  # Flask file that runs app

- data
|   - disaster_categories.csv  # data to process 
|   - disaster_messages.csv  # data to process
|   - process_data.py
|   - DisasterResponse.db   # database to save clean data to

- models
|   - train_classifier.py
|   - classifier.pkl  # saved model 

- README.md
```


## 5. Licensing, Authors, and Acknowledgements <a name='licensing'></a>

Must give credit to Appen for the data. You can find the Licensing for the data and other descriptive information at the Appen website [here](https://appen.com/).

Otherwise, feel free to use the code here as you would like!