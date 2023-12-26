# core module
import json
import pathlib
import pandas as pd
import joblib

# text module
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

# flask
from flask import Flask
from flask import render_template, request

# visualization libraries
import plotly
from plotly.graph_objs import Bar

# database module
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


database_filepath = "../data/disaster_response.db"
model_filepath = "../models/classifier.pkl"

db_path = pathlib.Path(database_filepath)
tb_name = db_path.stem + "_table"

client = create_engine("sqlite:///" + db_path.as_posix())

# load the previosuly saved table into a dataframe
dataframe = pd.read_sql_table(tb_name, client)

# loading the model
model_path = pathlib.Path(model_filepath)
model = joblib.load(model_path)


# index webpage to display some visuals and to receive user input text for model
@app.route("/")
@app.route("/index")
def index():
    # extracting data needed for visuals
    genre_count = (
        dataframe.groupby("genre").count()["message"].sort_values(ascending=False)
    )
    genre_names = list(genre_count.index)

    category_names = dataframe.iloc[:, 4:].columns
    category_count = sorted((dataframe.iloc[:, 4:] != 0).sum().values, reverse=True)

    df_social = dataframe.loc[dataframe["genre"] == "news"]
    social_category_count = sorted(
        (df_social.iloc[:, 4:] != 0).sum().values, reverse=True
    )

    # creating the visuals
    graphs = [
        # GRAPH 1 - genre graph
        {
            "data": [Bar(x=genre_names, y=genre_count)],
            "layout": {
                "title": "Distribution of Message Genres",
                "yaxis": {"title": "No. of Messages"},
                "xaxis": {"title": "Genre"},
            },
        },
        # GRAPH 2 - category graph
        {
            "data": [Bar(x=category_names, y=category_count)],
            "layout": {
                "title": "Distribution of Message Categories",
                "yaxis": {"title": "No. of Messages"},
                "xaxis": {"title": "Category", "tickangle": 35},
            },
        },
        # GRAPH 3 - social messages graph
        {
            "data": [Bar(x=category_names, y=social_category_count)],
            "layout": {
                "title": "Distribution of Message Categories for Social Messages",
                "yaxis": {"title": "No. of Social Messages"},
                "xaxis": {"title": "Category", "tickangle": 35},
            },
        },
    ]

    # encoding plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # rendering web page with plotly graphs
    return render_template("master.html", ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route("/go")
def go():
    # save user input in query
    query = request.args.get("query", "")

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(dataframe.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        "go.html", query=query, classification_result=classification_results
    )


def main():
    app.run(host="0.0.0.0", port=3001, debug=True)


if __name__ == "__main__":
    main()
