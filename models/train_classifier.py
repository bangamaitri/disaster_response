# core module
import pandas as pd
import pathlib
import nltk

# sklearn pipelines
from sklearn.pipeline import Pipeline, FeatureUnion

# sklearn model selection
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report

# database module
from sqlalchemy import create_engine

# text module
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# sklearn model training
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier


nltk.download(["stopwords", "punkt", "wordnet", "averaged_perceptron_tagger"])


def load_data(database_filepath=None):
    """Load data from SQLite DB

    Arguments:
    databse_filepath - path of SQLite destination DB

    Output:
    data dictionary object consisting of -
    X_train and y_train - training dataset
    X_test and y_test - testing dataset
    category_names - list of categories names
    """
    db_path = pathlib.Path(database_filepath)
    tb_name = db_path.stem + "_table"

    client = create_engine("sqlite:///" + db_path.as_posix())

    # load the previosuly saved table into a dataframe
    dataframe = pd.read_sql_table(tb_name, client)
    print(f"Dataframe (row, coloumns) : {dataframe.shape}")

    # defining X and y
    X, y = dataframe["message"], dataframe.iloc[:, 4:]
    dataframe_split = train_test_split(X, y, test_size=0.2)

    # return data object
    data = {
        "category": y.columns,
        "X_train": dataframe_split[0],
        "X_test": dataframe_split[1],
        "y_train": dataframe_split[2],
        "y_test": dataframe_split[3],
    }
    print(f"Training set has {data['X_train'].shape} samples.")
    print(f"Testing set has {data['X_test'].shape} samples.")

    return data


def tokenize(text):
    """Tokenize the text messages

    Arguments:
    text - text message to be tokenized
    Output:
    clean_tokens - list of tokens extracted from given text
    """
    # replacing urls with placeholder
    url_place_holder_string = "urlplaceholder"
    url_regex = (
        "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )

    # local module import
    import re

    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, url_place_holder_string)
    tokens = word_tokenize(text)

    # removing stop words
    tokens = [t for t in tokens if t not in stopwords.words("english")]

    # lemmatizing the text and saving the final tokens in a list
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model(data=None):
    """Builds the ML model after performing grid search

    Arguments:
    data dictionary object consisting of -
    X_train and y_train - training dataset
    X_test and y_test - testing dataset
    category_names - list of categories names
    Output:
    ML model after performing grid search
    """
    # defining pipeline having the classifier as RandomForestClassifier.
    # Previously used Naive Bayes but this provides better results for various categories.

    pipeline_clean = Pipeline(
        [
            (
                "vect",
                CountVectorizer(tokenizer=tokenize, token_pattern=None),
            ),
            ("tfidf", TfidfTransformer()),
        ]
    )

    pipeline = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "text_pipeline",
                            pipeline_clean,
                        )
                    ]
                ),
            ),
            ("clf", MultiOutputClassifier(RandomForestClassifier())),
        ]
    )

    # defining parameters for grid search
    parameter_grid = {
        "features__text_pipeline__vect__max_features": [1000, 3000],
        "features__text_pipeline__vect__ngram_range": ((1, 1), (1, 2)),
        "clf__estimator__n_estimators": [10, 20],
    }

    # training
    search = GridSearchCV(
        pipeline, parameter_grid, verbose=2, cv=5, n_jobs=1, return_train_score=True
    )
    search.fit(data["X_train"], data["y_train"])

    # model selection
    model = search.best_estimator_
    print(pd.DataFrame(search.cv_results_))

    # return best model
    return model


def evaluate_model(data, model):
    """Evaluates the model by predicting it on the test set and prints out the model performance consisting of the f1 score, precision and recall for each category.

    Arguments:
    model - the ML model
    data dictionary object consisting of -
    X_train and y_train - training dataset
    X_test and y_test - testing dataset
    category_names - list of categories names
    """
    # preciting the model on test features
    y_pred = model.predict(data["X_test"])

    # determining the classification report and defining zero_division as 0 to avoid the warning
    class_report = classification_report(
        data["y_test"], y_pred, target_names=data["category"], zero_division=0
    )
    print(class_report)


def save_model(model, model_filepath):
    """Saves the ML model as a pickle file

    Arguments:
    model - ML model
    model_filepath - path of the .pkl file to be saved
    """
    # local module import
    import joblib

    model_path = pathlib.Path(model_filepath)
    joblib.dump(model, open(model_path, "wb"))


def main():
    # local module import
    import sys

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print("\nLoading data...\n    DATABASE: {}".format(database_filepath))
        data = load_data(database_filepath)

        print("\nTraining model...")
        model = build_model(data)

        print("\nEvaluating model...")
        evaluate_model(data, model)

        print("\nSaving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print("\nJob Completed!")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/disaster_response.db classifier.pkl"
        )


if __name__ == "__main__":
    main()
