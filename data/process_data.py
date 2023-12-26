# core module
import pandas as pd
import pathlib

# database module
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Loads the messages and categories data and stores the
    result in a dataframe

    Arguments:
    messages_filepath - path of csv file consisting of messages
    categories_filepath - path of csv file consisting of categories
    Output:
    dataframe - combined data consisting of messages and categories"""

    # reading messages and categories files
    message_path = pathlib.Path(messages_filepath)
    categories_path = pathlib.Path(categories_filepath)
    messages = pd.read_csv(message_path)
    categories = pd.read_csv(categories_path)

    # merging messages and categories file on id column
    dataframe = pd.merge(messages, categories, on="id")
    return dataframe


def clean_data(dataframe=None):
    """Cleans the combined dataframe

    Arguments:
    dataframe - dataframe consisting of messages and categories
    Output:
    dataframe - cleaned up dataframe consisting of messages and categories"""

    # creating a dataframe of the 36 individual category columns
    categories = dataframe["categories"].str.split(pat=";", expand=True)
    # using first row to extract column names for categories
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split("-")[0])
    # renaming the columns of `categories`
    categories.columns = category_colnames

    # converting category value to 0 or 1
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)

    # replacing category column in df with new category columns
    dataframe = dataframe.drop("categories", axis=1)
    dataframe = pd.concat([dataframe, categories], axis=1)

    # removing duplicates
    dataframe = dataframe.drop_duplicates()

    # deleting child_alone column as it consists of all zeroes
    dataframe = dataframe.drop(["child_alone"], axis=1)

    # converting value 2 in related column to 1 as majority of the rows have value 1
    dataframe["related"] = dataframe["related"].map(lambda x: 1 if x == 2 else x)

    return dataframe


def save_data(dataframe, database_filename):
    """Saves the clean dataframe in a SQLite DB

    Arguments:
    dataframe - cleaned up dataframe consisting of messages and categories
    database_filename - path of SQLite DB
    """
    # creating SQLite engine object and defining table name
    db_path = pathlib.Path(database_filename)
    client = create_engine("sqlite:///" + db_path.as_posix())
    tb_name = db_path.stem + "_table"
    # saving the cleaned dataset into SQLite DB
    dataframe.to_sql(tb_name, client, index=False, if_exists="replace")


def main():
    # local module import
    import sys

    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(
            "Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(
                messages_filepath, categories_filepath
            )
        )
        dataframe = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        dataframe = clean_data(dataframe)

        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(dataframe, database_filepath)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python process_data.py "
            "disaster_messages.csv disaster_categories.csv "
            "disaster_response.db"
        )


if __name__ == "__main__":
    main()
