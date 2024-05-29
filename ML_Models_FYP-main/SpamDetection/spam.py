import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_data(file_path):
    """
    Load data from a CSV file.

    Parameters:
    file_path (str): Path to the CSV file.

    Returns:
    pandas.DataFrame: Loaded DataFrame.
    """
    return pd.read_csv(file_path)


def preprocess_data(df):
    """
    Preprocess the DataFrame by handling missing values and converting labels to numeric values.

    Parameters:
    df (pandas.DataFrame): Input DataFrame.

    Returns:
    pandas.DataFrame: Processed DataFrame.
    """
    df = df.where((pd.notnull(df)), '')
    df.loc[df['Category'] == 'spam', 'Category'] = 0
    df.loc[df['Category'] == 'ham', 'Category'] = 1
    return df


def train_model(X_train_features, Y_train):
    """
    Train a logistic regression model.

    Parameters:
    X_train_features (scipy.sparse.csr_matrix): Features of the training data.
    Y_train (numpy.ndarray): Labels of the training data.

    Returns:
    sklearn.linear_model.LogisticRegression: Trained logistic regression model.
    """
    model = LogisticRegression()
    model.fit(X_train_features, Y_train)
    return model


def predict_review(model, feature_extraction, review):
    """
    Predict whether a review is spam or not.

    Parameters:
    model (sklearn.linear_model.LogisticRegression): Trained logistic regression model.
    feature_extraction (sklearn.feature_extraction.text.TfidfVectorizer): Fitted TfidfVectorizer.
    review (str): Review to be predicted.

    Returns:
    str: Prediction result.
    """
    review = [review]
    input_features = feature_extraction.transform(review)
    prediction = model.predict(input_features)
    return "The Provided review is NOT SPAM" if prediction[0] == 1 else "The Provided review is a SPAM"


def main():
    df = load_data('traindata.csv')
    df = preprocess_data(df)

    X = df['Message']
    Y = df['Category']

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=3)

    feature_extraction = TfidfVectorizer(
        min_df=1, stop_words='english', lowercase=True)

    X_train_features = feature_extraction.fit_transform(X_train)
    X_test_features = feature_extraction.transform(X_test)

    Y_train = Y_train.astype('int')
    Y_test = Y_test.astype('int')

    model = train_model(X_train_features, Y_train)

    prediction_on_training_data = model.predict(X_train_features)
    accuracy_on_training_data = accuracy_score(
        Y_train, prediction_on_training_data)

    print("Accuracy of Training and Testing:")
    print("Training data accuracy:", accuracy_on_training_data * 100)

    prediction_on_test_data = model.predict(X_test_features)
    accuracy_on_test_data = accuracy_score(
        Y_test, prediction_on_test_data)
    print("Testing data accuracy:", accuracy_on_test_data * 100)


    # This part can be removed since we dont have to use it anymore
    while True:
        inp = input("Do you want to test a review? (Y/N): ")
        if inp.lower() == "n":
            break
        elif inp.lower() == "y":
            user_input = input("Enter a review: ")
            print(predict_review(model, feature_extraction, user_input))
        else:
            print("Invalid input!")


if __name__ == "__main__":
    main()
