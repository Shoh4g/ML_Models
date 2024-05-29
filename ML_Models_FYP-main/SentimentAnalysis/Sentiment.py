import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax
import ssl
import nltk

def load_model_and_tokenizer(model_name):
    """
    Load the sentiment analysis model and tokenizer.

    Parameters:
    model_name (str): Name of the pre-trained model.

    Returns:
    transformers.AutoModelForSequenceClassification, transformers.AutoTokenizer: Model and Tokenizer objects.
    """
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download('all')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    return model, tokenizer

def get_sentiment_scores(model, tokenizer, example):
    """
    Get sentiment scores for the given text using a pre-trained model.

    Parameters:
    model (transformers.AutoModelForSequenceClassification): Pre-trained model.
    tokenizer (transformers.AutoTokenizer): Tokenizer object.
    example (str): Input text.

    Returns:
    dict: Dictionary containing sentiment scores.
    """
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    return scores_dict

def main():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    model, tokenizer = load_model_and_tokenizer(model_name)
    
     # This part can be removed since we dont have to use it anymore
    while True:
        inp = input("Do you want to test the Sentiment of a review? (Y/N): ")
        if inp.lower() == "n":
            break
        elif inp.lower() == "y":
            user_input = input("Enter a review: ")
            scores = get_sentiment_scores(model, tokenizer, user_input)
            print()
            print(user_input)
            print()
            print(scores)
            print()

if __name__ == "__main__":
    main()
