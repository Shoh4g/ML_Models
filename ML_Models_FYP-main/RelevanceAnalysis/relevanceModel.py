from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import spacy

# Load BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Load Spacy English model
nlp = spacy.load("en_core_web_sm")


def polarity_scores_roberta(example):
    """
    Get sentiment scores for the given text using the roberta model.

    Parameters:
    example (str): Input text.

    Returns:
    float: Sentiment score.
    """
    # Importing necessary libraries
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from scipy.special import softmax
    import ssl
    import nltk

    # BERT model and tokenizer
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download('all')

    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    total = scores_dict['roberta_neg'] + scores_dict['roberta_pos']
    return total


def calculate_similarity(text1, text2):
    """
    Calculate cosine similarity between two texts using BERT embeddings.

    Parameters:
    text1 (str): First text.
    text2 (str): Second text.

    Returns:
    float: Cosine similarity between the embeddings.
    """
    maxlength = max(len(text1.split()), len(text2.split()))
    input_ids1 = tokenizer.encode(text1, add_special_tokens=True, max_length=maxlength, truncation=True,
                                   padding='max_length')
    input_ids2 = tokenizer.encode(text2, add_special_tokens=True, max_length=maxlength, truncation=True,
                                   padding='max_length')
    input_ids1 = torch.tensor(input_ids1).unsqueeze(0)
    input_ids2 = torch.tensor(input_ids2).unsqueeze(0)
    with torch.no_grad():
        outputs1 = model(input_ids1)
        embeddings1 = outputs1.last_hidden_state
        outputs2 = model(input_ids2)
        embeddings2 = outputs2.last_hidden_state
    embeddings1 = embeddings1[0].numpy()
    embeddings2 = embeddings2[0].numpy()
    similarity = cosine_similarity(embeddings1, embeddings2)
    return similarity[0][0]


def count_action_verbs(text):
    """
    Count the occurrence of action verbs in the text.

    Parameters:
    text (str): Input text.

    Returns:
    int: Count of action verbs.
    """
    action_verbs = ["Prepare", "Plan", "Organize", "Attend", "Engage", "Review", "Practice", "Seek", "Network", "Adapt",
                    "Balance", "Stay Motivated", "Reflect", "Utilize", "Manage", "Take Notes", "Stay Informed",
                    "Achieve", "Learn", "Understand",
                    "Analyze", "Discuss", "Participate", "Collaborate", "Research", "Experiment", "Adapt", "Focus",
                    "Prioritize",
                    "Problem-solve", "Communicate", "Listen", "Support", "Guide", "Clarify", "Simplify",
                    "Demonstrate", "Encourage", "Coach",
                    "Motivate", "Inspire", "Challenge", "Evaluate", "Monitor", "Assess", "Evaluate", "Revise",
                    "Improve", "Master", "Apply",
                    "Share", "Exchange", "Connect", "Relate", "Collaborate", "Integrate", "Assist", "Serve",
                    "Offer", "Recommend", "Suggest",
                    "Advise", "Mentor", "Guide", "Help", "Empower", "Assist", "Familiarize", "Inform", "Direct",
                    "Influence", "Lead", "Inspire",
                    "Encourage", "Promote", "Facilitate", "Foster", "Advocate", "Support", "Nurture", "Empathize",
                    "Inspire", "Motivate",
                    "Resolve", "Tackle", "Overcome", "Face", "Conquer", "Manage", "Maintain", "Handle", "Address",
                    "Cope", "Adapt", "Flourish",
                    "Thrive", "Excel", "Achieve", "Succeed", "Triumph"]
    doc = nlp(text)
    verbs_in_text = [token.text for token in doc if token.text in action_verbs]
    return len(verbs_in_text)


def calculate_text_ratio(text1, text2):
    """
    Calculate the ratio of length of text2 to text1.

    Parameters:
    text1 (str): First text.
    text2 (str): Second text.

    Returns:
    float: Ratio of length of text2 to text1.
    """
    return len(text2) / len(text1)


def main(k):
    """
    Main function to calculate relevance scores of reviews.

    Parameters:
    k (int): Number of top relevant reviews to display.
    """
    json_file_path = 'reviews.json'
    try:
        with open(json_file_path, 'r') as file:
            json_data = json.load(file)
    except json.decoder.JSONDecodeError as e:
        print("Error decoding JSON:", e)
        print("JSON file path:", json_file_path)
        raise 

    course_description = json_data["afm102"]['description']
    course_reviews = json_data["afm102"]['reviews']
    relevance_scores = {}

    for review in course_reviews:
        score = calculate_similarity(course_description, review) + count_action_verbs(review) + calculate_text_ratio(
            course_description, review) + polarity_scores_roberta(review)
        relevance_scores[review] = score

    relevance_scores = dict(sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True))

    print()
    print("Course Description:", course_description)
    print()
    print("Top", k, "relevant reviews are: ")
    print()
    i = 1
    for key, value in relevance_scores.items():
        if i > k:
            break
        print(i, " : ", key)
        print()
        i += 1


if __name__ == "__main__":
    main(5)
