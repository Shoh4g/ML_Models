import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

def give_rec(title):
    """
    Generate recommendations for a given course title.

    Parameters:
    title (str): Title of the course.

    Returns:
    pd.Series: Series of recommended courses.
    """
    # Read the courses data
    courses = pd.read_csv("cs_courses - Sheet1.csv")
    courses['Overview'] = courses['Overview'].fillna('')
    
    # Vectorize the course overviews
    tfv = TfidfVectorizer(min_df=3, max_features=None,
                          strip_accents='unicode', analyzer='word',
                          token_pattern=r'\w{1,}',
                          ngram_range=(1, 3),
                          stop_words='english')
    tfv_matrix = tfv.fit_transform(courses['Overview'])
    
    # Compute sigmoid kernel
    sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
    
    # Map course titles to their indices
    indices = pd.Series(courses.index, index=courses['Course_name']).drop_duplicates()
    
    # Get index of the input course title
    idx = indices[title]
    
    # Get sigmoid scores for courses
    sig_scores = list(enumerate(sig[idx]))
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
    sig_scores = sig_scores[1:6]  # Exclude the input course itself
    course_indices = [i[0] for i in sig_scores]
    
    # Return recommended courses
    return courses['Course_name'].iloc[course_indices]


if __name__ == "__main__":
    # List of courses to get recommendations for
    courses = ["Machine learning", "Artificial intelligence", "Introduction to data structures and algorithms",
               "Implementation testing and maintenance of software systems", "Computer game design and programming"]
    
    # Generate recommendations for each course and print
    for course in courses:
        recommendations = give_rec(course)
        print("Course: ", course)
        print(recommendations)
        print()
