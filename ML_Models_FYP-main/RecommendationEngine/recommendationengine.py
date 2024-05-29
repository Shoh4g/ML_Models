import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

def recommend_courses():
    """
    Generate recommendations for courses based on student's profile and available courses.
    
    Returns:
    tuple: A tuple containing two elements - a dictionary of recommended courses categorized by type and a list of elective courses to be taken.
    """
    with open('studyplan.json') as json_file:
        original_data = json.load(json_file)

    with open('studentProfile.json') as json_file:
        student_transcript = json.load(json_file)

    # Determine the current year and next year based on the transcript
    current_year = None
    if student_transcript.get("first_year") and not student_transcript.get("second_year"):
        current_year = "first_year"
        next_year = "second_year"
    elif student_transcript.get("second_year") and not student_transcript.get("third_year"):
        current_year = "second_year"
        next_year = "third_year"
    elif student_transcript.get("third_year") and not student_transcript.get("fourth_year"):
        current_year = "third_year"
        next_year = "fourth_year"
    else:
        return "Student has completed all years."

    # Retrieve courses available for the next academic year    
    next_year_courses = original_data.get(next_year)
    electives_to_be_taken = []

    # Filter out courses already completed by the student
    for courses in student_transcript.get(current_year, {}).values():
        for course in courses:
            if course in next_year_courses.values():
                next_year_courses = list(filter(lambda x: x != course, next_year_courses))

    # If elective courses are available, recommend them
    for elective_taken in student_transcript[current_year]["discipline_elective_courses"]:
        electives_to_be_taken.append(give_rec(elective_taken))
        
    next_year_courses.popitem()

    return (next_year_courses, electives_to_be_taken)


import pandas as pd
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
    recommendations = recommend_courses()

    for course_type, courses in recommendations[0].items():
        print(course_type + ":")
        for course in courses:
            print("-", course)
            
    print(recommendations[1])