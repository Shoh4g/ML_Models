import math
import json

def load_courses_data(file_path='coursesgrade.json'):
    """
    Load courses data from a JSON file.

    Parameters:
    file_path (str): Path to the JSON file containing courses data.

    Returns:
    dict: Dictionary containing courses data.
    """
    with open(file_path) as json_file:
        courses_data = json.load(json_file)
    return courses_data

def find_closest_courses(preferences, courses_data, num_courses=5):
    """
    Find the closest courses based on user preferences.

    Parameters:
    preferences (dict): User preferences for course selection.
    courses_data (dict): Dictionary containing courses data.
    num_courses (int): Number of closest courses to return.

    Returns:
    list: List of closest courses.
    """
    closest_courses = []
    
    for course, scores in courses_data.items():
        distance = math.sqrt(sum((preferences[metric] - scores[metric]) ** 2 for metric in preferences))
        closest_courses.append((course, distance))
    
    closest_courses.sort(key=lambda x: x[1])
    
    result = [course[0] for course in closest_courses[:num_courses]]

    return result 

if __name__ == "__main__":
    # Load user preferences
    user_preferences = {
        "Difficulty": 1,
        "Usefulness": 5,
        "Workload": 3,
        "Good grade": 4
    }

    # Load courses data
    courses_data = load_courses_data()

    # Find the closest courses
    closest_courses = find_closest_courses(user_preferences, courses_data)
    print(closest_courses)
