import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel


studyplan = {
  "first_year": {
    "engineering_core_courses": [
      "MATH1851 Calculus and ordinary differential equations",
      "MATH1853 Linear algebra, probability and statistics",
      "ENGG1300 Fundamental mechanics",
      "ENGG1310 Electricity and electronics",
      "ENGG1320 Engineers in the modern world",
      "ENGG1330 Computer programming I",
      "ENGG1340 Computer programming II"
    ]
  },
  "second_year": {
    "introductory_discipline_core_courses": [
      "COMP2119 Introduction to data structures and algorithms",
      "COMP2120 Computer organization",
      "COMP2121 Discrete mathematics",
      "COMP2396 Object-oriented programming and Java"
    ],
    "discipline_elective_courses": []
  },
  "third_year": {
    "advanced_discipline_core_courses": [
      "COMP3230 Principles of operating systems",
      "COMP3234 Computer and communication networks",
      "COMP3250 Design and analysis of algorithms",
      "COMP3278 Introduction to database management systems",
      "COMP3297 Software engineering"
    ],
    "discipline_elective_courses": []
  },
  "fourth_year": {
    "discipline_elective_courses": [],
    "capstone_experience": "COMP4801 Final year project",
    "university_requirements": [
      "CAES9542 Technical English for Computer Science"
    ]
  }
}

cs_courses = [
    {
        "ID": 1,
        "Course_name": "Technical English for Computer Science",
        "Overview": "Running alongside Computer Science project based courses, this one semester, 6-credit course will build and consolidate final year CS and Computing and data analytics students\u2019 ability to compose technical reports, and make technical oral presentations. The focus of this course is on helping students to report on the progress of their Final Year Project in an effective, professional manner in both written and oral communication. Topics include accessing, abstracting, analyzing, organizing and summarizing information; making effective grammatical and lexical choices; technical report writing; and technical presentations"
    },
    {
        "ID": 2,
        "Course_name": "Computer programming",
        "Overview": "This is an introductory course in computer programming. Students will acquire basic Python programming skills, including syntax, identifiers, control statements, functions, recursions, strings, lists, dictionaries, tuples and files. Searching and sorting algorithms, such as sequential search, binary search, bubble sort, insertion sort and selection sort, will also be covered."
    },
    {
        "ID": 3,
        "Course_name": "Programming technologies",
        "Overview": "This course covers intermediate to advanced computer programming topics on various technologies and tools that are useful for software development. Topics include Linux shell commands, shell scripts, C/C++ programming, and separate compilation techniques and version control. This is a self-learning course; there will be no lecture and students will be provided with self-study materials. Students are required to complete milestone-based self-assessment tasks during the course. This course is designed for students who are interested in Computer Science / Computer Engineering. "
    },
    {
        "ID": 4,
        "Course_name": "Introduction to data structures and algorithms",
        "Overview": "Arrays, linked lists, trees and graphs; stacks and queues; symbol tables; priority queues, balanced trees; sorting algorithms; complexity analysis."
    },
    {
        "ID": 5,
        "Course_name": "Computer organization",
        "Overview": "Introduction to computer organization and architecture; data representations; instruction sets; machine and assembly languages; basic logic design and integrated devices; the central processing unit and its control; memory and caches; I/O and storage systems; computer arithmetic."
    },
    {
        "ID": 6,
        "Course_name": "Discrete mathematics",
        "Overview": "This course provides students a solid background on discrete mathematics and structures pertinent to computer science. Topics include logic; set theory; mathematical reasoning; counting techniques; discrete probability; trees, graphs, and related algorithms; modeling computation. "
    },
    {
        "ID": 7,
        "Course_name": "Object-oriented programming and Java",
        "Overview": "Introduction to object-oriented programming; abstract data types and classes; inheritance and polymorphism; object-oriented program design; Java language and its program development environment; user interfaces and GUI programming; collection class and iteration protocol; program documentation."
    },
    {
        "ID": 8,
        "Course_name": "Principles of operating systems",
        "Overview": "Operating system structures, process and thread, CPU scheduling, process synchronization, deadlocks, memory management, file systems, I/O systems and device driver, mass-storage structure and disk scheduling, case studies. "
    },
    {
        "ID": 9,
        "Course_name": "Computer architecture",
        "Overview": "Introduction to computer design process; performance and cost analysis; instruction set design; datapath and controller design; pipelining; memory system; I/O design; GPU architecture and programming; introduction to advanced topics."
    },
    {
        "ID": 10,
        "Course_name": "Computer and communication networks",
        "Overview": "Network structure and architecture; reference models; stop and wait protocol; sliding window protocols; character and bit oriented protocols; virtual circuits and datagrams; routing; flow control; congestion control; local area networks; issues and principles of network interconnection; transport protocols and application layer; and examples of network protocols. "
    },
    {
        "ID": 11,
        "Course_name": "Compiling techniques",
        "Overview": "Lexical analysis; symbol table management; parsing techniques; error detection; error recovery; error diagnostics; run-time memory management; optimization; code generation. "
    },
    {
        "ID": 12,
        "Course_name": "Design and analysis of algorithms",
        "Overview": "The course studies various algorithm design techniques, such as divide and conquer, and dynamic programming. These techniques are applied to design novel algorithms from various areas of computer science. Topics include: advanced data structures; graph algorithms; searching algorithms; geometric algorithms; overview of NP-complete problems. "
    },
    {
        "ID": 13,
        "Course_name": "Functional programming",
        "Overview": "The course teaches the basics of functional programming using the language Haskell. The main goal is introduce students to fundamental programming concepts such as recursion, abstraction, lambda expressions and higher-order functions and data types. The course will also study the mathematical reasoning involved in the design of functional programs and techniques for proving properties about functions so defined. With the adoption of lambda expressions recent versions of Java, C++ or C#, functional programming and related programming techniques are becoming increasingly more relevant even for programmers of languages that are not traditionally viewed as functional. This course is important to introduce students to such techniques. "
    },
    {
        "ID": 14,
        "Course_name": "Principles of programming languages",
        "Overview": "Syntax and semantics specification; data types; data control and memory management; expressions, precedence and associativity of operators; control structures; comparative study of existing programming languages; advanced topics such as polymorphism, programming paradigms, exception handling and concurrency. "
    },
    {
        "ID": 15,
        "Course_name": "Artificial intelligence",
        "Overview": "This is an introduction course on the subject of artificial intelligence. Topics include: intelligent agents; search techniques for problem solving; knowledge representation; logical inference; reasoning under uncertainty; statistical models and machine learning. "
    },
    {
        "ID": 16,
        "Course_name": "Computer graphics",
        "Overview": "Overview of graphics hardware, basic drawing algorithms, 2-D transformations, windowing and clipping, interactive input devices, curves and surfaces, 3-D transformations and viewing, hidden-surface and hidden-line removal, shading and colour models, modelling, illumination models, image synthesis, computer animation. "
    },
    {
        "ID": 17,
        "Course_name": "Introduction to database management systems",
        "Overview": "This course studies the principles, design, administration, and implementation of database management systems.  Topics include: entity-relationship model, relational model, relational algebra, database design and normalization, database query languages, indexing schemes, integrity and concurrency control. "
    },
    {
        "ID": 18,
        "Course_name": "Software engineering",
        "Overview": "This course introduces the fundamental principles and methodologies of software engineering.  It covers the software process, and methods and tools employed in the modern software development, with focus on the analysis, design, implementation and testing of contemporary object-oriented systems.  The use   15   of the UML and contemporary frameworks are emphasized.  The course includes a team-based project in which students apply their new knowledge to a full lifecycle of iterative and incremental development. "
    },
    {
        "ID": 19,
        "Course_name": "Legal aspects of computing",
        "Overview": "To introduce students to the laws affecting computing and the legal issues arising from the technology.  Contents include: the legal system of Hong Kong; copyright protection for computer programs; intellectual property issues on the Internet; data privacy; computer-related crimes; codes of professional conduct for computer professionals. "
    },
    {
        "ID": 20,
        "Course_name": "Machine learning",
        "Overview": "This course introduces algorithms, tools, practices, and applications of machine learning. Topics include core methods such as supervised learning (classification and regression), unsupervised learning (clustering, principal component analysis), Bayesian estimation, neural networks; common practices in data pre-processing, hyper-parameter tuning, and model evaluation; tools/libraries/APIs such as scikitlearn, Theano/Keras, and multi/many-core CPU/GPU programming"
    },
    {
        "ID": 21,
        "Course_name": "Quantum information and computation",
        "Overview": "This course offers a gentle introduction to the interdisciplinary field of quantum information and computation. We will start from the basic principles of quantum theory and become familiar with the counterintuitive notions of quantum superposition and entanglement. Once the basics have been covered, we will explore the cornerstones of quantum information theory: quantum cloning machines, quantum teleportation, quantum state discrimination, quantum error correction, quantum cryptography and data compression.  Finally, we will provide an overview of quantum computation and of the main quantum algorithms, including Shor\u2019s algorithm for prime factorization in polynomial time and Grover's quantum search algorithm."
    },
    {
        "ID": 22,
        "Course_name": "Computer vision",
        "Overview": "This course introduces the principles, mathematical models and applications of computer vision. Topics include: image processing techniques, feature extraction techniques, imaging models and camera calibration techniques, stereo vision, and motion analysis. "
    },
    {
        "ID": 23,
        "Course_name": "Electronic commerce technology",
        "Overview": "This course aims to help students to understand the technical and managerial challenges they will face as electronic commerce becomes a new locus of economics activities. Topics include Internet and WWW technology, information security technologies, public-key crypto-systems, public-key infrastructure, electronic payment systems, and electronic commerce activities in different sectors. "
    },
    {
        "ID": 24,
        "Course_name": "Modern technologies on World Wide Web",
        "Overview": "Selected network protocols relevant to the World Wide Web (e.g., HTTP, DNS, IP); World Wide Web; technologies for programming the Web (e.g, HTML, style sheets, PHP, JavaScript, Node.js.; other topics of current interest (AJAX, HTML5, web services, cloud computing). "
    },
    {
        "ID": 25,
        "Course_name": "Advanced database systems",
        "Overview": "The course will study some advanced topics and techniques in database systems, with a focus on the system and algorithmic aspects.  It will also survey the recent development and progress in selected areas.  Topics include: query optimization, spatial-spatiotemporal data management, multimedia and time-series data management, information retrieval and XML, data mining. "
    },
    {
        "ID": 26,
        "Course_name": "Computer game design and programming",
        "Overview": "This course introduces the concepts and techniques for computer game design and development. Topics include: game history and genres, game design process, game engine, audio and visual design, 2D and 3D graphics, physics, optimization, camera, network, artificial intelligence and user interface design. Students participate in group projects to gain hands-on experience in using common game engine in the market."
    },
    {
        "ID": 27,
        "Course_name": "Interactive mobile application design and programming",
        "Overview": "This course aims at introducing the design and development issues of mobile apps. Students will learn the basic principles, constraints and lifecycle of mobile apps. Then they will learn how to use modern object-oriented languages for the development and different design patterns. Next they will learn various development issues such as graphics, touch events, handling of concurrency, sensors, location   17   services and server connection. Students will also participate in both individual assignments and group project to practice ideation, reading, writing, coding and presentation throughout this course.  "
    },
    {
        "ID": 28,
        "Course_name": "Applied deep learning",
        "Overview": "An introduction to algorithms and applications of deep learning.  The course helps students get handson experience of building deep learning models to solve practical tasks including image recognition, image generation, reinforcement learning, and language translation. Topics include: machine learning theory; optimization in deep learning; convolutional neural networks; recurrent neural networks; generative adversarial networks; reinforcement learning; self-driving vehicle. "
    },
    {
        "ID": 29,
        "Course_name": "Advanced algorithm analysis",
        "Overview": " This class introduces advanced mathematical techniques for analyzing the complexity and correctness of algorithms.  NP-complete problems are believed to be not solvable in polynomial time and we study how approximation algorithms could give near optimal solutions. In particular, we will see that probability theory gives us a very powerful tool to tackle problems that are otherwise hard to solve. "
    },
    {
        "ID": 30,
        "Course_name": "Algorithmic game theory",
        "Overview": "Strategic behaviors of users are of increasingly importance in today\u2019s computational problems, from data analysis (where a user may manipulate his data) to routing (where a user may strategically choose a path instead of the one that the algorithm specifies). This is an undergraduate advanced algorithm course that covers various topics at the interface of theoretical computer science and economics, seeking to provide the basic concepts and techniques, both economic and algorithmic ones, that would allow to students to design algorithms that achieve the desirable outcomes in the presence of strategic behaviors of users.  This course focuses on three topics: 1) mechanism design, a study on incentivizing users to truthfully report their data for a given computational task; 2) price of anarchy in games, a systematic approach to quantify the inefficiency caused by users\u2019 strategic behaviors; and 3) algorithms and complexity theory for learning and computing Nash and market equilibria. The course will also cover some selected advanced topics such as the use of data of past user behaviors in auction design, and case studies of some important applications including online advertisement auctions and kidney exchange market.  "
    },
    {
        "ID": 31,
        "Course_name": "Bioinformatics",
        "Overview": "The goal of the course is for students to be grounded in basic bioinformatics concepts, algorithms, tools, and databases. Students will be leaving the course with hands-on bioinformatics analysis experience and empowered to conduct independent bioinformatics analyses. We will study: 1) algorithms, especially those for sequence alignment and assembly, which comprise the foundation of the rapid development of bioinformatics and DNA sequencing; 2) the leading bioinformatics tools for comparing and analyzing genomes starting from raw sequencing data; 3) the functions and organization of a few essential bioinformatics databases and learn how they support various types of bioinformatics analysis. "
    },
    {
        "ID": 32,
        "Course_name": "Statistical learning",
        "Overview": "The challenges in learning from big and complicated data have led to significant advancements in the statistical sciences. This course introduces supervised and unsupervised learning, with emphases on the theoretical underpinnings and on applications in the statistical programming environment R. Topics include linear methods for regression and classification, model selection, model averaging, basic expansions and regularization, kernel smoothing methods, additive models and tree-based methods. We will also provide an overview of neural networks and random forests. "
    },
    {
        "ID": 33,
        "Course_name": "Cyber security",
        "Overview": "This course introduces the principles, mechanisms and implementation of cyber security and data protection. Knowledge about the attack and defense are included. Topics include notion and terms of cyber security; network and Internet security, introduction to encryption: classic and modern encryption technologies; authentication methods; access control methods; cyber attacks and defenses (e.g. malware, DDoS). "
    },
    {
        "ID": 34,
        "Course_name": "Robotics",
        "Overview": "This course provides an introduction to mathematics and algorithms underneath state-of-the-art robotic systems. The majority of these techniques are heavily based on probabilistic reasoning and optimization \u2013 two areas with wide applicability in modern AI. We will also cover some basic knowledge about robotics, namely geometry, kinematics, dynamics, control of a robot, as well as the mathematical tools required to describe the spatial motion of a robot will be presented. In addition, we will cover perception, planning, and learning for a robotic system, with the obstacle avoidance and robotic arm manipulation as typical examples.  "
    },
    {
        "ID": 35,
        "Course_name": "Cryptography",
        "Overview": "This course offers a gentle introduction to the field of cryptography. We will start from the basic principles of confidentiality, integrity and authentication.  After that, we will go through some fundamental cryptographic primitives like hash function, symmetric key encryption, public key encryption and digital signatures.  Finally, we will introduce the basics of quantum cryptography including quantum key distribution and random number generation."
    },
    {
        "ID": 36,
        "Course_name": "Distributed and parallel computing",
        "Overview": "This course introduces the basic concepts and modern software architectures on distributed and parallel computing. Topics include: computer network primitives, distributed transactions and two-phase commits, webservices, parallelism and scalability models, distributed consistency models, distributed fault-tolerance, actor and monads, Facebook photo cache, Amazon key-value stores, Google Mapreduce, Spark, and TensorFlow. "
    },
    {
        "ID": 37,
        "Course_name": "Artificial intelligence applications",
        "Overview": "This course focuses on practical applications of AI technologies.  The course comprises two main components: students first acquire the basic know-how of the state-of-the-art AI technologies, platforms and tools (e.g., TensorFlow, PyTorch, scikit-learn) via self-learning of designated materials including code examples and open courseware.  Students will then explore practical AI applications and complete a course project which implements an AI-powered solution to a problem of their own choice. This course is designed for students who are interested in experimenting with some typical AI problems before delving deeper into the AI foundations and theories.  "
    },
    {
        "ID": 38,
        "Course_name": "Implementation testing and maintenance of software systems",
        "Overview": "This course examines the theory and practice of software implementation, testing and maintenance. Topics in implementation include: detailed design issues and implementation strategies; coding style and standards; the review process; pattern implementation and reuse. Testing covers strategies and techniques for unit and component testing; integration testing; system, performance and acceptance testing; test documentation and test management. Topics in maintenance include maintenance techniques, tools and metrics; software rejuvenation; and refactoring. Prerequisite: COMP3297 or IIMT3602 "
    },
    {
        "ID": 39,
        "Course_name": "Software quality and project management",
        "Overview": "Topics in software quality include: software quality models; quality assurance; software quality metrics; quality reviews, inspections and audits. Topics in project management include: project planning, cost estimation and scheduling; project monitoring and control; agile, traditional and extreme process models and their management; risk analysis; configuration management and control; software acquisition; contract management; and process improvement.  "
    },
    {
        "ID": 40,
        "Course_name": "Scientific computing",
        "Overview": "This course provides an overview and covers the fundamentals of scientific and numerical computing. Topics include numerical analysis and computation, symbolic computation, scientific visualization, architectures for scientific computing, and applications of scientific computing. "
    }
]

def recommend_courses():
    """
    Generate recommendations for courses based on student's profile and available courses.
    
    Returns:
    tuple: A tuple containing two elements - a dictionary of recommended courses categorized by type and a list of elective courses to be taken.
    """
    
    original_data = studyplan

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
    courses = pd.DataFrame(cs_courses)
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