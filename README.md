
### Project Description
As education has grown to rely more on technology, vast amounts of data has become available for examination and prediction. Logs of student activities, grades, interactions with teachers and fellow students, and more, are now captured in real time through learning management systems like **Canvas** and **Edmodo**. This is especially true for online classrooms, which are becoming popular even at the primary and secondary school level. Within all levels of education, there exists a push to help increase the likelihood of student success, without watering down the education or engaging in behaviors that fail to improve the underlying issues. Graduation rates are often the criteria of choice, and educators seek new ways to predict the success and failure of students early enough to stage effective interventions.

A local school district has a goal to reach a 95% graduation rate by the end of the decade by identifying students who need intervention before they drop out of school. As a software engineer contacted by the school district, your task is to model the factors that predict how likely a student is to pass their high school final exam, by constructing an intervention system that leverages supervised learning techniques. The board of supervisors has asked that you find the most effective model that uses the least amount of computation costs to save on the budget. You will need to analyze the dataset on students' performance and develop a model that will predict the likelihood that a given student will pass, quantifying whether an intervention is necessary.

[Udacity Nanodegree Machine Learning webpage with a details about the program I participate in](https://www.udacity.com/course/machine-learning-engineer-nanodegree-by-google--nd009)


# Machine Learning Engineer Nanodegree
## Supervised Learning
## Project 2: Building a Student Intervention System

Welcome to the second project of the Machine Learning Engineer Nanodegree! In this notebook, some template code has already been provided for you, and it will be your job to implement the additional functionality necessary to successfully complete this project. Sections that begin with **'Implementation'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section and the specifics of the implementation are marked in the code block with a `'TODO'` statement. Please be sure to read the instructions carefully!

In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.  

>**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

### Question 1 - Classification vs. Regression
*Your goal for this project is to identify students who might need early intervention before they fail to graduate. Which type of supervised learning problem is this, classification or regression? Why?*

**Answer: **

I would say that this is a typical classification problem because labels(outcomes) are descrete. There is only two possible outcomes. Whether student needs intervention or not.

## Exploring the Data
Run the code cell below to load necessary Python libraries and load the student data. Note that the last column from this dataset, `'passed'`, will be our target label (whether the student graduated or didn't graduate). All other columns are features about each student.


```python
# Import libraries
import pandas as pd
import numpy as np

# Read student data
student_data = pd.read_csv("student-data.csv")
print("Student data read successfully!")
```

    Student data read successfully!


### Implementation: Data Exploration
Let's begin by investigating the dataset to determine how many students we have information on, and learn about the graduation rate among these students. In the code cell below, you will need to compute the following:
- The total number of students, `n_students`.
- The total number of features for each student, `n_features`.
- The number of those students who passed, `n_passed`.
- The number of those students who failed, `n_failed`.
- The graduation rate of the class, `grad_rate`, in percent (%).



```python
# TODO: Calculate number of students
n_students = len(student_data.index)

# TODO: Calculate number of features
n_features = len(student_data.columns)-1

# TODO: Calculate passing students
n_passed = len(student_data[student_data['passed'] == 'yes'].index)

# TODO: Calculate failing students
n_failed = len(student_data[student_data['passed'] == 'no'].index)

# TODO: Calculate graduation rate
grad_rate = float(n_passed)/n_students*100

# Print the results
print("Total number of students: {}".format(n_students))
print("Number of features: {}".format(n_features))
print("Number of students who passed: {}".format(n_passed))
print("Number of students who failed: {}".format(n_failed))
print("Graduation rate of the class: {:.2f}%".format(grad_rate))
```

    Total number of students: 395
    Number of features: 30
    Number of students who passed: 265
    Number of students who failed: 130
    Graduation rate of the class: 67.09%


## Preparing the Data
In this section, we will prepare the data for modeling, training and testing.

### Identify feature and target columns
It is often the case that the data you obtain contains non-numeric features. This can be a problem, as most machine learning algorithms expect numeric data to perform computations with.

Run the code cell below to separate the student data into feature and target columns to see if any features are non-numeric.


```python
# Extract feature columns
feature_cols = list(student_data.columns[:-1])

# Extract target column 'passed'
target_col = student_data.columns[-1] 

# Show the list of columns
print("Feature columns:\n{}".format(feature_cols))
print("\nTarget column: {}".format(target_col))

# Separate the data into feature data and target data (X_all and y_all, respectively)
X_all = student_data[feature_cols]
y_all = student_data[target_col]

# Show the feature information by printing the first five rows
print("\nFeature values:")
print(X_all.head())
```

    Feature columns:
    ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']
    
    Target column: passed
    
    Feature values:
      school sex  age address famsize Pstatus  Medu  Fedu     Mjob      Fjob  \
    0     GP   F   18       U     GT3       A     4     4  at_home   teacher   
    1     GP   F   17       U     GT3       T     1     1  at_home     other   
    2     GP   F   15       U     LE3       T     1     1  at_home     other   
    3     GP   F   15       U     GT3       T     4     2   health  services   
    4     GP   F   16       U     GT3       T     3     3    other     other   
    
        ...    higher internet  romantic  famrel  freetime goout Dalc Walc health  \
    0   ...       yes       no        no       4         3     4    1    1      3   
    1   ...       yes      yes        no       5         3     3    1    1      3   
    2   ...       yes      yes        no       4         3     2    2    3      3   
    3   ...       yes      yes       yes       3         2     2    1    1      5   
    4   ...       yes       no        no       4         3     2    1    2      5   
    
      absences  
    0        6  
    1        4  
    2       10  
    3        2  
    4        4  
    
    [5 rows x 30 columns]


### Preprocess Feature Columns

As you can see, there are several non-numeric columns that need to be converted! Many of them are simply `yes`/`no`, e.g. `internet`. These can be reasonably converted into `1`/`0` (binary) values.

Other columns, like `Mjob` and `Fjob`, have more than two values, and are known as _categorical variables_. The recommended way to handle such a column is to create as many columns as possible values (e.g. `Fjob_teacher`, `Fjob_other`, `Fjob_services`, etc.), and assign a `1` to one of them and `0` to all others.

These generated columns are sometimes called _dummy variables_, and we will use the [`pandas.get_dummies()`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html?highlight=get_dummies#pandas.get_dummies) function to perform this transformation. Run the code cell below to perform the preprocessing routine discussed in this section.


```python
def preprocess_features(X):
    ''' Preprocesses the student data and converts non-numeric binary variables into
        binary (0/1) variables. Converts categorical variables into dummy variables. '''
    
    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():
        # If data type is non-numeric, replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            # Example: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix = col)  
        
        # Collect the revised columns
        output = output.join(col_data)
    
    return output

X_origin = X_all
y_origin = y_all

X_all = preprocess_features(X_all)

print("Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns)))
```

    Processed feature columns (48 total features):
    ['school_GP', 'school_MS', 'sex_F', 'sex_M', 'age', 'address_R', 'address_U', 'famsize_GT3', 'famsize_LE3', 'Pstatus_A', 'Pstatus_T', 'Medu', 'Fedu', 'Mjob_at_home', 'Mjob_health', 'Mjob_other', 'Mjob_services', 'Mjob_teacher', 'Fjob_at_home', 'Fjob_health', 'Fjob_other', 'Fjob_services', 'Fjob_teacher', 'reason_course', 'reason_home', 'reason_other', 'reason_reputation', 'guardian_father', 'guardian_mother', 'guardian_other', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']


### Implementation: Training and Testing Data Split
So far, we have converted all _categorical_ features into numeric values. For the next step, we split the data (both features and corresponding labels) into training and test sets. In the following code cell below, you will need to implement the following:
- Randomly shuffle and split the data (`X_all`, `y_all`) into training and testing subsets.
  - Use 300 training points (approximately 75%) and 95 testing points (approximately 25%).
  - Set a `random_state` for the function(s) you use, if provided.
  - Store the results in `X_train`, `X_test`, `y_train`, and `y_test`.


```python
# TODO: Import any additional functionality you may need here
from sklearn import cross_validation
# TODO: Set the number of training points
num_train = 300

# Set the number of testing points
num_test = X_all.shape[0] - num_train

# TODO: Shuffle and split the dataset into the number of training and testing points above
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_all, y_all,
                                                                     train_size=num_train,
                                                                     test_size=num_test,
                                                                     random_state=True)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))
```

    Training set has 300 samples.
    Testing set has 95 samples.


## Training and Evaluating Models
In this section, you will choose 3 supervised learning models that are appropriate for this problem and available in `scikit-learn`. You will first discuss the reasoning behind choosing these three models by considering what you know about the data and each model's strengths and weaknesses. You will then fit the model to varying sizes of training data (100 data points, 200 data points, and 300 data points) and measure the F<sub>1</sub> score. You will need to produce three tables (one for each model) that shows the training set size, training time, prediction time, F<sub>1</sub> score on the training set, and F<sub>1</sub> score on the testing set.

### Question 2 - Model Application
*List three supervised learning models that are appropriate for this problem. What are the general applications of each model? What are their strengths and weaknesses? Given what you know about the data, why did you choose these models to be applied?*

**Answer:**

In this project I decided to choose Decision Tree, Support Vector Machines and Naive Bayes classifiers. The reason why I have chosen exactly these models is written below, under each model's description.

- Decision Tree - http://scikit-learn.org/stable/modules/tree.html :

Decision Tree Classifier builds a model which makes a prediction based on decision rules inferred from the data features.

Decision tree is a simple and efficient classifier. As we know, our input data could have some non-numeric values, like job type or gender. Decision Tree is a well known model which can handle such data without any preprocessing. So if we will see that this model provides good results, we can use it to build our system without preprocessing which will simplify our code logic.

    Advantages:

        * Very simple for understanding and visualising model. 
        * Very fast in prediction(Logarighmic cost in the number of data points used to train model). 
        * Easly handles numerical and categorical inputs. Moreover, model can process categorical data without creating a dummy features. Although we created dummy-features to process our data, decision tree classifier can handle even original features.

    Disadvantages:

        * As will be shown in default DecisionTreeClassifier below, model may become overfitted which doesn't generalize data well. But that’s where ensemble methods like random forests (or boosted trees) come in.
        * Decision tree learners create biased trees if some classes dominate. It is therefore recommended to balance the dataset prior to fitting with the decision tree.
        * Decision trees can be unstable because small variations in the data might result in a completely different tree being generated. This problem is mitigated by using decision trees within an ensemble.
        
- Support Vector Machines - http://scikit-learn.org/stable/modules/svm.html :

SVM classifier builds a linear decision boundary in high dimensional space. If data cannot be splitted linearly classifier creates additional dimension using provided kernel method where data could be splitted linearly. 

This prediction model is known as a good classifier in high-dimensional space(Each student had 30 features before preprocessing and 48 after it. From my point of view it's quite big amount of features). This is the reason why I have chosen this model.

    Advantages:
    
        * Effective on data with big amount of features.
        * Effective if amount of features is higher than amount of samples.
        * Flexible. Various kernel methods. Possible to define custom kernel functions.
        
    Disadvantages:
        
        * Poor performance if amount of features is much higher than sample size.
        * Model doesn't directly provide probability estimates.
        
- Naive Bayes http://scikit-learn.org/stable/modules/naive_bayes.html :

Naive Bayes is a kind of probablistic classifier which calculates a probability that given input corresponds to some label. Based on Bayesian theorem. Assumes that all features between input are independent(for instance, it expects that there is no correlation between you gender and age while models calculate whether student needs intervention or not).

I have chosen Naive Bayes classifier because it has good time performance, requires a small amount of data for trainig. And it also efficient for models with big amount of features.

    Advantages:
        * Known as a good classifier in real-world situations.
        * Requires a small amount of training data to estimate the necessary parameters well.
        
    Disadvantages:
        * Although naive Bayes is known as a good classifier, it is known to be a bad estimator.
        * Although it requires a small amount of data for training, data should be representative. Because, if a given class and feature value never occur together in the training data, then the frequency-based probability estimate will be zero. As far as I know, to prevent it we could add some fake data to the dataset. I learned this technique on one of Sebastian's Intro to Statistic courses.

### Setup
Run the code cell below to initialize three helper functions which you can use for training and testing the three supervised learning models you've chosen above. The functions are as follows:
- `train_classifier` - takes as input a classifier and training data and fits the classifier to the data.
- `predict_labels` - takes as input a fit classifier, features, and a target labeling and makes predictions using the F<sub>1</sub> score.
- `train_predict` - takes as input a classifier, and the training and testing data, and performs `train_clasifier` and `predict_labels`.
 - This function will report the F<sub>1</sub> score for both the training and testing data separately.


```python
from time import time
from sklearn.metrics import f1_score

def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    
    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    
    # Print the results
    print("Trained model in {:.4f} seconds".format(end - start))

    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()
    
    # Print and return results
    print("Made predictions in {:.4f} seconds.".format(end - start))
    return f1_score(target, y_pred, pos_label='yes')


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''
    
    # Indicate the classifier and the training set size
    print("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))
    
    # Train the classifier
    train_classifier(clf, X_train, y_train)
    
    # Print the results of prediction for both training and testing
    print("F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train)))
    print("F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test)))
```

### Implementation: Model Performance Metrics
With the predefined functions above, you will now import the three supervised learning models of your choice and run the `train_predict` function for each one. Remember that you will need to train and predict on each classifier for three different training set sizes: 100, 200, and 300. Hence, you should expect to have 9 different outputs below — 3 for each model using the varying training set sizes. In the following code cell, you will need to implement the following:
- Import the three supervised learning models you've discussed in the previous section.
- Initialize the three models and store them in `clf_A`, `clf_B`, and `clf_C`.
 - Use a `random_state` for each model you use, if provided.
- Create the different training set sizes to be used to train each model.
 - *Do not reshuffle and resplit the data! The new training points should be drawn from `X_train` and `y_train`.*
- Fit each model with each training set size and make predictions on the test set (9 in total).  
**Note:** Three tables are provided after the following code cell which can be used to store your results.


```python
# TODO: Import the three supervised learning models from sklearn
from sklearn import tree
from sklearn import svm
from sklearn import naive_bayes

# TODO: Initialize the three models
clf_A = tree.DecisionTreeClassifier()
clf_B = svm.SVC()
clf_C = naive_bayes.MultinomialNB()

# TODO: Set up the training set sizes
X_train_100 = X_train[:100]
y_train_100 = y_train[:100]

X_train_200 = X_train[:200]
y_train_200 = y_train[:200]

X_train_300 = X_train
y_train_300 = y_train

# TODO: Execute the 'train_predict' function for each classifier and each training set size
print("##### DECISION TREE CLASSIFIER #####")
train_predict(clf_A, X_train_100, y_train_100, X_test, y_test)
print("\n")
train_predict(clf_A, X_train_200, y_train_200, X_test, y_test)
print("\n")
train_predict(clf_A, X_train_300, y_train_300, X_test, y_test)
print("\n\n\n\n")
print("##### SUPPORT VECTOR MACHINES CLASSIFIER #####")
train_predict(clf_B, X_train_100, y_train_100, X_test, y_test)
print("\n")
train_predict(clf_B, X_train_200, y_train_200, X_test, y_test)
print("\n")
train_predict(clf_B, X_train_300, y_train_300, X_test, y_test)
print("\n\n\n\n")
print("##### NAIVE BAYES CLASSIFIER #####")
train_predict(clf_C, X_train_100, y_train_100, X_test, y_test)
print("\n")
train_predict(clf_C, X_train_200, y_train_200, X_test, y_test)
print("\n")
train_predict(clf_C, X_train_300, y_train_300, X_test, y_test)
```

    ##### DECISION TREE CLASSIFIER #####
    Training a DecisionTreeClassifier using a training set size of 100. . .
    Trained model in 0.0019 seconds
    Made predictions in 0.0004 seconds.
    F1 score for training set: 1.0000.
    Made predictions in 0.0002 seconds.
    F1 score for test set: 0.6261.
    
    
    Training a DecisionTreeClassifier using a training set size of 200. . .
    Trained model in 0.0015 seconds
    Made predictions in 0.0002 seconds.
    F1 score for training set: 1.0000.
    Made predictions in 0.0005 seconds.
    F1 score for test set: 0.7943.
    
    
    Training a DecisionTreeClassifier using a training set size of 300. . .
    Trained model in 0.0040 seconds
    Made predictions in 0.0003 seconds.
    F1 score for training set: 1.0000.
    Made predictions in 0.0002 seconds.
    F1 score for test set: 0.6721.
    
    
    
    
    
    ##### SUPPORT VECTOR MACHINES CLASSIFIER #####
    Training a SVC using a training set size of 100. . .
    Trained model in 0.0012 seconds
    Made predictions in 0.0008 seconds.
    F1 score for training set: 0.8591.
    Made predictions in 0.0009 seconds.
    F1 score for test set: 0.8333.
    
    
    Training a SVC using a training set size of 200. . .
    Trained model in 0.0056 seconds
    Made predictions in 0.0025 seconds.
    F1 score for training set: 0.8581.
    Made predictions in 0.0012 seconds.
    F1 score for test set: 0.8408.
    
    
    Training a SVC using a training set size of 300. . .
    Trained model in 0.0080 seconds
    Made predictions in 0.0049 seconds.
    F1 score for training set: 0.8584.
    Made predictions in 0.0019 seconds.
    F1 score for test set: 0.8462.
    
    
    
    
    
    ##### NAIVE BAYES CLASSIFIER #####
    Training a MultinomialNB using a training set size of 100. . .
    Trained model in 0.0008 seconds
    Made predictions in 0.0002 seconds.
    F1 score for training set: 0.8209.
    Made predictions in 0.0002 seconds.
    F1 score for test set: 0.7647.
    
    
    Training a MultinomialNB using a training set size of 200. . .
    Trained model in 0.0009 seconds
    Made predictions in 0.0002 seconds.
    F1 score for training set: 0.8099.
    Made predictions in 0.0002 seconds.
    F1 score for test set: 0.8333.
    
    
    Training a MultinomialNB using a training set size of 300. . .
    Trained model in 0.0017 seconds
    Made predictions in 0.0004 seconds.
    F1 score for training set: 0.8019.
    Made predictions in 0.0004 seconds.
    F1 score for test set: 0.8148.


### Tabular Results
Edit the cell below to see how a table can be designed in [Markdown](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet#tables). You can record your results from above in the tables provided.

** Classifer 1 - Decision Tree**  

| Training Set Size | Prediction Time (train) | Prediction Time (test) | F1 Score (train) | F1 Score (test) |
| :---------------: | :---------------------: | :--------------------: | :--------------: | :-------------: |
| 100               |      0.0002 seconds     |     0.0005 seconds     |     1.0000       |     0.6500      |
| 200               |      0.0004 seconds     |     0.0001 seconds     |     1.0000       |     0.7231      |
| 300               |      0.0058 seconds     |     0.0001 seconds     |     1.0000       |     0.7031      |


** Classifer 2 - Support Vector Machines**  

| Training Set Size | Prediction Time (train) | Prediction Time (test) | F1 Score (train) | F1 Score (test) |
| :---------------: | :---------------------: | :--------------------: | :--------------: | :-------------: |
| 100               |      0.0006 seconds     |     0.0006 seconds     |     0.8591       |     0.8333      |
| 200               |      0.0032 seconds     |     0.0016 seconds     |     0.8581       |     0.8408      |
| 300               |      0.0058 seconds     |     0.0020 seconds     |     0.8584       |     0.8462      |


** Classifer 3 - Naive Bayes Classifier**  

| Training Set Size | Prediction Time (train) | Prediction Time (test) | F1 Score (train) | F1 Score (test) |
| :---------------: | :---------------------: | :--------------------: | :--------------: | :-------------: |
| 100               |      0.0001 seconds     |     0.0001 seconds     |     0.8209       |     0.7647      |
| 200               |      0.0001 seconds     |     0.0001 seconds     |     0.8099       |     0.8333      |
| 300               |      0.0001 seconds     |     0.0001 seconds     |     0.8019       |     0.8148      |

## Choosing the Best Model
In this final section, you will choose from the three supervised learning models the *best* model to use on the student data. You will then perform a grid search optimization for the model over the entire training set (`X_train` and `y_train`) by tuning at least one parameter to improve upon the untuned model's F<sub>1</sub> score. 

### Question 3 - Chosing the Best Model
*Based on the experiments you performed earlier, in one to two paragraphs, explain to the board of supervisors what single model you chose as the best model. Which model is generally the most appropriate based on the available data, limited resources, cost, and performance?*

**Answer: **

From my point of view, in this case Naive Bayes Classifier shows the best result. It built upon assumption that all features are independent: P(X_i|y, X_1,...,X_i-1,X_i+1,...X_n) = P(X_i|y). The model trains extremely fast and provides good prediction accuracy. It has a bit less accuracy comparing to SVM, but the learning speed is dramatically faster as amount of data growth. So from my point of view this minor difference in accuracy and major difference in time performance could be a reasanable point to choose Naive Bayes classifier. The only disadvantage of NBC is that if 

Speaking about Decision Tree Classifier it showed the worst result. The time complexity also growth quite fast and it provides the lowest accuracy rate on test data. Moreover the accuracy on training set is maximum(f1 score equails 1), which means that this model is overfitted.

### Question 4 - Model in Layman's Terms
*In one to two paragraphs, explain to the board of directors in layman's terms how the final model chosen is supposed to work. For example if you've chosen to use a decision tree or a support vector machine, how does the model go about making a prediction?*

**Answer: **

Naive Bayes Classifier gives a probability that answer to your question is correct one, rather than just give you some answer. Naive Bayes Classifier (NBC) is based on simple Bayesian Theorem, which is following _Posterior = (Likelihood * Prior) / Evidence_.

To explain what does it mean let's build an example. Suppose we want to know whether 15 y. o. student will pass exams or not. First for all we need some historical data of other students. Using this data we will calculate likelihood, prior and evidence. In this case:

**Prior** - Probability that student will pass exams.
**Likelihood** - Probability that student who will pass exams is 15 y. o. 
**Posterior** - Probability that student is 15 y. o.

Let's put it in a table for easier calculations. This table contains historical data of previous students: 

| Age | Passed |
| :-: | :----: |
| 15  | no     |
| 17  | yes    |
| 17  | yes    |
| 17  | no     |
| 16  | yes    |
| 16  | no     |
| 15  | no     |
| 15  | no     |
| 16  | yes    |
| 15  | yes    |


                                                Frequemcy Table   
  
| Age | yes | no  |                  
| :-: | :-: | :-: |              
| 15  |  1  |  3  |              
| 16  |  2  |  1  |              
| 17  |  2  |  1  |              


                                                Likelihood table
  
  
| Age     | yes   | no    |                  
| :-:     | :-:   | :-:   |              
| 15      |  1    |  3    |  4/10    | 0.4 - P(15) - Probability that student is 15 y. o. **Evidence**. |  
| 16      |  2    |  1    |  3/10    |
| 17      |  2    |  1    |  3/10    |
| **ALL** |  5    |  5    |
|         |  5/10 = 0.5 - P(yes) - Probability that studentwill pass pass exams. **Prior**. |
|         |  1/5 = 0.2 - P(15&#124;yes) - Probability that student who will pass exams is 15 y. o. **Posterior**. |

All these steps above happened during training phase. Now, it's time to predict whether 15 years old student will pass exams or not. To do that we have to calculate probability for both cases. The answer will be those one, that has the highest probability.

P(yes|15) = P(15|yes) * P(yes) / P(15) = (1/5) * (0.5) / (4/10) = 0.25

P(no|15) = P(15|no) * P(no)/P(15) = (3/5) * (0.5) / (4/10) = 0.75

Based on the calculations above 15 y. o. student will not pass the exams with probability of 75%. These calculations happening during test phase.

### Implementation: Model Tuning
Fine tune the chosen model. Use grid search (`GridSearchCV`) with at least one important parameter tuned with at least 3 different values. You will need to use the entire training set for this. In the code cell below, you will need to implement the following:
- Import [`sklearn.grid_search.gridSearchCV`](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html) and [`sklearn.metrics.make_scorer`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html).
- Create a dictionary of parameters you wish to tune for the chosen model.
 - Example: `parameters = {'parameter' : [list of values]}`.
- Initialize the classifier you've chosen and store it in `clf`.
- Create the F<sub>1</sub> scoring function using `make_scorer` and store it in `f1_scorer`.
 - Set the `pos_label` parameter to the correct value!
- Perform grid search on the classifier `clf` using `f1_scorer` as the scoring method, and store it in `grid_obj`.
- Fit the grid search object to the training data (`X_train`, `y_train`), and store it in `grid_obj`.


```python
# TODO: Import 'gridSearchCV' and 'make_scorer'
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer, f1_score

# TODO: Create the parameters list you wish to tune
parameters = {'alpha':[0, 0.3, 0.6, 1.0], 'fit_prior':[True, False]}

# TODO: Initialize the classifier
clf = naive_bayes.MultinomialNB()

# TODO: Make an f1 scoring function using 'make_scorer' 
f1_scorer = make_scorer(f1_score, pos_label="yes")

# TODO: Perform grid search on the classifier using the f1_scorer as the scoring method
grid_obj = GridSearchCV(clf, parameters, f1_scorer)

# TODO: Fit the grid search object to the training data and find the optimal parameters
grid_obj = grid_obj.fit(X_train, y_train)

# Get the estimator
clf = grid_obj.best_estimator_

# Report the final F1 score for training and testing after parameter tuning
print("Tuned model has a training F1 score of {:.4f}.".format(predict_labels(clf, X_train, y_train)))
print("Tuned model has a testing F1 score of {:.4f}.".format(predict_labels(clf, X_test, y_test)))
```

    Made predictions in 0.0002 seconds.
    Tuned model has a training F1 score of 0.8019.
    Made predictions in 0.0002 seconds.
    Tuned model has a testing F1 score of 0.8148.


### Question 5 - Final F<sub>1</sub> Score
*What is the final model's F<sub>1</sub> score for training and testing? How does that score compare to the untuned model?*

**Answer: **

The best estimator provided by GridSearchCV gives same F1 score as model without tuning, which means that MultinomialNB configuration is the best one.

> **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  
**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.
