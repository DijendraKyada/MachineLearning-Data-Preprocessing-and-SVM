# -*- coding: utf-8 -*-
"""
Created on 2 July 2019

@author: Joe Skufca

This file provides a guided effort through the Heart Disease Homework assignment.

The goal of this assignment is to better understand how to tackle some
preprocessing steps, with particular focus on scaling.

Additionally, it intends to explore the use of cross-validation as a tool
for improved accuracy assessments and tuning of parameters.


"""
#%% Action

"""
For your homework, save a copy of this file under the filename

    FirstNameLastNameHDDHW.py
    
    
Edit as necessary to complet the requested work.

"""

#%% libraries and modules that we may need

# standards for data and computation
import pandas as pd
import numpy as np

# use seaborn plotting defaults
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# preprocessing from scikit learn
import sklearn.preprocessing as sklp

# Note: other commands may be loaded as neeeded.


#%% Task 1 - read as data frames
"""
Ensure you have downloaded the data from our course page.  
The data was taken from the UCI Machine Learning Library
Cleveland Heart Disease Data

""" 

dfCleve=pd.read_csv("cleve.txt", header=None,skiprows=20,sep='\s+')
dfProcessed=pd.read_csv("processed.cleveland.data", header=None)

#%%  Task 1a   
""" ACTION REQUIRED: Write code that looks at the first 5 rows of each dataset
to get some notion
that you have correctly loaded the data
""" 

# Insert code below
dfCleve.head(5)
dfProcessed.head(5)


#%% Task 2a  Convert from dfCleve to dfProcessed

# I will build as individual fixes, with later consideration of a pipeline

# inspect the object before we start
dfCleve.dtypes
dfCleve.describe()
"""
ACTION REQUIRED:  Which fields are already numeric?

* Your answer here
0     float64
1      object
2      object
3     float64
4     float64
5      object
6      object
7     float64
8      object
9     float64
10     object
11     object
12     object
13     object
14     object
dtype: object

Fields already numeric: age, trestbps, chol, thalach and oldpeak
"""

#%% Task 2b Buiding the numeric dataset from the cleve.txt
"""
NOTE -  Assignment statements in Python do not copy objects, 
they create bindings between a target and an object. 
A copy is sometimes needed so one can change one copy without changing the other.
"""
dfCP=dfCleve.copy() # create the new object

"""
LabelEncoder 

This object converts categorical to numeric and is a convenient way to convert
most variables to a numeric code.  However, it only assigns numbers in
alphabetical order, starting with 0.

To achieve the same encoding as dfProcessed, only a few of the variables satisfy.  
"""
le=sklp.LabelEncoder() #instantiate the object


dfCP[1]=le.fit_transform(dfCleve[1]) # number assigned is alphabetical
dfCP[5]=le.fit_transform(dfCleve[5])
dfCP[8]=le.fit_transform(dfCleve[8])


"""ACTION REQUIRED: Provide a brief description of what 

             fit_transform() 
             
achieves.

* YOUR ANSWER HERE 
For trianing set you need to perform both some calculation and transformation. fit_transformation() does 
some calculation and then do transformation for example, calculating the means of columns from some data and then 
replacing the missing values. Transform replaces the missing values with a number. By default this number
is the means of columns of some data that you choose. by fit the imputer calculates the means of columns
from some data, and by transform it applies those means to some data (which is just replacing missing values 
with the means). If both these data are the same (i.e. the data for calculating the means and the data that 
means are applied to) you can use fit_transform which is basically a fit followed by a transform.
Why we might need to transform data? For various reasons, many real world datasets contain missing values,
often encoded as blanks, NaNs or other placeholders. Such datasets however are incompatible with 
scikit-learn estimators which assume that all values in an array are numerical.   
What does it mean fitting model on training data and transforming to test data? The fit of an imputer has 
nothing to do with fit used in model fitting. So using imputer's fit on training data just calculates means
of each column of training data. Using transform on test data then replaces missing values of test data 
with means that were calculated from training data.

In summary, fit performs the training, transform changes the data in the pipeline in order to pass it on to
the next stage in the pipeline, and fit_transform does both the fitting and the transforming in one possibly 
optimized step 
"""


#%% Task 2c Buiding the numeric dataset 
"""
To match our encoding to that in the dfProcessed, we need a specific conversion
from category (text) to number.  

The pandas 'map' function may be the easiest way, converting IAW a 
dictionary which specifies the mapping.

"""

dfCP[2]=dfCleve[2].map({'angina':1,'abnang':2,'notang':3,'asympt':4})

""" ACTION REQUIRED: refer to the first 20 lines of "cleve.txt" to find the
desired mappings, then write code that achieves that mapping for columns

6,10,11,12,14

INSERT CODE BELOW.
"""
dfCP[6]=dfCleve[6].map({'norm':0,'abn':1,'hyp':2})
dfCP[10]=dfCleve[10].map({'up':1,'flat':2,'down':3})
dfCP[11]=dfCleve[11].map({'0.0':0,'1.0':1,'2.0':2,'3.0':3})
dfCP[12]=dfCleve[12].map({'norm':3,'fix':6,'rev':7})
dfCP[14]=dfCleve[14].map({'H':0,'S1':1,'S2':2,'S3':3,'S4':4})

dfCP.head()
#%%  2c continued  

""" Note that dfProcessed does not include a 

"buff/sick" column, so we remove it from dfCP
"""

dfCP.drop(columns=13,inplace=True)

"""
ACTION REQUIRED:  Why do we need the option "inplace=True"?

* YOUR ANSWER HERE
When inplace=True is passed, the data is renamed in place (it returns nothing). When inplace=False is passed,
performs the operation and returns a copy of the object. You use inplace=True, if you don't want to save the 
updated data to the same variable. Basically, 'False' creates a new copy which we have to assign back to 
dataframe. When it is 'True', there is no need to assign back to dataframe, because it is on the same copy.
"""

#%%  NO ACTION REQUIRED - INFORMATIONAL

"""
You should inspect to feel comfortable that you have properly encoded to 
achieve the same result in dfCP as dfProcessed.

NOTE - They will not match exactly.   The row ordering in the dataframes will
be slightly different.

"""

#%% Task 3 incorporating one-hot encoding

"""
As we have discussed - one-hot encoding is often an "obvious" need.  
Consequently, pandas includes a dataframe method to covert categrorical variables
to one-hot encoded dummy variables.
"""

dfOnehot=pd.get_dummies(dfCleve.dropna())

#%% Task 4 Scaling of numeric variables

# let's use MinMax to scale to unit output using 

scaler=sklp.MinMaxScaler()

scaler.fit(dfOnehot) # fit the scaler to the test data

# Not appropriate to use full scaled dataset
dfScaled=scaler.transform(dfOnehot)

"""
Above, I used the MinMax scaler.  

(a) Identify three other scalers availble in sklearn.prepocessing
(b) Briefly describe the difference between MinMaxScaler and StandardScaler.

** YOUR ANSWER HERE **
a) StandardScaler, MinMaxScaler, MaxAbsScaler and RobustScaler are the scalers
provided by sklearn.

b) Sklearn its main scaler, the StandardScaler, uses a strict definition of standardization to standardize 
data. It purely centers the data by using the following formula, where u is the mean and s is the standard
deviation. x_scaled = (x — u) / s. The MinMaxScaler transforms features by scaling each feature to a given
range. This range can be set by specifying the feature_range parameter (default at (0,1)). This scaler works
better for cases where the distribution is not Gaussian or the standard deviation is very small. However, 
it is sensitive to outliers, so if there are outliers in the data, you might want to consider another scaler.
x_scaled = (x-min(x)) / (max(x)–min(x)). Importing and using the MinMaxScaler works — in exactly the same way
as the StandardScaler. The only difference sits in the parameters on initiation of a new instance.
"""


#%% Task 5 comparing performance

from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import cross_val_score

"""
What metric do you thing is most appropriate for this evaluation?  
Provide a short justification for your choice.

** YOUR ANSWER HERE **
There different type of evaluation you can perform:
1.Classification Accuracy.
2.Logarithmic Loss.
3.Area Under ROC Curve.
4.Confusion Matrix.
5.Classification Report

I think the Accuracy is the most appropriate metric for this evaluation. Because, Accuracy returns single
value which is more precise compared to other metric types. Also, it treats all errors the same.

"""


#%% Analysis with no manipulation of the data

""" For consistency - I will use dfCP as the starting point for these
analyses.

Additionally, I will construct a temporary dataframe, df1, to allow
resuse of some of my code lines.
"""

df1=dfCP.dropna()
X=df1.drop(columns=14)
y=(df1[14]>0)

"""
X  is my predictor variables
y  is the response variable - 
"""

mdl1=svm.SVC(C=1,gamma='auto')

scores = cross_val_score(mdl1, X, y, cv=10)
np.mean(scores)

"""
Provide a brief explanation of what is accomplished by the line

    scores = cross_val_score(mdl1, X, y, cv=10)

to include the meaning of the parameter choice  "cv=10"

** YOUR ANSWER HERE **
The cross_val_score() is the simplest way to make cross-validation using sklearn.
mdl1 = The object to use to fit the data
X = the data to fit
y = The target variable to try to predict in the case of supervised learning
cv = Determines the cross-validation splitting strategy.

"cross_val_score" splits the data into 10 folds, in this case, since cv=10. 
Then for each fold it fits the data on 9 folds and scores the 10th fold. 
Then it gives you the 10 scores from which you can calculate a mean and variance 
for the score. You crossval to tune parameters and get an estimate of the score. 
This includes fitting, in fact it includes 10 fits!

This can be broken down into the following 7 steps in detail:
1.Split the dataset (X and y) into K=10 equal partitions (or "folds")
2.Train the mdl1 model on union of folds 2 to 10 (training set)
3.Test the model on fold 1 (testing set) and calculate testing accuracy
4.Train the mdl1 model on union of fold 1 and fold 3 to 10 (training set)
5.Test the model on fold 2 (testing set) and calculate testing accuracy
6.It will do this 8 more times
7.When finished, it will return the 10 testing scores as an array
"""

#%% (Scenario) Evaluating performance with just one-hot  encoding

df1=dfOnehot.copy()
X=df1.loc[:,:'12_rev']
y=(df1['13_sick']>0)

"""
My command for creating predictor variable `X' and response variable 'y'
is somewhat different.

Justify/explain what I am doing.

** YOUR ANSWER HERE **
We separate the data into the predictor variables and the response.
X will be your feature matrix. It contains all the features (predictors) that will predict the 
target variable y. The features are the independent variables while the y is the dependent variable.
y -> is a conditional, which returns the values from column '13_sick' where the 
values are greater than 0.
"""

#%% One-hot (continued)

mdl1=svm.SVC(C=1,gamma='auto')

scores = cross_val_score(mdl1, X, y, cv=10)

""" Add a line of code so that you can see the resultant mean score.
"""

# INSERT CODE HERE
np.mean(scores)


#%% onehot and scaling
df1=scaler.transform(dfOnehot.copy()) # operate on a copy, not original data


# EDIT THE CODE BELOW
X=df1[:,0:30]
y=(df1[:,31]>0)

""" Correct the code above so that it selects the appropriate columns """



mdl1=svm.SVC(C=1,gamma='auto')

scores = cross_val_score(mdl1, X, y, cv=10)
np.mean(scores)


#%% (Scenario)  Just scaling, no one-hot

df1=dfCP.dropna()

#  EDIT CODE BELOW TO PROPERLY SELECT PREDICTOR AND RESPONSE
X=df1.drop(columns=14)
y=(df1[14]>0)

scaler2=sklp.MinMaxScaler()

scaler2.fit(X) # fit the scaler to the test data

Xs=scaler2.transform(X)

"""
Add code below to evaluate performace using the Xs data as predictor
"""

# INSERT CODE HERE
mdl1=svm.SVC(C=1, gamma='auto')

scores=cross_val_score(mdl1, Xs, y, cv=10)
np.mean(scores)

#%% Parameter tuning

"""
In the various code sections above, I wrote the code as if to use default
values of parameters.  

Explain why it would be appropriate to tune the parameter for each dataset
before making accuracy comparisons.

** YOUR ANSWER HERE.
Fine tuning machine learning predictive model is a crucial step to improve accuracy of the forecasted 
results. One of the key tasks is to fine-tune the parameter for each dataset before making accuracy 
comparisons. This will further enhance the accuracy. It is important to feed more data as soon as it is 
available and test the accuracy of the model on continuous basis so that the performance and accuracy 
can be further optimised.

"""

#%% onehot and scalinG AND full paramter tune

"""
You may have done some manual tuning of parameter, but let's explore
the built in capability to find good parameters.
"""

from sklearn.model_selection import GridSearchCV

df1=scaler.transform(dfOnehot.copy())
X=df1[:,0:30]
y=(df1[:,31]>0)

Cs = [0.001, 0.01, 0.1, 1, 10,100]
gammas = [0.001, 0.01,.05, 0.1, .15,  1]
param_grid = {'C': Cs, 'gamma' : gammas}

my_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=10)
my_search.fit(X, y)
print(my_search.best_params_)
print(my_search.best_score_)

"""
Modify Cs and gammas to identify what you think are good choices for these
parameters.

What was the best choice that you found?

** INSERT ANSWER HERE. **
best choice for parameters: {'C': 1, 'gamma': 0.05}
best score: 0.854516129032258

"""
