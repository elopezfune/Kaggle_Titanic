# # Titanic: Machine Learning from Disaster

# This is the legendary Titanic ML competition-the best, first challenge for you to dive into ML competitions and familiarize yourself with how the Kaggle platform works.
# 
# The competition is simple: use machine learning to create a model that predicts which passengers survived the Titanic shipwreck.
# 
# 
# The Challenge: The sinking of the Titanic is one of the most infamous shipwrecks in history.
# 
# On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew. While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.
# 
# In this challenge, we ask you to build a predictive model that answers the question: "what sorts of people were more likely to survive?" using passenger data (ie name, age, gender, socio-economic class, etc).
# 
# 
# Overview
# 
# The data has been split into two groups:
# 
#     training set (train.csv)
#     test set (test.csv)
# 
# The training set should be used to build your machine learning models. For the training set, we provide the outcome (also known as the “ground truth”) for each passenger. Your model will be based on “features” like passengers’ gender and class. You can also use feature engineering to create new features.
# 
# The test set should be used to see how well your model performs on unseen data. For the test set, we do not provide the ground truth for each passenger. It is your job to predict these outcomes. For each passenger in the test set, use the model you trained to predict whether or not they survived the sinking of the Titanic.
# 
# We also include gender_submission.csv, a set of predictions that assume all and only female passengers survive, as an example of what a submission file should look like.
# 
# Using the patterns you find in the train.csv data, predict whether the other 418 passengers on board (found in test.csv) survived.



# For numerical computations and linear algebra
import numpy as np 

# For database manipulation
import pandas as pd   
pd.options.mode.chained_assignment = None




# For data visualization
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', size=16) # To use big fonts...
matplotlib.rc('figure', facecolor='white')  # To set white as the background color for plots
# To show the plots in-line   
get_ipython().run_line_magic('matplotlib', 'inline')




# Loadinf the provided database and to show some of its entries
dataframe_train = pd.read_csv("train.csv")
dataframe_tests = pd.read_csv("test.csv")
dataframe_tests["Survived"]=-1
dataframe = pd.concat([dataframe_train,dataframe_tests],axis=0)
dataframe = dataframe.reset_index()
dataframe = dataframe.drop("index",axis=1)

# Remove duplicate rows
dataframe = dataframe.drop_duplicates()


# # Missing Data


def check_missing_values(df,cols=None,axis=0):
    ### This function check out for missing values in each column
    ## Arguments:
                #df: data frame
                #cols: list. List of column names
                #axis: int. 0 means column and 1 means row
    
    # This function returns the missing info as a dataframe 
    
    if cols != None:
        df = df[cols]
    missing_num = df.isnull().sum(axis).to_frame().rename(columns={0:'missing_num'})
    missing_num['missing_percent'] = df.isnull().mean(axis)*100
    result = missing_num.sort_values(by='missing_percent',ascending = False)
    return result[result["missing_percent"]>0.0]


to_drop_missing = check_missing_values(dataframe,cols=None,axis=0)
to_drop_missing = to_drop_missing[to_drop_missing["missing_percent"]>=30.0].index
dataframe = dataframe.drop(to_drop_missing,axis=1)


#Impute missing values using LSTM from DataWig

import datawig

dataframe = datawig.SimpleImputer.complete(dataframe)


# # Label Encoder
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for el in dataframe.drop(["PassengerId"],axis=1).select_dtypes(include=['object']).columns:    
    dataframe[el]=le.fit_transform(dataframe[el].values)


# Train-Test Split

from sklearn.model_selection import train_test_split

to_train = dataframe[dataframe["Survived"]!=-1]
to_tests = dataframe[dataframe["Survived"]==-1]




# # Random Forest


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.optimize import minimize, differential_evolution, basinhopping

def Random_Forest_Classifier(params):
    n_estimators, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, min_impurity_decrease, ccp_alpha = params
    n_estimators = round(n_estimators)
    min_samples_split = round(min_samples_split)
    min_samples_leaf = min_samples_leaf
    
    model = RandomForestClassifier(n_estimators=n_estimators,criterion='gini', max_depth=None,
                                   min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf,
                                   min_weight_fraction_leaf=min_weight_fraction_leaf,
                                   max_features='auto', max_leaf_nodes=None,
                                   min_impurity_decrease=min_impurity_decrease, min_impurity_split=None,
                                   bootstrap=True, oob_score=False, n_jobs=None, random_state=None,
                                   verbose=0, warm_start=False, class_weight=None, ccp_alpha=ccp_alpha,
                                   max_samples=None)
    
    model = model.fit(to_train.drop(["PassengerId","Survived"],axis=1).values,to_train["Survived"].values)
    return model





def metric_function(params):
    n_estimators, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, min_impurity_decrease, ccp_alpha = params
    model  = Random_Forest_Classifier(params)
    Y_true = to_train["Survived"].values
    Y_pred = model.predict(to_train.drop(["PassengerId","Survived"],axis=1).values)
    mse = mean_squared_error(Y_true, Y_pred)
    print("The MSE is: ", mse)
    auc = roc_auc_score(Y_true, Y_pred)
    print("The AUC is: ", auc)
    return mse/auc 


#This is the optimizer of the model
x0 = [100.0, 2.0, 0.2, 0.1, 0.1, 0.1]
boundary_min = [10.0, 2.0, 0.1, 0.001, 0.001, 0.001]
boundary_max = [1000.0, 50.0, 0.5, 0.5, 50.0, 50.0]
bounds = [(low,high) for low, high in zip(boundary_min,boundary_max)]
opt = differential_evolution(metric_function, bounds=bounds,seed=1)
#opt = minimize(metric_function,x0,bounds=boundary)
# use method L-BFGS-B because the problem is smooth and bounded
#minimizer_kwargs = dict(method="L-BFGS-B", bounds=bounds)
#opt = basinhopping(metric_function, x0, minimizer_kwargs=minimizer_kwargs)

print(opt)

#model = Random_Forest_Classifier(opt.x)
#Y_predict=model.predict(to_tests.drop(["PassengerId","Survived"],axis=1))
#predictions = to_tests.copy()
#predictions["Survived"] = Y_predict
#predictions = predictions[["PassengerId","Survived"]]
#predictions.to_csv("log_gender_submission.csv",index=False)
#predictions



