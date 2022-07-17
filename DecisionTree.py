# Implementation of decision tree classifier
# Importing the required packages
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import warnings
#ignore warnings
warnings.filterwarnings('ignore')

# Function importing Dataset
def importdata():
    data = pd.read_csv('learnDataset.csv')
    # Printing the dataset shape
    print ("Dataset Length: ", len(data))
    print ("Dataset Shape: ", data.shape)
    # Printing the dataset obseravtions
    print ("Dataset: ",data.head())
    return data


def splitdataset(data):
    # Separating the target variable
    X = data.iloc[:,1:].values
    Y = data.iloc[:,:1].values

   # Splitting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 100)
    return X, Y, X_train, X_test, y_train, y_test

# Function to perform training with giniIndex.
def train_using_gini(X_train, X_test, y_train):

    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=4, min_samples_leaf=2)

    # Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini 


# Function to perform training with entropy.
def train_using_entropy(X_train, X_test, y_train):
    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth = 4, min_samples_leaf = 2)
    # Performing training
    clf_entropy.fit(X_train, y_train)
    return clf_entropy




# Function to make predictions
def prediction(X_test, clf_object):
    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred

# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: ",confusion_matrix(y_test, y_pred))
    print ("Accuracy : ",accuracy_score(y_test,y_pred)*100)
    print("Report : ",classification_report(y_test, y_pred))



#Driver Code
def main():
    #Building Phase
    data  = importdata()

    #Factorizing data.
    data['class'],_unique_classes =  pd.factorize(data['class'])
    # print(data['class'].head(10))
    # print(_unique_classes)

    # data.info()
    Xf = data.iloc[:,1:]
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)

    clf_gini = train_using_gini(X_train, X_test, y_train)
    clf_entropy = train_using_entropy(X_train, X_test, y_train)
    # Operational Phase
    print("Results Using Gini Index:")
    # Prediction using gini
    y_pred_gini = prediction(X_test, clf_gini)
    cal_accuracy(y_test, y_pred_gini)
    print("Results Using Entropy:")
    # Prediction using entropy
    y_pred_entropy = prediction(X_test, clf_entropy)
    cal_accuracy(y_test, y_pred_entropy)


if __name__=="__main__":
    main()

