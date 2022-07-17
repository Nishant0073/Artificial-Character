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
from sklearn.neighbors import KNeighborsClassifier
import warnings
#ignore warnings
warnings.filterwarnings('ignore')

# Function importing Dataset
def importdata():
    data = pd.read_csv('Dataset.csv')
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
def train_using_knn(X_train, X_test, y_train):

    # Creating the classifier object
    clf_knn = KNeighborsClassifier(n_neighbors = 7)


    # Performing training
    clf_knn.fit(X_train, y_train)
    return clf_knn

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

    clf_knn = train_using_knn(X_train, X_test, y_train)
    # Operational Phase
    print("Results Using KNN Index:")
    # Prediction using gini
    y_pred_knn = prediction(X_test, clf_knn)
    cal_accuracy(y_test, y_pred_knn)


if __name__=="__main__":
    main()

