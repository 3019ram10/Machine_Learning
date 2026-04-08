import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Binarizer

# ---------------------------------------
# Load Dataset
# ---------------------------------------

iris=load_iris()

X=iris.data
y=iris.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

# ---------------------------------------
# Weighted KNN
# ---------------------------------------

weighted_knn=KNeighborsClassifier(n_neighbors=5,weights='distance')

weighted_knn.fit(X_train,y_train)

pred_weighted=weighted_knn.predict(X_test)

print("Weighted KNN Accuracy:",accuracy_score(y_test,pred_weighted))


# ---------------------------------------
# Bernoulli Naive Bayes
# ---------------------------------------

binarizer=Binarizer()

X_binary=binarizer.fit_transform(X)

X_train_b,X_test_b,y_train_b,y_test_b=train_test_split(X_binary,y,test_size=0.3,random_state=42)

bnb=BernoulliNB()

bnb.fit(X_train_b,y_train_b)

pred_bnb=bnb.predict(X_test_b)

print("Bernoulli NB Accuracy:",accuracy_score(y_test_b,pred_bnb))


# ---------------------------------------
# Voting Classifier (KNN + NB)
# ---------------------------------------

knn=KNeighborsClassifier()

nb=BernoulliNB()

voting=VotingClassifier(estimators=[('knn',knn),('nb',nb)],voting='hard')

voting.fit(X_train_b,y_train_b)

pred_vote=voting.predict(X_test_b)

print("Voting Classifier Accuracy:",accuracy_score(y_test_b,pred_vote))


# ---------------------------------------
# Hyperparameter Tuning using GridSearch
# ---------------------------------------

param_grid={'n_neighbors':[1,3,5,7,9]}

grid=GridSearchCV(KNeighborsClassifier(),param_grid,cv=5)

grid.fit(X_train,y_train)

print("Best K:",grid.best_params_)
print("Best Accuracy:",grid.best_score_)


# ---------------------------------------
# Misclassified Instances
# ---------------------------------------

pred_final=grid.predict(X_test)

misclassified=[]

for i in range(len(y_test)):

    if pred_final[i]!=y_test[i]:
        misclassified.append((X_test[i],y_test[i],pred_final[i]))

print("Misclassified Samples:")

for m in misclassified[:5]:
    print(m)