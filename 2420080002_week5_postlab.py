import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from math import log2

# -----------------------------
# Manual Information Gain Function
# -----------------------------
def information_gain(parent,left,right):

    def entropy(labels):
        classes=np.unique(labels)
        ent=0
        for c in classes:
            p=np.sum(labels==c)/len(labels)
            ent-=p*log2(p)
        return ent

    parent_entropy=entropy(parent)

    w_left=len(left)/len(parent)
    w_right=len(right)/len(parent)

    gain=parent_entropy-(w_left*entropy(left)+w_right*entropy(right))

    return gain


# Example dataset
data=pd.DataFrame({
    'Feature':[1,2,3,4,5,6],
    'Class':[0,0,0,1,1,1]
})

parent=data['Class']
left=data[data['Feature']<=3]['Class']
right=data[data['Feature']>3]['Class']

print("Manual Information Gain:",information_gain(parent,left,right))


# -----------------------------
# Load Iris Dataset
# -----------------------------
iris=load_iris()
X=iris.data
y=iris.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


# -----------------------------
# Compare Different Max Depths
# -----------------------------
depths=[1,2,3,4,5]

for d in depths:

    model=DecisionTreeClassifier(max_depth=d)
    model.fit(X_train,y_train)

    pred=model.predict(X_test)

    acc=accuracy_score(y_test,pred)

    print("Depth:",d," Accuracy:",acc)


# -----------------------------
# Cost Complexity Pruning
# -----------------------------
path=DecisionTreeClassifier().cost_complexity_pruning_path(X_train,y_train)

ccp_alphas=path.ccp_alphas

for alpha in ccp_alphas[:5]:

    pruned_tree=DecisionTreeClassifier(ccp_alpha=alpha)
    pruned_tree.fit(X_train,y_train)

    pred=pruned_tree.predict(X_test)

    print("Alpha:",alpha," Accuracy:",accuracy_score(y_test,pred))


# -----------------------------
# Decision Tree Regressor
# -----------------------------
regressor=DecisionTreeRegressor(max_depth=3)

regressor.fit(X_train,y_train)

pred_reg=regressor.predict(X_test)

print("Sample Regression Output:",pred_reg[:5])


# -----------------------------
# Export Decision Rules
# -----------------------------
tree_model=DecisionTreeClassifier(max_depth=3)

tree_model.fit(X_train,y_train)

rules=export_text(tree_model,feature_names=list(iris.feature_names))

print("Decision Rules:\n")
print(rules)