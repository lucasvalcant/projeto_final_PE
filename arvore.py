import pandas as pd
import os
import numpy as np
from sklearn import tree

train = pd.read_csv('interin_data/database_train.csv')
y_train = train['caro']
x_train = train.drop(['caro'], axis=1).values
decision_tree = tree.DecisionTreeClassifier(min_samples_leaf=5, min_samples_split=15, min_impurity_decrease = 0.0001, max_depth=15)
decision_tree.fit(x_train, y_train)

with open("discrete_15.dot", 'w') as f:
    f = tree.export_graphviz(decision_tree,
                             out_file=f,
                             max_depth=15,
                             impurity= True,
                             feature_names = list(train.drop(['caro'], axis=1)),
                             class_names = ["0", "1", "2"],
                             rounded = True,
                             filled = True)