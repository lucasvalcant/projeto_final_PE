import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import tree

train = pd.read_csv('interin_data/database_discrete.csv')
y_train = train['caro']
x_train = train.drop(['caro'], axis=1).values
decision_tree = tree.DecisionTreeClassifier(max_depth = 10)
decision_tree.fit(x_train, y_train)

with open("discrete_10.dot", 'w') as f:
    f = tree.export_graphviz(decision_tree,
                             out_file=f,
                             max_depth=10,
                             impurity= True,
                             feature_names = list(train.drop(['caro'], axis=1)),
                             class_names = ['False', 'True'],
                             rounded = True,
                             filled = True)