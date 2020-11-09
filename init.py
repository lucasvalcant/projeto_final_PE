import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
'''
Define onde buscar o database
'''
path = os.path.abspath(__file__)
path = os.path.dirname(path)
print(path)
#Carrega o dataframe
df = pd.read_excel(path+os.sep+"raw_data"+os.sep+"database.xlsx")
#Corta o dataframe só com a coluna caro
df2 = df[["caro"]]
def add_kmeans(column,cluster,plot = False):

    kmeans = KMeans(n_clusters=cluster, random_state=0)
    df_cuted = df[[column]].values
    kmeans.fit(df_cuted)
    cuted_list = []

    for line in df_cuted:
        discrete = kmeans.predict([line])
        cuted_list.append(discrete[0])


    df2.insert(0, "discretization_"+column, cuted_list, True)
    x = range(0, len(df[["price"]]))
    if plot:
        sns.scatterplot(x=x, y=column, data=df, hue="discretization_"+column)
        plt.show()


def add_2kmeans(column1,column2,cluster, plot = False):

    kmeans = KMeans(n_clusters=cluster, random_state=0)
    df_cuted = df[[column1,column2]].values
    kmeans.fit(df_cuted)
    cuted_list = []

    for line in df_cuted:
        discrete = kmeans.predict([line])
        cuted_list.append(discrete[0])


    df2.insert(0, "discretization_"+column1+"_"+column2, cuted_list, True)


add_2kmeans("lat", "long", 2, plot=False)
add_kmeans("sqft_living", 2, plot=False)
add_kmeans("sqft_lot", 2, plot=False)
add_kmeans("bathrooms", 2, plot=False)
add_kmeans("bedrooms", 2, plot=False)
add_kmeans("floors", 2, plot=False)
add_kmeans("view", 2, plot=False)
add_kmeans("condition", 2, plot=False)
add_kmeans("grade", 3, plot=False)
add_kmeans("sqft_above", 2, plot=False)
add_kmeans("sqft_basement", 2, plot=False)
add_kmeans("yr_built", 2, plot=False)
add_kmeans("yr_renovated", 2, plot=False)
add_kmeans("sqft_living15", 2, plot=False)
add_kmeans("sqft_lot15", 2, plot=False)




df2 = df2[:-10]
df_validation = df2.tail(10)
df_validation.to_excel(path+os.sep+"interin_data"+os.sep+"database_validation.xlsx",index=False)
df_validation.to_csv(path+os.sep+"interin_data"+os.sep+"database_validation.csv", index=False)
df2.to_excel(path+os.sep+"interin_data"+os.sep+"database_train.xlsx",index=False)
df2.to_csv(path+os.sep+"interin_data"+os.sep+"database_train.csv", index=False)
print("oi")
