import pandas as pd
import os
import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.cluster import KMeans

path = os.path.abspath(__file__)
path = os.path.dirname(path)
print(path)
df = pd.read_excel(path+os.sep+"raw_data"+os.sep+"database.xlsx")
kmeans = KMeans(n_clusters=2, random_state=0)
df_cut = df[["lat", "long"]].values
kmeans.fit(df_cut)
disc_list = []

for line in df_cut:

    disc = kmeans.predict([line])
    disc_list.append(disc[0])
    print(disc)

df.insert(2, "discretization_lat_and_long", disc_list, True)

sns.scatterplot(x = "lat", y = "long", data = df, hue = "discretization_lat_and_long")

plt.show()

df.to_excel(path+os.sep+"interin_data"+os.sep+"database_latlong.xlsx")

print("oi")
