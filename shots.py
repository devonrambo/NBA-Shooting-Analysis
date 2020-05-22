import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.cluster import AffinityPropagation
from statistics import mean
import datetime
from dateutil.parser import parse
from pygam import LinearGAM, s, f

shotlog = pd.read_csv('shot_logs.csv')

# Stylistic Shooter Comparisons

print(shotlog.head(10))

player_list = shotlog['player_name'].unique()

print(player_list)
print(len(player_list))

# Make functions for players respective metrics

touch_time_list = []

def med_touch_time(x):
	for i in player_list:
		group = shotlog[shotlog['player_name'] == i]
		med = group.TOUCH_TIME.median()
		touch_time_list.append(med)

med_touch_time(player_list)

dribbles_list = []

def avg_dribble_time(x):
	for i in player_list:
		group = shotlog[shotlog['player_name'] == i]
		avg = group.DRIBBLES.mean()
		dribbles_list.append(avg)

avg_dribble_time(player_list)

shot_dist_list = []

def med_shot_distance(x):
	for i in player_list:
		group = shotlog[shotlog['player_name'] == i]
		med = group.SHOT_DIST.median()
		shot_dist_list.append(med)

med_shot_distance(player_list)

def_dist_list = []

def med_def_distance(x):
	for i in player_list:
		group = shotlog[shotlog['player_name'] == i]
		med = group.CLOSE_DEF_DIST.median()
		def_dist_list.append(med)

med_def_distance(player_list)

# make the features array

features_array = np.column_stack((dribbles_list, def_dist_list, shot_dist_list))

print(features_array)

print(features_array[5])

norm_features = normalize(features_array, axis = 0)

print(norm_features)

print(norm_features[5])

# create the model

model = AffinityPropagation()

model.fit(norm_features)

labels = model.predict(norm_features)

print(labels)


print(player_list.shape)
print(labels.shape)

clusters = zip(player_list, labels)
clusters = list(clusters)

clusters = np.array(clusters)

sorted_clusters = clusters[np.argsort(clusters[:, 1])]

print(sorted_clusters)

print(model.cluster_centers_indices_)
print(model.cluster_centers_)
print(model.n_iter_)


# make the plot

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

ax.scatter(norm_features[:,0], norm_features[:,1], norm_features[:,2], c = labels, cmap = 'rainbow', alpha = .5)
ax.text(norm_features[30,0], norm_features[30,1], norm_features[30,2], 'cluster0', alpha = .5)
ax.text(norm_features[46,0], norm_features[46,1], norm_features[46,2], 'cluster1', alpha = .5)
ax.text(norm_features[49,0], norm_features[49,1], norm_features[49,2], 'cluster2', alpha = .5)
ax.text(norm_features[71,0], norm_features[71,1], norm_features[71,2], 'cluster3', alpha = .5)
ax.text(norm_features[72,0], norm_features[72,1], norm_features[72,2], 'cluster4', alpha = .5)
ax.text(norm_features[91,0], norm_features[91,1], norm_features[91,2], 'cluster5', alpha = .5)
ax.text(norm_features[106,0], norm_features[106,1], norm_features[106,2], 'cluster6', alpha = .5)
ax.text(norm_features[135,0], norm_features[135,1], norm_features[135,2], 'cluster7', alpha = .5)
ax.text(norm_features[137,0], norm_features[137,1], norm_features[137,2], 'cluster8', alpha = .5)
ax.text(norm_features[199,0], norm_features[199,1], norm_features[199,2], 'cluster9', alpha = .5)
ax.text(norm_features[216,0], norm_features[216,1], norm_features[216,2], 'cluster10', alpha = .5)
ax.text(norm_features[236,0], norm_features[236,1], norm_features[236,2], 'cluster11', alpha = .5)
ax.text(norm_features[254,0], norm_features[254,1], norm_features[254,2], 'cluster12', alpha = .5)
ax.set_xlabel('Dribbles per Shot')
ax.set_ylabel('Distance from Defender')
ax.set_zlabel('Shot Distance')
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.set_title('2014-2015 NBA Shooting Style Clusters')

plt.show()




