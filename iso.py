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


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

shotlog = pd.read_csv('shot_logs.csv')

on_shotlog = shotlog[shotlog['TOUCH_TIME'] >= 2]

off_shotlog = shotlog[shotlog['TOUCH_TIME'] <= 1.5]

on_player_list = on_shotlog['player_name'].unique()

on_player_id_list = on_shotlog['player_id'].unique()

on_player_id_list = on_player_id_list.tolist()

on_defender_list = on_shotlog['CLOSEST_DEFENDER'].unique()

on_defender_id_list = on_shotlog['CLOSEST_DEFENDER_PLAYER_ID'].unique()

on_defender_id_list = on_defender_id_list.tolist()

iso_points_allowed = []

iso_shots_defended = []

iso_points_scored = []

iso_shots_taken = []



# defensive functions to fill out lists

def iso_defended(x):
	for i in on_defender_list:
		group = on_shotlog[on_shotlog['CLOSEST_DEFENDER'] == i]
		count = group.shape[0]
		iso_shots_defended.append(count)

iso_defended(on_defender_list)


def points_per_possession_on(x):
	for i in on_defender_list:
		group = on_shotlog[on_shotlog['CLOSEST_DEFENDER'] == i]
		avg = group.PTS.mean()
		iso_points_allowed.append(avg)


points_per_possession_on(on_defender_id_list)


# offenseive functions to fill out lists

def points_per_possession_offense_on(x):
	for i in on_player_list:
		group = on_shotlog[on_shotlog['player_name'] == i]
		avg = group.PTS.mean()
		iso_points_scored.append(avg)

points_per_possession_offense_on(on_player_list)

def iso_shots(x):
	for i in on_player_list:
		group = on_shotlog[on_shotlog['player_name'] == i]
		count = group.shape[0]
		iso_shots_taken.append(count)

iso_shots(on_player_list)


# creating new df with points per possession defensive metrics

df = pd.DataFrame({'player': on_defender_list, 'player_id': on_defender_id_list, 'CLOSEST_DEFENDER_PLAYER_ID' : on_defender_id_list, 'PPP_Allowed': iso_points_allowed, 'Shots_Defended': iso_shots_defended,})
df = df[df['Shots_Defended'] >= 100]
df = df.sort_values(by = 'PPP_Allowed', ascending = False)

# creating new df with points per possesion offensive metric

df2 = pd.DataFrame({'player': on_player_list, 'player_id': on_player_id_list, 'PPP_Scored': iso_points_scored, 'Shots_Taken' : iso_shots_taken})
df2 = df2[df2['Shots_Taken'] >= 100]
df2 = df2.sort_values(by = 'PPP_Scored', ascending = True)

# merging dataframes

df3 = pd.merge(df, df2, how='outer', on='player_id')
df4 = pd.merge(on_shotlog,df3[['player_id','PPP_Scored']],on='player_id', how='left')
df5 = pd.merge(df4,df3[['CLOSEST_DEFENDER_PLAYER_ID','PPP_Allowed']],on='CLOSEST_DEFENDER_PLAYER_ID', how='left')

# calculating net offensive rating and net deffensive rating per possession
median_PPP_Allowed = df5.PPP_Allowed.median(skipna = True)


median_PPP_Scored = df5.PPP_Scored.median(skipna = True)

df5['PPP_Allowed'] = df5['PPP_Allowed'].fillna(median_PPP_Allowed)
df5['PPP_Scored'] = df5['PPP_Scored'].fillna(median_PPP_Scored)


df5['offensive_rating'] = df5['PTS'] - df5['PPP_Allowed']
df5['defensive_rating'] = df5['PPP_Scored'] - df5['PTS']


# adding the net offensive and defensive ratings to the table that contains the stats per player (df3)

offensive_rating_list = []

def offensive_rating(x):
	for i in on_player_list:
		group = df5[df5['player_name'] == i]
		avg = group.offensive_rating.mean()
		offensive_rating_list.append(avg)

offensive_rating(on_player_list)

defensive_rating_list = []

def defensive_rating(x):
	for i in on_defender_list:
		group = df5[df5['CLOSEST_DEFENDER'] == i]
		avg = group.defensive_rating.mean()
		defensive_rating_list.append(avg)

defensive_rating(on_defender_list)

df_o_rating = pd.DataFrame({'player': on_player_list, 'player_id': on_player_id_list, 'offensive_rating': offensive_rating_list})
df_o_rating = df_o_rating.sort_values(by = 'offensive_rating', ascending = False)



df_d_rating = pd.DataFrame({'player': on_defender_list, 'player_id': on_defender_id_list, 'defensive_rating': defensive_rating_list})
df_d_rating = df_d_rating.sort_values(by = 'defensive_rating', ascending = False)


df_player = pd.merge(df3,df_o_rating[['player_id','offensive_rating']],on='player_id', how='left')
df_player = pd.merge(df_player,df_d_rating[['player_id','defensive_rating']],on='player_id', how='left')


# adding in the sum of offensive and defensive ratings based on volume to find total impact by player
# this will find who was efficient in the face of large volume (a much more valuable & difficult task)
df_player['O_Rating_Total'] = df_player['offensive_rating'] * df_player['Shots_Taken']
df_player['D_Rating_Total'] = df_player['defensive_rating'] * df_player['Shots_Defended']
del df_player['player_y']
df_player = df_player.sort_values(by = 'D_Rating_Total', ascending = False)
print(df_player[['player_x', 'D_Rating_Total']].head(20).reset_index())



search_name = input('Welcome to the 2014-2015 NBA Shooting Dashboard! \nThis Dashboard will return metrics based on isolation offense and defense \nFor our purposes isolation is definded as a player having possession of the ball for 2 or more seconds  \nThe total value added metric accounts for efficiency, volume & quality of opposing player \nEnter a player name in the format Lastname, Firstname to view their metrics: ')

def metrics_search(x):
	group = df_player[df_player['player_x'] == search_name]
	PPP_A = group.iloc[0]['PPP_Allowed']
	Shots_D = group.iloc[0]['Shots_Defended']
	PPP_S = group.iloc[0]['PPP_Scored']
	Shots_T = group.iloc[0]['Shots_Taken']
	sorted_D = df_player.sort_values(by = 'D_Rating_Total', ascending = False).reset_index()
	sorted_D = sorted_D['player_x'].tolist()
	sorted_O = df_player.sort_values(by = 'O_Rating_Total', ascending = False).reset_index()
	sorted_O = sorted_O['player_x'].tolist()
	O_Value_Rank = sorted_O.index(search_name)
	O_Value_Rank = O_Value_Rank + 1
	D_Value_Rank = sorted_D.index(search_name)
	D_Value_Rank = D_Value_Rank + 1
	PPP_A = round(PPP_A, 2)
	Shots_D = round(Shots_D, 0)
	PPP_S = round(PPP_S, 2)
	Shots_T = round(Shots_T, 0)
	print(' ')
	print(search_name)
	print(' ')
	print('Points per Possession Allowed: ' + str(PPP_A))
	print('Number of Shots Defended: ' + str(Shots_D))
	print('Points per Possesion Scored: ' + str(PPP_S))
	print('Number of Shots Taken: ' + str(Shots_T))
	print('Isolation Defense Value Added Rank: ' + str(D_Value_Rank))
	print('Isolation Offense Value Added Rank: ' + str(O_Value_Rank))


metrics_search(search_name)





