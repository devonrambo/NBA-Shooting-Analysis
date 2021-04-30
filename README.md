# NBA Shooting Analysis


### Description
This project has two segments: Clustering of players based on shot selection, and analysis of players based on isolation offense and defense. The data is from an amazing Kaggle database of the 2014-2015 NBA season (https://www.kaggle.com/dansbecker/nba-shot-logs)

### Motivation
I completed a couple courses on data analysis and machine learning in Python from Codecademy and wanted to test my hand at a project that was interesting to me.

### Methods
##### Shot Clustering
⦁	An affinity propagation clustering algorithm was used to classify players into different groups or “shooting styles”. <br />
⦁	The algorithm iterates through every shot taken and clusters the players based on three features of their shots: distance from the basket, distance from the nearest defender, and number of dribbles taken before the shots. <br />
⦁	The results were graphed in a 3D plot, and I attached a writeup for the different clusters. <br />
<br />

##### Isolation Offense & Defense Analysis
⦁	The database contains who the nearest defender was on every shot, as well as how long the player had possession before shooting.  <br />
⦁	Isolation was defined as having possession for two or more seconds in order to eliminate catch and shoot plays and focus on shots players create themselves.  <br />
⦁	I calculated every player’s points per possession on iso shots, points per possession allowed when defending iso shots, as well as quantity of shots taken and defended.  <br />
⦁	After that process, every shot was weighted based on the quality of the offensive and defensive player.  <br />
⦁	This was used to calculated Iso Offensive and Defensive Value Added.  <br />  <br />


### Sources

This was only possible due to the incredible database from DanB on Kaggle! The link is in the description.
