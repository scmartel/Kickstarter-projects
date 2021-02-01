# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 20:00:39 2020

@author: Sophie
"""

#### 3 MODELS WITH FEATURE SELECTION & CROSS VALIDATION ####
##for each model, the code that was used to select the features was commented out such that running the whole file would only print the scores of the 3 models
## simply remove comments to see the results of every step -- this was removed in the grading py file


#import the packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances


from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import plotly



################################### STEP 1 - FEATURE ENGINEERING & CLEANING ##############################################

#import dataset
kickstarter_df = pd.read_excel(r"C:\Users\Sophie\Downloads\Kickstarter.xlsx")
new_dataset = kickstarter_df.drop(['launch_to_state_change_days'], axis = 1) #drop column that was empty

#only successful or failed projects
states = ['failed', 'successful'] #only select successful or failed projects
kickstarters = new_dataset[new_dataset.state.isin(states)]

kickstarters = kickstarters.dropna() #drop empty rows
kickstarters = kickstarters.reset_index(drop=True)


############ Detecting the anomalies in the dataset ##############
model=IsolationForest(n_estimators=100, max_samples='auto', contamination=float(0.01),max_features=1.0, random_state = 0)
model.fit(kickstarters[['goal']])

#identify anomalies
kickstarters['score']=model.decision_function(kickstarters[['goal']])
kickstarters['anomaly']=model.predict(kickstarters[['goal']])
kickstarters.head(20)

#index of observations that are anomalies
anomaly=kickstarters.loc[kickstarters['anomaly']==-1]
anomaly_index=list(anomaly.index)

#only keep projects that are not anomalies
kickstarters = kickstarters[~kickstarters.index.isin(anomaly_index)]




############# Calculated fields ##################################

#1 - lenght of the name of the project
kickstarters['name_length'] = kickstarters['name'].str.len()

long_names = kickstarters.loc[kickstarters.name_length >= kickstarters['name_length'].median()]

kickstarters = kickstarters.assign(Long_Name = [1 if a >= kickstarters['name_length'].median() else 0 for a in kickstarters['name_length']])


#2 - converting the goals in USD
goals_US = []

I = range(len(kickstarters))

for i in I:
    currency = kickstarters.iloc[i, 7]
    goal = kickstarters.iloc[i, 2]
    usd_pledged = kickstarters.iloc[i, 15]
    static_usd_rate = kickstarters.iloc[i, 14]
    
    if currency == 'USD':
        goals_US.append(goal)
    else:
        goal_in_USD = goal*static_usd_rate
        goals_US.append(goal_in_USD)

kickstarters = kickstarters.assign(goals_in_USD = goals_US)
kickstarters['goals_in_USD'] = round(kickstarters['goals_in_USD'],2)



#3 - countries simplified
countries = ['USD', 'CA', 'GB', 'AU', 'DE', 'NL']
Country_simplified = []

I = range(len(kickstarters))

for i in I:
    country = kickstarters.iloc[i, 6]
        
    if country in countries:
        Country_simplified.append(country)
    else:
        Country_simplified.append('other')

kickstarters = kickstarters.assign(country_simplified = Country_simplified)


#4 - simplified categories
categories = ['Hardware', 'Web', 'Software', 'Gadgets', 'Plays', 'Apps', 'Wearables', 'Musical', 'Sound', 'Festivals', 'Robots', 'Flight', 'Experimental', 'Immersive', 'Markerspaces', 'Spaces']
Category_simplified = []

I = range(len(kickstarters))

for i in I:
    category = kickstarters.iloc[i, 16]
        
    if category in categories:
        Category_simplified.append(category)
    else:
        Category_simplified.append('other')

kickstarters = kickstarters.assign(category_simplified = Category_simplified)


##############################################################################
##############################################################################
##############################################################################
##############################################################################


############################################## STEP 2 - BUILDING THE MODELS ###################################################


############################### MODEL 1 - REGRESSION ###################################

#drop columns that can't be used or are not usefull
KS_part1 = kickstarters.drop(['score','anomaly','staff_pick','project_id','name', 'state', 'deadline','goal', 'country', 'pledged', 'disable_communication', 'currency', 'state_changed_at', 'created_at', 'launched_at', 'backers_count', 'static_usd_rate', 'category', 'spotlight', 'name_len', 'blurb_len','state_changed_at_weekday','state_changed_at_month',
       'state_changed_at_day', 'state_changed_at_yr', 'state_changed_at_hr','name_length'], axis = 1)


##### PCA FOR FEATURE SELECTION ###

# 1) create dummy variables
KS = pd.get_dummies(KS_part1, columns = ['country_simplified', 'category_simplified','deadline_weekday', 'created_at_weekday','launched_at_weekday','launched_at_weekday'])

# 2) seperate into X and Y
X = KS.loc[:, KS.columns != 'usd_pledged']
y = KS.usd_pledged


# 3) Transforming the data
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# 4) Splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=123)

# Fitting PCA
pca = PCA()
pca.fit_transform(X_std)

explained_var = np.cumsum(pca.explained_variance_ratio_)


# Plotting the amount of variation explained by PCA with different numbers of components
plt.plot(list(range(1, len(explained_var)+1)), explained_var)
plt.title('Amount of variation explained by PCA', fontsize=14)
plt.xlabel('Number of components')
plt.ylabel('Explained variance');

pca.explained_variance_ratio_


##### LASSO FOR FEATURE SELECTION ####

# 1) create dummy variables

KS = pd.get_dummies(KS_part1, columns = ['country_simplified', 'category_simplified','deadline_weekday', 'created_at_weekday','launched_at_weekday','launched_at_weekday'])

# 2) seperate into X and Y
X = KS.loc[:, KS.columns != 'usd_pledged']
y = KS.usd_pledged
# 2) seperate into X and Y
X = KS.loc[:, KS.columns != 'usd_pledged']
y = KS[['usd_pledged']]


# 3) Transforming the data
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

model = Lasso(alpha = 0.1, normalize= True)
model.fit(X_std, y)
model.coef_

coefs = pd.DataFrame(list(zip(X.columns,model.coef_)), columns = ['predictor','coefficient'])
coefs = coefs.sort_values('coefficient', ascending = [False])
nonneg = coefs[coefs['coefficient'] != 0]




##### RANDOM FOREST ######

KS = pd.get_dummies(KS_part1)


# 2) seperate into X and Y
X = KS.loc[:, KS.columns != 'usd_pledged']
y = KS.usd_pledged


from sklearn.ensemble import RandomForestRegressor
randomforest = RandomForestRegressor(random_state = 123)

model = randomforest.fit(X,y)

model.feature_importances_
#captures the impurity of the leaf nodes in the trees that is reduced by the predictors
#the higher the score, the more important the predictor is 

#better format 
df = pd.DataFrame(list(zip(X.columns,model.feature_importances_)), columns = ['predictor','feature importance'])
df = df.sort_values('feature importance', ascending = [False])

##common threshold is 0.05
df_top_preds = df[df['feature importance'] >= 0.01]
df_top_preds
preds = df_top_preds['predictor'].tolist()



### THE BEST FEATURES FOR REGRESSION ###
### this is the best one on linear regression
KS = pd.get_dummies(KS_part1)
PREDS2 = KS[['goals_in_USD',
 'create_to_launch_days',
 'created_at_day',
 'launched_at_weekday_Tuesday',
 'created_at_hr',
 'launch_to_deadline_days',
 'category_simplified_Sound',
 'launched_at_day',
 'launched_at_hr',
 'name_len_clean',
 'deadline_hr',
 'blurb_len_clean',
 'deadline_day',
 'created_at_month',
 'deadline_month',
 'deadline_weekday_Saturday',
 'created_at_yr',
 'deadline_weekday_Tuesday',
 'category_simplified_Wearables',
 'launched_at_weekday_Thursday',
 'category_simplified_Hardware',
 'launched_at_yr',
 'created_at_weekday_Monday',
 'deadline_weekday_Thursday',
           'usd_pledged']] #predicted variable

###hyperparameters tuning

# Construct variables
X = PREDS2.loc[:, PREDS2.columns != 'usd_pledged']
y = PREDS2.usd_pledged


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)


#Build the model 
GBT = GradientBoostingRegressor(random_state=0)

parameters = {
    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1],
    "max_depth":[3,5,8],
    "n_estimators":[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    }

gridS = GridSearchCV(GBT, parameters, n_jobs=-1, cv=5)
gridS.fit(X, y)

print('Best parameters found:\n', gridS.best_params_, '\nAccuracy score:\n', gridS.best_score_)


### GREADIENT BOOSTING REGRESSOR - best model performance out of all models tested
X = PREDS2.loc[:, PREDS2.columns != 'usd_pledged']
y = PREDS2.usd_pledged

# Separate the data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33, random_state = 0)

#with min max scaler 
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Run random forest
gbt = GradientBoostingRegressor(random_state = 0, n_estimators =100)
model1 = gbt.fit(X_train, y_train)


# Using the model to predict the results based on the test dataset
y_test_pred = model1.predict(X_test)

# Calculate the mean squared error of the prediction
mse = mean_squared_error(y_test, y_test_pred)
print('MSE of model 1 - gradient boosting regressor: ', mse)


### GBT cross validation

X = PREDS2.loc[:, PREDS2.columns != 'usd_pledged']
y = PREDS2.usd_pledged

#no need to standardize the model with GBT

# Separate the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33, random_state = 0)

#with min max scaler 
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

for i in range(2,10):
    gbt = GradientBoostingRegressor(random_state=0, min_samples_split=i, n_estimators = 100, learning_rate = 0.2)
    model5 = gbt.fit(X_train, y_train)
    negative_mse = cross_val_score(estimator=model5, X=X, y=y, scoring = 'neg_mean_squared_error', cv = 10)
    print('MSE: ', -np.average(negative_mse))



### FEATURE IMPORTANCE

importance = model1.feature_importances_

# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
    
# plot feature importance
from matplotlib import pyplot
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

columns = X.columns

for i in range(len(columns)):
    print(i, 'Feature: ', columns[i], ' Importance: ', importance[i])



##################################################################################################################
##################################################################################################################
##################################################################################################################



########################################## MODEL 2 - CLASSIFICATION #################################################

#dropping columns that cannot be used for classification 
KS_part2 = kickstarters.drop(['staff_pick','project_id','name', 'usd_pledged','deadline','goal', 'country', 'pledged', 'disable_communication', 'currency', 'state_changed_at', 'created_at', 'launched_at', 'backers_count', 'static_usd_rate', 'category', 'spotlight', 'name_len', 'blurb_len','state_changed_at_weekday','state_changed_at_month',
       'state_changed_at_day', 'state_changed_at_yr', 'state_changed_at_hr','name_length', 'anomaly', 'score', 'staff_pick'], axis = 1)

#KS_part2.info()

#changing state to binary variable and dummifying predictors
KS= KS_part2.assign(state = [1 if a == 'successful' else 0 for a in KS_part2['state']])
KS = pd.get_dummies(KS)
#KS.info()


###### FEATURE SELECTION ######

X = KS.drop('state', axis = 1)
y = KS['state']

randomforest = RandomForestClassifier(random_state = 0)

model = randomforest.fit(X,y)

model.feature_importances_
#captures the impurity of the leaf nodes in thetrees that is reduced by the predictors
#the higher the score, the more importance the predictor is 

#better format 
df = pd.DataFrame(list(zip(X.columns,model.feature_importances_)), columns = ['predictor','feature importance'])
df = df.sort_values('feature importance', ascending = [False])


##common threshold is 0.05
df_top_preds = df[df['feature importance'] >= 0.01]
df_top_preds
preds = df_top_preds['predictor'].tolist()



PREDS5 = KS[['goals_in_USD',
 'create_to_launch_days',
 'created_at_day',
 'launched_at_hr',
 'created_at_hr',
 'deadline_day',
 'launch_to_deadline_days',
 'deadline_hr',
 'launched_at_day',
 'name_len_clean',
 'category_simplified_Web',
 'blurb_len_clean',
 'created_at_month',
 'deadline_month',
 'category_simplified_Software',
 'deadline_yr',
 'launched_at_yr',
 'category_simplified_Plays',
 'Long_Name',
            'state']]


##### GRADIENT BOOSTING CLASSIFIER MODEL ######

#predictors
X = PREDS5.drop('state', axis = 1)
y = PREDS5['state']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 123)

#Scale data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#build the model
clf = GradientBoostingClassifier(random_state=0, min_samples_split=7, n_estimators = 90)
model2 = clf.fit(X_train, y_train)

y_test_pred = model2.predict(X_test)

##Accuracy
print("Accuracy score (training) classifier model: {0:.3f}".format(clf.score(X_train, y_train)))
print("Accuracy score (validation) classifier model: {0:.3f}".format(clf.score(X_test, y_test)))

#precision 
print('Precision score classifier model:', precision_score(y_test, y_test_pred))

#recall
print('Recall score classifier model:',recall_score(y_test, y_test_pred))

#f1 score 
print('F1 score classifier model: ', f1_score(y_test, y_test_pred))

#### CROSS VALIDATION 

X = PREDS5.drop('state', axis = 1)
y = PREDS5['state']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_logit, y_logit, test_size = 0.33, random_state = 123)

#Scale data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#build the model
clf = GradientBoostingClassifier(random_state=0)
clf.fit(X_train, y_train)

for i in range(2,10):
    model2 = GradientBoostingClassifier(random_state=0, min_samples_split=i, n_estimators = 90)
    scores = cross_val_score(estimator=model2, X=X, y=y)
    print(i, ':', np.average(scores))


##### MODEL FEATURE IMPORTANCE

X = PREDS5.drop('state', axis = 1)
y = PREDS5['state']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 123)

#Scale data - in the example, he didnt scale the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model2 = GradientBoostingClassifier(random_state=0, min_samples_split=7, n_estimators = 90)
model_GBT = model2.fit(X_train, y_train)
importance = model_GBT.feature_importances_

# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
    
# plot feature importance
from matplotlib import pyplot
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


columns = X.columns

for i in range(len(columns)):
    print(i, 'Feature: ', columns[i], ' Importance: ', importance[i])




##################################################################################################################
##################################################################################################################
##################################################################################################################




#################################### MODEL 3 - CLUSTERING  ############################################

KS_part3 = kickstarters.drop(['project_id',
                              'name',
                              'deadline', #we have simplified
                              'goal', #we have USD version
                              'country', #we simplified
                              'disable_communication', #same for all observatinos
                              'currency', #no need
                              'state_changed_at', #correlated
                              'created_at', #simplified
                              'launched_at', #simplified
                              'static_usd_rate', #used for goal conversion but not useful predictor
                              'category',  #simplified
                              'name_len', #used the clean version
                              'blurb_len', #used the clean version
                              'state_changed_at_weekday', #state changes and deadlines are correlated 
                              'state_changed_at_month', 
                               'state_changed_at_day',
                              'state_changed_at_yr',
                              'state_changed_at_hr',
                              'Long_Name',
                              'name_length', #correlated
                              'anomaly', #anomaly detection
                              'score'], axis = 1)


KS= KS_part3.assign(state = [1 if a == 'successful' else 0 for a in KS_part3['state']])
KS = pd.get_dummies(KS)


########### FEATURE SELECTION


#### PCA #####

df = KS

class PCA(object):
    def __init__(self, n_features, q=None):
        self.q = q
        self.n_features = n_features

    def fit(self, X):
        if not self.q:
            self.q = X.shape[1]

        sc = StandardScaler()
        X = sc.fit_transform(X)

        pca = PCA(n_components=self.q).fit(X) # calculation Cov matrix is embeded in PCA
        A_q = pca.components_.T

        kmeans = KMeans(n_clusters=self.n_features).fit(A_q)
        clusters = kmeans.predict(A_q)
        cluster_centers = kmeans.cluster_centers_

        dists = defaultdict(list)
        for i, c in enumerate(clusters):
            dist = euclidean_distances([A_q[i, :]], [cluster_centers[c, :]])[0][0]
            dists[c].append((i, dist))

        self.indices_ = [sorted(f, key=lambda x: x[1])[0][0] for f in dists.values()]
        self.features_ = X[:, self.indices_]
        
# Usage
pfa = PCA(n_features=10)
pfa.fit(df)

# To get the transformed matrix
x = pfa.features_
print(x)

# To get the column indices of the kept features
column_indices = pfa.indices_
print(column_indices)

predictors = KS.iloc[:, [4, 19, 5, 45, 17, 33, 30, 34, 53, 40]]
predictors.columns



#################### ELBOW METHOD ##############################

distortions = []

#predictors
X = KS[['goals_in_USD', 'create_to_launch_days', 'launched_at_yr', 'spotlight', 'staff_pick']]


#standardize
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

#cluster inertia
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(X)
    distortions.append(kmeanModel.inertia_)
    
plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()


############## BUILDING THE MODEL - KMEANS ######################
#predictors
X = KS[['goals_in_USD', 'backers_count', 'launched_at_yr', 'spotlight']]


#standardize data
scaler = StandardScaler()
X_std = scaler.fit_transform(X)


kmeans = KMeans(n_clusters = 6) #hyperparameter k determined with elbow method
model = kmeans.fit(X_std)
labels = model.predict(X_std) 

# Plot cluster membership
#silhouette score for each observation and its label 
silhouette = silhouette_samples(X_std,labels)

#average silhouette score for the entire model
print('Silhouette score of clustering model: ', silhouette_score(X_std,labels))

#silhouette score for the different clusters 
df = pd.DataFrame({'label':labels,'silhouette':silhouette})

print('Average Silhouette Score for Cluster 0: ',np.average(df[df['label'] == 0].silhouette))
print('Average Silhouette Score for Cluster 1: ',np.average(df[df['label'] == 1].silhouette))
print('Average Silhouette Score for Cluster 2: ',np.average(df[df['label'] == 2].silhouette))
print('Average Silhouette Score for Cluster 3: ',np.average(df[df['label'] == 3].silhouette))
print('Average Silhouette Score for Cluster 4: ',np.average(df[df['label'] == 4].silhouette))
print('Average Silhouette Score for Cluster 5: ',np.average(df[df['label'] == 5].silhouette))






##################### THE REST OF THIS FILE IS CLUSTER INTERPRETATION (visualization) ################################


####### IDENTIFYING CLUSTERS FOR INTERPRETATION ########



clusters = X

#assign clusters label to each observation
clusters = clusters.assign(CN = labels)

#group the observations by cluster
grouped = clusters.groupby(['CN'], sort = True)

#compute sum for every column in every group 
averages = grouped.mean()

#average goal and backers in thousands and hundreds
averages['goals_in_USD_in_thousands'] = averages['goals_in_USD']/1000
averages['backers_count_in_thousands'] = averages['backers_count']/1000


averages.assign(cluster = [1,2,3,4,5,6])


### HEATMAP OF GOALS AND BACKERS COUNT

DF1 = averages[['goals_in_USD', 'backers_count']]

data = [go.Heatmap( z=DF1.values.tolist(), 
                   y=['cluster 1', 'cluster 2', 'cluster 3', 'cluster 4', 'cluster 5', 'cluster 6'],
                   x=['Goals USD in thousands', 'backers_count'],
                   colorscale='Viridis')]

plot = plotly.offline.iplot(data, filename='pandas-heatmap')
plot 


### HEATMAP OF SPOTLIGHT PROJECTS

DF1 = averages[['spotlight']]

data = [go.Heatmap( z=DF1.values.tolist(), 
                   y=['cluster 1', 'cluster 2', 'cluster 3', 'cluster 4', 'cluster 5', 'cluster 6'],
                   x=['spotlight'],
                   colorscale='Viridis')]

plotly.offline.iplot(data, filename='pandas-heatmap')


### SNAKE PLOT OF STANDARDIZED GOALS AND BACKERS

test = averages

test2 = np.log(test[['goals_in_USD', 'backers_count']])
test2 = test2.assign(cluster = [1,2,3,4,5,6])

trace1 = go.Scatter(
    y=test2.goals_in_USD,
    x=test2.cluster,
    mode = 'lines',
    name = 'goals in USD')

trace2 = go.Scatter(
    y=test2.backers_count,
    x=test2.cluster,
    mode = 'lines',
    name = 'backers count')


data = [trace1, trace2]
layout = go.Layout(title = 'Standardized goals in USD and backers count per cluster')
figure = go.Figure(data = data, layout = layout)
figure.show()


#CLUSTER'S GOALS


clusters = X

#assign clusters label to each observation
clusters = clusters.assign(CN = labels)
clusters

C1 = clusters[clusters['CN'] == 0]
#value counts
print("C1 goals between 0 - 10 000:",C1[(C1["goals_in_USD"] >= 0)&(C1["goals_in_USD"]<=10000)].count()[0])
print("C1 goals between 10 000 - 50 000:",C1[(C1["goals_in_USD"] >= 10000)&(C1["goals_in_USD"]<=50000)].count()[0])
print("C1 goals between 50 000 - 100 000:",C1[(C1["goals_in_USD"] >= 50000)&(C1["goals_in_USD"]<=100000)].count()[0])
print("C1 goals between 100 000 - 150 000:",C1[(C1["goals_in_USD"] >= 100000)&(C1["goals_in_USD"]<=15000)].count()[0])
print("C1 goals between 150 000 - 200 000:",C1[(C1["goals_in_USD"] >= 150000)&(C1["goals_in_USD"]<=200000)].count()[0])
print("C1 goals above 200 000:",C1[C1["goals_in_USD"] >= 200000].count()[0])

print("-" * 19 + "\ncluster 2:")

#cluster 2
C2 = clusters[clusters['CN'] == 1]
#value counts
print("C2 goals between 0 - 10 000:",C2[(C2["goals_in_USD"] >= 0)&(C2["goals_in_USD"]<=10000)].count()[0])
print("C2 goals between 10 000 - 50 000:",C2[(C2["goals_in_USD"] >= 10000)&(C2["goals_in_USD"]<=50000)].count()[0])
print("C2 goals between 50 000 - 100 000:",C2[(C2["goals_in_USD"] >= 50000)&(C2["goals_in_USD"]<=100000)].count()[0])
print("C2 goals between 100 000 - 150 000:",C2[(C2["goals_in_USD"] >= 100000)&(C2["goals_in_USD"]<=15000)].count()[0])
print("C3 goals between 150 000 - 200 000:",C2[(C2["goals_in_USD"] >= 150000)&(C2["goals_in_USD"]<=200000)].count()[0])
print("C4 goals above 200 000:",C2[C2["goals_in_USD"] >= 200000].count()[0])

print("-" * 19 + "\ncluster 3:")

#cluster 3
C3 = clusters[clusters['CN'] == 2]
#value counts
print("C3 goals between 0 - 10 000:",C3[(C3["goals_in_USD"] >= 0)&(C3["goals_in_USD"]<=10000)].count()[0])
print("C3 goals between 10 000 - 50 000:",C3[(C3["goals_in_USD"] >= 10000)&(C3["goals_in_USD"]<=50000)].count()[0])
print("C3 goals between 50 000 - 100 000:",C3[(C3["goals_in_USD"] >= 50000)&(C3["goals_in_USD"]<=100000)].count()[0])
print("C3 goals between 100 000 - 150 000:",C3[(C3["goals_in_USD"] >= 100000)&(C3["goals_in_USD"]<=15000)].count()[0])
print("C3 goals between 150 000 - 200 000:",C3[(C3["goals_in_USD"] >= 150000)&(C3["goals_in_USD"]<=200000)].count()[0])
print("C3 goals above 200 000:",C3[C3["goals_in_USD"] >= 200000].count()[0])

print("-" * 19 + "\ncluster 4:")

#cluster 4
C4 = clusters[clusters['CN'] == 3]
#value counts
print("C4 goals between 0 - 10 000:",C4[(C4["goals_in_USD"] >= 0)&(C4["goals_in_USD"]<=10000)].count()[0])
print("C4 goals between 10 000 - 50 000:",C4[(C4["goals_in_USD"] >= 10000)&(C4["goals_in_USD"]<=50000)].count()[0])
print("C4 goals between 50 000 - 100 000:",C4[(C4["goals_in_USD"] >= 50000)&(C4["goals_in_USD"]<=100000)].count()[0])
print("C4 goals between 100 000 - 150 000:",C4[(C4["goals_in_USD"] >= 100000)&(C4["goals_in_USD"]<=15000)].count()[0])
print("C4 goals between 150 000 - 200 000:",C4[(C4["goals_in_USD"] >= 150000)&(C4["goals_in_USD"]<=200000)].count()[0])
print("C4 goals above 200 000:",C4[C4["goals_in_USD"] >= 200000].count()[0])

print("-" * 19 + "\ncluster 5:")

#cluster 5
C5 = clusters[clusters['CN'] == 4]
#value counts
print("C5 goals between 0 - 10 000:",C5[(C5["goals_in_USD"] >= 0)&(C5["goals_in_USD"]<=10000)].count()[0])
print("C5 goals between 10 000 - 50 000:",C5[(C5["goals_in_USD"] >= 10000)&(C5["goals_in_USD"]<=50000)].count()[0])
print("C5 goals between 50 000 - 100 000:",C5[(C5["goals_in_USD"] >= 50000)&(C5["goals_in_USD"]<=100000)].count()[0])
print("C5 goals between 100 000 - 150 000:",C5[(C5["goals_in_USD"] >= 100000)&(C5["goals_in_USD"]<=15000)].count()[0])
print("C5 goals between 150 000 - 200 000:",C5[(C5["goals_in_USD"] >= 150000)&(C5["goals_in_USD"]<=200000)].count()[0])
print("C5 goals above 200 000:",C5[C5["goals_in_USD"] >= 200000].count()[0])

print("-" * 19 + "\ncluster 6:")

#cluster 5
C6 = clusters[clusters['CN'] == 5]
#value counts
print("C6 goals between 0 - 10 000:",C6[(C6["goals_in_USD"] >= 0)&(C6["goals_in_USD"]<=10000)].count()[0])
print("C6 goals between 10 000 - 50 000:",C6[(C6["goals_in_USD"] >= 10000)&(C6["goals_in_USD"]<=50000)].count()[0])
print("C6 goals between 50 000 - 100 000:",C6[(C6["goals_in_USD"] >= 50000)&(C6["goals_in_USD"]<=100000)].count()[0])
print("C6 goals between 100 000 - 150 000:",C6[(C6["goals_in_USD"] >= 100000)&(C6["goals_in_USD"]<=15000)].count()[0])
print("C6 goals between 150 000 - 200 000:",C6[(C6["goals_in_USD"] >= 150000)&(C6["goals_in_USD"]<=200000)].count()[0])
print("C6 goals above 200 000:",C6[C6["goals_in_USD"] >= 200000].count()[0])


### YEAR RELEASED

#cluster 1
C1 = clusters[clusters['CN'] == 0]
#value counts
print("2009-2011:",C1[(C1['launched_at_yr'] >= 2009)&(C1['launched_at_yr']<=2011)].count()[0])
print("2012-2013:",C1[(C1['launched_at_yr'] > 2011)&(C1['launched_at_yr']<=2013)].count()[0])
print("2014-2015:",C1[(C1['launched_at_yr'] > 2013)&(C1['launched_at_yr']<=2015)].count()[0])
print("2016-2017:",C1[(C1['launched_at_yr'] > 2015)&(C1['launched_at_yr']<= 2017)].count()[0])


print("-" * 19 + "\ncluster 2:")

#cluster 2
C2 = clusters[clusters['CN'] == 1]
#value counts
print("2009-2011:",C2[(C2['launched_at_yr'] >= 2009)&(C2['launched_at_yr']<=2011)].count()[0])
print("2012-2013:",C2[(C2['launched_at_yr'] > 2011)&(C2['launched_at_yr']<=2013)].count()[0])
print("2014-2015:",C2[(C2['launched_at_yr'] > 2013)&(C2['launched_at_yr']<=2015)].count()[0])
print("2016-2017:",C2[(C2['launched_at_yr'] > 2015)&(C2['launched_at_yr']<= 2017)].count()[0])

print("-" * 19 + "\ncluster 3:")

#cluster 3
C3 = clusters[clusters['CN'] == 2]
#value counts
print("2009-2011:",C3[(C3['launched_at_yr'] >= 2009)&(C3['launched_at_yr']<=2011)].count()[0])
print("2012-2013:",C3[(C3['launched_at_yr'] > 2011)&(C3['launched_at_yr']<=2013)].count()[0])
print("2014-2015:",C3[(C3['launched_at_yr'] > 2013)&(C3['launched_at_yr']<=2015)].count()[0])
print("2016-2017:",C3[(C3['launched_at_yr'] > 2015)&(C3['launched_at_yr']<= 2017)].count()[0])

print("-" * 19 + "\ncluster 4:")

#cluster 4
C4 = clusters[clusters['CN'] == 3]
#value counts
print("2009-2011:",C4[(C4['launched_at_yr'] >= 2009)&(C4['launched_at_yr']<=2011)].count()[0])
print("2012-2013:",C4[(C4['launched_at_yr'] > 2011)&(C4['launched_at_yr']<=2013)].count()[0])
print("2014-2015:",C4[(C4['launched_at_yr'] > 2013)&(C4['launched_at_yr']<=2015)].count()[0])
print("2016-2017:",C4[(C4['launched_at_yr'] > 2015)&(C4['launched_at_yr']<= 2017)].count()[0])

print("-" * 19 + "\ncluster 5:")

#cluster 5
C5 = clusters[clusters['CN'] == 4]
#value counts
print("2009-2011:",C5[(C5['launched_at_yr'] >= 2009)&(C5['launched_at_yr']<=2011)].count()[0])
print("2012-2013:",C5[(C5['launched_at_yr'] > 2011)&(C5['launched_at_yr']<=2013)].count()[0])
print("2014-2015:",C5[(C5['launched_at_yr'] > 2013)&(C5['launched_at_yr']<=2015)].count()[0])
print("2016-2017:",C5[(C5['launched_at_yr'] > 2015)&(C5['launched_at_yr']<= 2017)].count()[0])

print("-" * 19 + "\ncluster 6:")

#cluster 5
C6 = clusters[clusters['CN'] == 5]
#value counts
print("2009-2011:",C6[(C6['launched_at_yr'] >= 2009)&(C6['launched_at_yr']<=2011)].count()[0])
print("2012-2013:",C6[(C6['launched_at_yr'] > 2011)&(C6['launched_at_yr']<=2013)].count()[0])
print("2014-2015:",C6[(C6['launched_at_yr'] > 2013)&(C6['launched_at_yr']<=2015)].count()[0])
print("2016-2017:",C6[(C6['launched_at_yr'] > 2015)&(C6['launched_at_yr']<= 2017)].count()[0])

data = {'row_1': ['C1','2009-2011', 0], 
        'row_2': ['C1','2012-2013', 443],
        'row_3': ['C1','2014-2015', 5797],
        'row_4': ['C1','206-2017', 2291],
        'row_5': ['C2','2009-2011', 1], 
        'row_6': ['C2','2012-2013', 25],
        'row_7': ['C2','2014-2015', 68],
        'row_8': ['C2','206-2017', 51],
        'row_9': ['C3','2009-2011', 1], 
        'row_10': ['C3','2012-2013', 32],
        'row_11': ['C3','2014-2015', 328],
        'row_12': ['C3','206-2017', 124],
        'row_13': ['C4','2009-2011', 0], 
        'row_14': ['C4','2012-2013', 380],
        'row_15': ['C4','2014-2015', 2577],
        'row_16': ['C4','206-2017', 1239],
        'row_17': ['C5','2009-2011', 0], 
        'row_18': ['C5','2012-2013', 2],
        'row_19': ['C5','2014-2015', 4],
        'row_20': ['C5','206-2017', 3],
        'row_21': ['C6','2009-2011', 369], 
        'row_22': ['C6','2012-2013', 336],
        'row_23': ['C6','2014-2015', 0],
        'row_24': ['C6','206-2017', 0],
        
       }
years = pd.DataFrame.from_dict(data, orient='index')
years.columns = ['cluster','years', 'count']

fig = px.bar(years, x="cluster", y="count", color="years", title="Years of project launch")
fig.show()


##### SPOTLIGHT PROJECTS

#cluster 1
C1 = clusters[clusters['CN'] == 0]
#value counts
print("non spotlight: ",C1[C1['spotlight'] == 0].count()[0])
print("yes spotlight: ",C1[C1['spotlight'] == 1].count()[0])



print("-" * 19 + "\ncluster 2:")

#cluster 2
C2 = clusters[clusters['CN'] == 1]
#value counts
print("non spotlight: ",C2[C2['spotlight'] == 0].count()[0])
print("yes spotlight: ",C2[C2['spotlight'] == 1].count()[0])

print("-" * 19 + "\ncluster 3:")

#cluster 3
C3 = clusters[clusters['CN'] == 2]
#value counts
print("non spotlight: ",C3[C3['spotlight'] == 0].count()[0])
print("yes spotlight: ",C3[C3['spotlight'] == 1].count()[0])

print("-" * 19 + "\ncluster 4:")

#cluster 4
C4 = clusters[clusters['CN'] == 3]
#value counts
print("non spotlight: ",C4[C4['spotlight'] == 0].count()[0])
print("yes spotlight: ",C4[C4['spotlight'] == 1].count()[0])

print("-" * 19 + "\ncluster 5:")

#cluster 5
C5 = clusters[clusters['CN'] == 4]
#value counts
print("non spotlight: ",C5[C5['spotlight'] == 0].count()[0])
print("yes spotlight: ",C5[C5['spotlight'] == 1].count()[0])

print("-" * 19 + "\ncluster 6:")

#cluster 5
C6 = clusters[clusters['CN'] == 5]
#value counts
print("non spotlight: ",C6[C6['spotlight'] == 0].count()[0])
print("yes spotlight: ",C6[C6['spotlight'] == 1].count()[0])

#visualization
spot = {'row_1': ['C1','spotlight', 8531], 
        'row_2': ['C1','non-spotlight', 0],
        'row_5': ['C2','spotlight', 0], 
        'row_6': ['C2','non-spotlight', 145],
        'row_9': ['C3','spotlight', 458], 
        'row_10': ['C3','non-spotlight', 17],
        'row_13': ['C4','spotlight', 0], 
        'row_14': ['C4','non-spotlight', 4196],
        'row_17': ['C5','spotlight', 0], 
        'row_18': ['C5','non-spotlight', 9],
        'row_21': ['C6','spotlight', 394], 
        'row_22': ['C6','non-spotlight', 311],
       }
years = pd.DataFrame.from_dict(spot, orient='index')
years.columns = ['cluster','spotlight', 'count']

fig = px.bar(years, x="cluster", y="count", color="spotlight", title="Spotlight projects per cluster")
fig.show()


