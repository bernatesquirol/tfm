#!/usr/bin/env python
# coding: utf-8
# %%

# %%
import pandas as pd
import numpy as np
import utils
import importlib 
importlib.reload(utils)


# %%
def get_features_timeline(actions, party=False):
    features = {}
    if len(actions)<2: return {}
    users_freq_actions = utils.user_frequency(actions)
    if party:
        features['party']=get_political_party(users_freq_actions)
    if len(users_freq_actions)>1:
        features['fit_exp'], features['fit_k'], dummie, features['fit_lambda'] = utils.get_best_args(users_freq_actions, 'exponweib')    
        features['1_2_actions']=users_freq_actions.iloc[0]/users_freq_actions.iloc[1]
        if len(users_freq_actions)>2:
            features['2_3_actions']=users_freq_actions.iloc[1]/users_freq_actions.iloc[2]        
    timeline = actions[actions['type']!='Like']
    likes = actions[actions['type']=='Like']
    
    timeline_len = (actions['created_at'].max()-timeline['created_at'].min()).days
    if timeline_len>0:
        features['frequency_timeline']= len(timeline) / timeline_len
    likes_len = (actions['created_at'].max()-likes['created_at'].min()).days
    if likes_len>0:
        features['frequency_like']= len(likes) / likes_len
    social_tweets_count = len(timeline[timeline['type']!='Text'])
    if 'frequency_timeline' in features and 'frequency_like' in features:
        features['like_tweet_ratio'] = features['frequency_like']/features['frequency_timeline']
    if len(timeline):
        features['social_ratio'] = 100*social_tweets_count/len(timeline)
    if social_tweets_count>0:
        features['reply_sratio'] = 100*(len(timeline[timeline['type']=='Reply'])/social_tweets_count)
        features['rt_sratio'] = 100*(len(timeline[timeline['type']=='RT'])/social_tweets_count)
        features['mention_sratio'] = 100*(len(timeline[timeline['type']=='Mention'])/social_tweets_count)
        features['quoted_sratio'] = 100*(len(timeline[timeline['type']=='Quoted'])/social_tweets_count)
        freq_1 = utils.user_frequency(timeline, skip_ones=False)
        if len(freq_1)>1:
            features['num_outliers_1'] = len(freq_1[utils.relevant_outliers(freq_1)])
        freq_2 = utils.user_frequency(timeline, skip_ones=True)
        if len(freq_2)>1:
            features['num_outliers_2'] =len(freq_2[utils.relevant_outliers(freq_2)])
        sorted_features = sorted([features['reply_sratio'],features['rt_sratio'],features['mention_sratio']],reverse=True)
        if sorted_features[1]>0:
            features['1_2_ratios']=sorted_features[0]/sorted_features[1]
    return features


# %%
def load_users(db,root='./data', party=False):
    import os
    path = os.path.join(root,db)
    list_politicians_dict = {}
    for p in os.listdir(path):
        path_file = os.path.join(path,p)
        p_dict = get_features_timeline(pd.read_pickle(path_file).reset_index(), party=party)
        list_politicians_dict[p[:-4]]=p_dict
    activity_profiles = pd.DataFrame(list_politicians_dict).T
    user_profile = ['followers_count','friends_count', 'verified', 'statuses_count','favourites_count']
    profiles = pd.read_pickle(path+'.pkl')
    return profiles[user_profile].join(activity_profiles)


# %%
politicians = load_users('politicians', party=False)
random_friends = load_users('random-friends')
random_followers = load_users('random-followers')
journalists = load_users('journalists')


# %%
# outliers -> out
all_users = pd.concat([politicians, random_friends, random_followers, journalists], sort=False)


# %%
users = politicians.copy()


# %%
politicians.corr()['num_outliers_2']


# %%
def filtered_users(users):
    users_filtered = users.copy()
    users_filtered = users_filtered[users_filtered['followers_count']<=2454815].copy()
    users_filtered = users_filtered[users_filtered['frequency_timeline']<=100].copy()
    users_filtered = users_filtered[users_filtered['frequency_like']<=130].copy()
    users_filtered = users_filtered[users_filtered['num_outliers_2']<=25].copy()
    users_filtered = users_filtered[users_filtered['social_ratio']>=10].copy()
    return users_filtered


# %%
new_users = filtered_users(all_users)
len(new_users)


# %%
new_users['num_outliers_1'].sort_values(ascending=False).head(50)


# %%
new_users['num_outliers_1']['ierrejon']


# %%
len(new_users[new_users['social_ratio']<10])


# %%
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


# %%
utils.create_interactive_viz(all_users)


# %%
politicians.columns


# %%
df = pd.DataFrame(np.random.randn(100, 3))

from scipy import stats
df[(np.abs(stats.zscore(df.values)) < 3).all(axis=1)]


# %%
cluster_p.values


# %%
columns = ['followers_count', 'friends_count',  'statuses_count' ]
cluster_p = politicians[columns]


# %%
stats.zscore(cluster_p.values)


# %%
from scipy import stats
cluster_p = cluster_p[(np.abs(stats.zscore(cluster_p)) < 3).all(axis=1)].copy()


# %%
X = StandardScaler().fit_transform(cluster_p[-cluster_p.isnull().any(axis=1)])


# %%
db = DBSCAN(eps=0.3, min_samples=100).fit(X)


# %%
np.where(db.labels_!=-1)[0].shape


# %%
db.labels_

