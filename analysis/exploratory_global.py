#!/usr/bin/env python
# coding: utf-8
# %%
import os
import pandas as pd
import json
import numpy as np


# %%
import utils


# %% [markdown]
# # General patterns

# %%
import pandas as pd


# %%
from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual

# %%
# %load_ext autoreload

# %%
# %autoreload 2

# %%
# twitter_client = utils.TwitterClient()


# %%
politicians = utils.load_users('politicians', party=True)


# %%
random_friends = utils.load_users('random-friends')


# %%
random_followers = utils.load_users('random-followers')

# %%
journalists = utils.load_users('journalists')

# %% [markdown]
# # Plot interactive

# %%
politicians['type']=0
journalists['type']=1
random_friends['type']=2
random_followers['type']=3
all_users = pd.concat([politicians,journalists,random_friends, random_followers], sort=False)


# %%
all_users['party']=all_users['party'].fillna('--')
all_users_no_null = utils.filter_users(all_users[-all_users.isnull().any(axis=1)])


# %% [markdown]
# ### random-friends vs random-followers

# %%
utils.create_interactive_viz(pd.concat([all_users_no_null[all_users_no_null['type'].isin([2,3])]]))

# %% [markdown]
# # Distribution of outliers

# %%
from pylab import rcParams
rcParams['figure.figsize'] = 15, 8


# %%
import scipy.stats as st

# %%
import matplotlib.pyplot as plt


# %%
def mean_hist(series,ax, n=1000, prop=0.7, **kwargs):
    sample_size = int(prop*len(series))
    mean_hist = [np.sort(series.sample(sample_size).values) for i in range(n)]
    ax.hist(np.mean(mean_hist, axis=0), density=1, alpha=0.7, **kwargs)
    ax.legend([kwargs['label']])


# %%
politicians_f = utils.filter_users(politicians)
journalists_f = utils.filter_users(journalists)
random_followers_f = utils.filter_users(random_followers)
random_friends_f = utils.filter_users(random_friends)

# %%
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
r1 = mean_hist(random_friends_f['num_outliers_2'],  prop=0.5, n=100, label='random_following', ax=ax1)
r2 = mean_hist(random_followers_f['num_outliers_2'],  prop=0.5, n=100, label='random_followers',ax=ax2)
p = mean_hist(politicians_f['num_outliers_2'], prop=0.5, n=100, label='politicians', ax=ax3)
j = mean_hist(journalists_f['num_outliers_2'], prop=0.5, n=100,  label='journalists',ax=ax4)
plt.title('Outliers distribution')

# %% [markdown]
# ## Fit distributions

# %%
utils.plot_best_args(journalists_f['num_outliers_2'].value_counts().sort_index(), 'gamma', title='journalists')


# %%
utils.plot_best_args(politicians_f['num_outliers_2'].value_counts().sort_index(), 'gamma', title='politicians')

# %%
utils.plot_best_args(random_friends_f['num_outliers_2'].value_counts().sort_index(), 'gamma', 'random_friends')

# %%
utils.plot_best_args(random_followers_f['num_outliers_2'].value_counts().sort_index(), 'gamma', 'random_friends')

# %% [markdown]
# ## Dataframe p_values

# %%
dbs = [politicians_f, journalists_f, random_friends_f, random_followers_f]
names = ['politicians', 'journalists', 'random_friends', 'random_followers']
dict_pvalue = {}
dict_statistic = {}
for i, db_1 in enumerate(dbs):
    dict_pvalue[names[i]] = {}
    dict_statistic[names[i]] = {}
    for j, db_2 in enumerate(dbs):
        tests = utils.compare_two_distributions(db_1['num_outliers_2'], db_2['num_outliers_2'], prop=0.8, n=1000)
        dict_statistic[names[i]][names[j]]=tests[0]
        dict_pvalue[names[i]][names[j]]=tests[1]
    

# %%
pd.DataFrame.from_dict(dict_pvalue).round(7)

# %%

# %% [markdown]
# # Clustering

# %%
from sklearn.decomposition import PCA #Principal Component Analysis
from sklearn.manifold import TSNE #T-Distributed Stochastic Neighbor Embedding
from sklearn.cluster import KMeans #K-Means Clustering
from sklearn.preprocessing import StandardScaler #used for 'Feature Scaling'


# %%
all_users_no_null = politicians[-politicians.isnull().any(axis=1)]

# %%
plot(all_users_no_null, ['like_tweet_ratio', 'social_ratio', 'rt_sratio'], color_column)


# %%
def get_PCA(df, dim=3, columns=[]):
    pca = PCA(n_components=dim)
    if len(columns) == 0:
        columns = df.columns
    scaler = StandardScaler()
    df_pca = pd.DataFrame(scaler.fit_transform(df[columns].copy()), columns=columns, index=df.index )
    df_final = df.copy()
    columns_pca = ['PCA_{}'.format(i) for i in range(dim)]
    df_final[columns_pca] = pd.DataFrame(pca.fit_transform(df_pca),columns=columns_pca, index=df_pca.index)
    return df_final, columns_pca, pca


# %%
def get_tSNE(df, dim=3, perplexity=50, columns=[]):
    from sklearn.manifold import TSNE #T-Distributed Stochastic Neighbor Embedding
    tsne = TSNE(n_components=dim, perplexity=perplexity)
    if len(columns) == 0:
        columns = df.columns
    scaler = StandardScaler()
    df_tsne = pd.DataFrame(scaler.fit_transform(df[columns].copy()), columns=columns, index=df.index )
    df_final = df.copy()
    columns_tsne = ['tSNE_{}'.format(i) for i in range(dim)]
    df_final[columns_tsne] = pd.DataFrame(tsne.fit_transform(df_tsne),columns=columns_tsne, index=df_tsne.index)
    return df_final, columns_tsne, tsne


# %%
columns = ['frequency_timeline', 'frequency_like', 'like_tweet_ratio',
       'social_ratio', '1_2_ratios']


# %%
# all_users_no_null= all_users_no_null[all_users_no_null.index!='ierrejon'].copy()
len(all_users_no_null)

# %%
df_pca, columns_pca, pca = get_PCA(all_users_no_null,dim=3, columns=columns)
df_tsne, columns_tsne, tsne = get_tSNE(df_pca, dim=3, columns=columns)


# %%
df_tsne.columns

# %%
color_column = 'type'
#df_tsne = df_tsne[columns+columns_tsne+columns_pca+[color_column]]

# %%
plot(df_tsne, columns_pca, color_column)


# %%
df_pca_politicians, columns_pca, pca = get_PCA(politicians[-politicians.isnull().any(axis=1)],dim=3, columns=columns)

# %%
columns

# %%
pca.components_

# %%
plot(df_pca_politicians, columns_pca, 'party')

# %%
plot(df_tsne[df_tsne['type']==0], columns_pca, 'party')

# %%
plot(df_tsne, columns_tsne, column_to_color=color_column)


# %% [markdown]
# ## Clustering by distribution num outliers

# %%
st.wasserstein_distance


# %%
def load_user_distribution_action_users(db,root='./data', party=False):
    import os
    path = os.path.join(root,db)
    dict_distributions = {}
    dict_party = {}
    for p in os.listdir(path):
        path_file = os.path.join(path,p)
        actions = pd.read_pickle(path_file).reset_index()
        if len(actions)>2:     
            if party:
                dict_party[p[:-4]] = get_political_party(utils.user_frequency(actions))
            timeline = actions[actions['type']!='Like']
            u_freq = utils.user_frequency(timeline, skip_ones=True)
            if len(u_freq)>1:
                dict_distributions[p[:-4]] = u_freq
    return dict_distributions, dict_party


# %%
def load_db_profiles(db,root='./data', db_profiles=None, dict_party=None):
    db_profiles = pd.read_pickle('./data/'+db+'.pkl')
    user_profile = ['followers_count','friends_count', 'verified', 'statuses_count','favourites_count']
    if dict_party:
        series_party = pd.Series(dict_party)
        db_profiles.loc[series_party.index,'party']=series_party.fillna('--')
        user_profile+=['party']
    return db_profiles[user_profile]
    


# %%
key = 'politicians'

# %%
dist_politicians, dict_party_p = load_user_distribution_action_users('politicians') 
# dist_politicians, dict_party_f = load_user_distribution_action_users('') 
# dict_party

# %%
dist_journalists, dict_party_p = load_user_distribution_action_users('journalists') 

# %%
dist_random_followers, dict_party_r = load_user_distribution_action_users('random-followers') 

# %%
dist_random_friends, dict_party_r = load_user_distribution_action_users('random-friends') 

# %%
# dist_politicians
series_party = pd.Series(dict_party)

# %%
# series_party
series_party

# %%
# series_party = pd.Series(dict_party)
# df = pd.DataFrame(index=dist_politicians.keys())
# user_profile = ['followers_count','friends_count', 'verified', 'statuses_count','favourites_count','party']
# profiles = pd.read_pickle('./data/'+key+'.pkl')
# profiles.loc[series_party.index,'party']=series_party.fillna('--')

# %%
dict_big = {**dist_journalists, **dist_random_followers, **dist_random_friends, **dist_politicians}

# %%
len(dict_big)

# %%
df = pd.Series(index=list(dict_big.keys()))

# %%
df

# %%
df = pd.DataFrame(index=list(dict_big.keys()))
for i, id_u in enumerate(df.index):
    dist_u = dict_big[id_u]
    a = [st.wasserstein_distance(dict_big[id_v].values, dist_u.values) for id_v in df.index]
    df[id_u]=a

# %%
import umap
fit = umap.UMAP(n_components=2, random_state=42, metric='precomputed')

# %%
# df.head()

# %%
u = fit.fit_transform(df)

# %%
columns_umap=['UMAP_{}'.format(i) for i in range(len(u[0]))]

# %%
umap_df = pd.DataFrame(u, index=df.index, columns=columns_umap)
# umap_df = umap_df.join(profiles[user_profile])

# %%
umap_df.loc[dist_politicians.keys(), 'type']='politicians'

# %%
umap_df.loc[dist_random_followers.keys(), 'type']='random_followers'

# %%
umap_df.loc[dist_random_friends.keys(), 'type']='random_friends'

# %%
umap_df.loc[dist_journalists.keys(), 'type']='journalists'

# %%
plot(umap_df, columns_umap, column_to_color='type').write_html('dist_umap_type.html', auto_open=True)
