#!/usr/bin/env python
# coding: utf-8
# %%
import os
import pandas as pd
import json
import numpy as np


# %%
import utils


# %%
get_ipython().run_line_magic('load_ext', 'autoreload')

# %% [markdown]
# # Inspect one user only

# %%
import altair as alt


# %%
def get_user_final_timeline(user, type_user):
    with open(os.path.join(".\data\{}".format(type_user), user+'.json'),'r') as json_file:
            timeline=pd.DataFrame.from_dict(json.load(json_file))
    final = utils.get_retweet_and_quoted(timeline)
    screen_names = final['user'].apply(lambda x: x['screen_name'])
    all_rt_users = final['user'].apply(lambda x: x['screen_name']).value_counts()
    relevant_outlayers = utils.relevant_outlayers(all_rt_users)
    final['outlayer']=screen_names.apply(lambda user: relevant_outlayers[user])
    return final


# %%
def plot_top_users_time(user, type_user='politicians'):
    final = get_user_final_timeline(user, type_user)
    all_users_x_month = alt.Chart(final,width=400).mark_bar(
        cornerRadiusTopLeft=3,
        cornerRadiusTopRight=3
    ).encode(
        x='yearmonth(created_at):O',
        y=alt.Y('count():Q'),#, sort=alt.SortField(field="count():Q", op="distinct", order='ascending')),
        color=alt.Color('user.screen_name:N',sort='-y'),
        order=alt.Order('count():Q')
    )
    outlayer_transparency = alt.Chart(final, width=400).mark_bar(
        cornerRadiusTopLeft=3,
        cornerRadiusTopRight=3,
        opacity=0.7,
        color='black'
    ).encode(
        x='yearmonth(created_at):O',
        y=alt.Y('count():Q'),#, sort=alt.SortField(field="count():Q", op="distinct", order='ascending')),
        opacity=alt.Opacity('outlayer:O',sort='-y', scale=alt.Scale(range=[0.35,0])),
        order='order:N'
    ).transform_calculate(
        order="if (datum.outlayer, 1, 0)"
    )
    return alt.layer(all_users_x_month,outlayer_transparency).resolve_scale(color='independent')


# %%
def plot_top_rt_and_quote(user,type_user='politicians'):
    final = get_user_final_timeline(user, type_user)
    return alt.Chart(final).mark_bar(
        cornerRadiusTopLeft=3,
        cornerRadiusTopRight=3
    ).encode(
        x=alt.X('user.screen_name:N', sort='-y'),
        y=alt.Y('count():Q'),
        color='outlayer'
    )


# 

# %% [markdown]
# # General patterns

# %%
import os
import pandas as pd

from sklearn.decomposition import PCA #Principal Component Analysis
from sklearn.manifold import TSNE #T-Distributed Stochastic Neighbor Embedding
from sklearn.cluster import KMeans #K-Means Clustering
from sklearn.preprocessing import StandardScaler #used for 'Feature Scaling'

import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


# %%
from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual


# %%
def get_political_party(user_freq, user=None):
    top15 = user_freq[:15].index
    dict_parties = {
                    'psoe': ['PSOE','GaliciaCnSusana', 'socialistes_cat', 'PSLPSOE','astro_duque','psoedeandalucia','PSOELaRioja', 'PSOE_Zamora'],
                    'pp':['populares','PopularesHuesca','PPdePalencia','PPRMurcia', 'NNGG_Palos'],
                    'podemos': ['iunida', 'EnComu_Podem', 'PODEMOS', 'Podem_'],
                    'maspais':['compromis','MasPais_Es'],
                    'cc': ['coalicion'],
                    'cup':['cupnacional'],
                    'vox_es':['vox_es'],
                    'ciudadanos':['CiudadanosCs'],
                    'erc':['Esquerra_ERC','alexvallbal'],
                    'cdc': ['JuntsXCat','Pdemocratacat'],
                    'pnv': ['eajpnv'],
                    'bildu':['ehbildu'],
                    'upn':['upn_navarra'],
                    'prc':['prcantabria'],
                    'bng':['obloque']
                   }
    for key, values in dict_parties.items():
        if np.any(top15.isin(values)):
            return key
    return np.nan
            


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

# %%
def get_screen_names(path):
    return [name[:-4] for name in os.listdir(path)]


# %%
twitter_client = utils.TwitterClient()


# %%
politicians = load_users('politicians', party=True)


# %%
random_friends = load_users('random-friends')


# %%
random_followers = load_users('random-followers')

# %%
journalists = load_users('journalists')

# %% [markdown]
# # Plot interactive

# %%
politicians['type']=0
journalists['type']=1
random_friends['type']=2
random_followers['type']=3
all_users = pd.concat([politicians,journalists,random_friends, random_followers], sort=False)


# %%


# %%
all_users['party']=all_users['party'].fillna('--')
all_users_no_null = all_users[-all_users.isnull().any(axis=1)]


# %%
len(all_users_no_null)


# %%
import utils
import importlib 
importlib.reload(utils)


# %%
utils.create_interactive_viz(politicians)

# %% [markdown]
# # Visualization

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
def get_umap(df, dim=3, perplexity=50, columns=[]):
    from sklearn.manifold import TSNE #T-Distributed Stochastic Neighbor Embedding
    tsne = TSNE(n_components=dim, perplexity=perplexity)
    if len(columns) == 0:
        columns = df.columns
    scaler = StandardScaler()
    df_tsne = pd.DataFrame(scaler.fit_transform(df[columns].copy()), columns=columns, index=df.index )
    df_final = df.copy()
    columns_tsne = ['umap_{}'.format(i) for i in range(dim)]
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


# %%

# %% [markdown]
# # Distribution of outliers

# %%
from pylab import rcParams
rcParams['figure.figsize'] = 15, 8


# %%
import scipy.stats as st

# %%
check_prop_two_distributions(filter_30(politicians)['num_outliers_2'], filter_30(politicians)['num_outliers_2'])


# %%
compare_two_distributions(filter_30(politicians)['num_outliers_2'], filter_30(politicians)['num_outliers_2'], prop=0.7)


# %%
result2 = st.ks_2samp(filter_30(politicians)['num_outliers_2'].sample(150).values, filter_30(politicians)['num_outliers_2'].sample(150).values)

# %%
import importlib
importlib.reload(utils)

# %%
politicians_f = utils.filter_users(politicians)

# %%
journalists_f = utils.filter_users(journalists)

# %%
random_followers_f = utils.filter_users(random_followers)

# %%
random_friends_f = utils.filter_users(random_friends)

# %%

# %%
# politicians vs random
compare_two_distributions(filter_30(politicians)['num_outliers_2'], filter_30(random_friends)['num_outliers_2'], prop=0.8, n=1000)

# %%
# politicians vs journalists
compare_two_distributions(filter_30(politicians)['num_outliers_2'], filter_30(journalists)['num_outliers_2'], prop=0.8, n=1000)

# %%
# politicians vs journalists
compare_two_distributions(filter_30(random_friends)['num_outliers_2'], filter_30(journalists)['num_outliers_2'], prop=0.8, n=1000)

# %%
# random vs random
compare_two_distributions(filter_30(random_friends)['num_outliers_2'], filter_30(random_followers)['num_outliers_2'],  prop=0.8, n=1000)

# %%
compare_two_distributions(filter_30(politicians)['num_outliers_2'], filter_30(journalists)['num_outliers_2'], prop=0.6, n=10000)

# %%
# import matplotlib.pyplot as plt
importlib.reload(utils)


# %%
def mean_hist(series, n=1000, prop=0.7, **kwargs):
    sample_size = int(prop*len(series))
    mean_hist = [np.sort(series.sample(sample_size).values) for i in range(n)]
    plt.hist(np.mean(mean_hist, axis=0), density=1, alpha=0.3, **kwargs)


# %%
# p = mean_hist(politicians_f['num_outliers_2'], prop=0.5, n=100, label='politicians')
j = mean_hist(journalists_f['num_outliers_2'], prop=0.5, n=100,  label='journalists')
# r1 = mean_hist(random_friends_f['num_outliers_2'],  prop=0.5, n=100, label='random_friends')
r2 = mean_hist(random_followers_f['num_outliers_2'],  prop=0.5, n=100, label='random_followers')
plt.legend()#[p, j, r1, r2],['politicians', 'journalists', 'random_friends', 'random_followers'])
plt.title('Outliers distribution')

# %%
utils.plot_best_args(journalists_f['num_outliers_2'].value_counts().sort_index(), 'gamma', title='journalists')


# %%
utils.plot_best_args(politicians_f['num_outliers_2'].value_counts().sort_index(), 'gamma', title='politicians')

# %%
utils.plot_best_args(random_friends_f['num_outliers_2'].value_counts().sort_index(), 'gamma', 'random_friends')

# %%
utils.plot_best_args(random_followers_f['num_outliers_2'].value_counts().sort_index(), 'gamma', 'random_friends')

# %%
compare_two_distributions(politicians_f['num_outliers_2'], journalists_f['num_outliers_2'], prop=0.8, n=1000)

# %%
compare_two_distributions(politicians_f['num_outliers_2'], random_friends_f['num_outliers_2'], prop=0.8, n=1000)

# %%
compare_two_distributions(random_friends_f['num_outliers_2'], journalists_f['num_outliers_2'], prop=0.8, n=1000)

# %%
compare_two_distributions(random_friends_f['num_outliers_2'], random_followers_f['num_outliers_2'], prop=0.8, n=1000)

# %%
# utils.plot_best_args(politicians_f['num_outliers_2'], 'expoweib')

# %%
# p = mean_hist(filter_30(politicians)['num_outliers_2'], prop=0.5, n=100, label='politicians')
# j = mean_hist(filter_30(journalists)['num_outliers_2'], prop=0.5, n=100,  label='journalists')
# r1 = mean_hist(filter_30(random_friends)['num_outliers_2'],  prop=0.5, n=100, label='random_friends')
# r2 = mean_hist(filter_30(random_followers)['num_outliers_2'],  prop=0.5, n=100, label='random_friends')
plt.legend()#[p, j, r1, r2],['politicians', 'journalists', 'random_friends', 'random_followers'])
plt.title('Outliers distribution')

# %%
# plt.

# %%
mean_hist(filter_30(journalists)['num_outliers_2'], prop=0.5, n=100);pass

# %%
mean_hist(filter_30(random_friends)['num_outliers_2'])

# %%
mean_hist(filter_30(random_followers)['num_outliers_2'])

# %% [markdown]
# # Clustering by distribution num outliers

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
def plot(df, columns_to_plot, column_to_color=None, bins=None, columns_text=None):
    import plotly.express as px
    dim = len(columns_to_plot)
    name_index = 'index'
    if df.index.name!=None:
        name_index = df.index.name
    if columns_text==None:
        columns_text = df.columns
    df_plot = df.reset_index()
    color = None
    if column_to_color and column_to_color in df.columns:
        if column_to_color == 'k-means':
            kmeans = KMeans(n_clusters=bins)
            kmeans.fit(df.values)
            clusters = kmeans.predict(df)
            df_plot['k-means'] = pd.Series(clusters)
            color='k-means'
        elif bins:
            df_plot[column_to_color+'_bins'] = pd.qcut(df_plot[column_to_color], q=bins, labels=range(bins)).astype(int)
            color = column_to_color+'_bins'
        else:
            color = column_to_color
    if dim==3:
        fig = px.scatter_3d(df_plot, 
                        x=columns_to_plot[0],
                        y=columns_to_plot[1], 
                        z=columns_to_plot[2],
                        color=color,
                        hover_name=name_index,
                        hover_data=list(columns_text))
    else:
        fig = px.scatter(df_plot, 
                        x=columns_to_plot[0],
                        y=columns_to_plot[1],
                        color=color,
                        hover_name=name_index,
                        hover_data=list(columns_text))
    return fig


# %%
# %matplotlib inline

# %%
umap_df.columns

# %%
plot(umap_df, columns_umap, column_to_color='type').write_html('dist_umap_type.html', auto_open=True)

# %%
