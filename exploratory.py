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
import ipywidgets as widgets


# %%
def get_features_timeline(actions):
    features = {}
    if len(actions)<2: return {}
    users_freq_actions = utils.user_frequency(actions)
    if len(users_freq_actions)>1:
        features['fit_exp'], features['fit_k'], dummie, features['fit_lambda'] = utils.get_best_args(users_freq_actions, 'exponweib')    
        features['1_2_actions']=users_freq_actions.iloc[0]/users_freq_actions.iloc[1]
        if len(users_freq_actions)>2:
            features['2_3_actions']=users_freq_actions.iloc[1]/users_freq_actions.iloc[2]        
    timeline = actions[actions['type']!='Like']
    likes = actions[actions['type']=='Like']
    timeline_len = (timeline['created_at'].max()-timeline['created_at'].min()).days
    if timeline_len>0:
        features['frequency_timeline']= len(timeline) / timeline_len
    likes_len = (likes['created_at'].max()-likes['created_at'].min()).days
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
def load_users(db,root='./data'):
    import os
    path = os.path.join(root,db)
    list_politicians_dict = {}
    for p in os.listdir(path):
        path_file = os.path.join(path,p)
        p_dict = get_features_timeline(pd.read_pickle(path_file).reset_index())
        list_politicians_dict[p[:-4]]=p_dict
    activity_profiles = pd.DataFrame(list_politicians_dict).T
    user_profile = ['followers_count','friends_count', 'verified', 'statuses_count','favourites_count']
    profiles = pd.read_pickle(path+'.pkl')
    return profiles[user_profile].join(activity_profiles)


# %%
import utils
import importlib 
importlib.reload(utils)


# %%
twitter_client = utils.TwitterClient()


# %%
politicians = load_users('politicians')


# %%
random_friends = load_users('random-friends')


# %%
random_followers = load_users('random-followers')


# %% [markdown]
# # Plot interactive

# %%
def plot(df, columns_to_plot, column_to_color=None, bins=None):
    import plotly as py
    import plotly.graph_objs as go
    dim = len(columns_to_plot)
    name_index = 'index'
    if df.index.name!=None:
        name_index = df.index.name
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
            df_plot[column_to_color+'_bins'] = pd.qcut(df_plot[column_to_color], q=cutting_num, labels=range(cutting_num)).astype(int)
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
                        hover_data=list(df_plot.columns))
    else:
        fig = px.scatter(df_plot, 
                        x=columns_to_plot[0],
                        y=columns_to_plot[1],
                        color=color,
                        hover_name=name_index,
                        hover_data=list(df_plot.columns))
    return fig


# %%
def plot_interactive(df, columns_to_plot, column_to_color=None, bins=None, fig=None):
    import plotly as py
    import plotly.graph_objs as go
    dim = len(columns_to_plot)
    name_index = 'index'
    if df.index.name!=None:
        name_index = df.index.name
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
            df_plot[column_to_color+'_bins'] = pd.qcut(df_plot[column_to_color], q=cutting_num, labels=range(cutting_num)).astype(int)
            color = column_to_color+'_bins'
        else:
            color = column_to_color
    data = dict(
        x=df_plot[columns_to_plot[0]],
        y=df_plot[columns_to_plot[1]],
        mode='markers',
        text=df_plot[name_index]
    )
    
    if color:
        try:
            color_column = df_plot[color].astype(float)
        except:
            color_column = df_plot[color].factorize()[0]
        data['marker']=dict(
            color=color_column, #set color equal to a variable
            colorscale='Viridis', # one of plotly colorscales
            showscale=True
        )
        
    if dim==3:
        data['type']='scatter3d'
        data['z']=df_plot[columns_to_plot[1]]
    if fig:
        fig.data[0].x = data['x']
        fig.data[0].y = data['y']        
        fig.data[0].text = data['text']
        if 'z' in fig.data[0]:
            fig.data[0].z = data['z']
        if color:
            fig.data[0].marker = data['marker']
    else:
        fig = go.FigureWidget(data=[data], layout={})
        return fig
    #
    


# %%
def select_column_widget(df, description='', default=None, none_option=False):
    if default==None:
        default = df.columns[0]
    options = list(df.columns)
    if none_option:
        options.insert(0, '--none--')
        default = '--none--'
    multiple = widgets.Select(
        options=options,
        value=default,
        rows=len(df.columns)+1,
        description=description,
        disabled=False
    )
    return multiple


# %%
def create_interactive_viz(df, third_d=False):
    x_col = select_column_widget(df, 'x')
    y_col = select_column_widget(df, 'y')
    color = select_column_widget(df, 'color', None, none_option=True)
    if third_d:
        z_col = select_column_widget(df, 'z')
    def get_cols_to_display():
        if third_d: 
            return [x_col.value, y_col.value, z_col.value]
        return [x_col.value, y_col.value]
    fig = plot_interactive(df, get_cols_to_display(), color.value)
    def observer(c):
        plot_interactive(df, get_cols_to_display(), color.value, fig=fig)
    x_col.observe(observer, 'value', 'change')
    y_col.observe(observer, 'value', 'change')
    color.observe(observer, 'value', 'change')
    if third_d:
        z_col.observe(observer, 'value', 'change')
        display(widgets.HBox([x_col, y_col, z_col, color]))
    else:
        display(widgets.HBox([x_col, y_col, color]))
    return fig


# %%
politicians['type']=0
random_friends['type']=1
random_followers['type']=2
all_users = politicians


# %%
len(all_users)


# %%
all_users_no_null = all_users[-all_users.isnull().any(axis=1)]


# %%
len(all_users_no_null)


# %%
create_interactive_viz(all_users_no_null)


# %% [markdown]
# # Clustering

# %%
def get_PCA(df, dim=3, columns=[]):
    pca = PCA(n_components=dim)
    if len(columns) == 0:
        columns = df.columns
    df_pca = df[columns].copy()
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
    df_tsne = df[columns].copy()
    df_final = df.copy()
    columns_tsne = ['tSNE_{}'.format(i) for i in range(dim)]
    df_final[columns_tsne] = pd.DataFrame(tsne.fit_transform(df_tsne),columns=columns_tsne, index=df_tsne.index)
    return df_final, columns_tsne, tsne


# %%
columns = ['frequency_timeline', 'frequency_like', 'like_tweet_ratio',
       'social_ratio', 'reply_sratio', 'rt_sratio', 'mention_sratio',
       'quoted_sratio', '1_2_ratios', 'num_outliers_2']


# %%
scaler = StandardScaler()
all_users_no_null_standard = pd.DataFrame(scaler.fit_transform(all_users_no_null), columns=all_users_no_null.columns, index=all_users_no_null.index )


# %%
df_pca, columns_pca, pca = get_PCA(all_users_no_null_standard, columns=columns)
df_tsne, columns_tsne, tsne = get_tSNE(df_pca, dim=2, columns=columns)


# %%
plot(df_tsne, columns_pca, column_to_color='type')


# %%
plot(df_tsne, columns_tsne, column_to_color='type')


# %%

# %% [markdown]
# # Distribution of outliers

# %%
from pylab import rcParams
rcParams['figure.figsize'] = 15, 8


# %%
politicians[politicians['num_outliers_2']<30]['num_outliers_2'].sample(75).hist(bins='auto', density=True)


# %%
def filter_30(df):
    return df[df['num_outliers_2']<30]


# %%
filter_30(politicians)[['fit_exp','fit_k', 'fit_lambda']].describe() #'fit_exp'], features['fit_k'], dummie, features['fit_lambda'


# %%
a = [filter_30(random_friends).sample(300)[['fit_exp','fit_k', 'fit_lambda']].describe() for i in range(100)]
pd.concat(a).reset_index().groupby('index').mean()


# %%
b = [filter_30(random_followers).sample(300)[['fit_exp','fit_k', 'fit_lambda']].describe() for i in range(100)]
pd.concat(b).reset_index().groupby('index').mean()


# %%
random_friends[random_friends['num_outliers_2']<30]['num_outliers_2'].sample(300).hist(bins='auto', density=True)


# %%
random_followers[random_followers['num_outliers_2']<30]['num_outliers_2'].sample(300).hist(bins='auto', density=True)


# %%
all_users_no_null[all_users_no_null['num_outliers_2']<30]['num_outliers_2'].sample(300).hist(bins='auto', density=True)


# %%
all_users_no_null[all_users_no_null['num_outliers_2']>15].index


# %%
all_users_no_null['num_outliers_2'].max()


# %%
all_users_no_null[all_users_no_null['num_outliers_2']==72.0].iloc[0]


# %%
politicians['num_outliers_1'].hist()

