import os
import pandas as pd
import json
import numpy as np
import utils

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px


# %load_ext autoreload
# %autoreload 2

def update_data():
    path='..\\data\\politicians'
    general_df = pd.read_pickle(path+'.pkl')
    dict_sn_id = general_df['id']
    for f in os.listdir(path+'_sn'):
        path_file = os.path.join(path+'_sn',f)
        f_df = pd.read_pickle(path_file)
        try:
            new_timeline = pd.read_pickle(os.path.join('..\\data\\new_old',str(dict_sn_id[f[:-4]])+'.pkl'))
            new_df = pd.concat([f_df, new_timeline[new_timeline.index>f_df.index.max()]])
            new_df.to_pickle(os.path.join(path,str(dict_sn_id[f[:-4]])+'.pkl'))
        except:
            pass


# +
# p_pre, p_post = utils.load_users_covid('politicians')
# rfriends_pre, rfriends_post = utils.load_users_covid('random-friends')
# rfollowers_pre, rfollowers_post = utils.load_users_covid('random-followers')
# j_pre, j_post = utils.load_users_covid('journalists')
# rfriends_big_pre, rfriends_big_post = utils.load_users_covid('random-friends-big')
# rfollowers_big_pre, rfollowers_big_post = utils.load_users_covid('random-followers-big')
# rfollowers_big_post.to_pickle('random-followers-big_post.pkl')
# rfollowers_big_pre.to_pickle('random-followers-big_pre.pkl')
# -

rfollowers_big_post = pd.read_pickle('global_timeline/journalists_post.pkl')

rfollowers_big_post.to_pickle('random-followers-big_post.pkl')
rfollowers_big_pre.to_pickle('random-followers-big_pre.pkl')

rfriends_big_post.to_pickle('random-friends-big_post.pkl')
rfriends_big_pre.to_pickle('random-friends-big_pre.pkl')


def filter_df_dict(dict_of_df):
    return {k: utils.filter_users(v) for k,v in dict_of_df.items()}


user_profile = ['followers_count','friends_count', 'verified', 'statuses_count','favourites_count', 'screen_name']
profiles = pd.read_pickle('../data/random-followers-big.pkl').set_index('id')
rfollowers_big_pre_2 = profiles[user_profile].join(rfollowers_big_pre, how='inner')
rfollowers_big_post_2 =profiles[user_profile].join(rfollowers_big_post, how='inner')

user_profile = ['followers_count','friends_count', 'verified', 'statuses_count','favourites_count', 'screen_name']
profiles = pd.read_pickle('../data/random-friends-big.pkl').set_index('id')
rfriends_big_pre = profiles[user_profile].join(rfriends_big_pre, how='inner')
rfriends_big_post =profiles[user_profile].join(rfriends_big_post, how='inner')

rfollowers_big_pre = rfollowers_big_pre_2
rfollowers_big_post = rfollowers_big_post_2

rfriends_big_pre.index

# +
# rfriends_big_post.loc[rfriends_big_pre.index]
# -

np.nan in filtered_pre['random_friends_big'].index

filtered_pre = filter_df_dict(df_dict_pre)

df_dict_pre = {'politicians': p_pre, 'random_friends':rfriends_pre, 'random_followers':rfollowers_pre, 'journalists':j_pre, 'random_friends_big':rfriends_big_pre, 'random_followers_big': rfollowers_big_pre}

filtered_pre = filter_df_dict(df_dict_pre)
filtered_pre_index = {k:[i for i in v.index if i is not None] for k,v in df_dict_pre.items()}
filtered_post = {k:v.loc[filtered_pre_index[k]] for k, v in df_dict_post.items()}


# +
# list(filtered_pre.values())[0]
# -

def plot_all(dict_of_df, column):
    fig = make_subplots(rows=2, cols=3, 
                   shared_xaxes='all', 
                   shared_yaxes='all',
                   subplot_titles=list(dict_of_df.keys()))
    for i, value in enumerate(dict_of_df.items()):
        key, df = value
        fig.add_trace(go.Histogram(x=df[column], histnorm='probability density'), row=int(i/3)+1, col=(i%3)+1)
    fig.update_layout(height=600, width=800,
                      title_text=column,showlegend=False)
    return fig


def plot_all_func(dict_of_df, func, label, drop=False):
    for key, df in dict_of_df.items():
        df[label]=df.apply(func, axis=1)
    fig = plot_all(dict_of_df, label)
    if drop:
        for key, df in dict_of_df.items():
            dict_of_df[key]=df.drop(columns=[label])
    return fig


plot_all_func(filtered_pre, lambda x: x.rt_sratio, 'rt_social_ratio (PRE)', drop=True ).write_html('rt_sratio_pre.html')

plot_all_func(filtered_post, lambda x: x.rt_sratio, 'rt_social_ratio (POST)', drop=True ).write_html('rt_sratio_post.html')

plot_all_func(filtered_pre, lambda x: x.rt_sratio*x.social_ratio, 'rt_ratio (PRE)', drop=True ).write_html('rt_ratio_pre.html')

plot_all_func(filtered_post, lambda x:  x.rt_sratio*x.social_ratio, 'rt_ratio (POST)', drop=True ).write_html('rt_ratio_post.html')

# +
# politicians = utils.load_users('politicians', party=False)
# random_friends = utils.load_users('random-friends')
# random_followers = utils.load_users('random-followers')
# journalists = utils.load_users('journalists')

# +
# j_pre['followers_count'].hist()
# -

import math
plot_all_func(filtered_pre, lambda x: math.log(x.followers_count), 'followers count', drop=True )

import math
plot_all_func(filtered_pre, lambda x: math.log(x.friends_count) if x.friends_count>0  else np.nan, 'friends count', drop=True )





len(j_pre),len(j_post)


# +
# journalists = pd.read_pickle('../data/journalists.pkl')
# r_followers = pd.read_pickle('../data/random-followers.pkl')
# r_friends = pd.read_pickle('../data/random-friends.pkl')
# politicians = pd.read_pickle('../data/politicians.pkl')
# -

def plot_series(dict_of_df):


random_friends = utils.load_users('random-friends')

random_friends.shape

# +
from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = make_subplots(
    rows=3, cols=2,
    specs=[[{}, {}],
           [{"colspan": 2}, None]],
    subplot_titles=("First Subplot","Second Subplot", "Third Subplot"))

fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2]),
                 row=1, col=1)

fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2]),
                 row=1, col=2)
fig.add_trace(go.Scatter(x=[1, 2, 3], y=[2, 1, 2]),
                 row=2, col=1)

fig.update_layout(showlegend=False, title_text="Specs with Subplot Title")
fig.show()
# -

import plotly.express as px
px.histogram(random_followers, 'followers_count',log_y=True)

pd.concat([journalists.id, r_followers.id, r_friends.id, politicians.id]).to_pickle('all_ids.pkl')

# +
from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = make_subplots(rows=1, cols=2, shared_yaxes=True)

fig.add_trace(go.Bar(x=[1, 2, 3], y=[4, 5, 6], marker=dict(color=[4, 5, 6], coloraxis="coloraxis")),
              1, 1)

fig.add_trace(go.Bar(x=[1, 2, 3], y=[2, 3, 5], marker=dict(color=[2, 3, 5], coloraxis="coloraxis")),
              1, 2)

fig.update_layout(coloraxis=dict(colorscale='Bluered_r'), showlegend=False)
fig.show()
# -


