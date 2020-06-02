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


def load_saved_covid(key):
    return pd.read_pickle('../data/global_timeline/{}_pre.pkl'.format(key)), pd.read_pickle('../data/global_timeline/{}_post.pkl'.format(key))


# p_pre, p_post = utils.load_users_covid('politicians')
# rfriends_pre, rfriends_post = utils.load_users_covid('random-friends')
# rfollowers_pre, rfollowers_post = utils.load_users_covid('random-followers')
j_pre, j_post = utils.load_users_covid('journalists-new')
# rfriends_big_pre, rfriends_big_post = utils.load_users_covid('random-friends-big')
# rfollowers_big_pre, rfollowers_big_post = utils.load_users_covid('random-followers-big')
# rfollowers_big_post.to_pickle('random-followers-big_post.pkl')
# rfollowers_big_pre.to_pickle('random-followers-big_pre.pkl')

j_pre.to_pickle('journalists-new_pre.pkl')
j_post.to_pickle('journalists-new_post.pkl')

# +
# for j in os.listdir('../data/journalists'):
#     if j in os.listdir('../data/journalists-new'):
#         df_j_n = pd.read_pickle('../data/journalists-new/{}'.format(j))
#         df_j = pd.read_pickle('../data/journalists/{}'.format(j))
#         concat_j = pd.concat([df_j, df_j_n])
#         concat_j[~concat_j.index.duplicated()].to_pickle('../data/journalists-new/{}'.format(j))        

# +
# df_j_n.head()

# +
# j_pre.head()
# -



# +
# j_n_pre, j_n_post = utils.load_users_covid('journalists-new')
# -

j_n_pre_filtered = j_pre[j_pre.frequency_timeline>0]

j_n_pre_filtered.shape

# +
# tal = j_n_pre_filtered.sort_values('first_action', ascending = False).head(50)

# +
# tal
# -

pd.read_pickle('../data/journalists-new/161021621.pkl')

len([i for i in tal.index if i not in j_pre.index])

p_pre, p_post = load_saved_covid('politicians')
rfriends_pre, rfriends_post = load_saved_covid('random-friends')
rfollowers_pre, rfollowers_post = load_saved_covid('random-followers')
j_pre, j_post = load_saved_covid('journalists-new')
rfriends_big_pre, rfriends_big_post = load_saved_covid('random-friends-big')
rfollowers_big_pre, rfollowers_big_post = load_saved_covid('random-followers-big')

p_pre['type']='politicians'
p_post['type']='politicians'
rfriends_pre['type']='random-friends'
rfriends_post['type']='random-friends'
rfollowers_pre['type']='random-followers'
rfollowers_post['type']='random-followers'
rfriends_big_pre['type']='random-friends-big'
rfriends_big_post['type']='random-friends-big'
rfollowers_big_pre['type']='random-followers-big'
rfollowers_big_post['type']='random-followers-big'

p_pre['covid']=False
p_post['covid']=True
rfriends_pre['covid']=False
rfriends_post['covid']=True
rfollowers_pre['covid']=False
rfollowers_post['covid']=True
rfriends_big_pre['covid']=False
rfriends_big_post['covid']=True
rfollowers_big_pre['covid']=False
rfollowers_big_post['covid']=True

users = pd.concat([p_pre, p_post, rfriends_pre, rfriends_post, rfollowers_pre, rfollowers_post, rfriends_big_pre, rfriends_big_post, rfollowers_big_pre, rfollowers_big_post])

users.to_pickle('users_global.pkl')

users = pd.read_pickle('users_global.pkl')

users[users['type']=='random-friends'].last_action.max()

users.head(1)

# +
# random_followers_big.set_index('id').join(rfollowers_big_pre)#.to_pickle('../data/global_timeline/random-followers-big_pre.pkl')

# +
# random_followers_big.set_index('id').join(rfollowers_big_pre)

# +
# random_friends_big.set_index('id').join(rfriends_big_post).to_pickle('../data/global_timeline/random-friends-big_post.pkl')

# +
# rfriends_big_pre.columns

# +
# df_dict_pre = {'politicians': p_pre, 'random_friends':rfriends_pre, 'random_followers':rfollowers_pre, 'journalists':j_pre, 'random_friends_big':rfriends_big_pre, 'random_followers_big': rfollowers_big_pre}
# df_dict_post = {'politicians': p_post, 'random_friends':rfriends_post, 'random_followers':rfollowers_post, 'journalists':j_post, 'random_friends_big':rfriends_big_post, 'random_followers_big': rfollowers_big_post}
# -

users.shape


def plot_all(users, column):
    fig = make_subplots(rows=2, cols=3, 
                   shared_xaxes='all', 
                   shared_yaxes='all',
                   subplot_titles=list(users.type.unique()))
    for i, value in enumerate(users.groupby('type')):
        key, df = value
        fig.add_trace(go.Histogram(x=df[column], histnorm='probability density'), row=int(i/3)+1, col=(i%3)+1)
    fig.update_layout(height=600, width=800,
                      title_text=column,showlegend=False)
    return fig


def plot_all2d(users, column):
    fig = make_subplots(rows=2, cols=3, 
                   shared_xaxes='all', 
                   shared_yaxes='all',
                   subplot_titles=list(users.type.unique()))
    for i, value in enumerate(users.groupby('type')):
        key, df = value
        fig.add_trace(go.Histogram2d(x=df[column_x], y=df[column_y]), row=int(i/3)+1, col=(i%3)+1)
    fig.update_layout(height=600, width=800,
                      title_text=column_x+'/'+column_y,showlegend=False)
    return fig


def create_new_column(dict_of_df, func, label):
    for key, df in dict_of_df.items():
        df[label]=func(df)


# +
# def plot_all_new(dict_pre, dict_post, func, label, drop=False):
#     create_new_column(dict_pre, func, label)
#     create_new_column(dict_post, func, label)
# -

def compare_covid(users, keys, column, **kwargs):
    pre_labels = [k+'_pre' for k in keys]
    post_labels = [k+'_post' for k in keys]
    fig = make_subplots(rows=len(keys), cols=2, 
                   shared_xaxes='all', 
                   shared_yaxes='rows',
                   subplot_titles=np.hstack(list(zip(pre_labels, post_labels))))
    for i, value in enumerate(users.groupby('type')):
        key, df = value
        fig.add_trace(go.Histogram(x=df[df['covid']==False][column], histnorm='probability density'), row=i+1, col=1)
        fig.add_trace(go.Histogram(x=df[df['covid']==True][column], histnorm='probability density'), row=i+1, col=2)
    fig.update_layout(title_text=column,**kwargs)
    return fig


# ## Global stuff

# +
# j_pre.columns
# -

def get_log(value):
    import math
    if value>0:
        return math.log(value)
    else: return np.nan


# ### Compare groups

users['total_tweets']=(users['last_action']-users['first_action']).apply(lambda x:x.days)*users['frequency_timeline']

plot_all(users, 'total_tweets')

plot_all(filtered_pre, 'total_tweets')

plot_all(filtered_pre,'rt_sratio' )#.write_html('rt_sratio_post.html')

create_new_column(filtered_pre,  lambda x: x.rt_sratio*x.social_ratio/100, 'rt_ratio')
create_new_column(filtered_post,  lambda x: x.rt_sratio*x.social_ratio/100, 'rt_ratio')


def filter_dict(dict_raw, bool_rule):
    return {k: v[bool_rule(v)].copy() for k,v in dict_raw.items()}


# +
# at_least_700_followers = filter_dict(filtered_pre, lambda x: x.followers_count>700)
# -

plot_all(at_least_700_followers, 'rt_ratio' )#.write_html('rt_ratio_pre.html')

# +
# plot_all_func(filtered_post, lambda x:  x.rt_sratio*x.social_ratio, 'rt_ratio (POST)', drop=True ).write_html('rt_ratio_post.html')
# -

# ### Correlations

create_new_column(filtered_pre, lambda x: x['followers_count'].apply(get_log), 'followers_count_log')

plot_all_2d(filtered_pre, 'total_tweets', 'followers_count_log')

plot_all_2d(filtered_pre, 'num_outliers_2', 'followers_count_log')

filtered_pre['journalists'].columns

all_pd = pd.concat(list(filtered_pre.values()))

filtered_pre['journalists']['first_action'].max()

# ### Compare covid

import math
create_new_column(filtered_pre, lambda x: x['frequency_timeline'].apply(get_log), 'frequency_timeline_log')
create_new_column(filtered_post, lambda x: x['frequency_timeline'].apply(get_log), 'frequency_timeline_log')

compare_covid(filtered_pre, filtered_post, ['random_friends_big','random_followers_big'], 'frequency_timeline')

compare_covid(filtered_pre, filtered_post, list(filtered_pre.keys()), 'frequency_timeline_log', height=1000)#.write_html('../tfm-plots/frequency_timeline_log_covid.html')

compare_covid(filtered_pre, filtered_post, list(filtered_pre.keys()), 'reply_sratio', height=1000)

compare_covid(filtered_pre, filtered_post, list(filtered_pre.keys()), 'rt_ratio', height=1000)

# +
# politicians = utils.load_users('politicians', party=False)
# random_friends = utils.load_users('random-friends')
# random_followers = utils.load_users('random-followers')
# journalists = utils.load_users('journalists')
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


