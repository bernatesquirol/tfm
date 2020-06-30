#!/usr/bin/env python
# coding: utf-8
# %%
import utils
import json
import pandas as pd
import os


# %%
import matplotlib as plt


# %%
twitter_client = utils.TwitterClient()


# %% [markdown]
# This notebook is created in order to fetch different user data from twitter. We will use the following function

# %%
def get_light_user_data(twitter_client, user, path_file=None, get_likes=True):
    timeline = twitter_client.get_timeline(user)
    light_timeline = utils.get_light_timeline(timeline)
    del timeline
    if get_likes:
        light_likes = utils.get_light_likes(likes)
        del likes
        concat = pd.concat([light_timeline, light_likes])
        if path_file:
            concat.to_pickle(os.path.join(path_file,'{}.pkl'.format(user)))
        return concat
    else:
        return light_timeline


# %%

# %%
def get_users_data(screen_names):
    data = [twitter_client.try_call(lambda: twitter_client.twitter.show_user(screen_name=s_n), throw_errors=True) for s_n in screen_names]
    return pd.DataFrame(data, index=screen_names)


# %%
def save_users_database(path):
    import os
    indexes = []
    for p in os.listdir(path):
        indexes.append(p[:-4])
    get_users_data(indexes).to_pickle(path+'.pkl')
    


# %% [markdown]
# # Congress members

# %% [markdown]
# ## Fetch list

# %%
congress_members = twitter_client.twitter.get_list_members(list_id=utils.LIST_ALL_CONGRESS,count=5000)

# %% [markdown]
# ## Interactions

# %%
for c in congress_members['users']:
    try:
        get_light_user_data(twitter_client, c['screen_name'], path_file='../data/politicians')
    except:
        print('Error',c['name'])


# %% [markdown]
# ## All politicians db

# %%
# get user data for all users
save_users_database('../data/politicians')

# %%
save_users_database('../data/politicians')

# %% [markdown]
# # Find journalists

# %% [markdown]
# ## Create list of journalists

# %%
sources = ['el_pais', 
           'JotDownSpain', 
           'eldiarioes', 
           'elespanolcom', 
           'revistamongolia', 
           'la_ser', 
           '_infoLibre', 
           'EFEnoticias', 
           'elmundoes', 
           'elconfidencial', 
           'indpcom', 
           'ctxt_es', 
           'publico_es', 
           'ondacero_es', 
           'cuatro', 
           'LaVanguardia', 
           'europapress',
           'laSextaTV', 
           'rtve']

# %%
all_followers_dict = {}
for source in sources:
    list_following = twitter_client.get_friends_list(source)
    all_followers_dict[source]=list_following

# %%
import unidecode

# %%
key_words = ['corresponsal', 'periodist', 'redactor', 'xornalista' ]
key_neg_words = ['corresponsales', 'periodistas', 'redactores', 'xornalistas', 'sport', 'deport', 'chiringuito', 'diputad', 'parlament', 'gobierno' ]
journalists = {}
for key, users in all_followers_dict.items():
    for index, user in users.iterrows():
        description = unidecode.unidecode(user['description'].lower())
        if len([1 for kw in key_words if kw in description or kw in sources])>0 and len([1 for kw in key_neg_words if kw in description])==0:
            journalists[user['id']]=user

# %%
journalists_df = pd.DataFrame(journalists.values(), index=journalists.keys())#.sort_values('followers_count', ascending=False)

# %%
journalists_df.to_pickle('./journalists_raw.pkl')

# %%
journalists_df.columns

# %% [markdown]
# ## Interactions

# %%
# take = journalists_df.sample(905)

# %%
for s_n in take:
    get_light_user_data(twitter_client, s_n['screen_name'], path_file='../data/journalists')

# %% [markdown]
# ## All journalists db

# %%
journalists_df.to_pickle('all_journalists.pkl')

# %% [markdown]
# # Random users

# %% [markdown]
# ## Interactions

# %%
SPAIN_GEOCODE = '39.8952506,-3.4686377505768764,600000km'


# %%
def get_df_search():
    all_tweets_call = twitter_client.twitter.search(q='lang:ca OR lang:es', geocode=SPAIN_GEOCODE, result_type='recent', count=100)
    all_tweets = pd.DataFrame((map(lambda t: t['user'], all_tweets_call['statuses'])), columns=['created_at','followers_count','friends_count','favourites_count','statuses_count','screen_name'])
    all_tweets = all_tweets.groupby('screen_name').first()
    all_tweets['created_at'] = pd.to_datetime(all_tweets['created_at'],utc=True)
    all_tweets['frequency'] = all_tweets['statuses_count'] /  all_tweets['created_at'].apply(lambda x: (pd.Timestamp.utcnow() - x).days)
    all_tweets['outlier_frequency'] = utils.relevant_outliers(all_tweets['frequency'])
    return all_tweets


# %%
MIN_FOLLOWERS = 55
MIN_FOLLOWING = 95
MIN_ACTIONS = 1000


# %%
def get_light_sample(ids, wrong_id=None):
    import random
    if wrong_id:
        if (type(np.array(ids))!=np.array):
            ids = np.array(ids)
        ids = ids[~np.isin(ids,[wrong_id])]
    id_random = int(random.choice(ids))
    try:
        print('trying {}'.format(id_random))
        user_obj = twitter_client.try_call(lambda: twitter_client.twitter.show_user(user_id=id_random), throw_errors=True)
        # conditions on random user sample
        print(user_obj['screen_name'], id_random,user_obj['favourites_count']+user_obj['statuses_count'] )
        if user_obj['favourites_count']+user_obj['statuses_count']<MIN_ACTIONS:
            print('no min_actions {}'.format(id_random))
            return get_light_sample(ids, id_random)
        
        all_data = get_light_user_data(twitter_client, user_obj['screen_name'])
        # conditions on all data
        if len(all_data)<MIN_ACTIONS:
            print('no min_actions_2 {}'.format(id_random))
            return get_light_sample(ids, id_random)
        return all_data, user_obj['screen_name']
    except:
        print('wrong {}'.format(id_random))
        return get_light_sample(ids, id_random)


# %%
def execute_search():
    all_tweets = get_df_search()
    all_tweets['enough_followers'] = all_tweets['followers_count']>MIN_FOLLOWERS
    screen_name_1 = all_tweets[-all_tweets['outlier_frequency'] & all_tweets['enough_followers']].sample().index.tolist()[0]
    call_result = twitter_client.try_call(lambda: twitter_client.twitter.get_followers_ids(screen_name=screen_name_1, count=5000))
    if call_result and 'ids' in call_result:
        data, user = get_light_sample(call_result['ids'])
        data.to_pickle(os.path.join('data/random-followers','{}.pkl'.format(user)))
    all_tweets['enough_friends'] = all_tweets['friends_count']>MIN_FOLLOWING
    screen_name_2 = all_tweets[-all_tweets['outlier_frequency'] & all_tweets['enough_friends']].sample().index.tolist()[0]
    call_result = twitter_client.try_call(lambda: twitter_client.twitter.get_friends_ids(screen_name=screen_name_2, count=5000))
    if call_result and 'ids' in call_result:
        data, user = get_light_sample(call_result['ids'])
        data.to_pickle(os.path.join('data/random-friends','{}.pkl'.format(user)))


# %%
# do this from 3 different notebooks with different cred
# for i in range(50):
#     print('----> volta', i)
#     execute_search()
#     time.sleep(300)

# %% [markdown]
# ## All random followers/following db

# %%
save_users_database('../data/random-followers')

# %%
save_users_database('../data/random-friends')

# %% [markdown]
# ## Profiles

# %%
import pandas as pd
all_profiles = pd.read_pickle('../data/all_profiles.pkl')

# %%
import os
tls = os.listdir('../data/timelines')
len(tls)

# %%
models = os.listdir('../data/models')

# %%
tls = [int(i[:-4]) for i in tls if int(i[:-4]) in all_profiles.index]

# %%
len(tls)

# %%
models = [int(i[:-4]) for i in models if int(i[:-4]) in tls]

# %%
import shutil
for tl in tls:
    shutil.move('../data/timelines/{}.pkl'.format(tl), '../data/bad_timelines/{}.pkl'.format(tl))

# %%
all_profiles.head()

# %%
from tqdm import tqdm


# %%
pd.Series([i for i in tls if i not in all_profiles.index]).to_pickle('../data/missing_users.pkl')

# %%
dict_all_tls = {}
for tl_id in tqdm(tls):
    tl = pd.read_pickle('../data/timelines/{}.pkl'.format(tl_id))
    dict_all_tls[tl_id]=[tl.index.min(),tl.index.max()]

# %%
all_profiles['observed_start']=np.nan
all_profiles['observed_end']=np.nan

# %%
start_and_end = pd.Series(list(dict_all_tls.values()), index=dict_all_tls.keys())
all_profiles.loc[start_and_end.index, 'observed_start'] =start_and_end.apply(lambda x: x[0])
all_profiles.loc[start_and_end.index, 'observed_end']=start_and_end.apply(lambda x: x[1])

# %%
pd.concat([all_profiles,pd.read_pickle('./users_missing_no.pkl')]).to_pickle('../data/all_profiles.pkl')

# %%
mod = ['observed_start','observed_end', 'type_profile']
all_profiles[[c for c in all_profiles.columns if c not in mod]+mod].to_pickle('../data/all_profiles.pkl')

# %%
bad_index = all_profiles[all_profiles.type_profile.isnull()].index

# %%
values = {'journalists':'journalist',
          'journalists-new':'journalist',
          'politicians':'politician',
          'random-followers':'random-follower',
          'random-friends':'random-friend',
          'random-friends-big':'random-friend',
          'random-followers-big':'random-follower'}

# %%
all_files_dict = {}
for folder in ['journalists', 'journalists-new', 'politicians', 'random-followers', 'random-friends', 'random-friends-big', 'random-followers-big']:
    for i in os.listdir('../../tfm2/data_old/'+folder):
        id_u = int(i[:-4])
        if id_u in bad_index:
            all_files_dict[id_u]=values[folder]
#     print(values[folder])

# %%
len(bad_index),len(all_files_dict)

# %%
all_profiles.loc[list(all_files_dict.keys()), 'type_profile']=list(all_files_dict.values())

# %%
all_profiles.type_profile.unique()

# %%
import plotly.express as px
fig = px.scatter(all_profiles[all_profiles.type_profile.isin(['random-friend', 'random-follower']) & (all_profiles.followers_count<5000) & (all_profiles.friends_count<5000)], x='followers_count', y='friends_count', color='type_profile')

# %%
fig['data'][0]['marker']['opacity']=0.1
fig['data'][1]['marker']['opacity']=0.1

# %%
utils.plotly_to_tfm(fig, 'random-types-bias')

# %%
import utils

# %%
len(all_profiles)

# %%
850396856551186432 in bad_index

# %%
