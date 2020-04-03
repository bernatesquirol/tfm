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
import importlib
importlib.reload(utils)
twitter_client = utils.TwitterClient()


# %%

# %% [markdown]
#  # Extract congress members (a few hours)

# %%
def get_light_user_data(twitter_client, user, path_file=None):
    timeline = twitter_client.get_timeline(user)
    light_timeline = utils.get_light_timeline(timeline)
    del timeline
    likes = twitter_client.get_likes(user)
    light_likes = utils.get_light_likes(likes)
    del likes
    concat = pd.concat([light_timeline, light_likes])
    if path_file:
        concat.to_pickle(os.path.join(path_file,'{}.pkl'.format(user)))
    return concat


# %%
congress_members = twitter_client.twitter.get_list_members(list_id=utils.LIST_ALL_CONGRESS,count=5000)
for c in congress_members['users']:
    get_light_user_data(twitter_client, c['screen_name'], path_file='./data/politicians')


# %% [markdown]
# # Find journalists

# %%
sources = ['el_pais', 'JotDownSpain', 'eldiarioes', 'elespanolcom', 'revistamongolia', 'la_ser', '_infoLibre', 'EFEnoticias', 'elmundoes', 'elconfidencial', 'indpcom', 'ctxt_es', 'publico_es', 'ondacero_es', 'cuatro', 'LaVanguardia', 'europapress','laSextaTV', 'rtve']

# %%
source = sources[0]

# %%
list_following = twitter_client.twitter.get_friends_list(screen_name=source, count=200)

# %%
len(list_following['users'])

# %%

# %%
twitter_client.twitter['get_friends_list']

# %%
list_following.keys()

# %%
list_following['next_cursor']

# %%
journalists = {}

# %%
len(list_following['users'])

# %%

# %%
other_guys = {}

# %%
# bernat_followers = twitter_client.get_friends_list('el_pais')
all_followers_dict.keys()

# %%
# all_followers_dict = {}
# for source in sources:
#     list_following = twitter_client.get_friends_list(source)
#     all_followers_dict[source]=list_following

# %%

# %%
key_words = ['corresponsal', 'periodist', 'redactor', 'xornalista' ]
key_neg_words = ['corresponsales', 'periodistas', 'redactores', 'xornalistas', 'sport', 'deport', 'chiringuito', 'diputad', 'parlament', 'gobierno' ]
journalists = {}
for key, users in all_followers_dict.items():
    for index, user in users.iterrows():
        description = unidecode.unidecode(user['description'].lower())
        if len([1 for kw in key_words if kw in description or kw in sources])>0 and len([1 for kw in key_neg_words if kw in description])==0:
            journalists[user['id']]={
                                        'screen_name':user['screen_name'],
                                        'favourites_count': user['favourites_count'],
                                        'followers_count': user['followers_count'], 
                                        'friends_count': user['friends_count']
                                    }

# %%
import unidecode

# %%
journalists_df = pd.DataFrame(journalists.values(), index=journalists.keys()).sort_values('followers_count', ascending=False)

# %%
journalists_df.to_pickle('all_journalists.pkl')

# %%
len(journalists)

# %%
len(list_following['users'])

# %% [markdown]
# # Extract Journalists

# %%
2626*2/3

# %%
take = journalists_df['screen_name'][:875]

# %%

# %%

# %%
for s_n in take:
    get_light_user_data(twitter_client, s_n, path_file='./data/journalists')

# %%
