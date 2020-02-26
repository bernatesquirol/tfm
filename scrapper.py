#!/usr/bin/env python
# coding: utf-8
# %%
import ipywidgets as widgets
from ipywidgets import interact, interact_manual


# %%
import utils
import json
import pandas as pd
import os


# %%
import importlib
importlib.reload(utils)


# %%
import matplotlib as plt


# %%
twitter_client = utils.TwitterClient()


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

