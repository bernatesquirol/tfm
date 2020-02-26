#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import utils
import importlib
importlib.reload(utils)
import os
import numpy as np


# %%


import pandas as pd


# %%


SPAIN_GEOCODE = '39.8952506,-3.4686377505768764,600000km'


# %%





# Experiment: 1000 tweets sample -> 170 tweets/day -> outlier

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


# all_df = []
# for i in range(10):
#     tweets = twitter_client.twitter.search(q='lang:ca OR lang:es', geocode=SPAIN_GEOCODE, result_type='recent', count=100)
#     df_users = pd.DataFrame((map(lambda t: t['user'], tweets['statuses'])), columns=['created_at','followers_count','friends_count','favourites_count','statuses_count','screen_name'])
#     df_users['created_at'] = pd.to_datetime(df_users['created_at'],utc=True)
#     all_df.append(df_users)
#     time.sleep(60)


# %%


# all_tweets = pd.concat(all_df, sort=False)


# %%


# all_tweets = all_tweets.groupby('screen_name').first()


# %%


# import pandas as pd
#df_users = pd.DataFrame((map(lambda t: t['user'], tweets['statuses'])), columns=['created_at','followers_count','friends_count','favourites_count','statuses_count','screen_name'])
# all_tweets['created_at'] = pd.to_datetime(all_tweets['created_at'],utc=True)


# %%


# all_tweets['frequency'] = all_tweets['statuses_count'] /  all_tweets['created_at'].apply(lambda x: (pd.Timestamp.utcnow() - x).days)
# all_tweets['outlier_frequency'] = utils.relevant_outliers(all_tweets['frequency'])


# %%


# utils.outlier_num(all_tweets['frequency'])


# # Random walk

# %%


MIN_FOLLOWERS = 55
MIN_FOLLOWING = 95
MIN_ACTIONS = 1000
#quantile q=.1


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


#get_light_sample([902112076067430403, 944839492233555969])
#choice -> 0


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


import time


# %%


twitter_client = utils.TwitterClient('cred.txt')


# %%


for i in range(50):
    print('----> volta', i)
    execute_search()
    time.sleep(300)


