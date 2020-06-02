from twython import Twython, TwythonError, TwythonRateLimitError, TwythonAuthError
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.stats import stattools
import scipy.stats as st


import time

LIST_ALL_CONGRESS = '1193835814754639872'
SPAIN_GEOCODE = '39.8952506,-3.4686377505768764,600000km'


# # General pandas

def save_dataframe(file, timeline, columns_timestamp = ['created_at']):
    import json
    with open(file, 'w') as json_file:
        for col in columns_timestamp:
            if col in timeline.columns:
                timeline[col]=timeline[col].astype(str)  
        json.dump(timeline.to_dict(), json_file)  


# +
# dict_keys = {}
# for i in range(0,len(lines),3):
#     dict_keys[lines[i].strip(), lines[i+1].strip()]=None
# -

# # Twython client class

class TwitterClient():
    def __init__(self, file_name='cred.txt'):
        with open(file_name) as f:
            lines = f.readlines()
            self.APP_KEY = lines[0].strip()
            self.APP_SECRET = lines[1].strip()
            if len(lines)>3:
                self.USER_TOKEN = lines[2].strip()
                self.USER_SECRET = lines[3].strip()
            else:
                self.USER_TOKEN = None
                self.USER_SECRET = None
            self.reset()
    def reset(self):
            self.twitter = Twython(self.APP_KEY, self.APP_SECRET, self.USER_TOKEN, self.USER_SECRET)
    def get_timeline(self, screen_name=None, user_id=None):
        all_tweets = []
        results = []
        next_max_id = None
        print('Extracting timeline of: @{}'.format(screen_name or user_id))
        
        while next_max_id==None or (results!=None and len(results) != 0):
            try:
                results = self.try_call(lambda: self.twitter.get_user_timeline(screen_name=screen_name, user_id=user_id, count=200, max_id=next_max_id, tweet_mode='extended'))
            except:
                #something went wrong: internet or auth
                break
            if results:
                next_max_id = results[-1]['id'] - 1
                all_tweets+=results
            else:
                break
        print('Total tweets: {}'.format(len(all_tweets)))
        timeline_df = pd.DataFrame(all_tweets)
        if len(timeline_df)>0:
            timeline_df['created_at']=pd.to_datetime(timeline_df['created_at'])
        return timeline_df
    def get_likes(self,  screen_name=None, user_id=None):
        all_tweets = []
        results = []
        next_max_id = None
        print('Extracting likes of: @{}'.format(screen_name or user_id))        
        while next_max_id==None or (results!=None and len(results) != 0):
            try:
                results = self.try_call(lambda: self.twitter.get_favorites(screen_name=screen_name, user_id=user_id, count=200, max_id=next_max_id, tweet_mode='extended'))
            except:
                #something went wrong: internet or auth
                break
            if results:
                next_max_id = results[-1]['id'] - 1
                all_tweets+=results
            else:
                break
        print('Total likes: {}'.format(len(all_tweets)))
        likes_df = pd.DataFrame(all_tweets)
        if len(likes_df)>0:
            likes_df['created_at']=pd.to_datetime(likes_df['created_at'])
        return likes_df
    def get_friends_list(self,  screen_name=None, user_id=None):
        all_friends = []
        next_cursor = -1
        response = None
        print('Extracting friends of: @{}'.format(screen_name or user_id))        
        while next_cursor!=0 and (next_cursor==-1 or (response!=None and len(response['users']) != 0)):
            print(next_cursor, response and len(response['users']), len(all_friends))
            time.sleep(5)
            try:
                response = self.try_call(lambda: self.twitter.get_friends_list(screen_name=screen_name, user_id=user_id, count=200, cursor=next_cursor))
            except:
                #something went wrong: internet or auth
                break
            if response:
                next_cursor = response['next_cursor']
                all_friends+=response['users']
            else:
                break
        print('Total friends: {}'.format(len(all_friends)))
        friends_df = pd.DataFrame(all_friends)
        return friends_df
    def get_followers_list(self,  screen_name=None, user_id=None):
        all_friends = []
        next_cursor = -1
        response = None
        print('Extracting followers of: @{}'.format(screen_name or user_id))        
        while next_cursor!=0 and (next_cursor==-1 or (response!=None and len(response['users']) != 0)):
            print(next_cursor, response and len(response['users']), len(all_friends))
            time.sleep(5)
            try:
                response = self.try_call(lambda: self.twitter.get_followers_list(screen_name=screen_name, user_id=user_id, count=200, cursor=next_cursor))
            except:
                #something went wrong: internet or auth
                break
            if response:
                next_cursor = response['next_cursor']
                all_friends+=response['users']
            else:
                break
        print('Total followers: {}'.format(len(all_friends)))
        friends_df = pd.DataFrame(all_friends)
        return friends_df
    def show_users(self, screen_names):
        dict_user_details = {}
        for screen_name in screen_names:
            dict_user_details[screen_name] = self.try_call(lambda: self.twitter.show_user(screen_name=screen_name, include_entities=False))
        return dict_user_details
        
    def try_call(self, call, deep=1, throw_errors = False):
        try:
            response = call()
            return response
        except TwythonRateLimitError as e:
            time_to_wait = int(self.twitter.get_lastfunction_header('x-rate-limit-reset')) - int(time.time()) + 10
            print('Rate limit exceded. {}th try. Waiting {} seconds: {}'.format(deep, time_to_wait, time.strftime('%X', time.localtime(time.time()+time_to_wait))))            
            time.sleep(time_to_wait)
            return self.try_call(call, deep=deep+1)
        except TwythonAuthError as e:
            print('Auth error')
            print(str(e))
            if throw_errors:
                raise
            return
        except TwythonError as e:
            if deep>=6:
                return
            print('No internet. Waiting {} seconds'.format(10))
            print(str(e))
            if throw_errors:
                raise
            #time.sleep(10)
            return #self.try_call(call, deep=deep+1)



# ## Tweet methods

def get_type_tweet(tweet):
    if 'retweeted_status' in tweet.keys() and not pd.isnull(tweet['retweeted_status']):
        return 'RT'
    elif 'quoted_status' in tweet.keys() and not pd.isnull(tweet['quoted_status']):
        return 'Quoted'
    elif 'in_reply_to_status_id' in tweet.keys() and (not pd.isnull(tweet['in_reply_to_status_id']) or not pd.isnull(tweet['in_reply_to_user_id'])):
        return 'Reply'
    elif len(tweet.entities['user_mentions'])>0:
        return 'Mention'
    else:
        return 'Text'


def get_type_tweet_and_user(tweet):
    type_tweet = get_type_tweet(tweet)
    user = None
    if type_tweet == 'RT':
        user = tweet['retweeted_status']['user']['screen_name']
    elif type_tweet =='Quoted':
        user = tweet['quoted_status']['user']['screen_name']
    elif type_tweet =='Reply':
        user = tweet['in_reply_to_screen_name']        
    elif type_tweet == 'Mention':
        user = tweet.entities['user_mentions'][0]['screen_name']
    return (type_tweet, user)

# ## Timeline df methods



def get_retweet_quoted_users_count(timeline):
    if not 'utils-type' in timeline.columns:
        timeline['utils-type']=timeline.apply(get_type_tweet, axis=1)
    timeline['utils-type']


def get_retweets(timeline):
    if 'retweeted_status' not in timeline.columns:
        return pd.DataFrame()
    rts = timeline[-timeline['retweeted_status'].isnull()]['retweeted_status']
    return pd.DataFrame(list(rts.values), index=rts.index)


def get_quoted(timeline):
    if 'quoted_status' not in timeline.columns:
        return pd.DataFrame()
    quoted = timeline[-timeline['quoted_status'].isnull()]['quoted_status']
    return pd.DataFrame(list(quoted.values), index=quoted.index)

def get_retweet_and_quoted(timeline):
    rts = get_retweets(timeline)
    quotes = get_quoted(timeline)
    #if len([i for i in rts.index if i in quoted.index])==0:
        #raise Exception('Retweets th')
    final = rts.append(quotes, verify_integrity=True, sort=False)
    if len(rts)>0:
        final.loc[rts.index, 'type']='retweet'
    if len(quotes)>0:
        final.loc[quotes.index, 'type']='quoted'
    return final 

def get_retweet_and_quoted_count(timeline, skip_ones=True):
    return user_frequency(get_retweet_and_quoted(timeline)['user'], skip_ones=skip_ones)


def user_frequency(tweets, skip_ones=True):
    value_counts = tweets['screen_name'].value_counts()
    if skip_ones:
        value_counts = value_counts[value_counts>1].copy()
    return value_counts

# ## Features timeline
# Similar to
# https://academic.oup.com/jcmc/article/22/5/231/4666424


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


# +
# def get_features_timeline(actions, party=False):
#     features = {}
#     if len(actions)<2: return {}
#     users_freq_actions = user_frequency(actions)
#     if party:
#         features['party']=get_political_party(users_freq_actions)
#     if len(users_freq_actions)>1:
# #         features['fit_exp'], features['fit_k'], dummie, features['fit_lambda'] = get_best_args(users_freq_actions, 'exponweib')    
#         features['1_2_actions']=users_freq_actions.iloc[0]/users_freq_actions.iloc[1]
#         if len(users_freq_actions)>2:
#             features['2_3_actions']=users_freq_actions.iloc[1]/users_freq_actions.iloc[2]        
#     timeline = actions[actions['type']!='Like']
#     likes = actions[actions['type']=='Like']
#     features['last_action'] = timeline['created_at'].max()
#     features['first_action'] = timeline['created_at'].min()
#     timeline_len = (actions['created_at'].max()-timeline['created_at'].min()).days
#     if timeline_len>0:
#         features['frequency_timeline']= len(timeline) / timeline_len
#     likes_len = (actions['created_at'].max()-likes['created_at'].min()).days
#     if likes_len>0:
#         features['frequency_like']= len(likes) / likes_len
#     social_tweets_count = len(timeline[timeline['type']!='Text'])
#     if 'frequency_timeline' in features and 'frequency_like' in features:
#         features['like_tweet_ratio'] = features['frequency_like']/features['frequency_timeline']
#     if len(timeline):
#         features['social_ratio'] = 100*social_tweets_count/len(timeline)
#     if social_tweets_count>0:
#         features['reply_sratio'] = 100*(len(timeline[timeline['type']=='Reply'])/social_tweets_count)
#         features['rt_sratio'] = 100*(len(timeline[timeline['type']=='RT'])/social_tweets_count)
#         features['mention_sratio'] = 100*(len(timeline[timeline['type']=='Mention'])/social_tweets_count)
#         features['quoted_sratio'] = 100*(len(timeline[timeline['type']=='Quoted'])/social_tweets_count)
#         freq_1 = user_frequency(timeline, skip_ones=False)
#         if len(freq_1)>1:
#             features['num_outliers_1'] = len(freq_1[relevant_outliers(freq_1)])
#         freq_2 = user_frequency(timeline, skip_ones=True)
#         if len(freq_2)>1:
#             features['num_outliers_2'] =len(freq_2[relevant_outliers(freq_2)])
#         sorted_features = sorted([features['reply_sratio'],features['rt_sratio'],features['mention_sratio']],reverse=True)
#         if sorted_features[1]>0:
#             features['1_2_ratios']=sorted_features[0]/sorted_features[1]
#     return features
# -

def get_light_timeline(timeline):
    if len(timeline)==0:
        return timeline
    #timeline['created_at']=timeline['created_at'].astype(str)    
    timeline = timeline.set_index('created_at')
    type_user = timeline.apply(get_type_tweet_and_user, axis=1)
    timeline['type']=type_user.apply(lambda x: x[0])
    timeline['screen_name']=type_user.apply(lambda x: x[1])
    return timeline[['type','screen_name']]


def get_light_likes(likes):
    #likes['created_at']=likes['created_at'].astype(str)  
    if len(likes)==0:
        return likes
    likes = likes.set_index('created_at')
    likes['type'] = 'Like'
    likes['screen_name'] = likes.user.apply(lambda u: u['screen_name'])
    return likes[['type','screen_name']]


def get_light_timeline(timeline):
    #timeline['created_at']=timeline['created_at'].astype(str) 
    if len(timeline)==0:
        return timeline
    timeline = timeline.set_index('created_at')
    type_user = timeline.apply(get_type_tweet_and_user, axis=1)
    timeline['type']=type_user.apply(lambda x: x[0])
    timeline['screen_name']=type_user.apply(lambda x: x[1])
    return timeline[['type','screen_name']]


def filter_users(users, max_dict = {'followers_count':2454815, 
                                    'frequency_timeline':100,
#                                     'frequency_like':130, 
                                    'num_outliers_2':25},
                        min_dict = {'social_ratio':10}
                ):
    users_filtered = users.copy()
    for col, max_col in max_dict.items():
        users_filtered = users_filtered[users_filtered[col]<=max_col].copy()
    for col, min_col in min_dict.items():
        users_filtered = users_filtered[users_filtered[col]>=min_col].copy()
    return users_filtered


# # Database load

def get_screen_names(path):
    return [name[:-4] for name in os.listdir(path)]


def get_features_timeline(actions, party=False):
    features = {}
    if len(actions)<2: return {}
    users_freq_actions = user_frequency(actions)
    if party:
        features['party']=get_political_party(users_freq_actions)
    if len(users_freq_actions)>1:
#         features['fit_exp'], features['fit_k'], dummie, features['fit_lambda'] = get_best_args(users_freq_actions, 'exponweib')    
        features['1_2_actions']=users_freq_actions.iloc[0]/users_freq_actions.iloc[1]
        if len(users_freq_actions)>2:
            features['2_3_actions']=users_freq_actions.iloc[1]/users_freq_actions.iloc[2]        
    timeline = actions[actions['type']!='Like']
    likes = actions[actions['type']=='Like']
    features['last_action'] = timeline['created_at'].max()
    features['first_action'] = timeline['created_at'].min()
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
    features['rt_ratio'] = 100*(len(timeline[timeline['type']=='RT'])/len(timeline))
    if social_tweets_count>0:
        features['reply_sratio'] = 100*(len(timeline[timeline['type']=='Reply'])/social_tweets_count)
        features['rt_sratio'] = 100*(len(timeline[timeline['type']=='RT'])/social_tweets_count)
        features['mention_sratio'] = 100*(len(timeline[timeline['type']=='Mention'])/social_tweets_count)
        features['quoted_sratio'] = 100*(len(timeline[timeline['type']=='Quoted'])/social_tweets_count)
        freq_1 = user_frequency(timeline, skip_ones=False)
        if len(freq_1)>1:
            features['num_outliers_1'] = len(freq_1[relevant_outliers(freq_1)])
        freq_2 = user_frequency(timeline, skip_ones=True)
        if len(freq_2)>1:
            features['num_outliers_2'] =len(freq_2[relevant_outliers(freq_2)])
        sorted_features = sorted([features['reply_sratio'],features['rt_sratio'],features['mention_sratio']],reverse=True)
        if sorted_features[1]>0:
            features['1_2_ratios']=sorted_features[0]/sorted_features[1]
    return features


def get_pre_post_covid(db):
    return (db[db.created_at<=pd.Timestamp('15/03/2020 00:00').tz_localize('UTC')], db[db.created_at>pd.Timestamp('15/03/2020 00:00').tz_localize('UTC')])


def load_users(db,root='..\\data', party=False):
    import os
    path = os.path.join(root,db)
    list_politicians_dict = {}
    for p in os.listdir(path):
        path_file = os.path.join(path,p)
        p_dict = get_features_timeline(pd.read_pickle(path_file).reset_index(), party=party)
        list_politicians_dict[p[:-4]]=p_dict
    activity_profiles = pd.DataFrame(list_politicians_dict).T
    user_profile = ['followers_count','friends_count', 'verified', 'statuses_count','favourites_count']
    try:
        profiles = pd.read_pickle(path+'.pkl')
        return profiles[user_profile].join(activity_profiles)
    except:
        return activity_profiles


def load_users_covid(db,root='..\\data', party=False):
    import os
    path = os.path.join(root,db)
    list_politicians_dict_pre = {}
    list_politicians_dict_post = {}
    for p in os.listdir(path):
        try:
            path_file = os.path.join(path,p)
            timeline = pd.read_pickle(path_file).reset_index()
            if len(timeline)>0:
                timeline = timeline.rename(columns={'createdAt': 'created_at'})
                pre_timeline, post_timeline = get_pre_post_covid(timeline)
                p_dict_pre = get_features_timeline(pre_timeline, party=party)
                list_politicians_dict_pre[int(p[:-4])]=p_dict_pre
                p_dict_post = get_features_timeline(post_timeline, party=party)
                list_politicians_dict_post[int(p[:-4])]=p_dict_post
        except:
            print(p)
    activity_profiles_pre = pd.DataFrame(list_politicians_dict_pre).T
    activity_profiles_post = pd.DataFrame(list_politicians_dict_post).T
    
    user_profile = ['followers_count','friends_count', 'verified', 'statuses_count','favourites_count', 'screen_name']
    try:
        profiles = pd.read_pickle(path+'.pkl').set_index('id')
        return (profiles[user_profile].join(activity_profiles_pre, how='inner'), profiles[user_profile].join(activity_profiles_post, how='inner'))
    except:
        return activity_profiles_pre, activity_profiles_post

# +
# profiles.join(pre, 'id')
# -



# +
# pre.index.name='id'

# +
# a = (pre_timeline.groupby('created_at').count()>2)

# +
# a[a['type']]
# -

# # Stats utils

import scipy.stats as st


def outlier_num(count_series,m=1.5):
    q3=count_series.quantile(q=.75)
    q1=count_series.quantile(q=.25)
    #print(medcouple(values),q3+np.exp(3*medcouple(values))*m*(q3-q1))
    return q3+np.exp(3*stattools.medcouple(count_series))*m*(q3-q1)



def relevant_outliers(count_series,m=1.5):
    return count_series>outlier_num(count_series,m)


def plot_best_args(frequency, dist_name, title=None):
    y = np.concatenate([np.zeros(v)+i+1 for i, v in enumerate(frequency.values)])
    #first
    plt.subplot(131)
    dist = getattr(st, dist_name)
    plt.hist(y, density=True, alpha=0.5, bins=len(frequency))
    args = dist.fit(y, floc=0)
    x = np.linspace(y.min(), y.max(), 100)
    plt.plot(x, dist(*args).pdf(x))
    plt.title("{} fit on data".format(dist_name))
    
    #second
    plt.subplot(132)
    import statsmodels.distributions
    ecdf = statsmodels.distributions.ECDF(y)
    plt.plot(x, ecdf(x), label="Empirical CDF")
    plt.plot(x, dist(*args).cdf(x),label="{} fit".format(dist_name))
    plt.title("Cumulative failure intensity")
    plt.legend()
    #third
    plt.subplot(133)
    
    st.probplot(y, dist=dist(*args),plot=plt, fit=True)
    plt.title("{} QQ-plot".format(dist_name))
    
    #plt.show()
    if title: plt.suptitle(title)
    test = st.kstest(y, dist_name, args=args)    
    return args, test


def get_best_args(frequency, dist_name):
    y = np.concatenate([np.zeros(v)+i+1 for i, v in enumerate(frequency.values)])
    dist = getattr(st, dist_name)
    args = dist.fit(y, floc=0)
    return args


def compare_two_distributions(a1, a2, n=1000, prop=0.7):
    min_len = min(len(a1), len(a2))
    sample_size = int(prop*min_len)
    tests = [st.ks_2samp(a1.sample(sample_size).values, a2.sample(sample_size).values) for i in range(n)]
    return np.mean(np.array(list(map(lambda x: np.array([x.statistic, x.pvalue]), tests))), axis=0)



def check_prop_two_distributions(a1, a2, n=1000):
    dict_prop = {}
    for prop in [0.5, 0.6, 0.7, 0.8, 0.9]:
        dict_prop[prop]=compare_two_distributions(a1, a2, n, prop)
    return dict_prop

# +
# twitter_client = TwitterClient()
# user = 'elsa_artadi'
# timeline = twitter_client.get_timeline(user)
# likes = twitter_client.get_likes(user)

# +
# len(timeline['created_at'].unique())

# +
# len(pd.concat([get_light_timeline(timeline),get_light_likes(likes)]))

# +
# len(timeline)+len(likes)
# -
# # Visualization


# ## Single

import altair as alt


def get_user_final_timeline(timeline, types):
    final = timeline[timeline.type.isin(types)].copy()
    freq = final.screen_name.value_counts()
    relevant_outliers_result = relevant_outliers(freq)
    final['outlier']=final.screen_name.apply(lambda user: relevant_outliers_result[user] if user in relevant_outliers_result else False)
    return final


def plot_top_users_time(user, types=['RT', 'Mention', 'Text', 'Reply', 'Quoted', 'Like'] ):
    final = get_user_final_timeline(user, types)
    all_users_x_month = alt.Chart(final,width=400).mark_bar(
        cornerRadiusTopLeft=3,
        cornerRadiusTopRight=3
    ).encode(
        x='yearmonth(created_at):O',
        y=alt.Y('count():Q'),#, sort=alt.SortField(field="count():Q", op="distinct", order='ascending')),
        color=alt.Color('screen_name:N',sort='-y'),
        order=alt.Order('count():Q')
    )
    outlier_transparency = alt.Chart(final, width=400).mark_bar(
        cornerRadiusTopLeft=3,
        cornerRadiusTopRight=3,
        opacity=0.7,
        color='black'
    ).encode(
        x='yearmonth(created_at):O',
        y=alt.Y('count():Q'),#, sort=alt.SortField(field="count():Q", op="distinct", order='ascending')),
        opacity=alt.Opacity('outlier:O',sort='-y', scale=alt.Scale(range=[0.35,0])),
        order='order:N'
    ).transform_calculate(
        order="if (datum.outlier, 1, 0)"
    )
    return alt.layer(all_users_x_month,outlier_transparency).resolve_scale(color='independent')


def plot_top_rt_and_quote(user):
    final = get_user_final_timeline(user,  types=['RT', 'Quote'])
    return alt.Chart(final).mark_bar(
        cornerRadiusTopLeft=3,
        cornerRadiusTopRight=3
    ).encode(
        x=alt.X('screen_name:N', sort='-y'),
        y=alt.Y('count():Q'),
        color='outlier'
    )


# ## Global analysis

import ipywidgets as widgets
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


def plot(df, columns_to_plot, column_to_color=None, bins=None):
    import plotly as py
    import plotly.graph_objs as go
    import plotly.express as px
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


# ## Networks

import networkx as nx
def plotly_graph(G, labels=None, pos=None):
    import plotly.graph_objects as go
    edge_x = []
    edge_y = []
    if not pos:
        pos = nx.circular_layout(G)
    nx.set_node_attributes(G, pos, name='pos')
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=labels,
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))
    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append('# of connections: '+str(len(adjacencies[1])))

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text
    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                title='<br>Network graph made with Python',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    fig.show()

# # Bayes


import tensorflow as tf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime


def plot_tau(model, fig=None, **kwargs):
    if not fig:
        fig = go.Figure()    
    fig.add_trace(go.Histogram(x=model['tau'], histnorm='probability', name='tau'), **kwargs)
    return fig


def plot_lambdas(l1, l2, fig=None, **kwargs):
    if not fig:        
        fig = go.Figure()
    fig.add_trace(go.Histogram(x=l1, histnorm='probability', name='lambda_1'), **kwargs)
    fig.add_trace(go.Histogram(x=l2, histnorm='probability',  name='lambda_2'), **kwargs)
#     fig.update_layout(barmode='overlay')
#     fig.update_traces(opacity=0.75)
    return fig


def expected_texts_bipoisson(model, freq):
    n_count_data = len(freq)
    N_ = model['tau'].shape[0]
    day_range = tf.range(0,n_count_data,delta=1,dtype = tf.int32)
    day_range = tf.expand_dims(day_range,0)
    day_range = tf.tile(day_range,tf.constant([N_,1]))
    tau_samples_per_day = tf.expand_dims(model['tau_samples'],0)
    tau_samples_per_day = tf.transpose(tf.tile(tau_samples_per_day,tf.constant([day_range.shape[1],1])))
    tau_samples_per_day = tf.cast(tau_samples_per_day,dtype=tf.int32)
    ix_day = day_range < tau_samples_per_day
    lambda_1_samples_per_day = tf.expand_dims(model['lambda_1'],0)
    lambda_1_samples_per_day = tf.transpose(tf.tile(lambda_1_samples_per_day,tf.constant([day_range.shape[1],1])))
    lambda_2_samples_per_day = tf.expand_dims(model['lambda_2'],0)
    lambda_2_samples_per_day = tf.transpose(tf.tile(lambda_2_samples_per_day,tf.constant([day_range.shape[1],1])))
    expected_texts_per_day = ((tf.reduce_sum(lambda_1_samples_per_day*tf.cast(ix_day,dtype=tf.float32),axis=0) + tf.reduce_sum(lambda_2_samples_per_day*tf.cast(~ix_day,dtype=tf.float32),axis=0))/N_)
    return expected_texts_per_day


def plot_expected(model, freq, n=1, fig=None, **kwargs):
    if not fig:
        fig = go.Figure()
    fig.add_trace(go.Scatter(x=freq.index, y=expected_texts_bipoisson(model, freq) , mode='lines', name='expected # tweets'), **kwargs)
    fig.add_trace(go.Scatter(x=freq.index, y=freq.rolling(n).mean(), mode='lines', name='moving mean n=1'), **kwargs)
    fig.add_trace(go.Scatter(x=freq.index, y=freq.rolling(n).mean()+freq.rolling(n).std(),
        fill=None,
        mode='lines',
        line_color='rgba(250, 250, 0, 1)',
        opacity=0.01,
        ))
    fig.add_trace(go.Scatter(x=freq.index, y=freq.rolling(n).mean()-freq.rolling(n).std(),
        fill='tonexty', # fill area between trace0 and trace1
        mode='lines',
        line_color='rgba(250, 250, 0, 1)'))
    fig.add_trace(go.Bar(x=freq.index, y=freq.values, name='# tweets'), **kwargs)
    return fig


def plot_everything(model, freq, n=1):
    fig = make_subplots(rows=4, cols=1, specs=[[{}],[{}],[{"rowspan":2}],[None]])
    fig.update_layout(
        height=600)
    plot_lambdas(model['lambda_1'],model['lambda_2'], fig=fig, row=1, col=1)
    plot_tau(model, fig=fig, row=2, col=1)
    return plot_expected(model, freq, n=n, fig=fig, row=3, col=1, )
