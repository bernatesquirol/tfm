from twython import Twython, TwythonError, TwythonRateLimitError, TwythonAuthError
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.stats import stattools

import time

LIST_ALL_CONGRESS = '1193835814754639872'
SPAIN_GEOCODE = '39.8952506,-3.4686377505768764,600000km'


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
    def get_timeline(self, name):
        all_tweets = []
        results = []
        next_max_id = None
        print('Extracting timeline of: @{}'.format(name))
        
        while next_max_id==None or (results!=None and len(results) != 0):
            try:
                results = self.try_call(lambda: self.twitter.get_user_timeline(screen_name=name, count=200, max_id=next_max_id, tweet_mode='extended'))
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
        timeline_df['created_at']=pd.to_datetime(timeline_df['created_at'])
        return timeline_df
    def get_likes(self, name):
        all_tweets = []
        results = []
        next_max_id = None
        print('Extracting likes of: @{}'.format(name))        
        while next_max_id==None or (results!=None and len(results) != 0):
            try:
                results = self.try_call(lambda: self.twitter.get_favorites(screen_name=name, count=200, max_id=next_max_id, tweet_mode='extended'))
            except:
                #something went wrong: internet or auth
                break
            if results:
                next_max_id = results[-1]['id'] - 1
                all_tweets+=results
            else:
                break
        print('Total likes: {}'.format(len(all_tweets)))
        timeline_df = pd.DataFrame(all_tweets)
        timeline_df['created_at']=pd.to_datetime(timeline_df['created_at'])
        return timeline_df
    #def get_timeline_features(self, name):
    
    def try_call(self, call, deep=1):
        try:
            response = call()
            return response
        except TwythonRateLimitError as e:
            time_to_wait = int(self.twitter.get_lastfunction_header('x-rate-limit-reset')) - int(time.time()) + 1
            print('Rate limit exceded.{}th try. Waiting {} seconds'.format(deep, time_to_wait))            
            time.sleep(time_to_wait)
            return self.try_call(call, deep=deep+1)
        except TwythonAuthError as e:
            print('Auth error')
            print(str(e))
            return
        except TwythonError as e:
            if deep>=6:
                return
            print('No internet. Waiting {} seconds'.format(10))
            time.sleep(10)
            return self.try_call(call, deep=deep+1)

def get_type_tweet(tweet):
    if 'retweeted_status' in tweet.keys() and not pd.isnull(tweet['retweeted_status']):
        return 'RT'
    elif 'quoted_status' in tweet.keys() and not pd.isnull(tweet['quoted_status']):
        return 'Reply' #return 'Quoted'
    elif 'in_reply_to_status_id' in tweet.keys() and (not pd.isnull(tweet['in_reply_to_status_id']) or not pd.isnull(tweet['in_reply_to_user_id'])):
        return 'Quoted'
    elif len(tweet.entities['user_mentions'])>0:
        return 'Mention'
    else:
        return 'Text'


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
    return user_frequency(get_retweet_and_quoted(timeline), skip_ones=skip_ones)


def user_frequency(tweets, skip_ones=True):
    value_counts = tweets.user.apply(lambda x: x['screen_name']).value_counts()
    if skip_ones:
        value_counts = value_counts[value_counts>1].copy()
    return value_counts


def outlayer_num(count_series,m=1.5):
    q3=count_series.quantile(q=.75)
    q1=count_series.quantile(q=.25)
    #print(medcouple(values),q3+np.exp(3*medcouple(values))*m*(q3-q1))
    return q3+np.exp(3*stattools.medcouple(count_series))*m*(q3-q1)


def relevant_outlayers(count_series,m=1.5):
    return count_series>outlayer_num(count_series,m)

# create functions to get table 
# https://academic.oup.com/jcmc/article/22/5/231/4666424


def get_features_timeline(timeline):
    features = {}
    if timeline['created_at'].dtype=='O':
        timeline_df['created_at']=pd.to_datetime(timeline_df['created_at'])
    features['user'] = user
    features['f_f'] = timeline.iloc[0].user['followers_count']/timeline.iloc[0].user['friends_count']
    features['frequency']= len(timeline) / (timeline['created_at'].max()-timeline['created_at'].min()).days
    timeline['utils-type']=timeline.apply(get_type_tweet, axis=1)
    social_tweets_count = len(timeline[timeline['utils-type']!='Text'])
    features['social_ratio'] = 100*social_tweets_count/len(timeline)
    features['reply_sratio'] = 100*(len(timeline[timeline['utils-type']=='Reply'])/social_tweets_count)
    features['rt_sratio'] = 100*(len(timeline[timeline['utils-type']=='RT'])/social_tweets_count)
    features['mention_sratio'] = 100*(len(timeline[timeline['utils-type']=='Mention'])/social_tweets_count)
    features['quoted_sratio'] = 100*(len(timeline[timeline['utils-type']=='Quoted'])/social_tweets_count)
    sorted_features = sorted([features['reply_sratio'],features['rt_sratio'],features['mention_sratio']],reverse=True)
    features['f_s']=sorted_features[0]/sorted_features[1]
    return features
