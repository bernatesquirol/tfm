from twython import Twython, TwythonError, TwythonRateLimitError, TwythonAuthError
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.stats import stattools

import time

LIST_ALL_CONGRESS = '1193835814754639872'


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
    def get_timeline_df(self, name):
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
        return pd.DataFrame(all_tweets)
    
    def try_call(self, call, deep=1):
        try:
            response = call()
            return response
        except TwythonRateLimitError as e:
            time_to_wait = int(self.twitter.get_lastfunction_header('x-rate-limit-reset')) - int(time.time()) + 1
            print('Rate limit exceded. Waiting {} seconds'.format(time_to_wait))            
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

def get_retweets(timeline):
    if 'retweeted_status' not in timeline.columns:
        return pd.DataFrame()
    rts = timeline[-timeline['retweeted_status'].isnull() & -timeline['retweeted_status'].isnull()]['retweeted_status']
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

def get_retweet_and_quoted_count(timeline, normalize=False):
    value_counts = get_retweet_and_quoted(timeline).apply(lambda x: x['user']['screen_name']).value_counts()
    if normalize:
        value_counts /= len(timeline)
    return value_counts


def outlayer_num(count_series,m=1.5):
    q3=count_series.quantile(q=.75)
    q1=count_series.quantile(q=.25)
    #print(medcouple(values),q3+np.exp(3*medcouple(values))*m*(q3-q1))
    return q3+np.exp(3*stattools.medcouple(count_series))*m*(q3-q1)


def relevant_outlayers(count_series,m=1.5):
    return count_series>outlayer_num(count_series,m)



