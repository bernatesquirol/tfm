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
        
    def get_light_user_data(self, user, path_file=None):
        timeline = self.get_timeline(user)
        light_timeline = utils.get_light_timeline(timeline)
        del timeline
        likes = self.get_likes(user)
        light_likes = utils.get_light_likes(likes)
        del likes
        concat = pd.concat([light_timeline, light_likes])
        if path_file:
            concat.to_pickle(os.path.join(path_file,'{}.pkl'.format(user)))
        return concat
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

# create functions to get table 
# https://academic.oup.com/jcmc/article/22/5/231/4666424


def get_features_timeline(timeline):
    features = {}
    if timeline['created_at'].dtype=='O':
        timeline_df['created_at']=pd.to_datetime(timeline_df['created_at'])
    features['user'] = user
    #features['f_f'] = timeline.iloc[0].user['followers_count']/timeline.iloc[0].user['friends_count']
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


def filter_users(users):
    users_filtered = users.copy()
    users_filtered = users_filtered[users_filtered['followers_count']<=2454815].copy()
    users_filtered = users_filtered[users_filtered['frequency_timeline']<=100].copy()
    users_filtered = users_filtered[users_filtered['frequency_like']<=130].copy()
    users_filtered = users_filtered[users_filtered['num_outliers_2']<=25].copy()
    users_filtered = users_filtered[users_filtered['social_ratio']>=10].copy()
    return users_filtered


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


import ipywidgets as widgets


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
