import os


for i in os.listdir('data/journalists'):
    if os.path.isfile('data/journalists-new/'+i):
        full_timeline = pd.concat([pd.read_pickle('data/journalists-new/'+i),pd.read_pickle('data/journalists/'+i)])
        full_timeline[~full_timeline.index.duplicated()].to_pickle('data/timelines/'+i)

import pickle

pickle.load

import pandas as pd

pd.read_pickle('data/journalists/'+i)


def read(path, type_df):
    df = pd.read_pickle(path).set_index('id')
    df['type_profile'] = type_df
    return df


all_profiles = pd.concat([read('data/journalists.pkl', 'journalist'),
                          read('data/journalists-new.pkl', 'journalist'),
                          read('data/politicians.pkl', 'politician'), 
                          read('data/random-followers.pkl', 'random-follower'),
                          read('data/random-followers-big.pkl', 'random-follower'),
                          read('data/random-friends-big.pkl', 'random-friend'),
                          read('data/random-friends.pkl', 'random-friend')])

save_duplicates = []

for i,g in all_profiles[all_profiles.index.duplicated(keep=False)].reset_index().groupby('id'):
    if len(g.type_profile.unique())!=1:
        p_and_j = g.type_profile.isin(['journalist','politician'])
        if np.any(p_and_j):
            save_duplicates.append(g[p_and_j].iloc[0].name)
        else:
            save_duplicates.append(g.sample().name)

# +
# len(all_profiles[all_profiles.index.duplicated(keep=False)].reset_index().id.unique())

# +
# len(save_duplicates)

# +
# all_profiles[(~all_profiles.index.duplicated(keep=False)) | (all_profiles.index.isin(save_duplicates))].to_pickle('all_profiles.pkl')
# -

# all_profiles[(~all_profiles.index.duplicated(keep=False)) | (all_profiles.index.isin(save_duplicates))]
import pandas as pd
missed_some = []

pd.Series(missed_some).to_pickle('missed_some_may_update.pkl')

import os
for i in os.listdir('data/all_ids_to_update'):
    print(i)
    new = pd.read_pickle('data/all_ids_to_update/'+i)
    try:
        old = pd.read_pickle('data/timelines/'+i)
        if old.index.max()<=new.index.min():
            print('missed_some', i, old.index.max(), new.index.min())
            missed_some.append(i)
#         new_df = pd.concat([new[new.index>old.index.max()], old])
#         if len(old)>len(new_df):
#             print('something wrong')
#         else:
#             new_df.to_pickle('data/timelines/'+i)
    except:
        pass


def get_timeline_frequency(path):
    timeline = pd.read_pickle('data/timelines/'+path).sort_index(ascending=True).reset_index()
    if len(timeline)==0:
        return None
    timeline.created_at = timeline.created_at.apply(lambda ts: ts-datetime.timedelta(hours=ts.hour, minutes=ts.minute, seconds=ts.second))
    
    freq = timeline.created_at.value_counts(sort=False).loc[timeline.created_at.unique()]
    freq = freq[freq.index>pd.to_datetime('2019-09-01 00:00:00+00:00')]
    if len(freq)<50:
        return None
    missing_dates = pd.Series(0, index=[i for i in pd.date_range(freq.index.min(), periods=(freq.index.max()-freq.index.min()).days) if i not in freq.index])
    return pd.concat([freq, missing_dates]).sort_index()


import datetime

pd.read_pickle('data/timelines/1000764488.pkl')

freqs = {}
# for i in os.listdir('data/timelines')[17320:]:
#     freqs[i]=get_timeline_frequency(i)

import shutil

for timeline in skip_timelines:
    shutil.move('data/timelines/'+timeline, 'data/useless_timelines/'+timeline)

old[old.index>pd.Timestamp("2020-03-14").tz_localize('UTC')]

news = os.listdir('data/all_ids_to_update')

olds = os.listdir('data/timelines')

pd.Series([n for n in news if n not in olds]).to_pickle('data/news_not_old.pkl')

new.min()

pd.concat([new[new.index>old.index.max()], old])

new[new.index>old.index.max()]

import os

import pandas as pd
import utils

client = utils.TwitterClient()

# %load_ext autoreload

# %autoreload 2

ids = [i[:-4] for i in os.listdir('../data/models') if int(i[:-4]) not in profiles.index]

profiles = pd.read_pickle('../data/all_profiles.pkl')

client.twitter.show_user(user_id=ids[0], include_entities=False)

new_profiles = client.show_users_ids(ids)

import importlib
importlib.reload(utils)

import os
tls = os.listdir('../data/timelines')



def get_tl_ratios(id_t):
    tl = pd.read_pickle('../data/timelines/{}'.format(id_t))
    min_obs = pd.Timestamp('2019-09-11 00:00:00+0000', tz='UTC', freq='W-MON')
    max_obs = tl.index.max()
    all_zeros_week = pd.Series({min_obs+pd.Timedelta(days=i*7):0 for i in range(int((max_obs-min_obs).days/7))})
    dict_all = {}
    tl = tl[tl.type!='Like'].copy()
    for i, w in enumerate(all_zeros_week.index[1:]):
        prev_week = all_zeros_week.index[i-1]
        values = tl[(tl.index>prev_week)&(tl.index<w)].type.value_counts()
        if len(values)>0:
            dict_all[w]=values
    df = pd.DataFrame(dict_all).T
    if len(df)==0:
        return pd.DataFrame()
    df = df.fillna(0)
    total_sum=df.sum(axis=1)
    df = df.apply(lambda x: x/total_sum, axis=0)
    df['total'] = total_sum
    if 'Mention' in df.columns and 'Text' in df.columns:
        df['Text']+=df['Mention'].fillna(0)
    df.total = (df.total-df.total.min())/(df.total.max()-df.total.min())
    return df


from tqdm import tqdm

for tl_id in tqdm(not_done):
    get_tl_ratios(tl_id).to_pickle('../data/pickles/'+tl_id+'.pkl')

import pickle
pickle.dump(all_data, open('../data/picklepickle.pkl', 'wb'))

all_data={}
for 


# +

# df.to
# -
not_done = [t for t in tls if t[:-4] not in all_data]

len(not_done)

prova = get_tl_ratios(tls[6])

pkl_1 = pd.read_pickle('../data/pickles/'+pkl)



all_profiles = pd.read_pickle('../data/all_profiles_mod.pkl')

all_profiles[-all_profiles.observed_end.isnull()].observed_end.max()

all_zeros_week.loc[pkl_1.index]+=pkl_1.RT.values

all_profiles.type_profile.unique()

RT_count = RT_count_total.copy()

min_obs = pd.Timestamp('2019-09-11 00:00:00+0000', tz='UTC', freq='W-MON')
max_obs = pd.Timestamp('2020-05-16 00:00:00+0000', tz='UTC', freq='W-MON')
# RT_count = pd.Series({min_obs+pd.Timedelta(days=i*7):1 for i in range(int((max_obs-min_obs).days/7))})
RT_count_total = pd.Series({min_obs+pd.Timedelta(days=i*7):1 for i in range(int((max_obs-min_obs).days/7))})
# text_count = pd.Series({min_obs+pd.Timedelta(days=i*7):1 for i in range(int((max_obs-min_obs).days/7))})
# text_count_total = pd.Series({min_obs+pd.Timedelta(days=i*7):1 for i in range(int((max_obs-min_obs).days/7))})
# total_count = pd.Series({min_obs+pd.Timedelta(days=i*7):1 for i in range(int((max_obs-min_obs).days/7))})
# total_count_total = pd.Series({min_obs+pd.Timedelta(days=i*7):1 for i in range(int((max_obs-min_obs).days/7))})
for pkl in os.listdir('../data/pickles'):
    file = pd.read_pickle('../data/pickles/'+pkl)
    if 'RT' in file.columns:
#         RT_count.loc[file.index]+=file['RT']
        RT_count_total.loc[file.index]+=1
#     if 'Text' in file.columns:
#         text_count.loc[file.index]+=file['Text']
#         text_count_total.loc[file.index]+=1
#     if 'total' in file.columns:
#         file = file[-file.total.isnull()]
#         total_count.loc[file.index]+=file['total']
#         total_count_total.loc[file.index]+=1
    else:
        print(pkl)

RT_count_2 = RT_count/RT_count_total
text_count_2 = text_count/text_count_total
total_count_2 = total_count/total_count_total

# +
# RT_count_total
# -

import plotly.express as px
fig = px.bar(x=total_count[2:].index, y=total_count[2:].values/np.hstack([a,[a[-1]]]))
# total_count.plot()


fig = px.bar(x=total_count_2[2:].index, y=total_count_2[2:])

fig

fig.layout['xaxis']['title']['text']='Time'
fig.layout['yaxis']['title']['text']='Mean tweet distribution'


final_bias = np.hstack([a,[a[-1]]])

import utils
utils.plotly_to_tfm(fig, 'intro-overall-activity-2')

import plotly.graph_objects as go

fig = go.Figure()
colors = ['#636EFA','#EF553B','#14D09E', '#AB63FA']
fig.add_trace(go.Scatter(x=total_count_2[2:].index, y=total_count_2[2:].values, name='RT_ratio', marker_color=colors[2]))
# fig.add_trace(go.Scatter(x=text_count_2[2:].index, y=text_count_2[2:].values, name='text_ratio', marker_color=colors[1]))

utils.plotly_to_tfm(fig, 'intro-overall-ratio-rt')

# for g_id,g in  all_profiles[-all_profiles.dict_levels.isnull()].groupby('type_profile'):
g = all_profiles[-all_profiles.dict_levels.isnull()]
starting = g.dict_levels.apply(lambda x: list(x.keys())[0])


min_obs = pd.Timestamp('2019-09-02 00:00:00+0000', tz='UTC', freq='W-MON')
max_obs = pd.Timestamp('2020-05-11 00:00:00+0000', tz='UTC', freq='W-MON')
all_zeros_week = pd.Series({min_obs+pd.Timedelta(days=i*7):0 for i in range(int((max_obs-min_obs).days/7))})


def get_week_oct(date):
    if date<all_zeros_week.index[0]+pd.Timedelta(days=7):
        return all_zeros_week.index[0]
    for d in all_zeros_week.index[1:]:
        if date<d+pd.Timedelta(days=7):
            return d


tal = starting.apply(get_week_oct).value_counts().sort_index().cumsum()

px.line(y=tal.values, x=tal.index, title='Starting date bias')

for g_id,g in  all_profiles[-all_profiles.dict_levels.isnull()].groupby('type_profile'):
    starting = g.dict_levels.apply(lambda x: list(x.keys())[0])
    tal = starting.value_counts().sort_index().cumsum()
    fig = px.line(y=tal.values, x=tal.index, title='Starting date bias for '+g_id)
    fig.layout['xaxis']['title']['text']='observed_start date'
    fig.layout['yaxis']['title']['text']='#users'
    fig.show()





fig = px.line(y=tal.values, x=tal.index, title='Starting date bias')

fig.layout['xaxis']['title']['text']='observed_start date'

fig.layout['yaxis']['title']['text']='#users'

utils.plotly_to_tfm(fig, 'intro-bias')


