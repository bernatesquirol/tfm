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
        if old.index.max()<new.index.min():
            print('missed_some', i, old.index.max(), new.index.min())
            missed_some.append(i)
        new_df = pd.concat([new[new.index>old.index.max()], old])
        if len(old)>len(new_df):
            print('something wrong')
        else:
            new_df.to_pickle('data/timelines/'+i)
    except:
        pass


news = os.listdir('data/all_ids_to_update')

olds = os.listdir('data/timelines')

pd.Series([n for n in news if n not in olds]).to_pickle('data/news_not_old.pkl')

new.min()

pd.concat([new[new.index>old.index.max()], old])

new[new.index>old.index.max()]


