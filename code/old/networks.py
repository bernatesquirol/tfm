# ## Getting media & parties

from itertools import chain
import pandas as pd
import re
import os

journalists = pd.read_pickle('../data/journalists.pkl')

media = [x[2:].lower() for x in chain.from_iterable(journalists.description.apply(lambda x: re.findall(r' @[^ ]*\b', x) ).values)]

media_ordered = pd.Series(media).value_counts().head(50)

# +
# media_ordered
# -

politicians = pd.read_pickle('../data/politicians.pkl')

parties = [x[2:].lower() for x in chain.from_iterable(politicians.description.apply(lambda x: re.findall(r' @[^ ]*\b', x) ).values)]

# +
# parties
# -

parties_ordered = pd.Series(parties).value_counts().head(50)

# ## Link analysis

twitter_client = utils.TwitterClient('cred2.txt')

saved_user_timelines = {x:pd.read_pickle(os.path.join('../data',x)) for x in os.listdir('../data') if os.path.isfile(os.path.join('../data',x))}


def get_user(id_user, saved_user_timelines=saved_user_timelines):
    import os
    for file, db in saved_user_timelines.items():
        if id_user in db.id.values:
            screen_name = db[db.id==id_user].iloc[0].name
            return pd.read_pickle(os.path.join('..\\data',file[:-4],screen_name+'.pkl'))
    timeline = twitter_client.get_timeline(user_id=id_user)
    light_timeline = utils.get_light_timeline(timeline)
    return light_timeline


def get_user_screen_name(screen_name, saved_user_timelines=saved_user_timelines):
    import os
    for file, db in saved_user_timelines.items():
        if screen_name in db.index:
            screen_name = db.loc[screen_name].name
            return pd.read_pickle(os.path.join('..\\data',file[:-4],screen_name+'.pkl'))
    timeline = twitter_client.get_timeline(screen_name=screen_name)
    light_timeline = utils.get_light_timeline(timeline)
    return light_timeline


ego_id = 155350950

import utils


def get_outliers(ego_timeline, types=['RT']):
#     ego_timeline = get_user(id_user)
    if len(ego_timeline)==0:
        return ([],[])
    freq = ego_timeline[ego_timeline['type'].isin(types)].screen_name.value_counts()    
    freq_without_one = freq[freq>1].copy()    
    if len(freq_without_one)>1:
        return (freq_without_one[utils.relevant_outliers(freq_without_one)], freq_without_one)
    else: return ([],[])


outliers_politicians = politicians.id.apply(lambda x: np.array(get_outliers(get_user(x))))

outliers_count = np.array([[len(x), len(y)]for x,y in outliers_politicians])

x=np.vstack(outliers_count)[:,0]
y=np.vstack(outliers_count)[:,1]

outliers_x_unique = pd.DataFrame({'x':x,'y':y}, index=politicians.index)

outliers_x_unique.reset_index().head()

import plotly.express as px

fig = px.scatter(outliers_x_unique.reset_index(), x='x', y='y', hover_data=['x','y','index'])

fig.show()

outliers_politicians.loc['PilarVallugera']

utils.outlier_num(outliers.loc['PilarVallugera'][1],1.5)

politicians.id

# ## Create network

ego_user_id = politicians.loc['sanchezcastejon'].id



import networkx as nx


def get_outliers_timeline(ego_user_id, from_date=None, to_date=None):
    timeline = get_user(ego_user_id).reset_index()
    if not from_date:
        from_date=timeline.created_at.min()
    if not to_date:
        to_date=timeline.created_at.max()
    timeline_filtered = timeline[(timeline.created_at>=from_date) & (timeline.created_at<=to_date)]
    outliers_ego = get_outliers(timeline_filtered, types=['RT'])[0]
    outliers = {}
    bad_outliers = []
    for outlier_ego in outliers_ego.index.values:
        try:
            outlier = get_user_screen_name(outlier_ego)
            outliers[outlier_ego]=outlier            
        except:
            bad_outliers.append(outlier)
    return outliers, bad_outliers


outliers, bad_outliers = get_outliers_timeline(ego_user_id)


def get_network(ego_node, outliers):
    ego = nx.DiGraph()
    for outlier_node in outliers.keys():
        ego.add_edge(ego_node, outlier_node)
    nodes = list(ego.nodes())
    for outlier_key, outlier_timeline in outliers.items():
        outliers_outilers = get_outliers(outlier_timeline)[0]
        for outlier_outlier_key in outliers_outilers.index.values:
            if outlier_outlier_key in nodes:
                ego.add_edge(outlier_key, outlier_outlier_key)
    return ego


ego = get_network('sanchezcastejon', outliers)


def draw_ego(G):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import networkx as nx
    
    # G = nx.generators.directed.random_k_out_graph(10, 3, 0.5)
    pos = nx.layout.spring_layout(G)
    # edge_colors = range(2, M + 2)
    # edge_alphas = [(5 + i) / (M + 4) for i in range(M)]

    nx.draw_networkx_nodes(G, pos, node_size=100, node_color='yellow')
    nx.draw_networkx_edges(G, pos, node_size=100, arrowstyle='->',
                                   arrowsize=10, width=1)
    nx.draw_networkx_labels(G, pos, labels={n:n for n in G.nodes()})
    plt.show()


draw_ego(ego)



user = 'LauraBorras'
outliers, bad = get_outliers_timeline(politicians.id.loc[user])


draw_ego(get_network(user, outliers))


