import pandas as pd

import utils

import networkx as nx

import os

# ### Example P outliers diagram

tls = os.listdir('../data/timelines')

tls_ex = pd.read_pickle('..\data\\timelines\{}'.format(np.random.choice(tls)))

rt_counts = tls_ex[tls_ex.type=='RT'].screen_name.value_counts()

series = rt_counts[rt_counts>1]
nbins = series.unique().shape[0]

import plotly.express as px
fig = px.histogram(x=rt_counts[rt_counts>1], nbins=nbins*2, labels={'x':'P(Ru)', 'count':'Number of users', }, title='Number of users / RT count')



fig.add_shape(dict(
            type="line",
            x0=int(utils.outlier_num(rt_counts[rt_counts>1]))-0.5,
            y0=0,
            x1=int(utils.outlier_num(rt_counts[rt_counts>1]))-0.5,
            y1=120,
            line=dict(
                color="red",
                width=3
            )
))

tls_ex

covid = pd.Timestamp('15/03/2020 00:00').tz_localize('UTC')

# ## Creation of networks and properties in db

all_profiles = pd.read_pickle('../data/all_profiles.pkl')


def get_timeline(id_u, path='timelines'):
    if os.path.isfile('../data/{}/{}.pkl'.format(path, id_u)):
        tl = pd.read_pickle('../data/{}/{}.pkl'.format(path, id_u))
        return tl


def get_outliers(id_u, path='timelines'):
    if os.path.isfile('../data/{}/{}.pkl'.format(path, id_u)):
        tl = pd.read_pickle('../data/{}/{}.pkl'.format(path, id_u))
        if len(tl)>1:
            rt_counts = tl[(tl['type']=='RT')].screen_name.value_counts()
            rt_counts = rt_counts[rt_counts>1]
            if len(rt_counts)>1:
                return rt_counts[utils.relevant_outliers(rt_counts)].index    


dict_all_outliers = pd.read_pickle('../data/outliers_tls.pkl')

dict_all_outliers

from tqdm import tqdm

dict_all_outliers = {}
for i in tqdm(all_profiles.index):
    dict_all_outliers[i]=get_outliers(i, 'timelines')


dict_outliers_outliers = {}
for i in tqdm(os.listdir('../data/outliers')):
    dict_outliers_outliers[i[:-4]]=get_outliers(i[:-4], 'outliers')

id_u = i[:-4]
path= 'outliers'
tl = pd.read_pickle('../data/{}/{}.pkl'.format(path, id_u))
# rt_counts = tl[(tl.index<covid)&(tl['type']=='RT')].screen_name.value_counts()


outliers_1 = set(np.hstack(dict_all_outliers.values()))

dict_all_outliers = dict_all_outliers[~dict_all_outliers.index.isnull()]

all_profiles.loc[dict_all_outliers.index, 'outliers']=dict_all_outliers

# all_profiles
all_profiles

info_outliers = set([f[:-4] for f in os.listdir('../data/outliers/')])

diff = outliers_1.difference(info_outliers)

tls_outliers = set(dict_outliers_outliers.keys())

diff = outliers_1.difference(tls_outliers)

pd.Series(list(diff)).to_pickle('../data/missing_outliers.pkl')

len(tls_outliers)

import pickle
pickle.dump(dict_outliers_outliers, open('../data/outliers_outliers_pre', "wb"))

all_profiles['outliers'] = all_profiles.head().apply(lambda x: get_outliers(x.name), axis=1)

a = pd.read_pickle('../../tfm2/data_old/all_index.pkl')

all_outliers = pd.Series(dict_all_outliers)
all_outliers = all_outliers[all_outliers.index.isin(all_profiles.index)]
all_outliers = all_outliers[~all_outliers.index.isnull()]

all_profiles.loc[all_outliers.index,'outliers']=all_outliers.values

outlier_length = all_profiles['outliers'].apply(lambda x: len(x) if type(x)==pd.core.indexes.base.Index else np.nan)

outliers_np = all_profiles['outliers'].apply(lambda x: np.array(x) if type(x)==pd.core.indexes.base.Index else np.nan)

# +
all_profiles['outliers'] = outliers_np

len(np.unique(np.hstack(outliers_np[-outliers_np.isnull()].values)))
# -

outlier_length.values).shape

outlier = all_profiles['outliers'].iloc[0]

type(outlier)

all_profiles.loc[all_outliers.index]

all_profiles.to_pickle('../data/all_profiles_mod.pkl')

list_all = []
for d1 in ['politicians','journalists-new', 'random-followers-big', 'random-followers', 'random-friends', 'random-friends-big']:
    list_all.append(os.listdir('../../tfm2/data_old/'+d1))

len(np.hstack(list_all))

#  Evolving

import pickle
dict_levels = pd.read_pickle('../data/models_levels.pickle')

len(os.listdir('../data/timelines'))

keys = list(dict_levels.keys())


def get_outliers_x_tau(timeline_id):
    tls_ex = pd.read_pickle('../data/timelines/{}.pkl'.format(timeline_id))
    if not timeline_id in dict_levels:
        return {}
    bins_index = pd.IntervalIndex.from_breaks(list(dict_levels[timeline_id].keys())+[tls_ex.index.max()])
    tls_ex['bin_value'] = pd.Series(pd.cut(tls_ex.index,bins=bins_index), index=tls_ex.index)
    dict_outliers = {}
    for g_id, g in tls_ex.groupby('bin_value'):
    #     dict_outliers[g_id]=g.utils.relevant_
        counts_rt = g[g['type']=='RT'].screen_name.value_counts()
        counts_rt = counts_rt[counts_rt>1]
        if len(counts_rt)>1:
            dict_outliers[g_id.left] = list(counts_rt[utils.relevant_outliers(counts_rt)].index)
        else:
            dict_outliers[g_id.left]=np.array([])
    return dict_outliers


all_outliers_x_tau = {}
for timeline in tqdm(tls):
    all_outliers_x_tau[timeline] = get_outliers_x_tau(timeline[:-4])

all_outliers_x_tau_series = pd.Series(list(all_outliers_x_tau.values()), index=[int(i[:-4]) for i in all_outliers_x_tau.keys()])

all_outliers_x_tau_series.iloc[0]

all_outliers_x_tau_series = all_outliers_x_tau_series[all_outliers_x_tau_series.index.isin(all_profiles.index)]

[i for i in os.listdir('../data/timelines') if i not in all_profiles.index]

from tqdm import tqdm
import pandas as pd

all_profiles = pd.read_pickle('../data/all_profiles.pkl')

list(dict_levels.keys())

all_profiles.loc[all_outliers_x_tau_series.index, 'outliers_x_level'] = all_outliers_x_tau_series

all_profiles['dict_levels']=np.nan

all_profiles.loc[[int(i) for i in dict_levels.keys()], 'dict_levels']=list(dict_levels.values())

for i, row in all_profiles[['outliers','dict_levels','outliers_x_level']].iterrows():
    print(row.outliers_x_level)
    break



dict_outliers

dict_levels_series = pd.Series(list(dict_levels.values()), index=[int(i) for i in dict_levels.keys()])

len(set(dict_levels_series.index).difference(set(all_profiles.index)))

dict_levels_series = dict_levels_series[dict_levels_series.index.isin(all_profiles.index)]

all_profiles.loc[dict_levels_series.index, 'dict_levels'] = dict_levels_series

all_profiles.to_pickle('../data/all_profiles_mod.pkl')

# ## Computations

import pandas as pd

all_profiles = pd.read_pickle('../data/all_profiles_mod.pkl')

all_profiles

all_profiles

tal = all_profiles.iloc[0]

all_profiles


def get_outliers(timeline_id, folder='timelines', start_date=None, end_date=None):
    tls_ex = pd.read_pickle('../data/{}/{}.pkl'.format(folder, timeline_id))
    if start_date:
        tls_ex = tls_ex[tls_ex.index>start_date].copy()
    if end_date:
        tls_ex = tls_ex[tls_ex.index<end_date].copy()
#     dict_outliers[g_id]=g.utils.relevant_
    if len(tls_ex)<=1:
        return []
    counts_rt = tls_ex[tls_ex['type']=='RT'].screen_name.value_counts()
    counts_rt = counts_rt[counts_rt>1]
    counts_rt = counts_rt[counts_rt.index]
    if len(counts_rt)<=1:
        return []
    return counts_rt[utils.relevant_outliers(counts_rt)].index


from tqdm import tqdm

dict_outliers = {}
for i in tqdm(all_profiles.index):
    if '{}.pkl'.format(i) in tls:
        outliers = get_outliers(i)
        if len(outliers)>0:
            dict_outliers[i] = outliers

all_profiles.loc[list(dict_outliers.keys()),'outliers_tl']=list(dict_outliers.values())

all_outliers_sn = os.listdir('../data/outliers')+os.listdir('../data/missing_outliers')

all_outliers_now = np.hstack(dict_outliers.values())

missing = [i for i in np.unique(all_outliers_now) if '{}.pkl'.format(i) not in all_outliers_sn]


pd.Series(missing).to_pickle('missing_outliers_2.pkl')

all_profiles.to_pickle('')


def create_network_timeline(profile):
    ego = nx.DiGraph()
    for n in profile.outliers_tl:
        ego.add_edge(profile.screen_name, n)
#     ego_count = len(profile.outliers_tl)
    between_outliers = 0
    reciprocal = 0
    for n in profile.outliers_tl:
        outliers_outlier = get_outliers(n, 'outliers')
#         if outliers_outlier:
        for o_o in outliers_outlier:
            if o_o in ego:
                if o_o == profile.screen_name:
                    reciprocal +=1
                else: 
                    between_outliers += 1
                ego.add_edge(n, o_o)
    return ego, between_outliers, reciprocal


from tqdm import tqdm

len(all_profiles)

del iter_profiles

dict_networks = {}
for i, profile in tqdm(all_profiles.iterrows()):
    if type(profile.outliers_tl)==pd.Index and len(profile.outliers_tl)>0:
        dict_networks[i]=create_network_timeline(profile)

all_profiles.loc[list(dict_networks.keys()), 'ego_network']=[a[0] for a in dict_networks.values()]

all_profiles.loc[list(dict_networks.keys()), 'edges_between_outliers']=[a[1] for a in dict_networks.values()]

all_profiles.loc[list(dict_networks.keys()), 'edges_reciprocal_ego']=[a[2] for a in dict_networks.values()]

all_profiles.to_pickle('../data/all_profiles_mod.pkl')

ego_dict = ego.edges(data=True)

ego_df = pd.DataFrame(ego_dict, columns=['source', 'target', 'attributes'])

ego_df.head()

len(ego_df)

ego_df['weight']=1

ego_df = ego_df.drop(columns=['attributes'])

d3fdgraph.plot_force_directed_graph(ego_df)

all_profiles.sample().iloc[0].ego_network

example_profile = all_profiles[all_profiles.type_profile=='journalist'].sample().iloc[0]
fig2 = utils.plot_directed(example_profile.ego_network, example_profile.screen_name)
fig2

e
fig = utils.plot_directed(example_profile.ego_network, example_profile.screen_name)

utils.plotly_to_tfm(fig2, 'networks-ego-example-5')

all_profiles['ego_transitivity']=all_profiles.ego_network.apply(lambda G: nx.transitivity(G) if not pd.isnull(G) else np.nan)

selected = ((all_profiles.edges_between_outliers+all_profiles.edges_reciprocal_ego)>0)

all_profiles.head()

has_network = -all_profiles.ego_network.isnull()#&all_profiles.ego_network.apply(lambda x: len(x)>0 if not pd.isnull(x) else False)


def sample_hist(fig, series, name, color, n=25, n_samples=100, **kwargs):
    fig.add_trace(go.Histogram(x=np.zeros(4),  marker_color=color, legendgroup=name,name=name, visible='legendonly', showlegend=True), **kwargs)
    for i in range(n):
        showlegend = False
#         if i==0:
#             showlegend=True
        fig.add_trace(go.Histogram(x=series.sample(n_samples), opacity=1/n,  marker_color=color, legendgroup=name,name='<span style="{}">{}</span>'.format('blue', name), showlegend=showlegend), **kwargs)
    return fig


colors = ['#636EFA','#EF553B','#14D09E', '#AB63FA']



all_profiles_with_network = all_profiles[has_network].copy()

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,5)

st.kstest(g['degree'].value_counts().sort_index().values, 'gamma', *[[1,2,3]])

 y = np.concatenate([np.zeros(v)+i+1 for i, v in enumerate(sample)])
st.probplot(y, dist=st.gamma(*args),plot=plt, fit=True)

import utils
all_args = []
for g_id, g in all_profiles_with_network.groupby('type_profile'):
#     print(len(g[g.degree==0]))
    sample = g[g.degree>1].degree.sample(200, replace=True).value_counts().sort_index()
    args = utils.get_best_args(sample, 'gamma')
    pvalue = st.kstest(sample, 'gamma',*[[args[0], args[2]]])
    print(pvalue)
    plt.plot(x, st.gamma(*args).pdf(x), label=k)
    all_args.append([g_id,args])
    utils.plot_best_args(sample, 'gamma', title='{} (a:{:.2f}, b:{:.2f}, pvalue:{:.2f})'.format(g_id, args[0], args[2], pvalue[1]))
    plt.show()

import scipy.stats as st

for k,args in all_args:
    x = np.linspace(0, 50, 100)
    plt.plot(x, st.gamma(*args).pdf(x), label=k)
    plt.legend()
plt.show()

all_profiles_with_network['degree'] = all_profiles[has_network].ego_network.apply(len)

from plotly.subplots import make_subplots
import plotly.graph_objects as go


def compare_groups_plot(df, column, title, groupby='type_profile'):
    
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True)
    for i, g in enumerate(df.groupby(groupby)):
        g_id, g = g
        sample_hist(fig, g[column], g_id, colors[i], col=1, row=i+1, n=20)
    fig.update_layout(title=title, height=500, barmode='overlay')
    return fig


def compare_groups(df, column, groupby='type_profile', **kwargs):
    list_values = []
    groups = list(df.groupby('type_profile'))
    for i, g in enumerate(groups):
        g_id_1, g_1 = g
        for g_id_2, g_2 in groups:
            value = utils.compare_two_distributions(g_1[column], g_2[column], **kwargs)
            list_values.append({'id_1': g_id_1,'id_2': g_id_2, 'value':value[0]})
    
    df = pd.DataFrame(list_values)    
    return df.pivot(index='id_1', columns='id_2')


fig = compare_groups_plot(all_profiles_with_network, 'degree','Degree of ego-network')

import importlib
importlib.reload(utils)
utils.plotly_to_tfm(fig, 'networks-degree-type')

fig

values = compare_groups(all_profiles_with_network, 'degree')


def get_edges_outliers_graph(G, name):
    return len([e for e in G.edges() if e[0]!=name and e[1]!=name and e[0]!=e[1]])


all_profiles_with_network['edges_between_outliers_2']=all_profiles_with_network.apply(lambda x: get_edges_outliers_graph(x.ego_network, x.screen_name), axis=1)

list(all_profiles_with_network.ego_network.loc[14048325].edges())

get_edges_outliers_graph(all_profiles_with_network.ego_network.loc[14048325], all_profiles_with_network.screen_name.loc[14048325])

utils.plot_directed(all_profiles_with_network.ego_network.loc[14048325], all_profiles_with_network.screen_name.loc[14048325])

all_profiles_with_network.ego_network.loc[14048325].edges()

all_profiles_with_network['network_max_connected'] = all_profiles_with_network.ego_network.apply(lambda x: (len(x)-1)*(len(x)-2))
all_profiles_with_network['network_outlier_connected']=all_profiles_with_network['edges_between_outliers_2']/all_profiles_with_network['network_max_connected']
all_profiles_with_network.loc[all_profiles_with_network['degree']<=2, 'network_outlier_connected']=np.nan

parties = all_profiles_with_network[all_profiles_with_network.type_profile=='politician'].outliers_tl.apply(lambda outl: utils.get_political_party(pd.Series(1, index=outl)))

all_profiles_with_network[(all_profiles_with_network.type_profile=='politician') & (all_profiles_with_network.party.isnull())].apply(lambda x: print(x.name, '\n' , x.description), axis=1)



dict_parties = {
    230807056:'podemos',
    1115297214325194755:'vox_es',
    92407680:'psoe',
    1013051904396595200: 'psoe',
    1373591724: np.nan,
    311979693 : 'psoe',
    1926442338 : 'psoe',
    755798880118243328 :'pp',
    11086042:'psoe',
    3050371833:'vox_es',
    844285125248651264:'psoe',
    3050371833:'vox_es',
    293944355: 'psoe',
    821764964: 'pnv',
    386439973: 'pp',
    4580450962:'psoe',
    527564981: 'pp',
    1128589504682569733: 'pp',
    270901461:'psoe',
    1013051904396595200: 'psoe',
    3300574151 : 'podemos',
    265938163 : 'pp',
    3083771273 : 'pp',
    359852036 : 'psoe',
    1724984834 : 'erc',
    3019568698 : 'podemos',
    1179187428 : 'pp',
    442932958 : 'psoe',
    807001880220073988: 'pp'
}

all_profiles_with_network.loc[list(dict_parties.keys()), 'party']=list(dict_parties.values())

all_profiles_with_network.loc[parties.index, 'party']=parties.values

import matplotlib.pyplot as plt

# compare_groups_plot(, 'network_outlier_connected','Outlier connections', 'party')
for p_id, p in all_profiles_with_network[all_profiles_with_network.type_profile=='politician'].groupby('party'):
    print(p_id, len(p))
    p.degree.hist(density=True)
    plt.show()

all_profiles_with_network[all_profiles_with_network.type_profile=='politician'].groupby

all_profiles_with_network[all_profiles_with_network['network_outlier_connected']>1][['degree','network_outlier_connected', 'edges_between_outliers', 'network_max_connected']]

fig = compare_groups_plot(all_profiles_with_network, 'network_outlier_connected','Outlier connections')

fig

utils.plotly_to_tfm(fig, 'networks-outlier-connectedness')

compare_groups(all_profiles_with_network, 'network_outlier_connected', n=2000)


def get_reciprocal_edges(G, name):
    return len([e for e in G.edges() if e[1]==name and e[0]!=e[1]])


all_profiles_with_network['edges_reciprocal_ego_2']=all_profiles_with_network.apply(lambda x: get_reciprocal_edges(x.ego_network, x.screen_name), axis=1)

all_profiles_with_network['ego_reciprocity']=all_profiles_with_network['edges_reciprocal_ego_2']/all_profiles_with_network.ego_network.apply(lambda x: (len(x)-1))

tal = all_profiles_with_network.friends_count>0
all_profiles_with_network.loc[tal,'ratio_f_f']=all_profiles_with_network[tal]['followers_count']/all_profiles_with_network[tal]['friends_count']

import plotly.express as px
px.scatter(all_profiles_with_network[(all_profiles_with_network.ego_reciprocity>0) &(all_profiles_with_network.type_profile=='journalist')], x='ego_clustering', y='ratio_f_f')

#

all_profiles_with_network[(all_profiles_with_network.type_profile=='random-follower') & (all_profiles_with_network.ego_reciprocity==0.125)].sort_values('followers_count', ascending=False)[['screen_name','degree','outliers_tl','followers_count']].head(20)

compare_groups(all_profiles_with_network, 'ego_reciprocity', n=1000)

all_profiles_with_network[all_profiles_with_network.ego_reciprocity>0]()

fig = compare_groups_plot(all_profiles_with_network[all_profiles_with_network.ego_reciprocity>0], 'ego_reciprocity','Ego reciprocity')
# utils.plotly_to_tfm(fig, 'networks-ego-reciprocity-type')
fig

all_profiles_with_network['ego_clustering'] = all_profiles_with_network.apply(lambda x: nx.clustering(x.ego_network, x.screen_name), axis=1)

fig = compare_groups_plot(all_profiles_with_network, 'ego_clustering','Ego clustering')
utils.plotly_to_tfm(fig, 'networks-ego-clustering-type')

compare_groups(all_profiles_with_network, 'ego_clustering', n=1000)

# ##### Manual clustering

# +
# utils.plot_directed(all_profiles_with_network.ego_network.iloc[0], all_profiles_with_network.screen_name.iloc[0])
# -

count_triangles_directed(G)

utils.plot_directed(G_ex, 1)

for i in range(3,10):
    G_ex = nx.DiGraph()
    G_ex.add_edges_from(itertools.permutations(list(range(i)), 2))
    print(count_triangles_directed(G_ex, True))

import itertools


def count_triangles_directed(g, cycles=False): 
    nodes = list(g.nodes())
    count_Triangle = 0
    for i in nodes: 
        for index,j in enumerate(nodes): 
            for k in nodes[index:]: 
                # check the triplet if it satisfies the condition 
                if (i!=j and i !=k and j !=k) and \
                    (nx.has_path(g, i, j) and nx.has_path(g, i, k) and (nx.has_path(g, j, k) or nx.has_path(g, k, j)) \
                    or (nx.has_path(g, j, i) and nx.has_path(g, k, i) and (nx.has_path(g, j, k) or nx.has_path(g, k, j)))) \
                    and not (cycles or nx.has_path(g, i, j) and nx.has_path(g, j, k) and nx.has_path(g, k, i)):
                        count_Triangle += 1
    return count_Triangle/2



count_triangles_directed(G_ex)

G = all_profiles_with_network.ego_network.iloc[0]

list(nx.cluster._directed_triangles_and_degree_iter(G))

utils.plot_directed(G.subgraph(['albamoraleda','antonialaborde','EduLorenGarcia']),'albamoraleda')

# +

list(nx.cluster._directed_triangles_and_degree_iter(G.subgraph(['albamoraleda','antonialaborde','EduLorenGarcia'])))
# -

all_profiles_with_network['clustering_network'] = all_profiles_with_network.apply(lambda x: sum([t[3]/2 for t in nx.cluster._directed_triangles_and_degree_iter(x.ego_network) ])/3, axis=1) #if t[0]!=x.screen_name

compare_groups_plot(all_profiles_with_network, 'clustering_network','Outlier connectness')

from plotly.subplots import make_subplots
import plotly.graph_objects as go
fig = make_subplots(rows=4, cols=1, shared_xaxes=True)
for i, g in enumerate(all_profiles[selected].groupby('type_profile')):
    g_id, g = g
    fig.add_trace(go.Histogram2d(x=g.edges_reciprocal_ego, y=g.ego_network.apply(len), name=g_id),col=1, row=i+1, )
fig.update_layout(title='Degree of ego-network', height=700)

from plotly.subplots import make_subplots
import plotly.graph_objects as go
fig = make_subplots(rows=4, cols=1, shared_xaxes=True)
for i, g in enumerate(all_profiles[has_network].groupby('type_profile')):
    g_id, g = g
    fig.add_trace(go.Histogram(x=g['edges_between_outliers']/g.ego_network.apply(lambda x: (len(x)-1)*(len(x)-2)), name=g_id),col=1, row=i+1, )
fig.update_layout(title='Edges between outliers', height=700)

from plotly.subplots import make_subplots
import plotly.graph_objects as go
fig = make_subplots(rows=4, cols=1, shared_xaxes=True)
for i, g in enumerate(all_profiles[has_network].groupby('type_profile')):
    g_id, g = g
    fig.add_trace(go.Histogram(x=g['edges_between_outliers'], name=g_id),col=1, row=i+1, )
fig.update_layout(title='Edges between outliers', height=700)

from plotly.subplots import make_subplots
import plotly.graph_objects as go
fig = make_subplots(rows=4, cols=1, shared_xaxes=True)
all_profiles[has_network&&g.ego_network.apply(len)>
for i, g in enumerate(].groupby('type_profile')):
    g_id, g = g
    fig.add_trace(go.Histogram(x=g['ego_transitivity'], name=g_id),col=1, row=i+1, )
fig.update_layout(title='Ego transitivity', height=700)

import utils

import importlib
importlib.reload(utils)


# # Evolution

def get_outliers_tl(tl,start_date=None, end_date=None):
    if start_date:
        tl = tl[tl.index>start_date].copy()
    if end_date:
        tl = tl[tl.index<end_date].copy()
    if len(tl)>1:
        rt_counts = tl[(tl['type']=='RT')].screen_name.value_counts()
        rt_counts = rt_counts[rt_counts>1]
        if len(rt_counts)>1:
            return rt_counts[utils.relevant_outliers(rt_counts)].index    


def get_outliers_proportion(tl, start_date=None, end_date=None):
    if start_date:
        tl = tl[tl.index>start_date].copy()
    if end_date:
        tl = tl[tl.index<end_date].copy()
    rt_tl = tl[(tl['type']=='RT')]
    if len(tl)>1:
        rt_counts = rt_tl.screen_name.value_counts()
        rt_counts = rt_counts[rt_counts>1]
        rt_counts
        if len(rt_counts)>1:
            return utils.outlier_num(rt_counts)/len(rt_tl)


def get_rt_tl(tl, start_date=None, end_date=None):
    if start_date:
        tl = tl[tl.index>start_date].copy()
    if end_date:
        tl = tl[tl.index<end_date].copy()
    rt_tl = tl[(tl['type']=='RT')]
    if len(tl)>1:
        rt_value_counts = rt_tl.screen_name.value_counts()
        return rt_value_counts[rt_value_counts>1]



profile = all_profiles[all_profiles.index==257371804].iloc[0]



# +
# all_profiles.loc[list(dict_outliers.keys()), 'outliers_oct']=list(dict_outliers.values())
# -

def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union



def get_indicators_week(profile):



def get_dict_week_stability(profile):
    tl = get_timeline(profile.name)
    dates = list(profile.dict_levels.keys())+[profile.observed_end]
    all_periods = []
    days_until_monday = 7-dates[0].dayofweek
    start_date = dates[0]
    


    current_breakpoint = 0
    proportion_breakpoint = get_outliers_proportion(tl,start_date=dates[current_breakpoint], end_date=dates[current_breakpoint+1])

    if days_until_monday != 7:
        end_date = (dates[0]+pd.Timedelta(days=7-dates[0].dayofweek ))
        start_date = end_date-pd.Timedelta(days=7)

    end_date = start_date+pd.Timedelta(days=7)
    last_outliers = None
    all_outliers = set()
    dict_weeks = {}
    while start_date<profile.observed_end:
        if start_date>dates[current_breakpoint+1]:
            current_breakpoint+=1
            proportion_breakpoint = get_outliers_proportion(tl,start_date=dates[current_breakpoint], end_date=dates[current_breakpoint+1])
        tl_week = get_rt_tl(tl, start_date=start_date, end_date=end_date)
        week_outliers = []    
        if tl_week is not None and len(tl_week)>0:
            week_outliers = list(tl_week[tl_week.apply(lambda x: x/sum(tl_week))>proportion_breakpoint].index)
#         print(week_outliers, proportion, set(week_outliers))
        new_ones = 0
        if tl_week is not None and len(week_outliers)>0:
            new_ones = len(week_outliers) - len(all_outliers.intersection(set(week_outliers)))
    #         new_ones_proportion = 1-/len(week_outliers)
            all_outliers = all_outliers.union(set(week_outliers))
        if last_outliers is not None and (len(week_outliers)>0 or len(last_outliers)>0):
            jacc = jaccard_similarity(week_outliers, last_outliers)
        else:
            jacc = np.nan
#         print(start_date.day,'/',start_date.month,week_outliers, 'jaccard:', jacc)
        dict_weeks[start_date] = {'jaccard':jacc, 'new_ones':new_ones, 'outliers_len':len(week_outliers), 'outliers':week_outliers, 'num_rt':sum(tl_week), 'num_rters':len(tl_week)}
        last_outliers = week_outliers
        start_date = end_date
        end_date += pd.Timedelta(days=7)
    return dict_weeks

from tqdm import tqdm

all_dict = {}
for i, profile in tqdm(all_profiles_with_network[-all_profiles_with_network.dict_levels.isnull()].iterrows()):
    all_dict[i]=get_dict_week_stability(profile)


all_profiles_with_network.loc[list(all_dict.keys()),'ego_network_evolution']=list(all_dict.values())

all_profiles_with_network['ego_network_evolution'].head()

all_profiles_with_network.to_pickle('../data/all_profiles_with_network.pkl')

all_profiles_with_network = pd.read_pickle('../data/all_profiles_with_network.pkl')

# ## starting date biases

min_obs = pd.Timestamp('2019-09-02 00:00:00+0000', tz='UTC', freq='W-MON')
# max_obs = all_profiles_with_network.observed_start.max()

# last_monday = (max_obs-pd.Timedelta(days=max_obs.weekday()))
min_obs = pd.Timestamp('2019-09-02 00:00:00+0000', tz='UTC', freq='W-MON')
max_obs = pd.Timestamp('2020-05-11 00:00:00+0000', tz='UTC', freq='W-MON')

all_zeros_week = pd.Series({min_obs+pd.Timedelta(days=i*7):0 for i in range(int((max_obs-min_obs).days/7))})

all_zeros_week


def get_week_oct(date):
    if date<all_zeros_week.index[0]+pd.Timedelta(days=7):
        return all_zeros_week.index[0]
    for d in all_zeros_week.index[1:]:
        if date<d+pd.Timedelta(days=7):
            return d


def get_bias(observed_start):
    week_tal = observed_start.apply(get_week_oct)
    weeks_started_freq = week_tal.value_counts()
    weeks_started_freq = pd.concat([weeks_started_freq,all_zeros_week[~all_zeros_week.index.isin(weeks_started_freq)]]).sort_index()
    cumsum = weeks_started_freq.cumsum()
    return cumsum/cumsum.max()



# ## plot evolution

jaccard_df = all_profiles_with_network[-all_profiles_with_network.ego_network_evolution.isnull()].apply(lambda x: pd.DataFrame(x.ego_network_evolution).T['jaccard'], axis=1)

outliers_df = all_profiles_with_network[-all_profiles_with_network.ego_network_evolution.isnull()].apply(lambda x: pd.DataFrame(x.ego_network_evolution).T['outliers'], axis=1)

new_ones_df = all_profiles_with_network[-all_profiles_with_network.ego_network_evolution.isnull()].apply(lambda x: pd.DataFrame(x.ego_network_evolution).T['new_ones'], axis=1)

all_profiles_with_network.type_profile.value_counts()

values_to_plot = {}
for g_id, g in all_profiles_with_network.groupby('type_profile'):
    indexes = [i for i in g.index if i and not pd.isnull(i) and i in jaccard_df.index]
    bias_twitter = get_bias(g.observed_start)
    values_to_plot[g_id]={
        'jaccard':jaccard_df.loc[indexes].mean(axis=0),
        'jaccard_var':jaccard_df.loc[indexes].var(axis=0),
#         'outliers':outliers_df.loc[indexes].mean(axis=0)/bias_twitter_all,
#         'new_outliers':new_ones_df.loc[indexes].mean(axis=0)/bias_twitter_all,
        'bias': bias_twitter
    }

from plotly.subplots import make_subplots
import plotly.graph_objects as go
fig = make_subplots(rows=2, cols=2, shared_xaxes=True)
# keys = list(values_to_plot.keys())
for i,k_v in enumerate(values_to_plot.items()):
    k,v=k_v
    fig.add_trace(go.Bar(x=v['jaccard'].index,y=v['jaccard'].values[1:], name=k),col=i%2+1, row=int(i/2)+1 )
fig.update_layout(title='Jaccard corrected per types', height=400)

utils.plotly_to_tfm(fig, 'networks-jaccard-corrected-types')

from plotly.subplots import make_subplots
import plotly.graph_objects as go
fig = make_subplots(rows=4, cols=1, shared_xaxes=True)
# keys = list(values_to_plot.keys())
for i,k_v in enumerate(values_to_plot.items()):
    k,v=k_v
    fig.add_trace(go.Bar(x=v['jaccard'].index,y=v['outliers'].values[1:], name=k, marker_color=colors[i]),col=1, row=i+1 )
    fig.add_trace(go.Bar(x=v['jaccard'].index,y=v['new_outliers'].values[1:], name=k, marker_color='white', opacity=0.6, showlegend=False),col=1, row=i+1,)
    fig.update_layout(barmode='overlay')
fig.update_layout(title='Number of outliers per types', height=800)

utils.plotly_to_tfm(fig, 'networks-count-outlier-corrected-types')

jaccard_df.columns.max()

jaccard_df_mean = jaccard_df.mean(axis=0)

bias_twitter_all = pd.concat([bias_twitter,jaccard_df_mean[~jaccard_df_mean.index.isin(bias_twitter.index)].apply(lambda x:1)]).sort_index()

import plotly.graph_objects as go
def plot_series(series, title, name=None):
    fig = go.Figure([go.Bar(x=series.index, y=series.values, name=name)])
    fig.update_layout(title=title)
    return fig


series = jaccard_df.mean(axis=0)
fig = plot_series(series, 'Jaccard corrected')

utils.plotly_to_tfm(fig, 'networks-jaccard-corrected-total')

# (outliers_df.mean(axis=0)/bias_twitter_all).plot()
series = outliers_df.mean(axis=0)/bias_twitter_all
fig = plot_series(series, 'Number of outliers corrected', name='Total outliers')

series = (new_ones_df.mean(axis=0)/bias_twitter_all)
fig.add_trace(go.Bar(x=series.index, y=series.values, name='New outliers', marker_color='white', opacity=0.6))
fig.update_layout(barmode='overlay')

utils.plotly_to_tfm(fig, 'networks-count-outlier-corrected')

# +
# all_profiles_with_network[all_profiles_with_network.party=='vox_es']

# +

first_monday = 
all_periods = []
for i, d in enumerate(dates):
    print(i, d)
    if i == 0 and d.dayofweek!=0:
# -


profile = all_profiles_with_network[all_profiles_with_network.screen_name=='Santi_ABASCAL'].iloc[0]
profile

profile.observed_start

get_timeline(profile.name).sort_index()

ines_tl = get_timeline(profile.name)
ines_tl[ines_tl.type!='Like'].sort_index()

# +
profile = all_profiles[all_profiles.screen_name=='InesArrimadas'].iloc[0]
# profile = all_profiles[all_profiles.index==257371804].iloc[0]
# profile = all_profiles[has_network&-all_profiles.dict_levels.isnull()].sample().iloc[0]

# for name, profile in all_profiles[has_network&-all_profiles.dict_levels.isnull()].sample(10).iterrows():
all_data_points = []
tl = get_timeline(profile.name)
dates = list(profile.dict_levels.keys())+[profile.observed_end]
last_outliers = []
for i, v in enumerate(dates[1:]):
    print(profile.dict_levels[dates[i]],)
    start_date = dates[i]
    proportion = get_outliers_proportion(tl,start_date=start_date, end_date=v)
    end_date = start_date+pd.Timedelta(days=7)
    all_rt_tl = get_rt_tl(tl, start_date=start_date, end_date=v)
    print( '\n', '---','PERIOD:', i,'from ',start_date.day,'/',start_date.month,'\nlevel: ',profile.dict_levels[dates[i]], '\nproportion:', proportion, '\n num_rt', sum(all_rt_tl), len(all_rt_tl))
    print('outliers:\n',all_rt_tl[all_rt_tl.apply(lambda x: x/sum(all_rt_tl))>proportion].index.values, )
    print('outliers:\n', get_outliers_tl(tl, start_date=start_date, end_date=v ))
    while start_date<v:
        tl_week = get_rt_tl(tl, start_date=start_date, end_date=end_date)
#         if tl_week
        week_outliers = []
        if tl_week is not None and len(tl_week)>0:
            week_outliers = list(tl_week[tl_week.apply(lambda x: x/sum(tl_week))>proportion].index)
        if len(week_outliers)>0 or len(last_outliers)>0:
            jacc = jaccard_similarity(week_outliers, last_outliers)
        else:
            jacc = np.nan
        print(start_date.day,'/',start_date.month,week_outliers, 'jaccard:', jacc)
        last_outliers = week_outliers
        start_date = end_date
        end_date += pd.Timedelta(days=7)
    
#     break
# break
#     import plotly.express as px
#     px.scatter(pd.DataFrame(all_data_points), x=0, y=1).show()
# -

profile = all_profiles[all_profiles.screen_name=='InesArrimadas'].iloc[0]

profile.dict_levels = {pd.Timestamp('2019-07-02 00:00:00+0000', tz='UTC'): 5.92622950819673}

ines = pd.DataFrame(get_dict_week_stability(profile)).T

# +
# ines.head(100)

# +
# ines_period
# -

ines_period = ines[(ines.index>pd.Timestamp('2019-03-11').tz_localize('UTC'))&(ines.index<pd.Timestamp('2020-04-11').tz_localize('UTC'))]
px.bar(ines_period.reset_index(), y='jaccard', x='index')

ines_period = ines[(ines.index>pd.Timestamp('2019-09-11').tz_localize('UTC'))]
ines_period.index.name='time'
fig1 = px.bar(ines_period.reset_index(), y='jaccard', x='time', title='Ines Arrimadas jaccard index')

import utils
utils.plotly_to_tfm(fig1,'networks-ines-1')

ines_period = ines[(ines.index>pd.Timestamp('2019-09-11').tz_localize('UTC'))]
# px.bar(ines_period.reset_index(), y='new_ones', x='index')

ines_tl = get_timeline(profile.name)

dict_types = {}
for i, w in enumerate(ines_period.index[1:]):
    prev_week = ines_period.index[i-1]
    dict_types[w]=ines_tl[(ines_tl.index>prev_week)&(ines_tl.index<w)].type.value_counts()



fig = plot_series(ines_period.outliers_len, 'Ines Arrimadas outliers weekly', name='Total outliers')
fig.add_trace(go.Bar(x=ines_period.index, y=ines_period.new_ones.values, name='New outliers', marker_color='white', opacity=0.45))
fig.update_layout(barmode='overlay')
utils.plotly_to_tfm(fig,'networks-ines-2')

ines_pre.new_ones.mean(),ines_pre.num_rt.mean(), ines_pre.num_rters.mean(), ines_pre.jaccard.mean()

ines_pre.new_ones.mean(),ines_pre.num_rt.mean(), ines_pre.num_rters.mean(), ines_pre.jaccard.mean()

ines_activity = pd.DataFrame(dict_types).T

ines_activity['Text']+=ines_activity['Mention']
ines_activity = ines_activity.drop(columns=['Like', 'Mention'])

ines_activity['other']=ines_activity.sum(axis=1)

ines_activity['other']-=ines_activity['Text']+ines_activity['RT']

# +
# ines_activity.loc[ines_activity[ines_activity.type=='Mention'], 'type']='Text'

# +
# ines_activity = ines_activity.apply(lambda x: x/x.total, axis=1)
# -

ines_activity.head()

df = ines_activity.reset_index().melt(id_vars="index")
df = df[df['variable'].isin(['RT', 'Text', 'other'])]
fig = px.bar(df, y='value', x='index', color='variable')
fig.update_layout()

utils.plotly_to_tfm(fig, 'networks-ines-3')

ines_pre.jaccard.mean()

ines_post = ines[(ines.index>pd.Timestamp('2019-11-11').tz_localize('UTC'))&(ines.index<pd.Timestamp('2020-01-11').tz_localize('UTC'))]

ines_post.new_ones.mean(),ines_post.num_rt.mean(), ines_post.num_rters.mean(), ines_post.jaccard.mean()

ines[(ines.index>pd.Timestamp('2019-11-11').tz_localize('UTC'))&(ines.index<pd.Timestamp('2020-02-11').tz_localize('UTC'))].new_ones.mean()

# +
# 257371804
# 0.027 proportion?

# +
# all_data_points
# -



tl = get_timeline(example_profile.name)

pd.DataFrame(all_data_points)




for 
get_outliers_tl(tl)

del dict_outliers

all_profiles[['outliers_x_level', 'outliers_oct']].iloc[0].apply(print)

with_breaks = all_profiles[(-all_profiles['outliers_x_level'].isnull())&(-all_profiles.outliers_oct.isnull())]

breaks = with_breaks.outliers_x_level.apply(len)

with_breaks[breaks>1][['dict_levels', 'outliers_x_level', 'outliers_oct']].iloc[4].apply(print)

breaks.hist()

dict_tau_slices = []
for i, profile in tqdm(with_breaks[breaks>1].iterrows()):
    items = list(profile.dict_levels.items())
    items += [(profile.observed_end,None)]
    for i,k_v in enumerate(items[:-1]):
        k,v=k_v
        k_1 = items[i+1][0]
        dict_tau_slices.append({'id_u': profile.name,'days':(k_1-k).days, 'level':v, 'outliers':profile.outliers_x_level[k]})

len(dict_tau_slices)

df_taus = pd.DataFrame(dict_tau_slices)

df_taus['len_outliers'] = df_taus.outliers.apply(len)

df_taus['len_outliers'].max()

df_taus.head(100)

import plotly.express as px
px.scatter(df_taus, x='level', y='len_outliers', opacity=0.3)


