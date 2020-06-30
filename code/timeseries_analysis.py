import pandas as pd
import os
import pickle
import numpy as np
tls = np.array(os.listdir('../data/models'))


import plotly.express as px
import plotly.graph_objects as go


from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# +
import rpy2.robjects.packages as rpackages

# import R's utility package
utils = rpackages.importr('utils')

# select a mirror for R packagesauto.arima
utils.chooseCRANmirror(ind=1)
# -

rpackages.importr('strucchange')

# +
packnames = ('ggplot2', 'hexbin')

# R vector of strings
from rpy2.robjects.vectors import StrVector
from rpy2 import robjects
# Selectively install what needs to be install.
# We are fancy, just because we can.
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))


# -
# ## Seasonality

def load_freq(user_id):
    return pickle.loads(open('../data/models/{}'.format(user_id), 'rb').read())['freq']


freq = pickle.loads(open('../data/models/{}'.format('1000092194961838080.pkl'), 'rb').read())['freq']

from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from tqdm import tqdm

dict_decompose = {}
for tl in tqdm(tls):
    freq = pickle.loads(open('../data/models/{}'.format(tl), 'rb').read())['freq']
    if len(freq)>=14:
        dict_decompose[tl]=seasonal_decompose(freq, period=7)

pickle.dump(dict_decompose, open('../data/temporality_tls.pickle', 'wb'))

import pandas as pd

plotly_series(pd.Series(acf(freq)), px.bar, ylabel='Overall mean acf', xlabel='k (period)', height=300,)

plotly_series(pd.Series(dict_decompose['1000092194961838080.pkl'].seasonal), px.bar, ylabel='Overall mean acf', xlabel='k (period)', height=300,)

ts = dict_decompose['1000092194961838080.pkl'].seasonal.index[0]


dict_real_df = pd.DataFrame.from_dict(dict_real)

dict_real_df.mean(axis=1).plot()



selection = dict_real_df[dict_real_df.index<pd.Timestamp('2020-03-01').tz_localize('UTC')].copy()

selection['day_of_week'] = selection.apply(lambda x:x.name.dayofweek, axis=1)


mean = selection.groupby(by=['day_of_week']).mean()

mean.to_pickle('day_of_week_plot.pkl')

from sklearn.preprocessing import MinMaxScaler

mean = pd.read_pickle('day_of_week_plot.pkl')

mean.apply(lambda x: list(x.values))

mean_series = pd.Series([list(a) for a in mean.T.values], index=mean.columns)

mean_scaled = MinMaxScaler().fit_transform(mean.values)

mean_scaled

mean.values

mean_scaled_series = pd.Series([list(a) for a in mean_scaled.T], index=mean.columns)

mean_scaled_series[mean_scaled_series.index.isnull()]

all_profiles_mod = pd.read_pickle('../data/all_profiles_mod.pkl')

mean_scaled_series.index[1465]

mean_series_f = mean_series[mean_series.index.isin(all_profiles_mod.index)]
mean_scaled_series_f = mean_scaled_series[mean_scaled_series.index.isin(all_profiles_mod.index)]

all_profiles_mod.loc[mean_series_f.index, 'mean_week_activity']=mean_series_f

all_profiles_mod.to_pickle('../data/all_profiles_mod.pkl')

dict_gs_1 = {}
for g_id, g in all_profiles_mod.groupby('type_profile'):
    values = g.sample(5000, replace=True)['mean_week_activity'].dropna().apply(np.array).values
    values = np.concatenate(values).reshape((values.shape[0],7))
    normalized = MinMaxScaler().fit_transform(values.T).T
    dict_gs_1[g_id] = np.nanmean(normalized, axis=0)

from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ### Analysis

# +
from plotly.subplots import make_subplots
import plotly.graph_objects as go
min_v , max_v=np.array(list(dict_gs.values())).min()-0.05, np.array(list(dict_gs.values())).max()+0.05
# fig = make_subplots(rows=4, cols=1,  shared_xaxes=True, )
F
for i,k_v in enumerate(dict_gs_1.items()):
    k,v=k_v
    fig.add_trace(go.Scatter(
        x=['Monday','Tuesday','Wednesday', 'Thursday', 'Friday','Saturday','Sunday'],
        y=v,
        name=k
    ))

fig.update_layout(height=400, title_text="Weekly activities normalized")
fig.show()
# -

plotly_to_tfm(fig, 'timeseries-seasonal-type')



pickle.dump(dict_real, open('../data/activity_week.pickle', 'wb'))

v.index[0]

dict_real = {int(k[:-4]):v.seasonal[[]:7] for k,v in dict_decompose.items()}

del dict_decompose

dict_real = {}
for k,v in dict_decompose.items():
    first_day_week = v.seasonal.index[0].dayofweek
    padding = 7-first_day_week
    dict_real[int(k[:-4])]=v.seasonal[padding:padding+7]

pickle

pd.Series(dict_decompose['1000092194961838080.pkl'].seasonal)[:7]

pd.Series(dict_decompose['1000092194961838080.pkl'].seasonal)[7:14]

values = list(dict_decompose.values())

from scipy import stats
stats.shapiro(values[2126].observed.dropna())

# +
# acf(tls[1])
# acorr_ljungbox(freq, period=7, return_df=True)
# -

# %timeit acf(freq)

acf(freq, fft=True)

choosen = np.random.choice(tls, 700)

acfs = [acf(load_freq(i), fft=True) for i in ]

acfs_df = pd.DataFrame(acfs)

acfs_df.head()


def get_max_index(x):
    return [i for i, v in enumerate(x[1:-1]) if x[i]<x[i+1] and x[i+1]>x[i+2]]


list_p = acfs_df.apply(lambda x: [i+1 for i, v in enumerate(x[1:-1]) if x[i]<x[i+1] and x[i+1]>x[i+2]], axis=1) #[i for i in enumerate(x[1:-1]) if ] # if x[i-1]<x[i] and x[i]>x[i+1]

# plt.hist(np.hstack(list_p), bins=30)
week_seasonality = list_p.apply(lambda p: np.any([(i)%7==0 for i in p]))

# +
# list_p.head(50)
# -



week_seasonality.value_counts()

vals = acfs_df[-week_seasonality].sample(1).values

np.any([(i+1)%7==0 for i in vals[0]])



# +
# plt.plot(vals[0])

# +
# get_max_index(series_to_plot)

# +
# series_to_plot.index.name
# -

def plotly_series(series, plotly_func, ylabel=None, xlabel=None, **kwargs):
    if not ylabel:
        ylabel = 'to_plot'
    index= 'index'
    df = pd.DataFrame(series.values, index=series.index, columns=[ylabel]).reset_index()
    if xlabel:
        index=xlabel
        df = df.rename(columns={'index':index})
    return plotly_func(df, y=ylabel, x=index, **kwargs)


series_to_plot = (acfs_df.sum()/len(acfs_df))


def plotly_to_tfm(fig, name):
    fig.write_html('../tfm-plots/{}.html'.format(name), config={"responsive": True})


plot_acf = lambda series_to_plot: plotly_series(series_to_plot, px.bar, ylabel='Overall mean acf', xlabel='k (period)', height=300)

# +

plotly_to_tfm(plotly_series(series_to_plot, px.bar, ylabel='Overall mean acf', xlabel='k (period)', height=300))
# -



seasonal_decompose(freq, period=7).plot()
pass

seasonal_decompose(freq, period=7).plot()
pass

# ## Calculate Breakpoints

import pickle

from statsmodels.tools.eval_measures import bic

tls[5]


rpackages.importr('strucchange')

# +
# rpackages.importr('forecast')
# -

from tqdm import tqdm


def get_mean(id_user):
    model = pickle.loads(open('../data/models/{}'.format(id_user), 'rb').read())
    freq = model['freq']
    return freq.mean()


def get_breakpoints_and_levels(id_user):
    model = pickle.loads(open('../data/models/{}'.format(id_user), 'rb').read())
    freq = model['freq']
    formula = robjects.Formula('freq_tweet ~ 1')
    env = formula.environment
    env['freq_tweet'] = robjects.r['ts'](robjects.FloatVector(freq.values),  start=freq.min())
    breakpoints = robjects.r['breakpoints'](formula)
    fitted = robjects.r['fitted'](breakpoints, breaks=len(breakpoints[0]))
    return {freq.index[int(i)]:fitted[int(i)] for i in [0.0]+list(breakpoints[0])}


dict_all_breakpoints = {}
for i in tqdm(tls):
    try:
        dict_all_breakpoints[i[:-4]]=get_breakpoints_and_levels(i)
    except: 
        pass

len(dict_all_breakpoints)

levels_file = open("../data/models_levels.pickle", 'wb') 

pickle.dump(dict_all_breakpoints, levels_file)

dict_all_breakpoints_2 = pickle.load(open("../data/models_levels.pickle", 'rb'))

len(dict_all_breakpoints_2)

dict_mean = {}
for i in tqdm(tls):
    try:
        dict_mean[i[:-4]]=get_mean(i)
    except: 
        pass

mean_file = open("./models_mean.pkl", 'wb') 

pickle.dump(dict_mean, mean_file)

# ## Visualize Breakpoints

# #### Example

id_user = tls[5]

id_user

model = pickle.loads(open('../data/models/{}'.format(id_user), 'rb').read())

freq = model['freq']

plotly_to_tfm(plot_acf(pd.Series(acf(freq))), 'timeseries-example-acf')

fig_xp_1 = plotly_series(freq, px.bar, ylabel='# activities', xlabel='time')['data'][0]
fig_xp_1['showlegend']=True
fig_xp_1['name']='# activities'
fig_xp_1_go = go.Figure(fig_xp_1, {'legend_orientation':"h"})
# plotly_to_tfm(fig_xp_1_go, 'timeseries-example-breakpoints')
# fig_xp_1_go

formula = robjects.Formula('freq_tweet ~ 1')
env = formula.environment

env['freq_tweet'] = robjects.r['ts'](robjects.FloatVector(freq.values),  start=freq.min())

env = formula.environment
env['freq_tweet'] = robjects.r['ts'](robjects.FloatVector(freq.values),  start=freq.min())
breakpoints = robjects.r['breakpoints'](formula)

dict_levels = {freq.index[int(i)]:fitted[int(i)] for i in [0.0]+list(breakpoints[0])}

dict_levels

items = list(dict_levels.items())
for i, k_v in enumerate(items):
    if (i==len(dict_levels)-1):
        break
    k,v=k_v    
    k_1, v_1 = items[i+1]
    print(k_1.date(), (v_1-v)/freq.mean())

items = list(dict_levels.items())
for i, k_v in enumerate(items):
    if (i==len(dict_levels)-1):
        break
    k,v=k_v    
    k_1, v_1 = items[i+1]
    print(k_1.date(), (v_1-v))


# graphics = rpackages.importr("graphics")
# graphics.plot(breakpoints)
def get_breakpoints_levels_series(freq, dict_levels):
    relevant_dates = list(dict_levels.keys())+[freq.index.max()]
    levels = pd.Series(np.nan, index=freq.index)
    for i, interval in enumerate(zip(relevant_dates[:-1],relevant_dates[1:])):
        levels.loc[interval[0]:interval[1]]=list(dict_levels.values())[i]
    return levels


levels = get_breakpoints_levels_series(freq, dict_levels)

import pandas as pd
all_profiles_mod = pd.read_pickle('../data/all_profiles_mod.pkl')

all_profiles_mod['dict_levels_len']=all_profiles_mod['dict_levels'].dropna().apply(len)

from plotly.subplots import make_subplots
import plotly.graph_objects as go
fig = make_subplots(rows=2, cols=2, shared_xaxes=True)
for i, g in enumerate(all_profiles_mod.groupby('type_profile')):
    g_id, g = g
    fig.add_trace(go.Histogram(x=g['dict_levels_len'],histnorm='probability density', name=g_id),col=i%2+1, row=int(i/2)+1 )
fig.update_layout(title='Distribution of number of breakpoints', height=350)

all_profiles_mod[all_profiles_mod['observed_start']<pd.Timestamp('2019-11-01').tz_localize('utc')]

plotly_to_tfm(fig, 'timeseries-breakpoints-analysis')


def get_mean_from_dict_levels(dict_levels, end):
    start = list(dict_levels.keys())[0]
    count = 0
    last_k = start
    items = list(dict_levels.items())+[[pd.Timestamp(end),'dummy']]
    for i,k_v in enumerate(items[:-1]):
        print(k_v)
        k,v=k_v
        print(k,v, items[i+1][0]-items[i][0])
        count+=(items[i+1][0]-items[i][0]).days*v
    return count/(end-start).days


get_mean_from_dict_levels(all_profiles_mod['dict_levels'].iloc[0], all_profiles_mod['observed_end'].iloc[0])

all_profiles_mod.apply(lambda x:x.dict_levels, axis=1)

fig = go.Figure(None,{'legend_orientation':"h"})
fig.add_trace(fig_xp_1)
fig_xp_2 = plotly_series(levels, px.line, ylabel='# activities', xlabel='time')['data'][0]
fig_xp_2['line']['color']='orange'
fig_xp_2['name']='# acts. simplified'
fig_xp_2['showlegend']=True
fig.add_trace(fig_xp_2)
fig.show()

plotly_to_tfm(go.Figure(fig), 'timeseries-example-breakpoints-s')

import IPython.display
def plot(rObjs):
    if type(rObjs)!=list:
        rObjs=[rObjs]
    graphics = rpackages.importr("graphics")
    from rpy2.robjects.lib import grdevices
    with robjects.lib.grdevices.render_to_bytesio(grdevices.png, width=1024, height=896, res=150) as img:
        for rObj in rObjs:
            key, arg, kwargs = rObj['key'], rObj['arg'], rObj['kwargs'] if 'kwargs' in rObj else {}
            if key=='plot':
                graphics.plot(arg, **kwargs)
            elif key=='lines':
                graphics.lines(arg, **kwargs)
    IPython.display.display(IPython.display.Image(data=img.getvalue(), format='png', embed=True))


plot({'key':'plot', 'arg':breakpoints})

objsPlot = [{'key': 'plot', 'arg': env['freq_tweet'], 'kwargs':{'ylab':'freq'}},
           {'key': 'lines', 'arg': fitted, 'kwargs':{'col':4}}]

import pickle
dict_all_breakpoints = pickle.loads(open("../data/models_levels.pickle", 'rb') .read())
dict_mean = pickle.loads(open("../data/models_mean.pickle", 'rb').read())

all_profiles_mod.type_profile

# ## Example

df_breakpoints = pd.DataFrame(columns=['value'], index=pd.MultiIndex.from_product([[],[]],names=['u','d']))

for user,dict_levels in dict_all_breakpoints.items():
#     dict_all_breakpoints_mean[user] = {}
    items = list(dict_levels.items())
    for i, k_v in enumerate(items):
        if (i==len(dict_levels)-1):
            break
        k,v=k_v    
        k_1, v_1 = items[i+1]
        df_breakpoints.loc[(user, k_1.date()), 'value'] =(v_1-v)/dict_mean[user]

df_breakpoints_u = df_breakpoints.reset_index().set_index('u')

all_profiles_mod['dict_levels_len']=all_profiles_mod['dict_levels'].apply(lambda x: len(x)-1 if not pd.isnull(x) else x)

all_profiles_mod[['type_profile', 'dict_levels_len']].join(df_breakpoints_u, how='inner')
all_profiles_mod.index = all_profiles_mod.index.astype(str)

join_df = all_profiles_mod[['type_profile', 'dict_levels_len']].join(df_breakpoints_u, how='inner')

join_df['positive']=join_df['value']>0


def get_bias(g):
    starting = g[-g.dict_levels.isnull()].dict_levels.apply(lambda x: list(x.keys())[0])
    observed_end_max = g[-g.observed_end.isnull()].observed_end.max()
    i = starting.min()
    all_dates = [i]
    while i < observed_end_max:
        i+=pd.Timedelta(days=1)
        all_dates.append(i)
    all_dates_series = pd.Series(0,index=all_dates)
    starting_values = starting.value_counts()
    all_dates_series.loc[starting_values.index]=starting_values.values    
    cumsum_dates = all_dates_series.sort_index().cumsum()
    return cumsum_dates/cumsum_dates.min()


g = all_profiles_mod[(-all_profiles_mod.dict_levels_len.isnull()) & (all_profiles_mod.dict_levels_len>0)]









join_df

import datetime
dict_g = {}
for i,g in enumerate(join_df.groupby(by=['type_profile', 'positive'])):
    g_id, g = g
    groupby_obj = g.groupby('d')
    values_overall = groupby_obj.value.sum()
    values_overall.index.name = 'index'
    for x in range((values_overall.index.max()-values_overall.index.min()).days):
        date_try = values_overall.index.min() + datetime.timedelta(days=x)
        if date_try not in values_overall.index:
            values_overall.loc[date_try]=0
    if not g_id[0] in dict_g:
        dict_g[g_id[0]]=[]
    index_sorted = values_overall.sort_index()
    bias_g = get_bias(all_profiles_mod[all_profiles_mod.type_profile==g_id[0]])
    unbiased_series = pd.Series(index_sorted.reset_index().apply(lambda x: x[1]/bias_g[x[0]], axis=1).values, index=index_sorted.index)
    dict_g[g_id[0]].append(unbiased_series)

# +
from plotly.subplots import make_subplots
import plotly.graph_objects as go
fig = make_subplots(rows=4, cols=1, subplot_titles=list(dict_g.keys()), shared_xaxes=True)
for i, k_v in enumerate(dict_g.items()):
    k, v = k_v
    v0, v1 = v
    fig.add_trace(go.Bar(x=v1.index, y=v1.values, name="+", marker_color='#2ca02c' ),col=1, row=i+1)
    fig.add_trace(go.Bar(x=v0.index, y=v0.values,name="-", marker_color='#d62728'),col=1, row=i+1)
    
fig.update_layout(title="Overall J' per user-type", height=800)
# -

import utils
utils.plotly_to_tfm(fig, 'timeseries-sum-js-types-2')

import datetime
dict_g = {}
for i,g in enumerate(join_df.groupby(by=['positive'])):
    g_id, g = g
    values_overall = g.groupby('d').value.sum()
    values_overall.index.name = 'index'
    for x in range((values_overall.index.max()-values_overall.index.min()).days):
        date_try = values_overall.index.min() + datetime.timedelta(days=x)
        if date_try not in values_overall.index:
            values_overall.loc[date_try]=0
    index_sorted = values_overall.sort_index()
    bias_g = get_bias(all_profiles_mod)
    unbiased_series = pd.Series(index_sorted.reset_index().apply(lambda x: x[1]/bias_g[x[0]], axis=1).values, index=index_sorted.index)    
    dict_g[g_id]= unbiased_series

from plotly.subplots import make_subplots
import plotly.graph_objects as go
# fig = make_subplots(rows=2, cols=1)
fig = go.Figure(None,{'legend_orientation':"h"})
# for i, k_v in enumerate(dict_g.items()):
v0, v1 = dict_g.values()
fig.add_trace(go.Bar(x=v1.index, y=v1.values, name="Positive jumps (J')", marker_color='#2ca02c'))
fig.add_trace(go.Bar(x=v0.index, y=v0.values, name="Negative jumps (J')", marker_color='#d62728'))
fig.update_layout(title="Overall mean J' through time", height=500)

utils.plotly_to_tfm(fig, 'timeseries-sum-js-2')

colors = ['red']

df_breakpoints
df_breakpoints['positive']=df_breakpoints['value']>0

values_overall = df_breakpoints.groupby().reset_index().groupby('d').sum().value
# values_overall.index.name = 'index'



plotly_to_tfm(plotly_series(values_overall.sort_index(), px.bar, ylabel="sum of all J'", xlabel='time'),'timeseries-sum-js')

values_overall.index.min()

dif_df.iloc[0][-dif_df.iloc[0].isnull()]

breakpoints = pd.Series({k:list(v.items()) for k,v in dict_all_breakpoints.items()})

# ## arima

# level <- c(rep(0, break_point$breakpoints), rep(1, length(excess_ts) - break_point$breakpoints))
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
pandas2ri.activate()

robjects.Formula('adasd ~ 1')

formula = robjects.Formula('freq_tweet ~ 1')
env = formula.environment
env['freq_tweet'] = robjects.r['ts'](robjects.FloatVector(freq.values),  start=freq.min())
breakpoints = robjects.r['breakpoints'](formula)
fitted = robjects.r['fitted'](breakpoints, breaks=len(breakpoints[0]))

ro.globalenv['freq_tweet'] = env['freq_tweet'] 

rpackages.importr("forecast")

rpackages.importr("graphics")

graphics = rpackages.importr("graphics")

# +
# graphics.plot
# -

model_1 = robjects.r['auto.arima'](env['freq_tweet'], trace=True)

b_1 = robjects.r['LjungBoxTest'](robjects.r['residuals'](model_1), k = 1)

rpackages.importr('FitARMA')

robjects.r['checkresiduals'](model_1)


def get_dict_r(list_r):
    return dict(zip(list_r.names,list(list_r)))


model_2_1 = robjects.r['Arima'](env['freq_tweet'], 
                    order = robjects.IntVector([0,1,2]), 
                    seasonal =  robjects.r['list'](order = robjects.IntVector([0,0,1]), period = 7), 
                    )

b_2_1 = robjects.r['LjungBoxTest'](robjects.r['residuals'](model_2_1), k=1)

print(b_2_1)

print(get_dict_r(model_2)['call'])

model_2_2 = robjects.r['Arima'](env['freq_tweet'], 
                    order = robjects.IntVector([0,1,2]), 
                    seasonal =  robjects.r['list'](order = robjects.IntVector([1,0,0]), period = 7), 
                    )

b_2_2 = robjects.r['LjungBoxTest'](robjects.r['residuals'](model_2_2))

print(b_2_2)

robjects.r['checkresiduals'](model_2_1)

robjects.r['checkresiduals'](model_2_2)

model_3_1 = robjects.r['auto.arima'](env['freq_tweet'], trace=True, xreg=fitted, )

b_3_1= robjects.r['LjungBoxTest'](robjects.r['residuals'](model_3_1))

all_bs = ['','','','','',]

all_bs[0] = str(b_1)
all_bs[1] = str(b_2_1)
all_bs[2] = str(b_2_2)
all_bs[3] = str(b_3_1)
all_bs[4] = str(b_3_2)

bs_filtered = []
for tal in all_bs:
    new_b = []
    for num_spaces in range(1,10):
        tal = tal.replace(''.join([" " for i in range(num_spaces)]), " ")
    splittal = [l.split(' ')[-1] for l in tal.split('\n\r')][1:]
    print(tal)
    print(splittal)
    for i in splittal:
        new_b.append(float(i))
    bs_filtered.append(new_b)

for num_spaces in range(1,10):
    tal = tal.replace(''.join([" " for i in range(num_spaces)]), " ")

bs_filtered

bs_pd = pd.DataFrame(bs_filtered).T
bs_pd.index = [i for i in bs_pd.index]

bs_pd.loc[list(range(1,11))]

model_3_2 =  robjects.r['Arima'](
    env['freq_tweet'], 
    order = robjects.IntVector([0,0,1]), 
    xreg=fitted,
    seasonal =  robjects.r['list'](order = robjects.IntVector([1,0,0]), period = 7),
)


b_3_2= robjects.r['LjungBoxTest'](robjects.r['residuals'](model_3_2))

str(b_3_2)

plot(objsPlot)

import pmdarima as pm


len(freq.values)





arima1.to_dict()

arima2.to_dict()

# +


#     result = robjects.r['plot'](break_points)

# from IPython.display import Image, display
# data = b.getvalue()
# display(Image(data=data, format='png', embed=True))

# -

pip install pmdarima

from statsmodels.tsa.statespace.sarimax import SARIMAX

list(fitted)

sarimax = SARIMAX(freq.values, exog=list(fitted))

sarimax_result = sarimax.fit()

_, pvalue = sarimax_result.test_serial_correlation('ljungbox')[0]

import pandas as pd

from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):

    #Determing rolling statistics
    rolmean = timeseries.rolling(window=7).mean()
    rolstd = timeseries.rolling(window=7).std()

    #Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)


test_stationarity(freq)


