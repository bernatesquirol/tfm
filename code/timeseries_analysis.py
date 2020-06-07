import pandas as pd
import os
import pickle
tls = np.array(os.listdir('../data/models'))


import plotly.express as px
import plotly.graph_objects as go


from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# +
import rpy2.robjects.packages as rpackages

# import R's utility package
utils = rpackages.importr('utils')

# select a mirror for R packages
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


freq = pickle.loads(open('../data/models/{}'.format(tls[6]), 'rb').read())['freq']


from statsmodels.tsa.stattools import acf

# %timeit acf(freq, fft=True)

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


# +

plotly_to_tfm(plotly_series(series_to_plot, px.bar, ylabel='Overall mean acf', xlabel='k (period)', height=300))
# -



seasonal_decompose(freq, period=7).plot()
pass

seasonal_decompose(freq, period=7).plot()
pass


def plot_tau(model, fig=None, **kwargs):
    if not fig:
        fig = go.Figure()    
    fig.add_trace(go.Histogram(x=model['tau'], histnorm='probability', name='tau'), **kwargs)
    return fig


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


# +
# dict_all_breakpoints = {}
# for i in tqdm(tls):
#     try:
#         dict_all_breakpoints[i[:-4]]=get_breakpoints_and_levels(i)
#     except: 
#         pass
# -

levels_file = open("./models_levels2.pickle", 'wb') 

pickle.dump(dict_all_breakpoints, levels_file)

# +
# dict_mean = {}
# for i in tqdm(tls):
#     try:
#         dict_mean[i[:-4]]=get_mean(i)
#     except: 
#         pass
# -

mean_file = open("./models_mean.pkl", 'wb') 

pickle.dump(dict_mean, mean_file)

# ## Visualize Breakpoints

# #### Example

id_user = tls[5]

id_user

model = pickle.loads(open('../data/models/{}'.format(id_user), 'rb').read())

freq = model['freq']

fig_xp_1 = plotly_series(freq, px.bar, ylabel='# activities', xlabel='time')['data'][0]
fig_xp_1['showlegend']=True
fig_xp_1['name']='# activities'
fig_xp_1_go = go.Figure(fig_xp_1, {'legend_orientation':"h"})
# plotly_to_tfm(fig_xp_1_go, 'timeseries-example-breakpoints')
# fig_xp_1_go

plotly_series(freq, px.bar, ylabel='# activities', xlabel='time')['data'][0]

formula = robjects.Formula('freq_tweet ~ 1')
env = formula.environment

env['freq_tweet'] = robjects.r['ts'](robjects.FloatVector(freq.values),  start=freq.min())

breakpoints = robjects.r['breakpoints'](formula)
fitted = robjects.r['fitted'](breakpoints, breaks=len(breakpoints[0]))

dict_levels = {freq.index[int(i)]:fitted[int(i)] for i in [0.0]+list(breakpoints[0])}

dict_levels





# graphics = rpackages.importr("graphics")
# graphics.plot(breakpoints)
def get_breakpoints_levels_series(freq, dict_levels):
    relevant_dates = list(dict_levels.keys())+[freq.index.max()]
    levels = pd.Series(np.nan, index=freq.index)
    for i, interval in enumerate(zip(relevant_dates[:-1],relevant_dates[1:])):
        levels.loc[interval[0]:interval[1]]=list(dict_levels.values())[i]
    return levels


levels = get_breakpoints_levels_series(freq, dict_levels)

levels

fig = go.Figure(None,{'legend_orientation':"h"})
fig.add_trace(fig_xp_1)
fig_xp_2 = plotly_series(levels, px.line, ylabel='# activities', xlabel='time')['data'][0]
fig_xp_2['line']['color']='orange'
fig_xp_2['name']='# acts. simplified'
fig_xp_2['showlegend']=True
fig.add_trace(fig_xp_2)
fig.show()

plotly_to_tfm(go.Figure(fig), 'timeseries-example-breakpoints-s')

fig_xp['data'][0]['name']='tal'

go.Line(
    x=[1, 2, 3, 4, 5],
    y=[1, 2, 3, 4, 5],
    name="Positive"
)

fig_xp['data'][0]['legendgroup']=None

fig_xp['data'][0]['showlegend']=True

fig_xp

px.line

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

dict_all_breakpoints = pickle.loads(open("./models_levels2.pickle", 'rb') .read())
dict_mean = pickle.loads(open("./models_mean.pkl", 'rb').read())

# ## Example

dif_df = pd.DataFrame(dif_breakpoints).T

dict_all_breakpoints['1000001503522934784'].keys()

dif_df.iloc[0][-dif_df.iloc[0].isnull()]

breakpoints = pd.Series({k:list(v.items()) for k,v in dict_all_breakpoints.items()})

# seasonality
pd.

# +
# level <- c(rep(0, break_point$breakpoints), rep(1, length(excess_ts) - break_point$breakpoints))

# +
# fitted[25]
# -

freq

# +
# levels
# -

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

np.sum(pvalue<0.05)

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


test_stationarity((freq-freq.shift(1)).dropna(inplace=False))


