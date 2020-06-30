# -*- coding: utf-8 -*-
import pandas as pd
import datetime
matplotlib_style = 'fivethirtyeight' #@param ['fivethirtyeight', 'bmh', 'ggplot', 'seaborn', 'default', 'Solarize_Light2', 'classic', 'dark_background', 'seaborn-colorblind', 'seaborn-notebook']
import matplotlib.pyplot as plt; plt.style.use(matplotlib_style)
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import pickle
import numpy as np
import os
import tensorflow as tf
import tensorflow_probability as tfp

# %load_ext autoreload
# %autoreload 2

tfd = tfp.distributions
tfb = tfp.bijectors


def session_options(enable_gpu_ram_resizing=True, enable_xla=False):
    """
    Allowing the notebook to make use of GPUs if they're available.

    XLA (Accelerated Linear Algebra) is a domain-specific compiler for linear
    algebra that optimizes TensorFlow computations.
    """
    config = tf.config
    gpu_devices = config.experimental.list_physical_devices('GPU')
    if enable_gpu_ram_resizing:
        for device in gpu_devices:
           tf.config.experimental.set_memory_growth(device, True)
    if enable_xla:
        config.optimizer.set_jit(True)
    return config

session_options(enable_gpu_ram_resizing=True, enable_xla=True)
# -
import pandas as pd
import datetime


def joint_log_prob(count_data, lambda_1, lambda_2, tau):
    tfd = tfp.distributions
 
    alpha = (1. / tf.reduce_mean(count_data))
    rv_lambda_1 = tfd.Exponential(rate=alpha)
    rv_lambda_2 = tfd.Exponential(rate=alpha)
 
    rv_tau = tfd.Uniform()
 
    lambda_ = tf.gather(
         [lambda_1, lambda_2],
         indices=tf.cast(tau * tf.cast(tf.size(count_data), dtype=tf.float32) <= tf.cast(tf.range(tf.size(count_data)), dtype=tf.float32), dtype=tf.int32))
    rv_observation = tfd.Poisson(rate=lambda_)
 
    return (
         rv_lambda_1.log_prob(lambda_1)
         + rv_lambda_2.log_prob(lambda_2)
         + rv_tau.log_prob(tau)
         + tf.reduce_sum(rv_observation.log_prob(count_data))
    )


# Define a closure over our joint_log_prob.
def unnormalized_log_posterior(lambda1, lambda2, tau, count_data):
    return joint_log_prob(count_data, lambda1, lambda2, tau)


# -

from functools import partial 


def fit_model(freq, num_burnin_steps=5000, num_results=20000, step_size = 0.2):
    count_data = tf.constant(freq.values, dtype=tf.float32)
    # wrap the mcmc sampling call in a @tf.function to speed it up
    @tf.function(autograph=False)
    def graph_sample_chain(*args, **kwargs):
        return tfp.mcmc.sample_chain(*args, **kwargs)
    
    # Set the chain's start state.
    initial_chain_state = [
        tf.cast(tf.reduce_mean(count_data), tf.float32) * tf.ones([], dtype=tf.float32, name="init_lambda1"),
        tf.cast(tf.reduce_mean(count_data), tf.float32) * tf.ones([], dtype=tf.float32, name="init_lambda2"),
        0.5 * tf.ones([], dtype=tf.float32, name="init_tau"),
    ]
    # Since HMC operates over unconstrained space, we need to transform the
    # samples so they live in real-space.
    unconstraining_bijectors = [
        tfp.bijectors.Exp(),       # Maps a positive real to R.
        tfp.bijectors.Exp(),       # Maps a positive real to R.
        tfp.bijectors.Sigmoid(),   # Maps [0,1] to R.  
    ]
    kernel = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=partial(unnormalized_log_posterior, count_data=count_data),
            num_leapfrog_steps=2,
            step_size=step_size,
            state_gradients_are_stopped=True),
        bijector=unconstraining_bijectors)
    
    kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel=kernel, num_adaptation_steps=int(num_burnin_steps * 0.8))
    [lambda_1_samples, lambda_2_samples, posterior_tau], kernel_results = graph_sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=initial_chain_state,
        kernel = kernel)
    tau_samples = tf.floor(posterior_tau * tf.cast(tf.size(count_data),dtype=tf.float32))
    #freq.index
    return { 'tau': np.array([pd.to_datetime(freq.index.values[int(t)]) for t in tau_samples]), 'tau_samples':tau_samples, 'lambda_1':lambda_1_samples, 'lambda_2': lambda_2_samples}



# # Timelines to perform the model on

# We perform the model on timelines from 01-09-2019, and inputing the null dates to 0. This may cause a problem with the not fetch data

def get_timeline_frequency(path):
    timeline = pd.read_pickle(PATH_DATA+'/timelines/'+path).sort_index(ascending=True).reset_index()
    timeline.created_at = timeline.created_at.apply(lambda ts: ts-datetime.timedelta(hours=ts.hour, minutes=ts.minute, seconds=ts.second))
    freq = timeline.created_at.value_counts(sort=False).loc[timeline.created_at.unique()]
    freq = freq[freq.index>pd.to_datetime('2019-09-01 00:00:00+00:00')]
    missing_dates = pd.Series(0, index=[i for i in pd.date_range(freq.index.min(), periods=(freq.index.max()-freq.index.min()).days) if i not in freq.index])
    return pd.concat([freq, missing_dates]).sort_index()


def fit_and_save_model(i, num_burnin_steps=5000, num_results=20000, step_size = 0.2, save=True):
    t0 = datetime.datetime.now()
    freq = get_timeline_frequency(i)
    model = fit_bipoisson_model(freq, num_burnin_steps, num_results, step_size)
    t1 = datetime.datetime.now()
    model_with_freq={'model':model, 'freq':freq, 'performance':(t1-t0).seconds}
    if save:
        with open(PATH_DATA+'/models/'+i, 'wb') as file:
            pickle.dump(model_with_freq, file, protocol=pickle.HIGHEST_PROTOCOL)
    return model_with_freq

# +
# we execute fit_and_save_model with all our profiles -> weeks to finish
# -

# # Plot model

def plot_lambdas(l1, l2, fig=None, **kwargs):
    if not fig:        
        fig = go.Figure()
    fig.add_trace(go.Histogram(x=l1, histnorm='probability', name='lambda_1'), **kwargs)
    fig.add_trace(go.Histogram(x=l2, histnorm='probability',  name='lambda_2'), **kwargs)
#     fig.update_layout(barmode='overlay')
#     fig.update_traces(opacity=0.75)
    return fig


def plot_tau(model, fig=None, **kwargs):
    if not fig:
        fig = go.Figure()    
    fig.add_trace(go.Histogram(x=model['tau'], histnorm='probability', name='tau'), **kwargs)
    return fig


def get_tau(model):
    return pd.Series(model['tau']).value_counts()/len(model['tau'])

def print_tau(model):
    print(pd.Series(model['tau']).value_counts()/len(model['tau']))


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
# expected_texts_per_day = tf.zeros(N_,n_count_data.shape[0])


# +
# freq =  model_with_freq_10k['freq']
# -
# # Model test

# +
# breakpoint_series
# -

import pandas as pd
import datetime


def create_artificial_dict(start, end, max_taus=5):
    taus_len = np.random.randint(max_taus)+1
    dates_synth = [0]+list(np.sort(np.random.choice(list(range(1,(end-start).days-1)), taus_len, replace=False)))
    freq_synth = np.random.choice(list(range(1,10)),taus_len+1, replace=False)
    all_time = [start + datetime.timedelta(days=x) for x in range((end-start).days+1)]
    return {all_time[k]: freq_synth[i] for i, k in enumerate(dates_synth)}


def get_breakpoints_levels_series(dict_levels, end):
    relevant_dates = list(dict_levels.keys())+[end]
    start = relevant_dates[0]
    all_time = [start + datetime.timedelta(days=x) for x in range((end-start).days+1)]
    levels = pd.Series(np.nan, index=all_time)
    for i, interval in enumerate(zip(relevant_dates[:-1],relevant_dates[1:])):
        levels.loc[interval[0]:interval[1]]=list(dict_levels.values())[i]
    return levels


import utils


def artificial_timeline(model, freq, n=1):
    min_date = freq.index.min()
    for i in range(n):
        tau_sample = np.random.choice(model['tau'])
        tau_sample_date_count = (tau_sample.tz_localize('UTC')-min_date).days
        values = tf.concat([tfd.Poisson(rate=np.random.choice(model['lambda_1'])).sample(sample_shape=tau_sample_date_count),
               tfd.Poisson(rate= np.random.choice(model['lambda_2'])).sample(sample_shape= (len(freq) - tau_sample_date_count))], axis=0).numpy()
        yield pd.Series(values, index=freq.index)


from scipy.stats import ks_2samp
def evaluate_changepoint_model(model, freq, n=5):
    all_tests = []
    for fake_timeline in artificial_timeline(model, freq, n):
        all_tests.append(ks_2samp(freq.values, fake_timeline.values))
    result = np.mean(np.vstack(all_tests), axis=0)
    return {'statistic': result[0], 'pvalue': result[1]}

def artificial_timeline(model, freq, n=1):
    min_date = freq.index.min()
    for i in range(n):
        tau_sample = np.random.choice(model['tau'])
        tau_sample_date_count = (tau_sample.tz_localize('UTC')-min_date).days
        values = tf.concat([tfd.Poisson(rate=np.random.choice(model['lambda_1'])).sample(sample_shape=tau_sample_date_count),
               tfd.Poisson(rate= np.random.choice(model['lambda_2'])).sample(sample_shape= (len(freq) - tau_sample_date_count))], axis=0).numpy()
        yield pd.Series(values, index=freq.index)


from scipy.stats import ks_2samp
def evaluate_changepoint_model(model, freq, n=5):
    all_tests = []
    for fake_timeline in artificial_timeline(model, freq, n):
        all_tests.append(ks_2samp(freq.values, fake_timeline.values))
    result = np.mean(np.vstack(all_tests), axis=0)
    return {'statistic': result[0], 'pvalue': result[1]}

def artificial_timeline_from_dict_levels(dict_levels, end):
    relevant_dates = list(dict_levels.keys())+[end]
    start = relevant_dates[0]
    all_time = [start + datetime.timedelta(days=x) for x in range((end-start).days)]
    list_all_poissons = []
    levels = list(dict_levels.values())
    zip_dates = zip(relevant_dates[:-1],relevant_dates[1:])
    for i, interval in enumerate(zip_dates):
        poi = tfd.Poisson(rate=float(levels[i])).sample(sample_shape=(interval[1]-interval[0]).days)
        list_all_poissons.append(poi)
    len_lists = [len(l) for l in list_all_poissons]
    values = tf.concat(list_all_poissons, axis=0).numpy()
    return pd.Series(values, index=all_time)

end = pd.Timestamp('2020-05-15 00:00:00+0000', tz='UTC')

dict_1 = {pd.Timestamp('2019-09-12 00:00:00+0000', tz='UTC'):3, pd.Timestamp('2020-03-09 00:00:00+0000', tz='UTC'):4}
poisson_dict_1 = artificial_timeline_from_dict_levels(dict_1, end)
model_1 = fit_bipoisson_model(poisson_dict_1)


dict_2 = {pd.Timestamp('2019-09-12 00:00:00+0000', tz='UTC'):3, pd.Timestamp('2020-03-09 00:00:00+0000', tz='UTC'):7}

poisson_dict_2 = artificial_timeline_from_dict_levels(dict_2, end)

model_2 = fit_bipoisson_model(poisson_dict_2)

dict_3 = {pd.Timestamp('2019-09-12 00:00:00+0000', tz='UTC'):3, pd.Timestamp('2020-01-22 00:00:00+0000', tz='UTC'):7}

poisson_dict_3 = artificial_timeline_from_dict_levels(dict_3, end)

model_3 = fit_bipoisson_model(poisson_dict_3)

dict_4 = {pd.Timestamp('2019-09-12 00:00:00+0000', tz='UTC'):1, pd.Timestamp('2020-01-12 00:00:00+0000', tz='UTC'):5, pd.Timestamp('2020-03-09 00:00:00+0000', tz='UTC'):2}
poisson_dict_4 = artificial_timeline_from_dict_levels(dict_4, end)
model_4 = fit_bipoisson_model(poisson_dict_4)

dict_5 = {pd.Timestamp('2019-09-12 00:00:00+0000', tz='UTC'):1, pd.Timestamp('2019-10-12 00:00:00+0000', tz='UTC'):5, pd.Timestamp('2020-03-09 00:00:00+0000', tz='UTC'):2}
poisson_dict_5 = artificial_timeline_from_dict_levels(dict_5, end)
model_5 = fit_bipoisson_model(poisson_dict_5)

dict_6 = {pd.Timestamp('2019-09-12 00:00:00+0000', tz='UTC'):1, 
          end-pd.Timedelta(days=round(3*(end-pd.Timestamp('2019-09-12 00:00:00+0000', tz='UTC')).days/4)):5, 
          end-pd.Timedelta(days=round(2*(end-pd.Timestamp('2019-09-12 00:00:00+0000', tz='UTC')).days/4)):1,
          end-pd.Timedelta(days=round((end-pd.Timestamp('2019-09-12 00:00:00+0000', tz='UTC')).days/4)):5
                                 }
poisson_dict_6 = artificial_timeline_from_dict_levels(dict_6, end)
model_6 = fit_bipoisson_model(poisson_dict_6)

dict_7 = {pd.Timestamp('2019-09-12 00:00:00+0000', tz='UTC'):1, 
          end-pd.Timedelta(days=round(2*(end-pd.Timestamp('2019-09-12 00:00:00+0000', tz='UTC')).days/3)):5, 
          end-pd.Timedelta(days=round((end-pd.Timestamp('2019-09-12 00:00:00+0000', tz='UTC')).days/3)):10
                                 }
poisson_dict_7 = artificial_timeline_from_dict_levels(dict_7, end)
model_7 = fit_bipoisson_model(poisson_dict_7)

relevant_dates = list(dict_6.keys())+[end]

all_time = [relevant_dates[0] + datetime.timedelta(days=x) for x in range((end-relevant_dates[0]).days)]
len(all_time), all_time[0], all_time[-1]





for m, tl in [[model_1, poisson_dict_1],[model_2, poisson_dict_2],[model_3, poisson_dict_3],[model_4, poisson_dict_4],[model_5, poisson_dict_5],[model_6, poisson_dict_6]]:
    print(evaluate_changepoint_model(m, tl, n=100))

dates = list(zip(relevant_dates[:-1],relevant_dates[1:]))

dates[0][1]-dates[0][0]+dates[1][1]-dates[1][0]+dates[2][1]-dates[2][0]+dates[3][1]-dates[3][0]





pd.Timestamp('2020-05-15 00:00:00+0000', tz='UTC')-pd.Timestamp('2019-09-12 00:00:00+0000', tz='UTC')



end

model_6

import importlib
importlib.reload(utils)


def plot_everything(model, freq, n=1):
    fig = make_subplots(rows=4, cols=1, specs=[[{}],[{}],[{"rowspan":2}],[None]])
    fig.update_layout(
        height=600)
    plot_lambdas(model['lambda_1'],model['lambda_2'], fig=fig, row=1, col=1)
    plot_tau(model, fig=fig, row=2, col=1)
#     plot_expected(model, freq, n=n, fig=fig, row=3, col=1, )    
    return fig


def plotly_to_tfm(fig, name):
    fig.write_html('../tfm-plots/{}.html'.format(name), config={"responsive": True})


fig = utils.plot_everything(model_1, poisson_dict_1)
fig.update_layout(title='Change λ from 3 to 4 at 10th Mar', margin=dict(l=0, r=0, t=40, b=0), height=400)

plotly_to_tfm(fig, 'bayesian-test-3-4')

fig = utils.plot_everything(model_2, poisson_dict_2)
fig.update_layout(title='Change λ from 3 to 7 at 10th Mar',margin=dict(t=40, b=0),height=400)

plotly_to_tfm(fig, 'bayesian-test-3-7')

fig = utils.plot_everything(model_3, poisson_dict_3)
fig.update_layout(title='Change λ from 3 to 7 at 22 Jan',margin=dict(t=40, b=0),height=400)

fig = utils.plot_everything(model_4, poisson_dict_4)
fig.update_layout(title='Change  λ from 1 to 5 at 12 Jan; 5 to 2 at 9 Mar',margin=dict(t=40, b=0),height=400)
plotly_to_tfm(fig, 'bayesian-test-3-7-2')

fig = utils.plot_everything(model_5, poisson_dict_5)
fig.update_layout(title='Change λ from 1 to 5 at 12 Oct; 5 to 2 at 9 Mar',margin=dict(t=40, b=0),height=400)
plotly_to_tfm(fig, 'bayesian-test-3-7-2-bis')

dict_6

fig = utils.plot_everything(model_6, poisson_dict_6)
fig.update_layout(title='Change λ from 1 to 5 at 13th Oct, 14 Apr', margin=dict(l=0, r=0, t=40, b=0), height=400)

plotly_to_tfm(fig, 'bayesian-test-1-5-1-5')

fig = utils.plot_everything(model_7, poisson_dict_7)
fig.update_layout(title='Change λ from 1 to 5 at 13th Oct, 14 Apr', margin=dict(l=0, r=0, t=40, b=0), height=400)


def dict_levels_to_string(dict_levels):
    items = list(dict_levels.items())
    #items[0][1]+'->'+
    return str(items[0][1])+'->'+'->'.join([str(i[0].date())+':'+str(i[1])for i in items[1:]])


df = pd.Series(dict_2, name='value_next').reset_index()
df['index'] = df['index'].apply(lambda x: x.date())

df.rename(columns={'index':'breakpoint'}).set_index('breakpoint')



fig.update_layout()

tfd.Exponential(rate=5).sample(len(breakpoint_series)).numpy()

model

fig.write_html('try_interactive.html')

evaluate_changepoint_model(model, poisson_dict_1)

# # Calculate bipoisson for all

missed_some = pd.read_pickle(PATH_DATA+'/missed_some_may_update.pkl')

# +
# models = {}
# for i in os.listdir('data/timelines'):
#     if i in missed_some.values:
#         continue
#     try:
#         print(i)
#         fit_and_save_model(i)
#     except KeyboardInterrupt:
#         print('stopping')
#         break
#     except:
#         print('bad {}'.format(i))
# -

import pickle
with open('../data/models/100774646.pkl', 'rb') as f:
    model = pickle.loads(f.read())

model['model']

# +
# model
# -

# # Calculate bipoisson for given username

all_profiles = pd.read_pickle('../data/all_profiles_mod.pkl')

ids_missed = missed_some.apply(lambda x: x[:-4]).values

id_to_model = '241080178' # all_profiles[all_profiles.type_profile=='politician'].sample().index[0]

model_with_freq_10k = fit_and_save_model('{}.pkl'.format(id_to_model), num_burnin_steps=5000, num_results=15000, step_size = 0.2, save=False)

plot_everything(model_with_freq_10k['model'], model_with_freq_10k['freq'], n=7)

plot_everything(model_with_freq_10k['model'], model_with_freq_10k['freq'], n=7)

# # Global analysis

all_profiles = pd.read_pickle('../data/all_profiles_mod.pkl')

import os
all_models = os.listdir('../data/models-light')

# +
# sample = np.random.choice(all_models, 1000)
# -

len(all_models)

# +
# sample
# -

profiles = all_profiles.loc[[int(i[:-4]) for i in all_models if int(i[:-4]) in all_profiles.index]].copy()

from tqdm import tqdm

models_light = {}
for i in tqdm(profiles.index):
    models_light[i]=pickle.load(open('../data/models-light/{}.pkl'.format(i), "rb"))

profiles.loc[list(models_light.keys()),'bayesian_model']=list(models_light.values())

profiles.to_pickle('../data/all_profiles_mod_bayes.pkl')

profiles = pd.read_pickle('../data/all_profiles_mod_bayes.pkl')

# +
# profiles.head()
# -

del models_light

profiles_with_model = profiles[-profiles.bayesian_model.isnull()]


# ## Total actions

# +
# total_actions = profiles['bayesian_model'].apply(lambda x: x['freq']).sum().reset_index().resample('W-Mon', on='index').sum()[0]
# fig = go.Figure()
# fig.add_trace(go.Bar(x=total_actions.index, y=total_actions.values))
# fig
# -

def hist_round_dict(hist, dict_all={}): 
    total = sum(hist[0])
    for i, v in enumerate(hist[1][1:]):
        index = int(round(np.mean([v,hist[1][i]])))
        if index not in dict_all:
            dict_all[index]=0
        dict_all[index]+=hist[0][i]/total
    return dict_all


def hist_mode(hist): 
    dict_tal = {np.mean([v,hist[1][i]]):hist[0][i] for i, v in enumerate(hist[1][1:])}
    return list(dict_tal.keys())[np.argmax(list(dict_tal.values()))]


def get_mean(id_u, folder='timelines'):
    if os.path.isfile('../data/{}/{}.pkl'.format(folder, id_u)):
        tls_ex = pd.read_pickle('../data/{}/{}.pkl'.format(folder, id_u))
        tls_ex = tls_ex[tls_ex.index>pd.to_datetime('2019-09-01 00:00:00+00:00')]
        temp = (tls_ex.index.max()-tls_ex.index.min()).days
        if temp>0:
            return len(tls_ex)/(tls_ex.index.max()-tls_ex.index.min()).days


from tqdm import tqdm

dict_mean_all = {}
for i in tqdm(profiles.index):
    mean = get_mean(i)
    if mean:
        dict_mean_all[i]=mean

len(dict_mean_all)


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


dict_sum = {}
for i_g, g in profiles.groupby('type_profile'):
    bias_g = get_bias(profiles[profiles.type_profile==i_g])
    dict_sum[i_g] = {}
    for i_r, r in tqdm(g.bayesian_model.iteritems()):
        if i_r not in dict_mean_all:
            continue
        sum_tau = np.sum(r['tau'][0])
        for i, t in enumerate(r['tau'][1][1:]):
            if t not in dict_sum[i_g]:
                dict_sum[i_g][t]={'value_tau':0, 'lambda_change_neg':0, 'lambda_change_pos':0}
            dict_sum[i_g][t]['value_tau']+=r['tau'][0][i]/sum_tau
            l_change = hist_mode(r['lambda_2'])-hist_mode(r['lambda_1'])
            if l_change>0:
                dict_sum[i_g][t]['lambda_change_pos'] += l_change/(dict_mean_all[i_r]*bias_g[t])
            else:
                dict_sum[i_g][t]['lambda_change_neg'] += l_change/(dict_mean_all[i_r]*bias_g[t])


dict_mean_all[i_r]

dict_len = profiles.type_profile.value_counts()

dict_dfs = {i:pd.DataFrame(list(v.values()),index=list(v.keys())) for i,v in dict_sum.items()}

from plotly.subplots import make_subplots
import plotly.graph_objects as go
fig = make_subplots(rows=4, cols=1, subplot_titles=list(dict_dfs.keys()), shared_xaxes=True)
for i, k_v in enumerate(dict_dfs.items()):
    k, df = k_v    
    df['value_tau_normalized']=df['value_tau']/dict_len[k]
    df['lambda_change_pos_tau']=df['lambda_change_pos']*df['value_tau_normalized']
    df['lambda_change_neg_tau']=df['lambda_change_neg']*df['value_tau_normalized']
    fig.add_trace(go.Bar(x=df.index, y=df['lambda_change_pos_tau'], name="+", marker_color='#2ca02c' ),col=1, row=i+1)
    fig.add_trace(go.Bar(x=df.index, y=df['lambda_change_neg_tau'],name="-", marker_color='#d62728'),col=1, row=i+1)
val_y = dict(
        autorange=True,
        showgrid=False,
        ticks='',
        showticklabels=False
    )
fig.update_layout(title="P(τ) * (λ2 - λ1) +-", height=800,
    yaxis=val_y, yaxis2=val_y,yaxis3=val_y, yaxis4=val_y)


# +
# fig
# -

def plotly_to_tfm(fig, name):
    fig.write_html('../tfm-plots/{}.html'.format(name), config={"responsive": True})


plotly_to_tfm(fig, 'bayesian-type-changes-2.html')

all_db = pd.concat(dict_dfs.values()).reset_index().groupby('index').sum()

fig = go.Figure()
all_db['value_tau_normalized']=all_db['value_tau']/dict_len[k]
all_db['lambda_change_pos_tau']=all_db['lambda_change_pos']*df['value_tau_normalized']
all_db['lambda_change_neg_tau']=all_db['lambda_change_neg']*df['value_tau_normalized']
fig.add_trace(go.Bar(x=all_db.index, y=all_db['lambda_change_pos_tau'], name="+", marker_color='#2ca02c' ))
fig.add_trace(go.Bar(x=all_db.index, y=all_db['lambda_change_neg_tau'],name="-", marker_color='#d62728'))
fig.update_layout(title="P(τ) * (λ2 - λ1) total", height=500, )

plotly_to_tfm(fig, 'bayesian-global-changes-2.html')
