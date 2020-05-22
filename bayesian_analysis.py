import pandas as pd
import datetime
matplotlib_style = 'fivethirtyeight' #@param ['fivethirtyeight', 'bmh', 'ggplot', 'seaborn', 'default', 'Solarize_Light2', 'classic', 'dark_background', 'seaborn-colorblind', 'seaborn-notebook']
import matplotlib.pyplot as plt; plt.style.use(matplotlib_style)
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots

import pickle

# +
import numpy as np
import os
# #%matplotlib inline
# import seaborn as sns; sns.set_context('notebook')
from IPython.core.pylabtools import figsize
#@markdown This sets the resolution of the plot outputs (`retina` is the highest resolution)
notebook_screen_res = 'retina' #@param ['retina', 'png', 'jpeg', 'svg', 'pdf']
# #%config InlineBackend.figure_format = notebook_screen_res

import tensorflow as tf

import tensorflow_probability as tfp
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


# # Timelines to perform the model on

# We perform the model on timelines from 01-09-2019, and inputing the null dates to 0. This may cause a problem with the not fetch data

def get_timeline_frequency(path):
    timeline = pd.read_pickle('data/timelines/'+path).sort_index(ascending=True).reset_index()
    timeline.created_at = timeline.created_at.apply(lambda ts: ts-datetime.timedelta(hours=ts.hour, minutes=ts.minute, seconds=ts.second))
    freq = timeline.created_at.value_counts(sort=False).loc[timeline.created_at.unique()]
    freq = freq[freq.index>pd.to_datetime('2019-09-01 00:00:00+00:00')]
    missing_dates = pd.Series(0, index=[i for i in pd.date_range(freq.index.min(), periods=(freq.index.max()-freq.index.min()).days) if i not in freq.index])
    return pd.concat([freq, missing_dates]).sort_index()


# $$ C_i \sim \text{Poisson}(\lambda)  $$
#
#
# $$
# \lambda = 
# \begin{cases}
# \lambda_1  & \text{if } t \lt \tau \cr
# \lambda_2 & \text{if } t \ge \tau
# \end{cases}
# $$
# \begin{align}
# &\lambda_1 \sim \text{Exp}( \alpha ) \\\
# &\lambda_2 \sim \text{Exp}( \alpha )
# \end{align}
#
# \begin{align}
# & \tau \sim \text{DiscreteUniform(1, len(timeline) ) }
# \end{align}

# +
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


def fit_bipoisson_model(freq, num_burnin_steps=5000, num_results=20000, step_size = 0.2):
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


def fit_and_save_model(i):
    t0 = datetime.datetime.now()
    freq = get_timeline_frequency(i)
    model = fit_bipoisson_model(freq)
    t1 = datetime.datetime.now()
    model_with_freq={'model':model, 'freq':freq, 'performance':(t1-t0).seconds}
    with open('data/models/'+i, 'wb') as file:
        pickle.dump(model_with_freq, file, protocol=pickle.HIGHEST_PROTOCOL)
    return model_with_freq


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
    fig.add_trace(go.Bar(x=freq.index, y=freq.values, name='# tweets'), **kwargs)
    return fig
# expected_texts_per_day = tf.zeros(N_,n_count_data.shape[0])


def plot_everything(model, freq, n=1):
    fig = make_subplots(rows=4, cols=1, specs=[[{}],[{}],[{"rowspan":2}],[None]])
    fig.update_layout(
        height=600)
    plot_lambdas(model['lambda_1'],model['lambda_2'], fig=fig, row=1, col=1)
    plot_tau(model, fig=fig, row=2, col=1)
    return plot_expected(model, freq, n=n, fig=fig, row=3, col=1, )


# # Calculate bipoisson for all

missed_some = pd.read_pickle('data/missed_some_may_update.pkl')

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

# # Calculate bipoisson for given username

all_profiles = pd.read_pickle('data/all_profiles.pkl')

ids_missed = missed_some.apply(lambda x: x[:-4]).values

id_to_model = all_profiles[all_profiles.type_profile=='politician'].sample().index[0]

model_with_freq = fit_and_save_model('{}.pkl'.format(id_to_model))

plot_everything(model_with_freq['model'], model_with_freq['freq'], n=7)

# # Global analysis

profiles = all_profiles.loc[[int(i[:-4]) for i in os.listdir('data/models') if int(i[:-4]) in all_profiles.index]].copy()

profiles['model'] = profiles.apply(lambda x: pickle.load(open('data/models/{}.pkl'.format(x.name), "rb")), axis=1)

# ## Total actions

total_actions = profiles['model'].apply(lambda x: x['freq']).sum().reset_index().resample('W-Mon', on='index').sum()[0]
fig = go.Figure()
fig.add_trace(go.Bar(x=total_actions.index, y=total_actions.values))
fig



# ## We have info

info_available = profiles['model'].apply(lambda x: x['freq'].apply(lambda x: 1)).sum().reset_index().resample('W-Mon', on='index').sum()[0]
fig = go.Figure()
fig.add_trace(go.Bar(x=info_available.index, y=info_available.values))
fig


# ## tau

def plot_taus_profiles(profiles, fig=None, name=None, **kwargs):
    taus = profiles['model'].apply(lambda m: get_tau(m['model']))
    taus_sum = taus.sum(axis=0)
    taus_weekly = taus_sum.reset_index().resample('W-Mon', on='index').sum()[0]
    if not fig:
        fig = go.Figure()
    fig.add_trace(go.Bar(x=taus_weekly.index, y=taus_weekly.values, name=name), **kwargs)
    return fig


plot_taus_profiles(profiles)

# #### by type

fig = make_subplots(rows=profiles.type_profile.unique().shape[0], cols=1)
for i, g in enumerate(profiles.groupby('type_profile')):
    id_g, group = g
    plot_taus_profiles(group, fig, name=id_g, row=i+1, col=1, )
fig

# #### by follower_count

fig = make_subplots(rows=4, cols=1)
profiles['followers_count_bins']=pd.qcut(profiles['followers_count'], q=4)
for i, g in enumerate(profiles.groupby('followers_count_bins')):
    id_g, group = g
    plot_taus_profiles(group, fig, name=str(id_g), row=i+1, col=1, )
fig

# #### by number of total tweets

fig = make_subplots(rows=4, cols=1)
profiles['statuses_count_bins']=pd.qcut(profiles['statuses_count'], q=4)
for i, g in enumerate(profiles.groupby('statuses_count_bins')):
    id_g, group = g
    plot_taus_profiles(group, fig, name=str(id_g), row=i+1, col=1, )
fig

# +
# pd.cut(profiles['created_at'], bins=5)
# -

profiles['created_at'] = profiles['created_at'].apply(lambda x: pd.to_datetime(x))

fig = make_subplots(rows=4, cols=1)
profiles['created_at_bins']=pd.qcut(profiles['created_at'], q=4)
for i, g in enumerate(profiles.groupby('created_at_bins')):
    id_g, group = g
    plot_taus_profiles(group, fig, name=str(id_g), row=i+1, col=1)
fig

profiles.columns

# ## lambdas

profiles['model'].apply(lambda m: m['model']['lambda_1'].numpy())

l1 =  profiles['model'].apply(lambda m: m['model']['lambda_1'].numpy())

l2 =  profiles['model'].apply(lambda m: m['model']['lambda_2'].numpy())

min_l1, max_l1 = [l1.apply(lambda m: m.min()).min(), l1.apply(lambda m: m.max()).max()]

min_l2, max_l2 = [l2.apply(lambda m: m.min()).min(), l2.apply(lambda m: m.max()).max()]

# l1 =  profiles['model'].apply(lambda m: m['model']['lambda_1'].numpy())
min_l, max_l = min(min_l1, min_l2), max(max_l1, max_l2)

import math
x_l = list(range(math.floor(min_l), math.ceil(max_l)))

x_l_log = np.logspace(np.log10(0.001+min_l), np.log10(math.ceil(max_l)), 50, endpoint=True)
x_l_log = np.hstack([[min_l],x_l_log])

hist_l1 = l1.apply(lambda m: np.histogram(m, density=True, bins=x_l_log)[0])
df_hist_l1 = pd.DataFrame(hist_l1.to_list(), index=hist_l1.index)

hist_l2 = l2.apply(lambda m: np.histogram(m, density=True, bins=x_l_log)[0])
df_hist_l2 = pd.DataFrame(hist_l2.to_list(), index=hist_l2.index)

# +
# hist_l1.values
# hist_l2
# -

df_hist_l1.shape

# +
# x_l_log = np.logspace(0.01, np.log(200), 200, endpoint=True)
# -

fig = go.Figure()
fig.add_trace(go.Bar(x=x_l,y=df_hist_l1.sum().values, name='lambda_1'))
fig.add_trace(go.Bar(x=x_l,y=df_hist_l2.sum().values, name='lambda_2'))
# fig.update_layout(barmode='overlay')
# fig.update_traces(opacity=0.75)
fig

mean_diff =  profiles['model'].apply(lambda m: np.mean((m['model']['lambda_1'].numpy()-m['model']['lambda_2'].numpy())))

# +
# taus_2.loc[1000016300][pd.Timestamp('2020-05-06')]
# -

taus_and_lambdas_diff = pd.DataFrame(taus.apply(lambda t: [i*mean_diff[t.name] for i in t.values], axis=1).to_list(), index=taus.index, columns=taus.columns)

fig = go.Figure()
fig.add_trace(go.Bar(x=to_plot.index, y=to_plot.values))
fig

to_plot = taus_and_lambdas_diff.sum()

taus.head().apply(lambda t: mean_diff[t.name], axis=1)


