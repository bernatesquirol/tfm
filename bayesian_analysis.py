import pandas as pd
import datetime
matplotlib_style = 'fivethirtyeight' #@param ['fivethirtyeight', 'bmh', 'ggplot', 'seaborn', 'default', 'Solarize_Light2', 'classic', 'dark_background', 'seaborn-colorblind', 'seaborn-notebook']
import matplotlib.pyplot as plt; plt.style.use(matplotlib_style)
import numpy as np

# +
warning_status = "ignore" #@param ["ignore", "always", "module", "once", "default", "error"]
import warnings
warnings.filterwarnings(warning_status)
with warnings.catch_warnings():
    warnings.filterwarnings(warning_status, category=DeprecationWarning)
    warnings.filterwarnings(warning_status, category=UserWarning)

import numpy as np
import os
#@markdown This sets the styles of the plotting (default is styled like plots from [FiveThirtyeight.com](https://fivethirtyeight.com/)
matplotlib_style = 'fivethirtyeight' #@param ['fivethirtyeight', 'bmh', 'ggplot', 'seaborn', 'default', 'Solarize_Light2', 'classic', 'dark_background', 'seaborn-colorblind', 'seaborn-notebook']
import matplotlib.pyplot as plt; plt.style.use(matplotlib_style)
import matplotlib.axes as axes;
from matplotlib.patches import Ellipse
# #%matplotlib inline
import seaborn as sns; sns.set_context('notebook')
from IPython.core.pylabtools import figsize
#@markdown This sets the resolution of the plot outputs (`retina` is the highest resolution)
notebook_screen_res = 'retina' #@param ['retina', 'png', 'jpeg', 'svg', 'pdf']
# #%config InlineBackend.figure_format = notebook_screen_res

import tensorflow as tf

import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

class _TFColor(object):
    """Enum of colors used in TF docs."""
    red = '#F15854'
    blue = '#5DA5DA'
    orange = '#FAA43A'
    green = '#60BD68'
    pink = '#F17CB0'
    brown = '#B2912F'
    purple = '#B276B2'
    yellow = '#DECF3F'
    gray = '#4D4D4D'
    def __getitem__(self, i):
        return [
            self.red,
            self.orange,
            self.green,
            self.blue,
            self.pink,
            self.brown,
            self.purple,
            self.yellow,
            self.gray,
        ][i % 9]
TFColor = _TFColor()

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


def get_timeline_frequency(path):
    timeline = pd.read_pickle('data/timelines/'+path).sort_index(ascending=True).reset_index()
    timeline.created_at = timeline.created_at.apply(lambda ts: ts-datetime.timedelta(hours=ts.hour, minutes=ts.minute, seconds=ts.second))
    freq = timeline.created_at.value_counts(sort=False).loc[timeline.created_at.unique()]
    freq = freq[freq.index>pd.to_datetime('2019-09-01 00:00:00+00:00')]
    missing_dates = pd.Series(0, index=[i for i in pd.date_range(freq.index.min(), periods=(freq.index.max()-freq.index.min()).days) if i not in freq.index])
    return pd.concat([freq, missing_dates]).sort_index()


# +
# get_timeline_frequency(i).index

# +
# len(get_timeline_frequency(i))
# -

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


def fit_bipoisson_model(values, num_burnin_steps=5000, num_results=20000, step_size = 0.2):
    count_data = tf.constant(values, dtype=tf.float32)
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
    [lambda_1_samples,lambda_2_samples, posterior_tau], kernel_results = graph_sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=initial_chain_state,
        kernel = kernel)
    tau_samples = tf.floor(posterior_tau * tf.cast(tf.size(count_data),dtype=tf.float32))
    #freq.index
    return { 'tau': np.array([pd.to_datetime(freq.index.values[int(t)]) for t in tau_samples]), 'tau_samples':tau_samples, 'lambda_1':lambda_1_samples, 'lambda_2': lambda_2_samples, 'kernel_results': kernel_results, 'kernel': kernel }


def plot_lambdas(l1, l2):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=l1, histnorm='probability', name='lambda_1'))
    fig.add_trace(go.Histogram(x=l2, histnorm='probability',  name='lambda_2'))
    fig.update_layout(barmode='overlay')
    fig.update_traces(opacity=0.75)
    return fig


def plot_tau(model):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=model['tau'], histnorm='probability', name='tau'))
    return fig


def print_tau(model):
    return pd.Series(model['tau']).value_counts()/len(model['tau'])


# +
# fig = plot_lambdas(model['lambda_1'],model['lambda_2'])
# -

def expected_texts_bipoisson(model, freq):
    model = item['model']
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


def plot_expected(model, freq):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freq.index, y=expected_texts_bipoisson(model, freq), mode='lines', name='# tweets'))
    fig.add_trace(go.Bar(x=freq.index, y=freq.values, name='expected # tweets'))
    return fig
# expected_texts_per_day = tf.zeros(N_,n_count_data.shape[0])


i = os.listdir('data/timelines')[0]
freq = get_timeline_frequency(i)

# +
# [tf.timestamp(a) for a in freq.index]
# -

model = fit_bipoisson_model(freq.values)

plot_expected(item['model'], item['freq'])

missed_some = pd.read_pickle('data/missed_some_may_update.pkl')

models = {}
for i in os.listdir('data/timelines'):
    if i in missed_some.values:
        continue
    try:
        t0 = datetime.datetime.now()
        freq = get_timeline_frequency(i)
        model = fit_bipoisson_model(freq.values)
        t1 = datetime.datetime.now()
        models[i]={'model':model, 'freq':freq, 'performance':(t1-t0).seconds}
    except KeyboardInterrupt:
        print('stopping')
        break
    except:
        print('bad {}'.format(i))


