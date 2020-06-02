import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15,8)
# %config InlineBackend.figure_format = 'retina'
import numpy as np
import pandas as pd

import pickle
import datetime

PATH_DATA = './data'


def get_timeline_frequency(path):
    timeline = pd.read_pickle(PATH_DATA+'/timelines/'+path).sort_index(ascending=True).reset_index()
    timeline.created_at = timeline.created_at.apply(lambda ts: ts-datetime.timedelta(hours=ts.hour, minutes=ts.minute, seconds=ts.second))
    freq = timeline.created_at.value_counts(sort=False).loc[timeline.created_at.unique()]
    freq = freq[freq.index>pd.to_datetime('2019-09-01 00:00:00+00:00')]
    missing_dates = pd.Series(0, index=[i for i in pd.date_range(freq.index.min(), periods=(freq.index.max()-freq.index.min()).days) if i not in freq.index])
    return pd.concat([freq, missing_dates]).sort_index()


all_profiles = pd.read_pickle(PATH_DATA+'/all_profiles.pkl')

# +
# model_with_freq_10k = fit_and_save_model('{}.pkl'.format(id_to_model), num_burnin_steps=5000, num_results=15000, step_size = 0.2, save=False)
# -

freq = get_timeline_frequency('{}.pkl'.format(241080178))

# ## TFP example bayesian switchpoint

disaster_data = freq.values
# days = np.arange(1851, 1962)
days = np.arange(len(disaster_data))
# plt.plot(days, disaster_data, 'o', markersize=8);
# plt.ylabel('Disaster count')
# plt.xlabel('days')
# plt.title('Mining disaster data set')
# plt.show()
freq.plot()


# +
def disaster_count_model(num_days, disaster_rate_fn):
    disaster_count = tfd.JointDistributionNamed(dict(
        e=tfd.Exponential(rate=1.),
        l=tfd.Exponential(rate=1.),
        s=tfd.Uniform(0., high=num_days),
        d_t=lambda s, l, e: tfd.Independent(
            tfd.Poisson(rate=disaster_rate_fn(np.arange(num_days), s, l, e)),
            reinterpreted_batch_ndims=1)
    ))
    return disaster_count

def disaster_rate_switch(ys, s, l, e):
    return tf.where(ys < s, e, l)

def disaster_rate_sigmoid(ys, s, l, e):
    return e + tf.sigmoid(ys - s) * (l - e)

model_switch = disaster_count_model(len(days), disaster_rate_switch)
model_sigmoid = disaster_count_model(len(days), disaster_rate_sigmoid)


# +
def target_log_prob_fn(model, s, e, l, disaster_data):
    return model.log_prob(s=s, e=e, l=l, d_t=disaster_data)

models = [model_switch, model_sigmoid]
print([target_log_prob_fn(m, 20., 2., 5, disaster_data).numpy() for m in models])  # Somewhat likely result
print([target_log_prob_fn(m, 60., 1., 5., disaster_data).numpy() for m in models])  # Rather unlikely result
print([target_log_prob_fn(m, -10., 1., 1., disaster_data).numpy() for m in models]) # Impossible result
# -

from functools import partial

# +
num_results = 10000
num_burnin_steps = 3000

@tf.function(autograph=False, experimental_compile=True)
def make_chain(target_log_prob_fn):
    kernel = tfp.mcmc.TransformedTransitionKernel(
       inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=target_log_prob_fn,
          step_size=0.05,
          num_leapfrog_steps=3),
       bijector=[
          # The switchpoint is constrained between zero and len(years).
          # Hence we supply a bijector that maps the real numbers (in a
          # differentiable way) to the interval (0;len(yers))
          tfb.Sigmoid(low=0., high=tf.cast(len(days), dtype=tf.float32)),
          # Early and late disaster rate: The exponential distribution is
          # defined on the positive real numbers
          tfb.Softplus(),
          tfb.Softplus(),
      ])
    kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel=kernel,
        num_adaptation_steps=int(0.8*num_burnin_steps))

    states = tfp.mcmc.sample_chain(
      num_results=num_results,
      num_burnin_steps=num_burnin_steps,
      current_state=[
          # The three latent variables
          tf.ones([], name='init_switchpoint'),
          tf.ones([], name='init_early_disaster_rate'),
          tf.ones([], name='init_late_disaster_rate'),
      ],
      trace_fn=lambda cs, kr: kr,
      kernel=kernel)
    return states[0]

switch_samples = [s for s in make_chain(
    lambda *args: target_log_prob_fn(model_switch, *args, disaster_data=disaster_data))]
sigmoid_samples = [s for s in make_chain(
    lambda *args: target_log_prob_fn(model_sigmoid, *args,  disaster_data=disaster_data))]

switchpoint, early_disaster_rate, late_disaster_rate = zip(
    switch_samples, sigmoid_samples)
# -

switch_samples


# +
def _desc(v):
    return '(median: {}; 95%ile CI: $[{}, {}]$)'.format(
      *np.round(np.percentile(v, [50, 2.5, 97.5]), 2))

for t, v in [
    ('Early disaster rate ($e$) posterior samples', early_disaster_rate),
    ('Late disaster rate ($l$) posterior samples', late_disaster_rate),
    ('Switch point ($s$) posterior samples', days[0] + switchpoint),
]:
    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True)
    for (m, i) in (('Switch', 0), ('Sigmoid', 1)):
        a = ax[i]
        a.hist(v[i], bins=50)
        a.axvline(x=np.percentile(v[i], 50), color='k')
        a.axvline(x=np.percentile(v[i], 2.5), color='k', ls='dashed', alpha=.5)
        a.axvline(x=np.percentile(v[i], 97.5), color='k', ls='dashed', alpha=.5)
        a.set_title(m + ' model ' + _desc(v[i]))
    fig.suptitle(t)
    plt.show()
# -

# ## TFP example multiple bayesian switchpoints

# +
import scipy.stats
true_rates = [40, 3, 20, 50]
true_durations = [10, 20, 5, 35]

observed_counts = freq.values.astype(np.float32)

plt.plot(observed_counts)

# +
num_states = 4

initial_state_logits = np.zeros([num_states], dtype=np.float32) # uniform distribution

daily_change_prob = 0.05
transition_probs = daily_change_prob / (num_states-1) * np.ones(
    [num_states, num_states], dtype=np.float32)
np.fill_diagonal(transition_probs,
                 1-daily_change_prob)

print("Initial state logits:\n{}".format(initial_state_logits))
print("Transition matrix:\n{}".format(transition_probs))

# +
# Define variable to represent the unknown log rates.
trainable_log_rates = tf.Variable(
  np.log(np.mean(observed_counts)) + tf.random.normal([num_states]),
  name='log_rates')

hmm = tfd.HiddenMarkovModel(
  initial_distribution=tfd.Categorical(
      logits=initial_state_logits),
  transition_distribution=tfd.Categorical(probs=transition_probs),
  observation_distribution=tfd.Poisson(log_rate=trainable_log_rates),
  num_steps=len(observed_counts))

# +
rate_prior = tfd.LogNormal(5, 5)

def log_prob():
    return (tf.reduce_sum(rate_prior.log_prob(tf.math.exp(trainable_log_rates))) +
         hmm.log_prob(observed_counts))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

@tf.function(autograph=False)
def train_op():
    with tf.GradientTape() as tape:
        neg_log_prob = -log_prob()
        grads = tape.gradient(neg_log_prob, [trainable_log_rates])[0]
        optimizer.apply_gradients([(grads, trainable_log_rates)])
        return neg_log_prob, tf.math.exp(trainable_log_rates)


# -

for step in range(100):
    loss, rates = [t.numpy() for t in train_op()]
    if step % 20 == 0:
        print("step {}: log prob {} rates {}".format(step, -loss, rates))
print("Inferred rates: {}".format(rates))
print("True rates: {}".format(true_rates))

# Runs forward-backward algorithm to compute marginal posteriors.
posterior_dists = hmm.posterior_marginals(observed_counts)
posterior_probs = posterior_dists.probs_parameter().numpy()


# +
def plot_state_posterior(ax, state_posterior_probs, title):
    ln1 = ax.plot(state_posterior_probs, c='blue', lw=3, label='p(state | counts)')
    ax.set_ylim(0., 1.1)
    ax.set_ylabel('posterior probability')
    ax2 = ax.twinx()
    ln2 = ax2.plot(observed_counts, c='black', alpha=0.3, label='observed counts')
    ax2.set_title(title)
    ax2.set_xlabel("time")
    lns = ln1+ln2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=4)
    ax.grid(True, color='white')
    ax2.grid(False)

fig = plt.figure(figsize=(10, 10))
plot_state_posterior(fig.add_subplot(2, 2, 1),
                     posterior_probs[:, 0],
                     title="state 0 (rate {:.2f})".format(rates[0]))
plot_state_posterior(fig.add_subplot(2, 2, 2),
                     posterior_probs[:, 1],
                     title="state 1 (rate {:.2f})".format(rates[1]))
plot_state_posterior(fig.add_subplot(2, 2, 3),
                     posterior_probs[:, 2],
                     title="state 2 (rate {:.2f})".format(rates[2]))
plot_state_posterior(fig.add_subplot(2, 2, 4),
                     posterior_probs[:, 3],
                     title="state 3 (rate {:.2f})".format(rates[3]))
plt.tight_layout()
