import pandas as pd

# +
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

  
def evaluate(tensors):
    """Evaluates Tensor or EagerTensor to Numpy `ndarray`s.
    Args:
    tensors: Object of `Tensor` or EagerTensor`s; can be `list`, `tuple`,
        `namedtuple` or combinations thereof.

    Returns:
        ndarrays: Object with same structure as `tensors` except with `Tensor` or
          `EagerTensor`s replaced by Numpy `ndarray`s.
    """
    if tf.executing_eagerly():
        return tf.contrib.framework.nest.pack_sequence_as(
            tensors,
            [t.numpy() if tf.contrib.framework.is_tensor(t) else t
             for t in tf.contrib.framework.nest.flatten(tensors)])
    return sess.run(tensors)


# -

import pickle
import datetime
import numpy as np

import utils

# %load_ext autoreload
# %autoreload 2

import os
import tqdm

with open('./data/models/620553.pkl', 'rb') as f:
    model_file = pickle.loads(f.read())

model, freq = model_file['model'], model_file['freq']


def artificial_timeline(model, freq, n=1):
    min_date = freq.index.min()
    for i in range(n):
        tau_sample = np.random.choice(model['tau'])
        tau_sample_date_count = (tau_sample.tz_localize('UTC')-min_date).days
        values = tf.concat([tfd.Poisson(rate=np.random.choice(model['lambda_1'])).sample(sample_shape=tau_sample_date_count),
               tfd.Poisson(rate= np.random.choice(model['lambda_2'])).sample(sample_shape= (len(freq) - tau_sample_date_count))], axis=0).numpy()
        yield pd.Series(values, index=freq.index)


artificial_timeline(model, freq)



from scipy.stats import ks_2samp
def evaluate_changepoint_model(model, freq, n=5):
    all_tests = []
    for fake_timeline in artificial_timeline(model, freq, n):
        all_tests.append(ks_2samp(freq.values, fake_timeline.values))
    result = np.mean(np.vstack(all_tests), axis=0)
    return {'statistic': result[0], 'pvalue': result[1]}


# %timeit evaluate_changepoint_model(model, freq, 15)

evaluate_changepoint_model(model, freq, 50)

all_results = {}
for i in tqdm.tqdm(os.listdir('./data/models')):
    with open('./data/models/'+i, 'rb') as f:
        model_file = pickle.loads(f.read())
        model, freq = model_file['model'], model_file['freq']
        result = evaluate_changepoint_model(model, freq, 50)
        all_results[i[:-4]]=result
        del model, freq, model_file

evaluate_df = pd.DataFrame(all_results).T

evaluate_df.pvalue.hist()

evaluate_df

evaluate_df.to_pickle('test_ks_models.pkl')

alright_index = evaluate_df[evaluate_df['pvalue']>0.2].index

bad_index = evaluate_df[evaluate_df['pvalue']<=0.2].index


def load_and_plot(i):
    print(i)
    with open('./data/models/'+i+'.pkl', 'rb') as f:
        model_file = pickle.loads(f.read())
        model, freq = model_file['model'], model_file['freq']
        return utils.plot_everything(model, freq, n=4)


all_profiles = pd.read_pickle('data/all_profiles.pkl')
all_profiles = all_profiles[all_profiles.index.astype(str).isin(evaluate_df.index)]

tal = evaluate_df[~evaluate_df.index.isin(all_profiles.index.astype(str))]

tal.index

1002255096854515714 in all_profiles.index

len(all_profiles)

len(evaluate_df)

all_profiles['pvalue']

# 911220797976580096
load_and_plot(np.random.choice(alright_index))

# lambda_2 te 2 probs
load_and_plot('112894609')

all_profiles.loc[2377330968]

# +
# april down 810992804977639424
# -



to_timestamp = np.vectorize(lambda x: (x - datetime.datetime(1970, 1, 1)).total_seconds())
from_timestamp = np.vectorize(lambda x: datetime.datetime.utcfromtimestamp(x))


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


n_count_data = len(freq)
N_ = model['tau'].shape[0]
day_range = tf.range(0,n_count_data,delta=1,dtype = tf.int32)
day_range = tf.expand_dims(day_range,0)

tau_samples_per_day = tf.expand_dims(model['tau_samples'],0)
tau_samples_per_day = tf.transpose(tf.tile(tau_samples_per_day,tf.constant([day_range.shape[1],1])))
tau_samples_per_day = tf.cast(tau_samples_per_day,dtype=tf.int32)
ix_day = day_range < tau_samples_per_day
lambda_1_samples_per_day = tf.expand_dims(model['lambda_1'],0)
lambda_1_samples_per_day = tf.transpose(tf.tile(lambda_1_samples_per_day,tf.constant([day_range.shape[1],1])))
lambda_2_samples_per_day = tf.expand_dims(model['lambda_2'],0)
lambda_2_samples_per_day = tf.transpose(tf.tile(lambda_2_samples_per_day,tf.constant([day_range.shape[1],1])))

((tf.reduce_sum(lambda_1_samples_per_day*tf.cast(ix_day,dtype=tf.float32),axis=0) +
  tf.reduce_sum(lambda_2_samples_per_day*tf.cast(~ix_day,dtype=tf.float32),axis=0))/N_)

lambda_1_samples_per_day

tf.tile(day_range,tf.constant([N_,1]))

day_range

hist, bin_edges = np.histogram(to_timestamp(model['model']['tau']),  bins=10)


hist, from_timestamp(bin_edges)

max(hist/sum(hist))

hist


np.sum(hist)


