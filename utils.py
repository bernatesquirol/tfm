import pandas as pd
import datetime
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import tensorflow as tf


def plot_tau(model, fig=None, **kwargs):
    if not fig:
        fig = go.Figure()    
    fig.add_trace(go.Histogram(x=model['tau'], histnorm='probability', name='tau'), **kwargs)
    return fig


def plot_lambdas(l1, l2, fig=None, **kwargs):
    if not fig:        
        fig = go.Figure()
    fig.add_trace(go.Histogram(x=l1, histnorm='probability', name='lambda_1'), **kwargs)
    fig.add_trace(go.Histogram(x=l2, histnorm='probability',  name='lambda_2'), **kwargs)
#     fig.update_layout(barmode='overlay')
#     fig.update_traces(opacity=0.75)
    return fig


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


def plot_everything(model, freq, n=1):
    fig = make_subplots(rows=4, cols=1, specs=[[{}],[{}],[{"rowspan":2}],[None]])
    fig.update_layout(
        height=600)
    plot_lambdas(model['lambda_1'],model['lambda_2'], fig=fig, row=1, col=1)
    plot_tau(model, fig=fig, row=2, col=1)
    return plot_expected(model, freq, n=n, fig=fig, row=3, col=1, )


