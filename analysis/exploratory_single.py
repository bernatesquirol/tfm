#!/usr/bin/env python
# coding: utf-8
# %%
import pandas as pd
import numpy as np
import scipy
import scipy.stats as st
import matplotlib.pyplot as plt
import altair as alt
get_ipython().run_line_magic('matplotlib', 'inline')


# %%
from pylab import rcParams
rcParams['figure.figsize'] = 15, 8


# %%
user = '../data/politicians/PabloIglesias.pkl'
actions = pd.read_pickle(user).reset_index()


# %%
actions.head()


# %%
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')




# %% [markdown]
# ## Timeline visualization

# %%
utils.plot_top_users_time(actions, types=['RT'])


# %%
utils.plot_top_rt_and_quote(actions)


# %% [markdown]
# ## Statistical analysis

# %%
timeline = actions[actions['type']!='Like']
likes = actions[actions['type']=='Like']


# %%
actions['type'].unique()


# %%
users_rt_and_quotes = utils.user_frequency(actions[actions['type'].isin(['RT','Quoted'])], False)


# %%
users_actions = utils.user_frequency(actions, False)


# %%
# users_actions


# %%
y_actions = np.concatenate([np.zeros(v)+i+1 for i, v in enumerate(users_actions)])
x_actions = np.arange(len(y_actions))


# %%
y_rts = np.concatenate([np.zeros(v)+i+1 for i, v in enumerate(users_rt_and_quotes.values)])
x_rts = np.arange(len(y_rts))


# %%
y_df = pd.DataFrame(y_actions, columns=['Data'])
y_df.describe()


# %%
plt.hist(y_actions, bins=len(users_actions), density=True)
plt.show()


# %%
plt.hist(y_rts, bins='auto', density=True)
plt.show()


# %% [markdown]
# https://risk-engineering.org/static/PDF/slides-data-analysis.pdf

# %%
def plot_best_args(frequency, dist_name):
    y = np.concatenate([np.zeros(v)+i+1 for i, v in enumerate(frequency.values)])
    #first
    plt.subplot(131)
    dist = getattr(scipy.stats, dist_name)
    plt.hist(y, density=True, alpha=0.5, bins=len(users_actions))
    args = dist.fit(y, floc=0)
    x = np.linspace(y.min(), y.max(), 100)
    plt.plot(x, dist(*args).pdf(x))
    plt.title("{} fit on data".format(dist_name))
    
    #second
    plt.subplot(132)
    import statsmodels.distributions
    ecdf = statsmodels.distributions.ECDF(y)
    plt.plot(x, ecdf(x), label="Empirical CDF")
    plt.plot(x, dist(*args).cdf(x),label="{} fit".format(dist_name))
    plt.title("Cumulative failure intensity")
    plt.legend()
    #third
    plt.subplot(133)
    from scipy.stats import probplot
    probplot(y, dist=dist(*args),plot=plt, fit=True)
    plt.title("{} QQ-plot".format(dist_name))
    
    #plt.show()
    return args


# %%
def get_best_args(frequency):
    y = np.concatenate([np.zeros(v)+i+1 for i, v in enumerate(frequency.values)])
    dist = getattr(scipy.stats, dist_name)
    args = dist.fit(y, floc=0)
    # (exp, k, loc, lam) 
    return args


# %%
users_actions


# %%
best_args = plot_best_args(users_actions, 'exponweib')


# %%
scipy.stats.kstest(y_actions, 'exponweib', args=best_args)


# %%
best_args_2 = plot_best_args(users_rt_and_quotes, 'exponweib')


# %%
scipy.stats.kstest(y_rts, 'exponweib', args=best_args_2)
# https://pythonhealthcare.org/2018/05/03/81-distribution-fitting-to-data/
