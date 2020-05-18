pip install bayesian_bootstrap

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
X = np.random.exponential(7, 4)

# +
from bayesian_bootstrap.bootstrap import mean, highest_density_interval, BayesianBootstrapBagging
posterior_samples = mean(X, 10000)
l, r = highest_density_interval(posterior_samples)

plt.title('Bayesian Bootstrap of mean')
sns.distplot(posterior_samples, label='Bayesian Bootstrap Samples')
plt.plot([l, r], [0, 0], linewidth=5.0, marker='o', label='95% HDI')
# -

from bayesian_bootstrap.bootstrap import bayesian_bootstrap
posterior_samples = bayesian_bootstrap(X, np.mean, 10000, 100)

X = np.random.normal(0, 1, 5).reshape(-1, 1)
y = X.reshape(1, -1).reshape(5) + np.random.normal(0, 1, 5)

m = BayesianBootstrapBagging(LinearRegression(), 10000, 1000)
m.fit(X, y)

import utils

p_pre, p_post = utils.load_users_covid('politicians')
rfriends_pre, rfriends_post = utils.load_users_covid('random-friends')
rfollowers_pre, rfollowers_post = utils.load_users_covid('random-followers')
# j_pre, j_post = utils.load_users_covid('journalists-new')
rfriends_big_pre, rfriends_big_post = utils.load_users_covid('random-friends-big')
rfollowers_big_pre, rfollowers_big_post = utils.load_users_covid('random-followers-big')
rfollowers_big_post.to_pickle('random-followers-big_post.pkl')
rfollowers_big_pre.to_pickle('random-followers-big_pre.pkl')


