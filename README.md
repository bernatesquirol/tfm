# Characterization of social behaviour in Twitter

The goal of this paper is to evaluate the differences of behaviour of users in Twitter before and after the Spanish 2020 confinement, that went from 13th March 2020 to May 2020. The dataset we'll analyse will be tweets from 18.000 Spanish twitter profiles, categorized in 3 types: politicians (about 300 profiles), journalists (800 profiles) and random profiles (16.000). Using Bayesian models we'll try to uncover the effect of the confinement and COVID-19 situation in those profiles. We'll count for each user the number of actions they perfomed in twitter, that means, every original tweet, every retweet, reply or quote.

The model we'll use a Poisson distribution to model the actions performed by the user, with a single switchpoint, where the Poisson will change parameters. So we'll have:

$$
\begin{align*}

\lambda_{1}^{(0)} &\sim \text{Exponential}(\text{rate}=\alpha) \\

\lambda_{2}^{(0)} &\sim \text{Exponential}(\text{rate}=\alpha) \\

\tau &\sim \text{Uniform}[\text{low}=0,\text{high}=1) \\

\text{for }  i &= 1\ldots N: \\

\lambda_i &= \begin{cases} \lambda_{1}^{(0)}, & \tau > i/N \\ \lambda_{2}^{(0)}, &   \tau \leq i/N\end{cases}\\

 X_i &\sim \text{Poisson}(\text{rate}=\lambda_i)

\end{align*}

$$

The model will fit a $\tau$ that will be the day of the _distribution change_. Our prior for this $\tau$ will be a uniform, as we want all days to have the same probability to be the switchpoint.

The parameter of the Poisson function, that will describe the number of actions for a given day, will be changed between $\lambda_1$ and $\lambda_2$and in the day $\tau$. As a prior we'll have $\lambda_i\sim \text{Exponential}(\alpha)$. 

https://colab.research.google.com/github/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Multiple_changepoint_detection_and_Bayesian_model_selection.ipynb#scrollTo=bvEpqBxvoleY​.
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTY0NTAzMDcyXX0=
-->