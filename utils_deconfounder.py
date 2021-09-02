import functools
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import random

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd

from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve


from sklearn.datasets import load_breast_cancer

tf.enable_v2_behavior()

plt.style.use("ggplot")


import numpy.random as npr
from scipy import sparse, stats
from scipy.special import expit, logit, logsumexp


def fit_ppca(x_train, latent_dim, mask, stddv_datapoints = 0.1, optimizer_steps = 500):
    
    num_datapoints, data_dim = x_train.shape
    
    def ppca_model(data_dim, latent_dim, num_datapoints, stddv_datapoints, mask):
        w = yield tfd.Normal(loc=tf.zeros([latent_dim, data_dim]),
                    scale=tf.ones([latent_dim, data_dim]),
                    name="w")  # parameter
        z = yield tfd.Normal(loc=tf.zeros([num_datapoints, latent_dim]),
                    scale=tf.ones([num_datapoints, latent_dim]),
                    name="z")  # local latent variable / substitute confounder
        x = yield tfd.Normal(loc=tf.multiply(tf.matmul(z, w), mask),
                        scale=stddv_datapoints * tf.ones([num_datapoints, data_dim]),
                        name="x")  # (modeled) data
        
    
    
    concrete_ppca_model = functools.partial(ppca_model,
                                            data_dim=data_dim,
                                            latent_dim=latent_dim,
                                            num_datapoints=num_datapoints,
                                            stddv_datapoints=stddv_datapoints,
                                            mask=mask)
    
    model = tfd.JointDistributionCoroutineAutoBatched(concrete_ppca_model)
    
    # Initialize w and z as a tensorflow variable
    w = tf.Variable(tf.random.normal([latent_dim, data_dim]))
    z = tf.Variable(tf.random.normal([num_datapoints, latent_dim]))
    
    # target log joint porbability
    target_log_prob_fn = lambda w, z: model.log_prob((w, z, x_train))
    
    # Initialize the parameters in the variational distribution
    
    qw_mean = tf.Variable(tf.random.normal([latent_dim, data_dim]))
    qz_mean = tf.Variable(tf.random.normal([num_datapoints, latent_dim]))
    qw_stddv = tfp.util.TransformedVariable(1e-4*tf.ones([latent_dim, data_dim]),
                                            bijector=tfb.Softplus())
    qz_stddv = tfp.util.TransformedVariable(1e-4*tf.ones([num_datapoints, latent_dim]),
                                            bijector=tfb.Softplus())
    
    # Variational model and surrogate posterior:
    def factored_normal_variational_model():
      qw = yield tfd.Normal(loc=qw_mean, scale=qw_stddv, name="qw")
      qz = yield tfd.Normal(loc=qz_mean, scale=qz_stddv, name="qz")
        
    
    surrogate_posterior = tfd.JointDistributionCoroutineAutoBatched(
        factored_normal_variational_model)
    
    
    losses = tfp.vi.fit_surrogate_posterior(
        target_log_prob_fn,
        surrogate_posterior=surrogate_posterior,
        optimizer=tf.optimizers.Adam(learning_rate=0.05),
        num_steps=optimizer_steps)
    
    w_mean_inferred = np.array(qw_mean)
    w_stddv_inferred = np.array(qw_stddv)
    z_mean_inferred = np.array(qz_mean)
    z_stddv_inferred = np.array(qz_stddv)
    
    return model, surrogate_posterior,w_mean_inferred, w_stddv_inferred, z_mean_inferred, z_stddv_inferred, losses    



def ppc_ppca(x_train, x_vad, mask, holdout_row, model, surrogate_posterior, 
             w_mean, w_stddv, z_mean, z_stddv, stddv_datapoints = 0.1, 
             n_rep = 100, n_eval = 100):
    
    num_datapoints, data_dim = x_train.shape
    
    holdout_gen = np.zeros((n_rep,*(x_train.shape)))
    
    posterior_samples = surrogate_posterior.sample(n_rep)
    _, _, x_generated = model.sample(value=(posterior_samples))
    
    # look only at the heldout entries
    holdout_gen = np.multiply(x_generated, mask)
    
    
    obs_ll = []
    rep_ll = []
    for j in range(n_eval):
        w_sample = npr.normal(w_mean, w_stddv)
        z_sample = npr.normal(z_mean, z_stddv)
        
        holdoutmean_sample = np.multiply(z_sample.dot(w_sample), mask)
        obs_ll.append(np.mean(stats.norm(holdoutmean_sample, \
                            stddv_datapoints).logpdf(x_vad), axis=1))
    
        rep_ll.append(np.mean(stats.norm(holdoutmean_sample, \
                            stddv_datapoints).logpdf(holdout_gen),axis=2))
        
    obs_ll_per_zi, rep_ll_per_zi = np.mean(np.array(obs_ll), axis=0), np.mean(np.array(rep_ll), axis=0)
    
    pvals = np.array([np.mean(rep_ll_per_zi[:,i] > obs_ll_per_zi[i]) for i in range(num_datapoints)])
    holdout_subjects = np.unique(holdout_row)
    overall_pval = np.mean(pvals[holdout_subjects])
    
    return overall_pval, holdout_subjects, rep_ll_per_zi, obs_ll_per_zi


def fit_pica(x_train, latent_dim, mask, stddv_datapoints = 0.1):
    
    num_datapoints, data_dim = x_train.shape
    
    def pica_model(data_dim, latent_dim, num_datapoints, stddv_datapoints, mask):
        w = yield tfd.Normal(loc=tf.zeros([latent_dim, data_dim]),
                    scale=tf.ones([latent_dim, data_dim]),
                    name="w")  # parameter
        z = yield tfd.Laplace(loc=tf.zeros([num_datapoints, latent_dim]),
                    scale=tf.ones([num_datapoints, latent_dim]),
                    name="z")  # local latent variable / substitute confounder
        x = yield tfd.Normal(loc=tf.multiply(tf.matmul(z, w), mask),
                        scale=stddv_datapoints * tf.ones([num_datapoints, data_dim]),
                        name="x")  # (modeled) data
        
    
    
    concrete_pica_model = functools.partial(pica_model,
                                            data_dim=data_dim,
                                            latent_dim=latent_dim,
                                            num_datapoints=num_datapoints,
                                            stddv_datapoints=stddv_datapoints,
                                            mask=mask)
    
    model = tfd.JointDistributionCoroutineAutoBatched(concrete_pica_model)
    
    # Initialize w and z as a tensorflow variable
    w = tf.Variable(tf.random.normal([latent_dim, data_dim]))
    z = tf.Variable(tf.random.normal([num_datapoints, latent_dim]))
    
    # target log joint porbability
    target_log_prob_fn = lambda w, z: model.log_prob((w, z, x_train))
    
    # Initialize the parameters in the variational distribution
    
    qw_mean = tf.Variable(tf.random.normal([latent_dim, data_dim]))
    qz_mean = tf.Variable(tf.random.normal([num_datapoints, latent_dim]))
    qw_stddv = tfp.util.TransformedVariable(1e-4*tf.ones([latent_dim, data_dim]),
                                            bijector=tfb.Softplus())
    qz_stddv = tfp.util.TransformedVariable(1e-4*tf.ones([num_datapoints, latent_dim]),
                                            bijector=tfb.Softplus())
    
    # Variational model and surrogate posterior:
    def factored_normal_variational_model():
      qw = yield tfd.Normal(loc=qw_mean, scale=qw_stddv, name="qw")
      qz = yield tfd.Normal(loc=qz_mean, scale=qz_stddv, name="qz")
        
    
    surrogate_posterior = tfd.JointDistributionCoroutineAutoBatched(
        factored_normal_variational_model)
    
    
    losses = tfp.vi.fit_surrogate_posterior(
        target_log_prob_fn,
        surrogate_posterior=surrogate_posterior,
        optimizer=tf.optimizers.Adam(learning_rate=0.05),
        num_steps=500)
    
    w_mean_inferred = np.array(qw_mean)
    w_stddv_inferred = np.array(qw_stddv)
    z_mean_inferred = np.array(qz_mean)
    z_stddv_inferred = np.array(qz_stddv)
    
    return model, surrogate_posterior,w_mean_inferred, w_stddv_inferred, z_mean_inferred, z_stddv_inferred, losses    


def ppc_pica(x_train, x_vad, mask, holdout_row, model, surrogate_posterior, 
             w_mean, w_stddv, z_mean, z_stddv, stddv_datapoints = 0.1, 
             n_rep = 100, n_eval = 100):
    
    num_datapoints, data_dim = x_train.shape
    
    holdout_gen = np.zeros((n_rep,*(x_train.shape)))
    
    posterior_samples = surrogate_posterior.sample(n_rep)
    _, _, x_generated = model.sample(value=(posterior_samples))
    
    # look only at the heldout entries
    holdout_gen = np.multiply(x_generated, mask)
    
    
    obs_ll = []
    rep_ll = []
    for j in range(n_eval):
        w_sample = npr.normal(w_mean, w_stddv)
        z_sample = npr.laplace(z_mean, z_stddv)
        
        holdoutmean_sample = np.multiply(z_sample.dot(w_sample), mask)
        obs_ll.append(np.mean(stats.norm(holdoutmean_sample, \
                            stddv_datapoints).logpdf(x_vad), axis=1))
    
        rep_ll.append(np.mean(stats.norm(holdoutmean_sample, \
                            stddv_datapoints).logpdf(holdout_gen),axis=2))
        
    obs_ll_per_zi, rep_ll_per_zi = np.mean(np.array(obs_ll), axis=0), np.mean(np.array(rep_ll), axis=0)
    
    pvals = np.array([np.mean(rep_ll_per_zi[:,i] > obs_ll_per_zi[i]) for i in range(num_datapoints)])
    holdout_subjects = np.unique(holdout_row)
    overall_pval = np.mean(pvals[holdout_subjects])
    
    return overall_pval, holdout_subjects, rep_ll_per_zi, obs_ll_per_zi


def fit_gmm(x_train, components, dtype = np.float64):
    
    num_samples, dims = x_train.shape
    
    class MVNCholPrecisionTriL(tfd.TransformedDistribution): 
      """Multivariate Normal with loc and (Cholesky) precision matrix."""
    
      def __init__(self, loc, chol_precision_tril, name=None):
        super(MVNCholPrecisionTriL, self).__init__(
            distribution=tfd.Independent(tfd.Normal(tf.zeros_like(loc),
                                                    scale=tf.ones_like(loc)),
                                         reinterpreted_batch_ndims=1),
            bijector=tfb.Chain([
                tfb.Shift(shift=loc),
                tfb.Invert(tfb.ScaleMatvecTriL(scale_tril=chol_precision_tril,
                                               adjoint=True)),
            ]),
            name=name)
    
    bgmm = tfd.JointDistributionNamed(dict(
      mix_probs=tfd.Dirichlet(
        concentration=tf.ones(components, dtype) / 10.),
      loc=tfd.Independent(
          tfd.Normal(
              loc=tf.zeros(dims, dtype),
              scale=tf.ones([components, dims], dtype)),
          reinterpreted_batch_ndims=2),
      precision=tfd.Independent(
          tfd.WishartTriL(
              df=30,
              scale_tril=np.stack([np.eye(dims, dtype = dtype)]*components),
              input_output_cholesky=True),
          reinterpreted_batch_ndims=1),
      s=lambda mix_probs, loc, precision: tfd.Sample(tfd.MixtureSameFamily(
          mixture_distribution=tfd.Categorical(probs=mix_probs),
          components_distribution=MVNCholPrecisionTriL(
              loc=loc,
              chol_precision_tril=precision)),
          sample_shape=num_samples)
    ))
    
    def joint_log_prob(observations, mix_probs, loc, chol_precision):
        return bgmm.log_prob(mix_probs = mix_probs, 
                             loc = loc, 
                             precision = chol_precision, 
                             s = observations)
    
    unnormalized_posterior_log_prob = functools.partial(joint_log_prob, x_train)
    
    initial_state = [
        tf.fill([components],
                value=np.array(1. / components, dtype),
                name='mix_probs'),
        tf.constant(0, shape = (components, dims), dtype = dtype,
                    name='loc'),
        tf.linalg.eye(dims, batch_shape=[components], dtype = dtype, name='chol_precision'),
    ]
    
    unconstraining_bijectors = [
        tfb.SoftmaxCentered(),
        tfb.Identity(),
        tfb.Chain([
            tfb.TransformDiagonal(tfb.Softplus()),
            tfb.FillTriangular(),
        ])]
    
    @tf.function(autograph=False)
    def sample_hmc():
      return tfp.mcmc.sample_chain(
        num_results=2000,
        num_burnin_steps=500,
        current_state=initial_state,
        kernel=tfp.mcmc.SimpleStepSizeAdaptation(
            tfp.mcmc.TransformedTransitionKernel(
                inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
                    target_log_prob_fn=unnormalized_posterior_log_prob,
                     step_size=0.065,
                     num_leapfrog_steps=5),
                bijector=unconstraining_bijectors),
             num_adaptation_steps=400),
        trace_fn=lambda _, pkr: pkr.inner_results.inner_results.is_accepted)
    
    [mix_probs, loc, chol_precision], is_accepted = sample_hmc()
    
    acceptance_rate = tf.reduce_mean(tf.cast(is_accepted, dtype=tf.float32)).numpy()
    mean_mix_probs = tf.reduce_mean(mix_probs, axis=0).numpy()
    mean_loc = tf.reduce_mean(loc, axis=0).numpy()
    mean_chol_precision = tf.reduce_mean(chol_precision, axis=0).numpy() # Cholesky lower triangular matrix
    
    precision = tf.linalg.matmul(mean_chol_precision, mean_chol_precision, transpose_b=True)
    #sigmasq = tf.linalg.inv(precision)
    
    z_post = np.array([np.log(mean_mix_probs[i]) - 
                       np.sum(np.matmul((x_train-mean_loc[i])**2, 
                              precision[i]), axis=1) #Multiply with the precision instead of dividind by the cov
                       for i in range(components)]).argmax(axis=0) 
    
    
    #x_post = mean_loc[z_post]
    mu_post = loc # (2000, 3, 8) i.e. (simulations, K, M)
    pi_post = mix_probs
    
    precision_post = tf.linalg.matmul(chol_precision, chol_precision, transpose_b=True)
    sigmasq_post = tf.linalg.inv(precision_post)
    sigma_post = tf.linalg.sqrtm(sigmasq_post)
    
    return z_post, mu_post, pi_post, precision_post, sigma_post, acceptance_rate


def ppc_gmm(x_train, x_vad, components, holdout_mask, 
            z_post, mu_post, pi_post, sigma_post, 
            n_rep = 10, n_eval = 10):
    
    K = components
    obs_ll = []
    rep_ll = []
    
    # First part of predictive checks:
    # n_rep is the number of datasets produced
    # We generate x as a multivariate random normal because the cov is non-diagonal, which is different from deconfounder public gmm
    
    holdout_row, holdout_col = np.where(holdout_mask > 0)
    holdout_gen = np.zeros([n_rep,x_train.shape[0], x_train.shape[1]])
    for i in range(n_rep):
        z_generated = npr.multinomial(n = 1, 
                                      pvals = np.array([npr.choice(np.array(pi_post)[:,i], size = 50, replace = False) for i in range(pi_post.shape[1])]).mean(1),
                                      size = x_train.shape[0]).argmax(axis=1)
        index = npr.choice(sigma_post.shape[0], 50, replace=False)
        mean = np.array([np.array([npr.choice(np.array(mu_post)[:,i,j], size = 50, replace = False) for j in range(mu_post.shape[2])]) for i in range(mu_post.shape[1])]).mean(2)[z_generated]
        cov = np.array(sigma_post)[index,:,:,:].mean(0)[z_generated]
        x_generated = []
        for k in range(x_train.shape[0]):
            x_generated_current = npr.multivariate_normal(mean = mean[k,:],
                                                         cov = cov[k,:]
                                                         )
            x_generated.append(x_generated_current)    
        x_generated = np.array(x_generated)
        # look only at the heldout entries
        holdout_gen[i] = np.multiply(x_generated, holdout_mask)
    
    
    # Second part of predictive checks:
    
    for j in range(n_eval):
        mu_sample = np.array(mu_post)[npr.choice(mu_post.shape[0], 1, replace=False),:,:].mean(0) # The last part (mean(0)) is simply yo colapse the first dimension and the same for the following two
        sigma_sample = np.array(sigma_post)[npr.choice(sigma_post.shape[0], 1, replace=False),:,:,:].mean(0)
        pi_sample = np.array(pi_post)[npr.choice(pi_post.shape[0], 1, replace=False),:].mean(0)
    
    
        obs_ll_cur = logsumexp(np.array([np.log(pi_sample[i]) -
                                         np.sum(np.matmul(np.multiply(x_vad-mu_sample[i],holdout_mask)**2, 
                                                          sigma_sample[i]), axis=1) #Multiply with the precision instead of dividind by the cov
                                         for i in range(K)]), axis=0) 
    
    
        rep_ll_cur = np.array([logsumexp(np.array([np.log(pi_sample[i]) - 
                                         np.sum(np.matmul(np.multiply(holdout_gen[j]-mu_sample[i],holdout_mask)**2, 
                                                          sigma_sample[i]), axis=1) #Multiply with the precision instead of dividind by the cov
                                         for i in range(K)]), axis=0) for j in range(holdout_gen.shape[0])])
        
        
    
        
        obs_ll.append(obs_ll_cur)
        rep_ll.append(rep_ll_cur)
        
    obs_ll_per_zi, rep_ll_per_zi = np.mean(np.array(obs_ll), axis=0), np.mean(np.array(rep_ll), axis=0)
    
    
    # Third and last part of predictive checks:
    
    pvals = np.array([np.mean(rep_ll_per_zi[:,i] < obs_ll_per_zi[i]) for i in range(len(obs_ll_per_zi))])
    holdout_subjects = np.unique(holdout_col)
    overall_pval = np.mean(pvals[holdout_subjects])
    
    return overall_pval

# Multiple (4) chains sampled: Used for graph, however, the code takes some time (5 min approx).
def fit_gmm2(x_train, components, dtype = np.float64):
    
    num_samples, dims = x_train.shape
    
    class MVNCholPrecisionTriL(tfd.TransformedDistribution): 
      """Multivariate Normal with loc and (Cholesky) precision matrix."""
    
      def __init__(self, loc, chol_precision_tril, name=None):
        super(MVNCholPrecisionTriL, self).__init__(
            distribution=tfd.Independent(tfd.Normal(tf.zeros_like(loc),
                                                    scale=tf.ones_like(loc)),
                                         reinterpreted_batch_ndims=1),
            bijector=tfb.Chain([
                tfb.Shift(shift=loc),
                tfb.Invert(tfb.ScaleMatvecTriL(scale_tril=chol_precision_tril,
                                               adjoint=True)),
            ]),
            name=name)
    
    bgmm = tfd.JointDistributionNamed(dict(
      mix_probs=tfd.Dirichlet(
        concentration=tf.ones(components, dtype) / 10.),
      loc=tfd.Independent(
          tfd.Normal(
              loc=tf.zeros(dims, dtype),
              scale=tf.ones([components, dims], dtype)),
          reinterpreted_batch_ndims=2),
      precision=tfd.Independent(
          tfd.WishartTriL(
              df=30,
              scale_tril=np.stack([np.eye(dims, dtype = dtype)]*components),
              input_output_cholesky=True),
          reinterpreted_batch_ndims=1),
      s=lambda mix_probs, loc, precision: tfd.Sample(tfd.MixtureSameFamily(
          mixture_distribution=tfd.Categorical(probs=mix_probs),
          components_distribution=MVNCholPrecisionTriL(
              loc=loc,
              chol_precision_tril=precision)),
          sample_shape=num_samples)
    ))
    
    def joint_log_prob(observations, mix_probs, loc, chol_precision):
        return bgmm.log_prob(mix_probs = mix_probs, 
                             loc = loc, 
                             precision = chol_precision, 
                             s = observations)
    
    unnormalized_posterior_log_prob = functools.partial(joint_log_prob, x_train)
    
    initial_state = [
        tf.fill([components],
                value=np.array(1. / components, dtype),
                name='mix_probs'),
        tf.constant(0, shape = (components, dims), dtype = dtype,
                    name='loc'),
        tf.linalg.eye(dims, batch_shape=[components], dtype = dtype, name='chol_precision'),
    ]
    
    unconstraining_bijectors = [
        tfb.SoftmaxCentered(),
        tfb.Identity(),
        tfb.Chain([
            tfb.TransformDiagonal(tfb.Softplus()),
            tfb.FillTriangular(),
        ])]
    
    @tf.function(autograph=False)
    def sample_hmc():
      return tfp.mcmc.sample_chain(
        num_results=2000,
        num_burnin_steps=500,
        current_state=initial_state,
        kernel=tfp.mcmc.SimpleStepSizeAdaptation(
            tfp.mcmc.TransformedTransitionKernel(
                inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
                    target_log_prob_fn=unnormalized_posterior_log_prob,
                     step_size=0.065,
                     num_leapfrog_steps=5),
                bijector=unconstraining_bijectors),
             num_adaptation_steps=400),
        trace_fn=lambda _, pkr: pkr.inner_results.inner_results.is_accepted)
    
    [mix_probs, loc, chol_precision], is_accepted = sample_hmc()
    [mix_probs1, loc1, chol_precision1], _ = sample_hmc()
    [mix_probs2, loc2, chol_precision2], _ = sample_hmc()
    [mix_probs3, loc3, chol_precision3], _ = sample_hmc()
    
    acceptance_rate = tf.reduce_mean(tf.cast(is_accepted, dtype=tf.float32)).numpy()
    mean_mix_probs = tf.reduce_mean(mix_probs, axis=0).numpy()
    mean_loc = tf.reduce_mean(loc, axis=0).numpy()
    mean_chol_precision = tf.reduce_mean(chol_precision, axis=0).numpy() # Cholesky lower triangular matrix
    
    precision = tf.linalg.matmul(mean_chol_precision, mean_chol_precision, transpose_b=True)
    #sigmasq = tf.linalg.inv(precision)
    
    z_post = np.array([np.log(mean_mix_probs[i]) - 
                       np.sum(np.matmul((x_train-mean_loc[i])**2, 
                              precision[i]), axis=1) #Multiply with the precision instead of dividind by the cov
                       for i in range(components)]).argmax(axis=0) 
    
    
    #x_post = mean_loc[z_post]
    mu_post = loc # (2000, 3, 8) i.e. (simulations, K, M)
    pi_post = mix_probs
    
    precision_post = tf.linalg.matmul(chol_precision, chol_precision, transpose_b=True)
    sigmasq_post = tf.linalg.inv(precision_post)
    sigma_post = tf.linalg.sqrtm(sigmasq_post)
    
    return z_post, mu_post, pi_post, precision_post, sigma_post, acceptance_rate, is_accepted, mix_probs1, mix_probs2, mix_probs3



def fit_dlgm(x_train, K = [7, 5, 2], prior_mean = 0, prior_stddv = 1, prior_mean_latent = 0, prior_stddv_latent = 1, prior_stddv_datapoints = 1):
    
    N, M = x_train.shape
    
    def deep_gaussian(N = N, M = M, K = K):
        
        w2 = yield tfd.Normal(loc = prior_mean * tf.ones([K[2], K[1]]), 
                              scale = prior_stddv * tf.ones([K[2], K[1]]), name="w2")
        w1 = yield tfd.Normal(loc = prior_mean * tf.ones([K[1], K[0]]), 
                              scale = prior_stddv * tf.ones([K[1], K[0]]), name="w1")
        w0 = yield tfd.Normal(loc = prior_mean * tf.ones([K[0], M]), 
                              scale = prior_stddv * tf.ones([K[0], M]), name="w0")
        
        
        z2 = yield tfd.Normal(loc = prior_mean_latent * tf.ones([N, K[2]]),
                              scale = prior_stddv_latent * tf.ones([N, K[2]]), 
                              name="z2")
        z1 = yield tfd.Normal(loc = tf.math.exp(tf.matmul(z2, w2)) / tf.math.reduce_sum(tf.math.exp(tf.matmul(z2, w2)), 
                                                                                        axis = 0), 
                              scale=tf.ones([N, K[1]]), 
                              name="z1")
        z0 = yield tfd.Normal(loc = tf.math.exp(tf.matmul(z1, w1)) / tf.math.reduce_sum(tf.math.exp(tf.matmul(z1, w1)),
                                                                                       axis = 0), 
                              scale=tf.ones([N, K[0]]), 
                              name="z0")
        
        
        x = yield tfd.Normal(loc = tf.math.exp(tf.matmul(z0, w0)) / tf.math.reduce_sum(tf.math.exp(tf.matmul(z0, w0)),
                                                                                      axis = 0),
                             scale = prior_stddv_datapoints*tf.ones([N, M]),
                             name = "x")
            
    
    
    concrete_deep_gaussian_model = functools.partial(deep_gaussian, N = N, M = M)
    
    model = tfd.JointDistributionCoroutineAutoBatched(concrete_deep_gaussian_model)

    
    # Initialize W and Z as a tensorflow variable
    w2 = tf.Variable(tf.random.normal([K[2], K[1]]))
    w1 = tf.Variable(tf.random.normal([K[1], K[0]]))
    w0 = tf.Variable(tf.random.normal([K[0], M]))
    
    z2 = tf.Variable(tf.random.normal([N, K[2]]))
    z1 = tf.Variable(tf.random.normal([N, K[1]]))
    z0 = tf.Variable(tf.random.normal([N, K[0]]))
    
    # target log joint (cond) probability
    target_log_prob_fn = lambda w2, w1, w0, z2, z1, z0: model.log_prob((w2, w1, w0, z2, 
                                                                    z1, z0, x_train))
    
    # Initialize the parameters in the variational distribution
    
    def trainable_parameters(shape):
        # Parameters:
        mean = tf.Variable(tf.random.normal(shape))
        stddv = tfp.util.TransformedVariable(tf.ones(shape), bijector=tfb.Softplus())
        return [mean, stddv]
    
    qw2_mean = trainable_parameters(w2.shape)[0]
    qw1_mean = trainable_parameters(w1.shape)[0]
    qw0_mean = trainable_parameters(w0.shape)[0]
    qz2_mean = trainable_parameters(z2.shape)[0]
    qz1_mean = trainable_parameters(z1.shape)[0]
    qz0_mean = trainable_parameters(z0.shape)[0]
    
    qw2_stddv = trainable_parameters(w2.shape)[1]
    qw1_stddv = trainable_parameters(w1.shape)[1]
    qw0_stddv = trainable_parameters(w0.shape)[1]
    qz2_stddv = trainable_parameters(z2.shape)[1]
    qz1_stddv = trainable_parameters(z1.shape)[1]
    qz0_stddv = trainable_parameters(z0.shape)[1]

    
    # Variational model and surrogate posterior:
    def factored_deep_gaussian_variational_model():
        qw2 = yield tfd.Normal(loc=qw2_mean, scale=qw2_stddv, name="qw2")
        qw1 = yield tfd.Normal(loc=qw1_mean, scale=qw1_stddv, name="qw1")
        qw0 = yield tfd.Normal(loc=qw0_mean, scale=qw0_stddv, name="qw0")
        
        qz2 = yield tfd.Normal(loc=qz2_mean, scale=qz2_stddv, name="qz2")
        qz1 = yield tfd.Normal(loc=qz1_mean, scale=qz1_stddv, name="qz1")
        qz0 = yield tfd.Normal(loc=qz0_mean, scale=qz0_stddv, name="qz0")    
    
    
    surrogate_posterior = tfd.JointDistributionCoroutineAutoBatched(
        factored_deep_gaussian_variational_model)
    
    losses = tfp.vi.fit_surrogate_posterior(
        target_log_prob_fn,
        surrogate_posterior=surrogate_posterior,
        optimizer=tf.optimizers.Adam(learning_rate=0.05),
        num_steps=500)
    
    w2_mean_inferred = qw2_mean
    w1_mean_inferred = qw1_mean
    w0_mean_inferred = qw0_mean
    z2_mean_inferred = qz2_mean
    z1_mean_inferred = qz1_mean
    z0_mean_inferred = qz0_mean # This is the one to use for causal discorvery!!
    
    w2_stddv_inferred = qw2_stddv
    w1_stddv_inferred = qw1_stddv
    w0_stddv_inferred = qw0_stddv
    z2_stddv_inferred = qz2_stddv
    z1_stddv_inferred = qz1_stddv
    z0_stddv_inferred = qz0_stddv
    
    return model, surrogate_posterior, w0_mean_inferred, w0_stddv_inferred, z0_mean_inferred, z0_stddv_inferred, losses    

def ppc_dlgm(x_train, x_vad, mask, holdout_row, model, surrogate_posterior, 
             w0_mean, w0_stddv, z0_mean, z0_stddv, stddv_datapoints = 1, 
             n_rep = 100, n_eval = 100):
    
    num_datapoints, data_dim = x_train.shape
    
    posterior_samples = surrogate_posterior.sample(n_rep)
    _, _, _, _, _, _, x_generated = model.sample(value=(posterior_samples))

    # look only at the heldout entries
    holdout_gen = np.multiply(x_generated, mask)
    
    
    obs_ll = []
    rep_ll = []
    for j in range(n_eval):
        w0_sample = npr.normal(np.array(w0_mean), np.array(w0_stddv))
        z0_sample = npr.normal(np.array(z0_mean), np.array(z0_stddv))
        
        holdoutmean_sample = np.multiply(z0_sample.dot(w0_sample), mask)
        obs_ll.append(np.mean(stats.norm(holdoutmean_sample, \
                            stddv_datapoints).logpdf(x_vad), axis=1))
    
        rep_ll.append(np.mean(stats.norm(holdoutmean_sample, \
                            stddv_datapoints).logpdf(holdout_gen),axis=2))
        
    obs_ll_per_zi, rep_ll_per_zi = np.mean(np.array(obs_ll), axis=0), np.mean(np.array(rep_ll), axis=0)  
    
    pvals = np.array([np.mean(rep_ll_per_zi[:,i] > obs_ll_per_zi[i]) for i in range(num_datapoints)])
    holdout_subjects = np.unique(holdout_row)
    overall_pval = np.mean(pvals[holdout_subjects])
    
    return overall_pval, holdout_subjects, rep_ll_per_zi, obs_ll_per_zi


def fit_pmf(x_train, latent_dim, mask, gamma_prior = 0.1):
    num_datapoints, data_dim = x_train.shape
    latent_dim = 5
    def pmf_model(data_dim, latent_dim, num_datapoints, mask, gamma_prior = 0.1):
        w = yield tfd.Gamma(concentration = gamma_prior * tf.ones([latent_dim, data_dim]),
                            rate = gamma_prior * tf.ones([latent_dim, data_dim]),
                            name="w")  # parameter
        z = yield tfd.Gamma(concentration = gamma_prior * tf.ones([num_datapoints, latent_dim]),
                            rate = gamma_prior * tf.ones([num_datapoints, latent_dim]),
                            name="z")  # local latent variable / substitute confounder
        x = yield tfd.Poisson(rate = tf.multiply(tf.matmul(z, w), mask),
                              name="x")  # (modeled) data
        
    
    
    concrete_pmf_model = functools.partial(pmf_model,
                                           data_dim=data_dim,
                                           latent_dim=latent_dim,
                                           num_datapoints=num_datapoints,
                                           mask=mask)
    
    model = tfd.JointDistributionCoroutineAutoBatched(concrete_pmf_model)
    
    
    # Initialize w and z as a tensorflow variable
    w = tf.Variable(tf.random.gamma([latent_dim, data_dim], alpha = 0.1))
    z = tf.Variable(tf.random.gamma([num_datapoints, latent_dim], alpha = 0.1))
    
    # target log joint porbability
    target_log_prob_fn = lambda w, z: model.log_prob((w, z, x_train))
    
    # Initialize the parameters in the variational distribution
    qw_conc = tf.random.uniform([latent_dim, data_dim], minval = 1e-5)
    qz_conc = tf.random.uniform([num_datapoints, latent_dim], minval = 1e-5)
    qw_rate = tf.maximum(tfp.util.TransformedVariable(tf.random.uniform([latent_dim, data_dim]),
                                                      bijector=tfb.Softplus()), 1e-5)
    qz_rate = tf.maximum(tfp.util.TransformedVariable(tf.random.uniform([num_datapoints, latent_dim]),
                                                      bijector=tfb.Softplus()), 1e-5)
    
    # Variational model and surrogate posterior:
    def factored_gamma_variational_model():
        qw = yield tfd.TransformedDistribution(distribution = tfd.Normal(loc = qw_conc,
                                                                         scale = qw_rate),
                                               bijector = tfb.Exp(),
                                               name = "qw")
        qz = yield tfd.TransformedDistribution(distribution = tfd.Normal(loc = qz_conc,
                                                                         scale = qz_rate),
                                               bijector = tfb.Exp(),
                                               name = "qz")
    
    surrogate_posterior = tfd.JointDistributionCoroutineAutoBatched(
        factored_gamma_variational_model)
    
    losses = tfp.vi.fit_surrogate_posterior(
        target_log_prob_fn,
        surrogate_posterior=surrogate_posterior,
        optimizer=tf.optimizers.Adam(learning_rate=0.05),
        num_steps=500)
    
    w_conc_inferred = np.array(qw_conc)
    w_rate_inferred = np.array(qw_rate)
    z_conc_inferred = np.array(qz_conc)
    z_rate_inferred = np.array(qz_rate)
    
    return model, surrogate_posterior,w_conc_inferred, w_rate_inferred, z_conc_inferred, z_rate_inferred, losses


def ppc_pmf(x_train, x_vad, mask, holdout_row, model, surrogate_posterior, 
             w_mean, w_stddv, z_mean, z_stddv, stddv_datapoints = 1, 
             n_rep = 100, n_eval = 100):
    
    num_datapoints, data_dim = x_train.shape
    
    holdout_gen = np.zeros((n_rep,*(x_train.shape)))
    
    posterior_samples = surrogate_posterior.sample(n_rep)
    _, _, x_generated = model.sample(value=(posterior_samples))
    
    # look only at the heldout entries
    holdout_gen = np.multiply(x_generated, mask)
    
    
    obs_ll = []
    rep_ll = []
    for j in range(n_eval):
        w_sample = npr.gamma(w_mean, w_stddv)
        z_sample = npr.gamma(z_mean, z_stddv)
        
        holdoutmean_sample = np.multiply(z_sample.dot(w_sample), mask)
        obs_ll.append(np.mean(stats.norm(holdoutmean_sample, \
                            stddv_datapoints).logpdf(x_vad), axis=1))
    
        rep_ll.append(np.mean(stats.norm(holdoutmean_sample, \
                            stddv_datapoints).logpdf(holdout_gen),axis=2))
        
    obs_ll_per_zi, rep_ll_per_zi = np.mean(np.array(obs_ll), axis=0), np.mean(np.array(rep_ll), axis=0)
    
    pvals = np.array([np.mean(rep_ll_per_zi[:,i] > obs_ll_per_zi[i]) for i in range(num_datapoints)])
    holdout_subjects = np.unique(holdout_row)
    overall_pval = np.mean(pvals[holdout_subjects])
    
    return overall_pval, holdout_subjects, rep_ll_per_zi, obs_ll_per_zi


def fit_causal_model(X, dfy, dfX_causal_inf, z_mean, data_dim_causal_inf):
    
    z_hat_ppca = z_mean
    X_aug_ppca = np.column_stack([X, z_hat_ppca])
    
    X_train_ppca, X_test_ppca, y_train_ppca, y_test_ppca = train_test_split(X_aug_ppca, 
                                                                            dfy, 
                                                                            test_size=0.2, 
                                                                            random_state=0)
    
    dcf_ppca_X_train = sm.add_constant(X_train_ppca)
    dcf_ppca_logit_model = sm.Logit(y_train_ppca, dcf_ppca_X_train)
    dcf_ppca_result = dcf_ppca_logit_model.fit_regularized(maxiter=5000)
    
    
    res = pd.DataFrame({"causal_mean": dcf_ppca_result.params[:data_dim_causal_inf+1], \
                  "causal_std": dcf_ppca_result.bse[:data_dim_causal_inf+1], \
                  "causal_025": dcf_ppca_result.conf_int()[:data_dim_causal_inf+1,0], \
                  "causal_975": dcf_ppca_result.conf_int()[:data_dim_causal_inf+1,1], \
                   "causal_pval": dcf_ppca_result.pvalues[:data_dim_causal_inf+1]})
    res["causal_sig"] = (res["causal_pval"] < 0.05)
    res = res.T
    #print(res.columns.shape)
    #print(dfX_causal_inf.columns.shape)
    res.columns = np.concatenate([["intercept"], np.array(dfX_causal_inf.columns)])
    res = res.T
    
    # make predictions with the causal model 
    dcf_ppca_X_test = X_test_ppca
    dcf_ppca_y_predprob = dcf_ppca_result.predict(sm.add_constant(dcf_ppca_X_test))
    dcf_ppca_y_pred = (dcf_ppca_y_predprob > 0.5)
    pred_report = classification_report(y_test_ppca, dcf_ppca_y_pred)
    
    nodcfX_train = sm.add_constant(X_train_ppca[:,:X.shape[1]])
    nodcflogit_model = sm.Logit(y_train_ppca, nodcfX_train)
    nodcfresult = nodcflogit_model.fit_regularized(maxiter=5000)

    res["noncausal_mean"] = np.array(nodcfresult.params)
    res["noncausal_std"] = np.array(nodcfresult.bse)
    res["noncausal_025"] = np.array(nodcfresult.conf_int()[:,0])
    res["noncausal_975"] = np.array(nodcfresult.conf_int()[:,1])
    res["noncausal_pval"] = np.array(nodcfresult.pvalues)
    res["noncausal_sig"] = (res["noncausal_pval"] < 0.05)

    res["diff"] = res["causal_mean"] - res["noncausal_mean"]
    res["pval_diff"] = res["causal_pval"] - res["noncausal_pval"]
    
    nodcfX_test = sm.add_constant(X_test_ppca[:,:X.shape[1]])
    nodcfy_predprob = nodcfresult.predict(nodcfX_test)
    nodcfy_pred = (nodcfy_predprob > 0.5)
    
    dcflogit_roc_auc = roc_auc_score(y_test_ppca, dcf_ppca_y_pred)
    dcffpr, dcftpr, dcfthresholds = roc_curve(y_test_ppca, dcf_ppca_y_predprob)
    nodcflogit_roc_auc = roc_auc_score(y_test_ppca, nodcfy_pred)
    nodcffpr, nodcftpr, nodcfthresholds = roc_curve(y_test_ppca, nodcfy_predprob)

    
    
    
    return dcf_ppca_result, nodcfresult, res, pred_report, dcflogit_roc_auc, dcffpr, dcftpr, nodcflogit_roc_auc, nodcffpr, nodcftpr     


