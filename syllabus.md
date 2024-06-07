# 1. Intro
Notebook: `lectures/intro.ipynb`
- JupyterLab
- Debugging
- Version control
- Python packages
## Exercise
- Use JupyterLab
- Monte Carlo estimate of pi

# 2. Probabilities
Notebook: `lectures/probabilities.ipynb`
- Different definitions of probability
- Set notation
- Outcomes, events
- Kolmogorov axioms
- Conditional probabilities and independence
- Bayes theorem
## Exercises
- Birthday problem
- Monty Hall problem

# 3. Random variables and probability distributions
Notebook: `lectures/random_variables_and_probability_distributions.ipynb`
- Random variables
- Probability distributions: discrete and continuous
- PDF and CDF
- Change of variables
- Inverse transform sampling
- Expectation
- Mean, variance, moments
- Joint, conditional, and marginal distributions
- Common probability distributions
    - Uniform
    - Binomial, multinomial
    - Poisson
    - Gaussian
    - Chi-squared
    - Cauchy
    - Power law
    - Central limit theorem
## Exercise
- Inverse transform sampling
- Derive Poisson from binomial distribution
- Distribution of sum of Gaussian
- General sum of independent RVs
- Distribution of chi-squared distribution

# 4. Introduction to Bayesian statistics
Notebook: `lectures/intro_to_bayes.ipynb`
- Bayes theorem
- Likelihood, prior, posterior
- Updating priors
- Prior and posterior predictive distributions
- Model comparison: evidences and Bayes ratio
- Bayesian line fitting
- MAP
- Posterior sampling
- Computing predictive distributions
## Exercises
- Fitting data
- Misspecified likelihood

# 5. Sampling from distributions 1
Notebook: `lectures/sampling.ipynb`
- Monte Carlo estimates of integrals
- Rejection sampling
- Markov chain Monte Carlo
- Metropolis-Hastings
## Exercises
- Implement rejection sampling
- Implement Metropolis-Hastings in n-d
- Show that Metropolis-Hastings satisfies detailed balance

# 6. Sampling from distributions 2
Notebook: `lectures/sampling_2.ipynb`
- Burn-in, convergence, and auto-correlation
- Slice sampling
- Nested sampling
- Application to model selection using Bayes' ratio on super novae data
## Exercises
- Implement nested or slice sampling
- Use emcee and dynesty
- Use dynesty to compare models

# 7. Model checking
Notebook: `lectures/model_checking.ipynb`
- Chi-square goodness-of-fit
- Posterior predictive checks
- Model comparison:
    - DIC
    - WAIC
    - Cross-validation
## Exercises
- Implement chi-square and posterior predictive checks
- Use DIC, WAIC, and Bayes ratio for model comparison

# 8. Estimators and data exploration
Notebook: `lectures/estimators_and_data_exploration.ipynb`
- Statistics and estimators
- Estimator bias and variance
- Statistics and their sampling distributions
    - Sample mean
    - Sample variance
    - Sample covariance
    - Correlation coefficient
- Correlation
    - Malmquist bias
- PCA
- Bootstrap
## Exercises
- Show that the sample variance estimator is unbiased
- Compute posterior on the correlation coefficient
- Check bootrap on case where exact sampling distribution is known

# 9. Fisher, Hamilton Monte Carlo, and JAX
Notebook: `lectures/fisher_hmc_and_jax.ipynb`
- Fisher information matrix
- Cramer-Rao bound
- Jeffreys prior
- JAX
- Hamiltonian Monte Carlo
## Exercises
- Use JAX to get Fisher information
- Experiment with HMC settings
- Use implementation of NUTS in tensorflow-probability

# 10. Simulation-based inference
Notebook: `lectures/simulation_based_inference.ipynb`
- Approximate Bayesian computation
- Neural density estimation
- Kullback-Leibler divergence
- Gaussian mixture models
- Loss functions and posteriors
- MLPs
- L_2, L_1, negative log likelihood loss
## Exercises
- Implement rejection ABC
- Show that the function that minimises the L_1 loss is the median
- Implement neural density estimation 

# 11. Recap
Notebook: `lectures/recap.ipynb`
- Recap of the course
- Worked example: cosmology inference on Type Ia supernovae data



