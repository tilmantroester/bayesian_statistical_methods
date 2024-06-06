import numpy as np

import scipy.optimize
import scipy.stats

import emcee
import corner

import matplotlib.pyplot as plt


def plot_data(x, y, y_err, models=None, subtract_y=None,
              axis_labels=["$x$", "$y$"], **data_kwargs):
    fig, ax = plt.subplots()

    data_kwargs = {**data_kwargs}
    if subtract_y is None:
        subtract_y = np.zeros_like(y)
    ax.errorbar(x, y-subtract_y, y_err, **data_kwargs, label="Data")

    if models is not None:
        for model_def in models:
            if "lower" in model_def:
                ax.fill_between(
                    model_def["x"],
                    model_def["lower"], model_def["upper"],
                    **model_def.get("style", {})
                )
            elif "y_err" in model_def:
                ax.errorbar(
                    model_def["x"], model_def["y"], model_def["y_err"],
                    **model_def.get("style", {})
                )
            elif "violin_samples" in model_def:
                violin_plots = ax.violinplot(
                    dataset=model_def["violin_samples"],
                    positions=model_def["x"], widths=2.0,
                    vert=True, showmeans=False, showextrema=False
                )
                style = model_def.get("style", {})                
                for body in violin_plots["bodies"]:
                    body.set(**style)
                    if "label" in style:
                        del style["label"]

            else:
                ax.plot(
                    model_def["x"], model_def["y"],
                    **model_def.get("style", {})
                )

    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.legend(frameon=False)

    return fig, ax


def analyse_data(data,
                 log_posterior_fn, model_fn, predict_fn, ppd_test_statistic_fn,
                 param_names,
                 derived_parameter_fn=None, derived_parameter_names=[],
                 derived_parameter_true=[],
                 theta_true=None,
                 theta_init=None, theta_init_std=None,
                 plot=True, model_x=None,
                 n_step_emcee=5000, n_ppd=500):

    x, y, sigma_y = data["x"], data["y"], data["y_err"]

    if theta_init is None:
        theta_init = theta_true
    theta_init = np.array(theta_init)

    if theta_init_std is None:
        theta_init_std = 0.1*theta_init

    if model_x is None:
        model_x = x

    if theta_true is not None:
        true_model_plot_def = [dict(
            x=model_x, y=model_fn(theta_true, model_x),
            style=dict(color="black", ls="--", label="True model")
        )]
    else:
        true_model_plot_def = []

    # Find the MAP
    def negative_log_posterior(theta, x, sigma_y, y):
        return -log_posterior_fn(theta, x, sigma_y, y)

    MAP_result = scipy.optimize.minimize(
        fun=negative_log_posterior,
        x0=theta_init,
        args=(x, sigma_y, y)
    )
    theta_MAP = MAP_result.x

    print("MAP results")
    for name, theta in zip(param_names, theta_MAP):
        print(f"{name}_MAP = {theta}")


    # Sample posterior with emcee

    # emcee requires some extra settings to run
    n_param = len(theta_init) # Number of parameter we are sampling
    n_walker = 3*n_param      # Number of walkers. This just needs to be larger than 2*n_param + 1
    n_step = n_step_emcee     # How many steps each walker will take. The number of samples will be n_walker*n_step

    # The starting point for each walker
    theta_init = theta_init + theta_init_std*np.random.normal(size=(n_walker, n_param))

    # compute derived parameters while sampling
    if derived_parameter_fn is not None:
        log_posterior_wrapper = lambda *args: (log_posterior_fn(*args), derived_parameter_fn(args[0]))
    else:
        log_posterior_wrapper = log_posterior_fn
    sampler = emcee.EnsembleSampler(
        nwalkers=n_walker, ndim=n_param,
        log_prob_fn=log_posterior_wrapper,
        args=(x, sigma_y, y)
    )
    state = sampler.run_mcmc(theta_init, nsteps=n_step, progress=True)

    # The samples will be correlated, this checks how correlated they are
    # We will discuss this once we come to MCMC methods
    integrated_autocorr_time = sampler.get_autocorr_time(quiet=True)
    print("Auto-correlation time of chain:")
    for name, value in zip(param_names, integrated_autocorr_time):
        print(f"{name} = {value:.1f}")

    max_autocorr_time = max(integrated_autocorr_time)

    # We need to discard the beginning of the chain (a few auto-correlation times)
    # to get rid of the initial conditions
    burn_in_config = dict(
        discard=int(5*max_autocorr_time),
        thin=int(max_autocorr_time/2),
        flat=True
    )
    chain = sampler.get_chain(
        **burn_in_config
    )

    # Make predictive distributions
    # Choose a small subsample of the chain for plotting purposes
    chain_samples = chain[np.random.choice(chain.shape[0], size=n_ppd)]
    # Evaluate the model at the sample parameters
    model_predictive = np.array(
        [model_fn(sample, model_x) for sample in chain_samples]
    )
    model_quantiles = np.quantile(
        model_predictive, q=[0.025, 0.16, 0.84, 0.975], axis=0
    )

    posterior_predictive = np.array(
        [predict_fn(sample, x, sigma_y) for sample in chain_samples]
    )
    posterior_predictive_quantiles = np.quantile(
        posterior_predictive, q=[0.025, 0.16, 0.84, 0.975], axis=0
    )

    if ppd_test_statistic_fn is not None:
        T_rep = np.array(
            [ppd_test_statistic_fn(y_rep, sample, x, sigma_y)
             for y_rep, sample in zip(posterior_predictive, chain_samples)]
        )
        T_data = np.array(
            [ppd_test_statistic_fn(y, sample, x, sigma_y)
             for sample in chain_samples]
        )
        goodness_of_fit_pte = np.sum((T_rep - T_data) > 0)/len(T_rep)

        print(f"PPD goodness of fit: {goodness_of_fit_pte:.2g} "
              f"(T_rep = {np.mean(T_rep):.2g}±{np.std(T_rep):.2g}, "
              f"(T_data = {np.mean(T_data):.2g}±{np.std(T_data):.2g})"
        )

    if derived_parameter_fn is not None:
        blobs = sampler.get_blobs(
            **burn_in_config
        )
        if blobs.ndim == 1:
            blobs = blobs.reshape(-1, 1)
        chain = np.concatenate((chain, blobs), axis=-1)

    print("Posterior results (mean±std)")
    for i, name in enumerate(param_names + derived_parameter_names):
        print(f"{name} = {np.mean(chain[:,i]):.2g}±{np.std(chain[:,i]):.2g}")

    if plot:
        # Make a corner plot
        fig = plt.figure()
        fig = corner.corner(
            chain,
            bins=40,
            labels=param_names + derived_parameter_names,
            truths=theta_true + derived_parameter_true,
            levels=1-np.exp(-0.5*np.array([1, 2])**2), # Credible contours corresponding to 1 and 2 sigma in 2D
            quantiles=[0.025, 0.16, 0.84, 0.975],
            fig=fig
        )
        plot_data(
            x=x, y=y, y_err=sigma_y,
            models=[
                dict(x=model_x, y=model_fn(theta_MAP, model_x),
                    style=dict(color="C2", label="MAP model")),
                
                dict(x=model_x, lower=model_quantiles[0], upper=model_quantiles[-1],
                    style=dict(color="C1", alpha=0.5, label="Model predictions")),
                dict(x=model_x, lower=model_quantiles[1], upper=model_quantiles[-2],
                    style=dict(color="C1", alpha=0.5)),

                # dict(x=x, lower=posterior_predictive_quantiles[0], upper=posterior_predictive_quantiles[-1],
                #     style=dict(color="grey", alpha=0.5, label="Posterior predictions")),
                # dict(x=x, lower=posterior_predictive_quantiles[1], upper=posterior_predictive_quantiles[-2],
                #     style=dict(color="grey", alpha=0.5)),
                
                dict(x=x, violin_samples=posterior_predictive,
                     style=dict(color="grey", alpha=0.5, label="Posterior predictions")),
            ] + true_model_plot_def,
            marker=".", linestyle="none"
        )
        if ppd_test_statistic_fn is not None:
            fig, ax = plt.subplots()
            ax.hist(T_rep, density=True, alpha=0.5, label="$T(y_\\mathrm{rep}, \\theta)$")
            ax.hist(T_data, density=True, alpha=0.5, label="$T(y, \\theta)$")
            ax.set_xlabel("$T$")
            ax.legend()

    return dict(
        MAP=theta_MAP,
        PPD=posterior_predictive,
        TPD=model_predictive,
        PPD_params=chain_samples,
        chain=chain
    )