import jax.numpy as jnp
import jax_cosmo

from likelihood import PantheonSH0ESLikelihood


class JAXPantheonSH0ESDataModel(PantheonSH0ESLikelihood):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.calibrator_distance = jnp.array(self.calibrator_distance)
        self.z_CMB = jnp.array(self.z_CMB)
        self.z_HEL = jnp.array(self.z_HEL)
        self.covariance = jnp.array(self.covariance)

    def model(self, params):
        # Doesn't matter for background
        Omega_b = 0.05
        Omega_c = params["Omega_m"] - Omega_b
        h = params["H0"] / 100.0
        Omega_k = params.get("Omega_k", 0.0)
        w0 = params.get("w0", -1.0)

        M = params["M"]

        cosmology = jax_cosmo.Cosmology(
            Omega_c=Omega_c, Omega_b=Omega_b, Omega_k=Omega_k,
            h=h, w0=w0, n_s=1.0, sigma8=0.8, wa=0.0
        )

        comoving_angular_diameter_distance = \
            jax_cosmo.background.angular_diameter_distance(
                cosmology, 1.0/(1.0+self.z_CMB)
            ) / h
        prediction = 5.0*jnp.log10((1.0+self.z_CMB)*(1.0+self.z_HEL)*comoving_angular_diameter_distance)+25.0

        return jnp.concatenate([self.calibrator_distance, prediction]) + M

    @property
    def data(self):
        return jnp.concatenate((self.calibrator_magnitude_data, self.magnitude_data))

    @property
    def z(self):
        return jnp.concatenate((self.z_CMB_calibrator, self.z_CMB))

    @property
    def data_error(self):
        return jnp.sqrt(jnp.diag(self.covariance))
