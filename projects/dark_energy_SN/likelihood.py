import numpy as np
import pandas as pd

import astropy.cosmology

# # Default is to use SN data from https://arxiv.org/abs/2202.04077
# # and Cepheids data from https://arxiv.org/abs/2112.04510
# default_data_file = https://github.com/PantheonPlusSH0ES/DataRelease/blob/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat
# default_covmat_file = https://github.com/PantheonPlusSH0ES/DataRelease/blob/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES_STAT%2BSYS.cov


class PantheonSH0ESLikelihood:
    """Likelihood for the Pantheon+SH0ES data set, based on the CosmoSIS
    likelihood https://github.com/PantheonPlusSH0ES/DataRelease/blob/main/Pantheon%2B_Data/5_COSMOLOGY/cosmosis_likelihoods/Pantheon%2BSH0ES_cosmosis_likelihood.py
    """
    def __init__(self, data_file_name=None, raw_data_file_name=None, raw_covariance_file_name=None):
        if raw_data_file_name is not None and raw_covariance_file_name is not None:
            self.load_raw_data_and_covariance(
                data_file_name=raw_data_file_name,
                covariance_file_name=raw_covariance_file_name
            )
            if data_file_name is not None:
                print(f"Writing loaded data to {data_file_name}")
                np.savez_compressed(
                    data_file_name,
                    magnitude_data=self.magnitude_data,
                    calibrator_magnitude_data=self.calibrator_magnitude_data,
                    calibrator_distance=self.calibrator_distance,
                    is_calibrator=self.is_calibrator,
                    z_CMB_calibrator=self.z_CMB_calibrator,
                    z_CMB=self.z_CMB,
                    z_HEL=self.z_HEL,
                    covariance_cholesky=self.covariance_cholesky.astype(dtype=np.float32),
                )
        elif data_file_name is not None:
            data = np.load(data_file_name)
            self.magnitude_data = data["magnitude_data"]
            self.calibrator_magnitude_data = data["calibrator_magnitude_data"]
            self.calibrator_distance = data["calibrator_distance"]
            self.is_calibrator = data["is_calibrator"]
            self.z_CMB_calibrator = data["z_CMB_calibrator"]
            self.z_CMB = data["z_CMB"]
            self.z_HEL = data["z_HEL"]
            self.covariance_cholesky = data["covariance_cholesky"]
            self.covariance = self.covariance_cholesky @ self.covariance_cholesky.T
            self.inverse_covariance = np.linalg.inv(self.covariance)

        self.magnitude_data_error = np.sqrt(np.diag(self.covariance))[~self.is_calibrator]
        self.calibrator_magnitude_data_error = np.sqrt(np.diag(self.covariance))[self.is_calibrator]
        self.covariance_no_calibrator = self.covariance[np.ix_(~self.is_calibrator, ~self.is_calibrator)]
        self.inverse_covariance_no_calibrator = np.linalg.inv(self.covariance_no_calibrator)

    def load_raw_data_and_covariance(self, data_file_name, covariance_file_name):
        data = pd.read_csv(data_file_name, delim_whitespace=True)
        origlen = len(data)

        ww = (data['zHD'] > 0.01) | (np.array(data['IS_CALIBRATOR'], dtype=bool))

        z_CMB = np.array(data['zHD'][ww]) # use the vpec corrected redshift for zCMB 
        z_HEL = np.array(data['zHEL'][ww])
        m_obs = np.array(data['m_b_corr'][ww])

        self.is_calibrator = np.array(data['IS_CALIBRATOR'][ww], dtype=bool)

        self.magnitude_data = m_obs[~self.is_calibrator]
        self.calibrator_magnitude_data = m_obs[self.is_calibrator]

        self.calibrator_distance = np.array(data['CEPH_DIST'][ww][self.is_calibrator])

        self.z_CMB_calibrator = z_CMB[self.is_calibrator]
        self.z_CMB = z_CMB[~self.is_calibrator]
        self.z_HEL = z_HEL[~self.is_calibrator]

        with open(covariance_file_name, "r") as f:
            _ = f.readline()
            n = int(len(z_CMB))
            covariance = np.zeros((n,n))
            ii = -1
            jj = -1
            for i in range(origlen):
                jj = -1
                if ww[i]:
                    ii += 1
                for j in range(origlen):
                    if ww[j]:
                        jj += 1
                    val = float(f.readline())
                    if ww[i]:
                        if ww[j]:
                            covariance[ii,jj] = val

        calib_idx = np.concatenate(np.where(self.is_calibrator)+np.where(~self.is_calibrator))
        self.covariance = covariance[np.ix_(calib_idx, calib_idx)]
        self.inverse_covariance = np.linalg.inv(self.covariance)
        self.covariance_cholesky = np.linalg.cholesky(self.covariance)

        data = np.concatenate((self.calibrator_magnitude_data, self.magnitude_data))
        assert np.isclose(m_obs @ np.linalg.inv(covariance) @ m_obs, data @ self.inverse_covariance @ data)

    def predict_magnitudes(self, H_0, Omega_m, Omega_de=None, M=-19.24):
        if Omega_de is None:
            cosmology = astropy.cosmology.FlatLambdaCDM(H0=H_0, Om0=Omega_m)
        else:
            cosmology = astropy.cosmology.LambdaCDM(H0=H_0, Om0=Omega_m, Ode0=Omega_de)

        comoving_angular_diameter_distance = cosmology.angular_diameter_distance(self.z_CMB)
        comoving_angular_diameter_distance = (comoving_angular_diameter_distance/astropy.units.Mpc).value

        prediction = 5.0*np.log10((1.0+self.z_CMB)*(1.0+self.z_HEL)*comoving_angular_diameter_distance)+25.
      
        # Add the absolute supernova magnitude and return
        prediction = prediction + M
        return prediction

    def predict_calibrator_magnitudes(self, M=-19.24):
        return self.calibrator_distance + M

    def model(self, H_0, Omega_m, Omega_de, M):
        model_calibrator = self.predict_calibrator_magnitudes(M)
        model = self.predict_magnitudes(H_0, Omega_m, Omega_de, M)

        return np.concatenate((model_calibrator, model))

    def data(self):
        return np.concatenate((self.calibrator_magnitude_data, self.magnitude_data))

    def log_likelihood(self, params):
        H_0, Omega_m, Omega_de, M = params

        prediction = self.model(H_0=H_0, Omega_m=Omega_m, Omega_de=Omega_de, M=M)
        data = self.data()
        residual = data - prediction

        chi2 = residual @ self.inverse_covariance @ residual
        return -0.5*chi2
