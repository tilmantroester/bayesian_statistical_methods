import numpy as np
import scipy.interpolate


pk_lib = {"BAHAMAS logT 7.6": {"pk_file": "BAHAMAS_logT_7.6/powtable_BAHAMAS_Theat7.6_nu0_WMAP9.dat",
                               "pk_dmo_file": "BAHAMAS_logT_7.6/powtable_DMONLY_2fluid_nu0_WMAP9_L400N1024.dat"},
          "BAHAMAS logT 7.8": {"pk_file": "BAHAMAS_logT_7.8/powtable_BAHAMAS_nu0_WMAP9.dat",
                               "pk_dmo_file": "BAHAMAS_logT_7.8/powtable_DMONLY_2fluid_nu0_WMAP9_L400N1024.dat"},
          "BAHAMAS logT 8.0": {"pk_file": "BAHAMAS_logT_8.0/powtable_BAHAMAS_Theat8.0_nu0_WMAP9.dat",
                               "pk_dmo_file": "BAHAMAS_logT_8.0/powtable_DMONLY_2fluid_nu0_WMAP9_L400N1024.dat"},
          "EAGLE": {"pk_file": "EAGLE/powtable_EAGLE_REF.dat",
                               "pk_dmo_file": "EAGLE/powtable_EAGLE_DMONLY_L100N1504.dat"},
          "COSMO-OWLS AGN": {"pk_file": "C-OWLS/powtable_C-OWLS_AGN_WMAP7.dat",
                             "pk_dmo_file": "C-OWLS/powtable_DMONLY_WMAP7_L400N1024.dat"},
          "COSMO-OWLS AGN logT 8.7 Planck": {"pk_file": "C-OWLS_AGN_Theat8.7_Planck2013/powtable_C-OWLS_AGN_Theat8.7_Planck2013.dat",
                                             "pk_dmo_file": "C-OWLS_AGN_Theat8.7_Planck2013/powtable_DMONLY_Planck2013_L400N1024.dat"},
          "COSMO-OWLS REF": {"pk_file": "C-OWLS_REF_WMAP7/powtable_C-OWLS_REF_WMAP7.dat",
                             "pk_dmo_file": "C-OWLS_REF_WMAP7/powtable_DMONLY_WMAP7_L400N1024.dat"},
          "Illustris": {"pk_file": "Illustris/powtable_Illustris-1.dat",
                        "pk_dmo_file": "Illustris/powtable_Illustris-1-DM.dat"},
          "TNG300": {"pk_file": "TNG300/powtable_TNG300-1.dat",
                     "pk_dmo_file": "TNG300/powtable_TNG300-1-DM.dat"},
          "Horizon-AGN": {"pk_file": "Horizon-AGN/powtable_Hz-AGN.dat",
                          "pk_dmo_file": "Horizon-AGN/powtable_Hz-DM.dat"},
          "SIMBA": {"pk_file": "SIMBA/powtable_SIMBA.dat",
                    "pk_dmo_file": "SIMBA/powtable_SIMBA_DM_L100N1024.dat"},
          }
                
z = 0.0

min_k = 1e-6
max_k = 1e3
for name, pk_def in pk_lib.items():
    d = np.loadtxt(pk_def["pk_file"])
    m = d[:, 0] == z
    print(name, m.sum())
    pk_lib[name]["pk"] = d[m, 2]
    pk_lib[name]["k"] = d[m, 1]
    pk_lib[name]["pk_dmo"] = np.loadtxt(pk_def["pk_dmo_file"], usecols=[2,])[m]

    min_k = max(min_k, min(pk_lib[name]["k"]))
    max_k = min(max_k, max(pk_lib[name]["k"]))

print("k-range:", min_k, max_k)

k = np.geomspace(min_k, max_k, 50)
r = {}
for name in pk_lib.keys():
    intp = scipy.interpolate.InterpolatedUnivariateSpline(
        x=pk_lib[name]["k"],
        y=pk_lib[name]["pk"]/pk_lib[name]["pk_dmo"])
    r[name] = intp(k)

np.savetxt(
    "power_spectrum_suppression.txt", 
    np.vstack((k, np.array(list(r.values())))).T,
    header="k, " + ", ".join(r.keys()) +"\n From https://powerlib.strw.leidenuniv.nl, http://arxiv.org/abs/1906.00968"
)
