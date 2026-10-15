import scipy.io as sio
from compute_TC_errors import *
import numpy as np
import matplotlib.pyplot as plt
import time

collocations_file = sio.loadmat('collocations_Jur2011_original.mat')
dataset = collocations_file.get('collocations_Jur2011_original')
dataset.shape  # check the matrix dimensions

np.set_printoptions(suppress=True) # to suppress scientific notation (only for numpy arrays)

for i, x in enumerate(dataset[0]):
    print(i+1,x)

# columns of dataset
# 1: buoy ID
# 2: buoy lat (deg)
# 3: buoy lon (deg)
# 4: buoy date (yyyymmdd)
# 5: buoy time (HHx100)
# 6: buoy speed (m/s)
# 7: buoy direction (deg), meteorological convention
# 8: scat lat (deg)
# 9: scat lon (deg)
# 10: scat date (yyyymmdd)
# 11: scat time (HHMMSS)
# 12: scat speed (m/s)
# 13: scat direction (deg), meteorological convention
# 14: nwp speed (m/s)
# 15: nwp direction (deg), meteorological convention


# -------------------------------------------------
# Data extraction (MATLAB → Python indexing)
# -------------------------------------------------

BUOY = dataset[:, 5:7]       # MATLAB 6:7
SCAT = dataset[:, 11:13]     # MATLAB 12:13
ECMWF = dataset[:, 13:15]    # MATLAB 14:15

dataset_reduced = np.hstack((BUOY, SCAT, ECMWF))
dataset_windspeed = dataset_reduced[:, [0, 2, 4]]

# Wind components
u1, v1 = spddir2uv(BUOY[:, 0], BUOY[:, 1])
u2, v2 = spddir2uv(SCAT[:, 0], SCAT[:, 1])
u3, v3 = spddir2uv(ECMWF[:, 0], ECMWF[:, 1])

dataset_u = np.column_stack((u1, u2, u3))
dataset_v = np.column_stack((v1, v2, v3))

# -------------------------------------------------
# Plot wind speeds
# -------------------------------------------------

plt.ion()
plt.figure()
plt.plot(BUOY[:, 0], label="BUOY")
plt.plot(SCAT[:, 0], label="SCAT")
plt.plot(ECMWF[:, 0], label="ECMWF")
plt.xlabel("collocated measurements")
plt.ylabel("wind speed (m/s)")
plt.title("collocated wind speeds")
plt.xlim(0, 100)
plt.ylim(0, 20)
plt.grid(True)
plt.legend()
#plt.show()
#time.sleep(3)
plt.savefig("r2_sensitivity_test_BUOY_ASCAT_ECMWF_windspeeds.png",dpi=200)
#plt.close()
#exit()

# -------------------------------------------------
# r2 sensitivity analysis
# -------------------------------------------------

r2_vector = np.arange(0.0, 2.0 + 0.1, 0.1)
n = len(r2_vector)

tic = time.time()

eps_cov_u_mod = np.zeros((n, 3))
eps_cov_u_scat = np.zeros((n, 3))
cov_u_xx = np.zeros((n, 6))
eps_cov_v_mod = np.zeros((n, 3))
eps_cov_v_scat = np.zeros((n, 3))

for i, r2 in enumerate(r2_vector):
    _, _, _, _, eps_u_scat, eps_v_scat, _, _ = \
        compute_TC_errors(dataset_reduced, r2, r2)

    eps_cov_u_scat[i, :] = eps_u_scat
    eps_cov_v_scat[i, :] = eps_v_scat

# Model-scale correction
eps_cov_u_mod[:, 0] = np.sqrt(eps_cov_u_scat[:, 0]**2 + r2_vector)
eps_cov_u_mod[:, 1] = np.sqrt(eps_cov_u_scat[:, 1]**2 + r2_vector)
eps_cov_u_mod[:, 2] = eps_cov_u_scat[:, 2]

eps_cov_v_mod[:, 0] = np.sqrt(eps_cov_v_scat[:, 0]**2 + r2_vector)
eps_cov_v_mod[:, 1] = np.sqrt(eps_cov_v_scat[:, 1]**2 + r2_vector)
eps_cov_v_mod[:, 2] = eps_cov_v_scat[:, 2]

toc = time.time()

print("elapsed time:",toc-tic,"seconds")
#exit()

# -------------------------------------------------
# Plot sensitivity results
# -------------------------------------------------

plt.ion()

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Left subplot – model scale
axes[0].plot(r2_vector, eps_cov_u_mod, marker='o')
axes[0].set_xlabel('$r^2$ (m$^2$/s$^2$)')
axes[0].set_ylabel('$\\mathrm{err}_u$ (m/s)')
axes[0].set_title('TC errors (new covariances formulation)\n\nat the model scale')
axes[0].legend(['Buoy', 'ASCAT 12.5', 'ECMWF'], loc='upper left')
axes[0].grid(True)
axes[0].set_ylim(0, 2)
axes[0].set_yticks(np.arange(0, 2.1, 0.1))

# Right subplot – scatterometer scale
axes[1].plot(r2_vector, eps_cov_u_scat, marker='o')
axes[1].set_xlabel('$r^2$ (m$^2$/s$^2$)')
axes[1].set_ylabel('$\\mathrm{err}_u$ (m/s)')
axes[1].set_title('\n\nat the scatterometer scale')
axes[1].legend(['Buoy', 'ASCAT 12.5', 'ECMWF'], loc='upper left')
axes[1].grid(True)
axes[1].set_ylim(0, 2)
axes[1].set_yticks(np.arange(0, 2.1, 0.1))

#plt.show()

plt.savefig("r2_sensitivity_test_buoy_ASCAT_ECMWF_new_covariances_dataset_u_notightlayout.png",dpi=200)
plt.tight_layout()
plt.savefig("r2_sensitivity_test_buoy_ASCAT_ECMWF_new_covariances_dataset_u.png",dpi=200)
plt.close()