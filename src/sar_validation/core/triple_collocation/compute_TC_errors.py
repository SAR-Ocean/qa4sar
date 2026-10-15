import numpy as np
import warnings

def spddir2uv(spd, wind_dir):
    """
    Convert wind speed and direction to u and v components.
    Meteorological convention: direction is where the wind comes FROM.
    0 deg = North, 90 deg = East.
    """
    rad = np.deg2rad(wind_dir)
    u = -spd * np.sin(rad)
    v = -spd * np.cos(rad)
    return u, v

def uv2spddir(u, v):
    """
    Convert u and v components to wind speed and direction.
    """
    spd = np.sqrt(u**2 + v**2)
    wind_dir = np.rad2deg(np.arctan2(-u, -v))
    # Ensure direction is in [0, 360)
    wind_dir = np.mod(wind_dir, 360)
    return spd, wind_dir

def compute_TC_errors(dataset, r2u, r2v, N_iter=10):
    """
    Compute triple collocation calibration coefficients and errors using the covariance formulation
    described in Vogelzang and Stoffelen (2021).
    
    input parameters:
    - dataset: dataset of buoy, ASCAT and ECMWF collocations (wind speed and wind direction pairs)
    - r2u: representativeness error in u
    - r2v: representativeness error in v
    - N_iter: maximum number of iterations
    
    output parameters:
    - a_u: scaling coefficient in u
    - a_v: scaling coefficient in v
    - b_u: bias coefficient in u
    - b_v: bias coefficient in v
    - eps_u: measurement errors in u for buoy, ASCAT and ECMWF
    - eps_v: measurement errors in v for buoy, ASCAT and ECMWF
    - dataset_cal: calibrated dataset
    - valid_values: boolean array of valid values
    """

    sigma_factor = 4.0  # for sigma test
    CF = 1e-6          # convergence factor
    convergence = False # check if convergence is reached
    verbose = False

    N = dataset.shape[0]
    dataset_cal = np.full((N, 6), np.nan)
    var_u = np.zeros(3)
    var_v = np.zeros(3)
    s2u = np.zeros(3)
    s2v = np.zeros(3)
    da_u = np.ones(3)
    da_v = np.ones(3)
    db_u = np.zeros(3)
    db_v = np.zeros(3)
    a_u = np.ones(3)
    a_v = np.ones(3)
    b_u = np.zeros(3)
    b_v = np.zeros(3)
    d2u = np.zeros(3)
    d2v = np.zeros(3)

    # Convert input speed and direction to U and V components
    u1, v1 = spddir2uv(dataset[:, 0], dataset[:, 1])
    u2, v2 = spddir2uv(dataset[:, 2], dataset[:, 3])
    u3, v3 = spddir2uv(dataset[:, 4], dataset[:, 5])
    u_in = np.column_stack((u1, u2, u3))
    v_in = np.column_stack((v1, v2, v3))

    valid_values = np.ones(N, dtype=bool)

    for i in range(N_iter):
        
        if verbose:
            print('calibration coefficients:')
            print(a_u)
            print(b_u)
        
        rejected = 0
        valid_values = np.ones(N, dtype=bool)
        
        # calibrate winds
        u_cal = (u_in - b_u) / a_u
        v_cal = (v_in - b_v) / a_v
        
        if verbose:
            print(f'u_cal(0, :): {u_cal[0, :]}')
        
        # update sigma test factors (using sample variance, ddof=1 matches Matlab's var)
        s2u[0] = np.var(u_cal[:, 0] - u_cal[:, 1], ddof=1)
        s2u[1] = np.var(u_cal[:, 1] - u_cal[:, 2], ddof=1)
        s2u[2] = np.var(u_cal[:, 2] - u_cal[:, 0], ddof=1)
        s2v[0] = np.var(v_cal[:, 0] - v_cal[:, 1], ddof=1)
        s2v[1] = np.var(v_cal[:, 1] - v_cal[:, 2], ddof=1)
        s2v[2] = np.var(v_cal[:, 2] - v_cal[:, 0], ddof=1)
        
        if verbose:
            print(f'sigma test variances: {s2u}')
        
        # sigma test (separated u and v)
        d2maxu = (sigma_factor**2) * s2u
        d2maxv = (sigma_factor**2) * s2v
        
        for k in range(N):
            for j in range(3):
                j1 = (j + 1) % 3
                d2u[j] = (u_cal[k, j] - u_cal[k, j1])**2
                d2v[j] = (v_cal[k, j] - v_cal[k, j1])**2
                
                if d2u[j] > d2maxu[j]:
                    rejected += 1
                    valid_values[k] = False
                    break
                
                if d2v[j] > d2maxv[j]:
                    rejected += 1
                    valid_values[k] = False
                    break
        
        if verbose:
            print(f'accepted & rejected: {np.sum(valid_values)} {rejected}')
            input("Press Enter to continue...")
        
        # select valid values
        u_cal_valid = u_cal[valid_values, :]
        v_cal_valid = v_cal[valid_values, :]
        
        # compute the moments and the covariances
        Mu_1 = np.mean(u_cal_valid[:, 0])
        Mu_2 = np.mean(u_cal_valid[:, 1])
        Mu_3 = np.mean(u_cal_valid[:, 2])
        
        Mu_11 = np.mean(u_cal_valid[:, 0] * u_cal_valid[:, 0])
        Mu_22 = np.mean(u_cal_valid[:, 1] * u_cal_valid[:, 1])
        Mu_33 = np.mean(u_cal_valid[:, 2] * u_cal_valid[:, 2])
        Mu_12 = np.mean(u_cal_valid[:, 0] * u_cal_valid[:, 1])
        Mu_13 = np.mean(u_cal_valid[:, 0] * u_cal_valid[:, 2])
        Mu_23 = np.mean(u_cal_valid[:, 1] * u_cal_valid[:, 2])
        
        Cu_11 = Mu_11 - Mu_1**2
        Cu_22 = Mu_22 - Mu_2**2
        Cu_33 = Mu_33 - Mu_3**2
        Cu_12 = Mu_12 - Mu_1 * Mu_2
        Cu_13 = Mu_13 - Mu_1 * Mu_3
        Cu_23 = Mu_23 - Mu_2 * Mu_3
        
        Mv_1 = np.mean(v_cal_valid[:, 0])
        Mv_2 = np.mean(v_cal_valid[:, 1])
        Mv_3 = np.mean(v_cal_valid[:, 2])
        Mv_11 = np.mean(v_cal_valid[:, 0] * v_cal_valid[:, 0])
        Mv_22 = np.mean(v_cal_valid[:, 1] * v_cal_valid[:, 1])
        Mv_33 = np.mean(v_cal_valid[:, 2] * v_cal_valid[:, 2])
        Mv_12 = np.mean(v_cal_valid[:, 0] * v_cal_valid[:, 1])
        Mv_13 = np.mean(v_cal_valid[:, 0] * v_cal_valid[:, 2])
        Mv_23 = np.mean(v_cal_valid[:, 1] * v_cal_valid[:, 2])
        
        Cv_11 = Mv_11 - Mv_1**2
        Cv_22 = Mv_22 - Mv_2**2
        Cv_33 = Mv_33 - Mv_3**2
        Cv_12 = Mv_12 - Mv_1 * Mv_2
        Cv_13 = Mv_13 - Mv_1 * Mv_3
        Cv_23 = Mv_23 - Mv_2 * Mv_3
        
        # compute the updates to the calibration scaling and to the calibration bias
        da_u[0] = 1.0
        da_u[1] = Cu_23 / Cu_13
        da_u[2] = (Cu_13 * Cu_23) / (Cu_12 * Cu_13 - Cu_23 * r2u)
        
        db_u[0] = 0.0
        db_u[1] = Mu_2 - da_u[1] * Mu_1
        db_u[2] = Mu_3 - da_u[2] * Mu_1
        
        da_v[0] = 1.0
        da_v[1] = Cv_23 / Cv_13
        da_v[2] = (Cv_13 * Cv_23) / (Cv_12 * Cv_13 - Cv_23 * r2v)
        
        db_v[0] = 0.0
        db_v[1] = Mv_2 - da_v[1] * Mv_1
        db_v[2] = Mv_3 - da_v[2] * Mv_1
        
        # update the calibration scaling and the calibration bias
        b_u = b_u + a_u * db_u
        a_u = a_u * da_u
        b_v = b_v + a_v * db_v
        a_v = a_v * da_v
        
        # check convergence
        if (np.all(np.abs(1.0 - da_u) < CF) and np.all(np.abs(db_u) < CF) and 
            np.all(np.abs(1.0 - da_v) < CF) and np.all(np.abs(db_v) < CF)):
            if verbose:
                print(f'accepted & rejected: {np.sum(valid_values)} {rejected}')
            convergence = True
            break

    if not convergence:
        warnings.warn(f"Convergence not reached after {N_iter} iterations.")

    # compute the error variances and standard deviations at the scale of the scatterometer
    var_u[0] = Cu_11 - Cu_12 * Cu_13 / Cu_23
    var_u[1] = (Cu_13 / Cu_23) * ((Cu_13 * Cu_22 / Cu_23) - Cu_12)
    var_u[2] = (Cu_12 / Cu_23) * (Cu_12 * Cu_33 / Cu_23 - Cu_13) + \
               (1 - 2 * Cu_12 * Cu_33 / (Cu_13 * Cu_23)) * r2u + \
               (Cu_33 / Cu_13**2) * r2u**2
               
    eps_u = np.sqrt(np.maximum(var_u, 0)) # Max with 0 to prevent issues with negative numerical errors

    var_v[0] = Cv_11 - Cv_12 * Cv_13 / Cv_23
    var_v[1] = (Cv_13 / Cv_23) * (Cv_13 * Cv_22 / Cv_23 - Cv_12)
    var_v[2] = (Cv_12 / Cv_23) * (Cv_12 * Cv_33 / Cv_23 - Cv_13) + \
               (1 - 2 * Cv_12 * Cv_33 / (Cv_13 * Cv_23)) * r2v + \
               (Cv_33 / Cv_13**2) * r2v**2
               
    eps_v = np.sqrt(np.maximum(var_v, 0))

    # Final calibration
    u_cal = (u_in - b_u) / a_u
    v_cal = (v_in - b_v) / a_v
    
    # Back to speed and direction for each collocation pair
    SPD1, DIR1 = uv2spddir(u_cal[:, 0], v_cal[:, 0])
    SPD2, DIR2 = uv2spddir(u_cal[:, 1], v_cal[:, 1])
    SPD3, DIR3 = uv2spddir(u_cal[:, 2], v_cal[:, 2])
    
    dataset_cal[:, 0] = SPD1
    dataset_cal[:, 1] = DIR1
    dataset_cal[:, 2] = SPD2
    dataset_cal[:, 3] = DIR2
    dataset_cal[:, 4] = SPD3
    dataset_cal[:, 5] = DIR3
    
    return a_u, a_v, b_u, b_v, eps_u, eps_v, dataset_cal, valid_values
