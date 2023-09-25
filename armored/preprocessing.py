import numpy as np
import pandas as pd
import jax.numpy as jnp

def format_data(df, species, metabolites, controls, obj_params=None, t_eval=None):
    '''
    Format data so that all experiments have same time length and time steps with
    NaNs filled in missing entries

    df is a dataframe with columns
    ['Experiments', 'Time', 'S_1', ..., 'S_ns', 'M_1', ..., 'M_nm', 'U_1', ..., 'U_nu']

    species := 'S_1', ..., 'S_ns'
    metabolites := 'M_1', ..., 'M_nm'
    controls := 'U_1', ..., 'U_nu'

    '''
    # concatenate all sytem variable names
    sys_vars = np.concatenate((species, metabolites, controls))

    # get experiment names
    experiments = df.Experiments.values

    # get unique experiments and number of time measurements
    unique_exps, counts = np.unique(experiments, return_counts=True)

    # determine time vector corresponding to longest sampled experiment
    if t_eval is None:
        exp_longest = unique_exps[np.argmax(counts)]
        exp_longest_inds = np.in1d(experiments, exp_longest)
        t_eval = df.iloc[exp_longest_inds]['Time'].values

    # initialize data matrix with NaNs
    D = np.empty([len(unique_exps), len(t_eval), len(sys_vars)])
    D[:] = np.nan

    # initialize matrix of objective parameters if provided
    if obj_params is not None:
        P = np.zeros([len(unique_exps), len(t_eval), len(obj_params)])

    # fill in data for each experiment
    exp_names = []
    N = 0
    effective_dimension = 0
    dimension = 0
    for i,exp in enumerate(unique_exps):
        exp_inds  = np.in1d(experiments, exp)
        comm_data = df.copy()[exp_inds]

        # count number of samples
        N += len(t_eval[1:])
        # pull species data
        Y_species = jnp.array(comm_data[species].values, float)
        # count effective dimension
        effective_dimension += np.sum(Y_species[0] > 0.)
        dimension += Y_species.shape[1]

        # store data
        exp_time = comm_data['Time'].values
        sampling_inds = np.in1d(t_eval, exp_time)
        D[i][sampling_inds] = comm_data[sys_vars].values
        exp_names += [exp]*len(t_eval)

        # store objective params
        if obj_params is not None:
            P[i][sampling_inds] = comm_data[obj_params].values

    # compute N, accounting for missing species
    N *= effective_dimension / dimension

    if obj_params is not None:
        return jnp.array(D), jnp.array(P), unique_exps, N
    else:
        return jnp.array(D), unique_exps, N

# define scaling functions
class ZeroMaxScaler():

    def __init__(self):
        pass

    def fit(self, X):
        # X has dimensions: (N_experiments, N_timepoints, N_variables)
        self.X_min = 0.
        self.X_max = np.nanmax(X, axis=0)
        self.X_range = self.X_max
        self.X_range[self.X_range==0.] = 1.
        return self

    def transform(self, X):
        # convert to 0-1 scale
        X_scaled = X / self.X_range[:,:X.shape[-1]]
        return X_scaled

    def inverse_transform(self, X_scaled):
        X = X_scaled*self.X_range[:,:X_scaled.shape[-1]]
        return X

    def inverse_transform_stdv(self, X_scaled):
        X = X_scaled*self.X_range[:,:X_scaled.shape[-1]]
        return X

    def inverse_transform_cov(self, COV_scaled):
        # transform covariance of scaled values to original scale based on:
        # var ( A x ) = A var (x) A^T
        # where COV_scaled is the var (x) and A is diag( range(x) )
        return np.einsum('tk,ntkl,tl->ntkl',self.X_range[1:], COV_scaled, self.X_range[1:])
