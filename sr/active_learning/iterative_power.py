"""Usage:   iterative_power.py --utility_file=utility_file --diversity_file=diversity_file --percentage=percentage
            iterative_power.py --help | -help | -h

Use shte utility value sand the diversity values in order to compute the best active learning samples
Arguments:
  utility_file      a file that contains utility values for the samples
  diversity_file    a file name that contains diversity for the smaples
  percentage        percentage of samples to select

Options:
  -h --help -h
"""
import time
import statistics
import random
from docopt import docopt
from pathlib import Path
import pandas as pd
import numpy as np
from frechetdist import frdist
from scipy.spatial.distance import directed_hausdorff

utility_scaler = 1
tolerance = 0.000001
monotonic_tolerance = 0.0000000001
MAX_ITER = 50

def read_utility(utility_file):
    '''
    Function to read the utility file and return a list containing utility values
    Args: utility file name
    Returns: list of utilities sorted by file name
    '''
    utility = pd.read_csv(utility_file)
    sorted_utility = utility.sort_values(by='filename')
    return sorted_utility['SR_L1'].to_list(), sorted_utility['filename'].to_list()

def read_diversity(diversity_file):
    '''
    Function to read the diversity file and return a list of numpys for diversity values
    Args: diversity file name
    Returns: list of diversities sorted by file name
    '''

    diversity = pd.read_csv(diversity_file, header=None,
                            names=['filename'] + [str(i) for i in range(0, 128)])
    sorted_diversity = diversity.sort_values(by='filename')
    diversity_values = sorted_diversity.drop('filename', axis='columns')
    return diversity_values.to_numpy()

def compute_diversity(diversity_values, utility_values, files):
    '''
    Function to read the encodded diversity
    values and compute distance between all point pairs
    Includes also creating diversity matrix with distances and the utility
    Args:
        diversity_values: list of numpys for encoding values
        utility_values: list of utility values
    Returns: the distance-and-utility matrix
    '''
    d_u_matrix = np.zeros((len(utility_values), len(utility_values)))
    start = time.time()
    for i in range(len(utility_values)):
        for j in range(i + 1, len(utility_values)):
            u = np.array(diversity_values[4*i: 4*(i+1)])
            v = np.array(diversity_values[4*j: 4*(j+1)])
            temp = max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
            d_u_matrix[i][j] = temp
            d_u_matrix[j][i] = temp
    print(f"Distance computation time {time.time() - start}")
    maxim = np.max(d_u_matrix)
    # maxim = 5 * np.std(d_u_matrix)
    d_u_matrix = d_u_matrix / maxim
    # maxim = 5 * statistics.stdev(utility_values)
    maxim = np.max(utility_values)
    for i in range(len(utility_values)):
        d_u_matrix[i][i] = utility_scaler * (utility_values[i] / maxim)

    # io.savemat("W.mat", {"W": d_u_matrix})
    return d_u_matrix

def iter_trunc_pow(d_u_matrix, k):
    '''
    Function to use the iterative truncated power algo
    Args:
        d_u_matrix: the diversity-and-utility matrix (W)
        k: number of points to select
    Returns: Indeces of k selected points
    '''

    # Selecting the initial solution
    sum_matrix = np.sum(d_u_matrix, axis=1)
    ind = np.argpartition(sum_matrix, -k)[-k:]
    x = np.zeros((d_u_matrix.shape[0], 1))
    # ind = random.sample(range(0, d_u_matrix.shape[0] - 1), k)
    x[np.array(ind)] = 1
    # power step
    s = np.matmul(d_u_matrix, x)
    g = np.multiply(s, 2)
    f = np.matmul(np.transpose(x), s)
    print(f"Iteration -1: Function value: {f}")
    # truncate step
    ind = np.argsort(g, axis=0, kind='stable')[-k:]
    # ind = np.argpartition(g, -k, axis=0)[-k:]
    x_t = np.zeros((d_u_matrix.shape[0], 1))
    x_t[np.array(ind)] = 1
    f_old = f
    for i in range(MAX_ITER):
        s_t = np.matmul(d_u_matrix, x_t)
        f_t = np.matmul(np.transpose(x_t), s_t)
        f = f_t
        # If there is any non-monotonicity, handle it byt adding lambda * I
        lambda1 = 0.0001
        while f < f_old - monotonic_tolerance:
            print(f"Fixing monotonicity for f_old: {f_old}, f: {f} lambda {lambda1}")
            g_t = np.add(g, np.multiply(x, 2 * lambda1))
            # ind = np.argpartition(g_t, -k, axis=0)[-k:]
            ind = np.argsort(g_t, axis=0, kind='stable')[-k:]
            x_t = np.zeros((d_u_matrix.shape[0], 1))
            x_t[np.array(ind)] = 1
            s_t = np.matmul(d_u_matrix, x_t)
            f_t = np.matmul(np.transpose(x_t), s_t)
            f = f_t
            lambda1 = lambda1 * 10
        print(f"Iteration {i}: Function value: {f}")
        # check if already converged and break
        if abs(f - f_old) < tolerance:
            break
        #Seelcting new x_t
        x = x_t
        g = np.multiply(s_t, 2)
        # ind = np.argpartition(g, -k, axis=0)[-k:]
        ind = np.argsort(g, axis=0, kind='stable')[-k:]
        x_t = np.zeros((d_u_matrix.shape[0], 1))
        x_t[np.array(ind)] = 1
        f_old = f

    return np.where(x == 1)


if __name__ == "__main__":
    args = docopt(__doc__)
    utility_file = Path(args["--utility_file"]).resolve()
    diversity_file = Path(args["--diversity_file"]).resolve()
    utility_values, files = read_utility(utility_file)
    percentage = float(args["--percentage"])
    diversity_values = read_diversity(diversity_file)
    d_u_matrix = compute_diversity(list(diversity_values), utility_values, files)
    k = int((percentage / 100) * len(utility_values))
    ans = iter_trunc_pow(d_u_matrix, k)
    print(ans[0])
