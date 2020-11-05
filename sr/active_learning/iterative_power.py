import sys
import pandas as pd
import numpy as np
import statistics
import random
import scipy.io as io

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

    diversity = pd.read_csv(diversity_file, header = None, names=['filename'] + [str(i) for i in range(0, 32)] )
    sorted_diversity = diversity.sort_values(by='filename')
    diversity_values = sorted_diversity.drop('filename' , axis='columns')
    return diversity_values.to_numpy()

def compute_diversity(diversity_values, utility_values, files):
    '''
    Function to read the encodded diversity values and compute distance between all point pairs
    Includes also creating diversity matrix with distances and the utility
    Args: 
        diversity_values: list of numpys for encoding values
        utility_values: list of utility values
    Returns: the distance-and-utility matrix
    '''
    d_u_matrix = np.zeros((len(utility_values), len(utility_values)))
    for i in range(len(utility_values)):
        for j in range(i+1, len(utility_values)):
            temp = np.sqrt(np.sum(np.square(np.subtract(diversity_values[i],
                                                        diversity_values[j]))))
            d_u_matrix[i][j] = temp
            d_u_matrix[j][i] = temp

    #ind = random.sample(range(0, d_u_matrix.shape[0] - 1), 30)
    #for i in ind:
    #    indices = np.argpartition(d_u_matrix[i], -4)[-4:]
    #    indices_l = np.argpartition(d_u_matrix[i], 5)[:5]
    #    print(files[i], end=' ')
    #    for x in indices:
    #        print(files[x], end=' ')
    #    for y in indices_l:
    #        if y != i:
    #            print(files[y], end=' ')
    #    print('  ')
    maxim = np.max(d_u_matrix)
    # maxim = 5 * np.std(d_u_matrix)
    d_u_matrix = d_u_matrix / maxim
    # maxim = 5 * statistics.stdev(utility_values)
    maxim = np.max(utility_values)
    for i in range(len(utility_values)):
        d_u_matrix[i][i] = utility_scaler * (utility_values[i]/ maxim)
    
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
    x = np.zeros((d_u_matrix.shape[0],1))
    #ind = random.sample(range(0, d_u_matrix.shape[0] - 1), k)
    x[np.array(ind)] = 1
    # power step
    # d_u_matrix = np.matmul(np.transpose(d_u_matrix), d_u_matrix)
    # d_u_matrix = np.array([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]])
    # x = np.array([0, 1, 1, 0, 1, 0])
    # x = x.reshape(6, 1)
    # k = 3

    # print(x.shape)
    s = np.matmul(d_u_matrix, x)
    g = np.multiply(s, 2)
    f = np.matmul(np.transpose(x), s)
    print(f)
    # truncate step
    ind = np.argsort(g, axis=0, kind='stable')[-k:]
    # ind = np.argpartition(g, -k, axis=0)[-k:]
    x_t = np.zeros((d_u_matrix.shape[0],1))
    x_t[np.array(ind)] = 1
    f_old = f
    for i in range(MAX_ITER): 
    
        s_t = np.matmul(d_u_matrix, x_t)
        f_t = np.matmul(np.transpose(x_t), s_t)
        f = f_t
        # print(f"Iteration {i}: Function value: {f}")
        # If there is any non-monotonicity, handle it byt adding lambda * I
        lambda1 = 0.0001
        while(f < f_old - monotonic_tolerance):
            print(f"Fixing monotonicity for f_old: {f_old}, f: {f} lambda {lambda1}")
            g_t = np.add(g, np.multiply(x, 2 * lambda1))
            # ind = np.argpartition(g_t, -k, axis=0)[-k:]
            ind = np.argsort(g_t, axis=0, kind='stable')[-k:]
            x_t = np.zeros((d_u_matrix.shape[0],1))
            x_t[np.array(ind)] = 1
            s_t = np.matmul(d_u_matrix, x_t)
            f_t = np.matmul(np.transpose(x_t), s_t)
            f = f_t
            lambda1 = lambda1 * 10
        # check if already converged and break
        print(f"Iteration {i}: Function value: {f}")
        if abs(f - f_old) < tolerance:
            break
     
        x = x_t
        g = np.multiply(s_t, 2)
        # ind = np.argpartition(g, -k, axis=0)[-k:]
        ind = np.argsort(g, axis=0, kind='stable')[-k:]
        x_t = np.zeros((d_u_matrix.shape[0], 1))
        x_t[np.array(ind)] = 1
        f_old = f

    return np.where(x==1)





if __name__ == "__main__":
    utility_file = sys.argv[1]
    utility_values, files = read_utility(utility_file)
    diversity_file = sys.argv[2]
    percentage = float(sys.argv[3])
    # utility_values = utility_values[:100]
    diversity_values = read_diversity(diversity_file)
    # diversity_values = diversity_values[:100]
    # files = files[:100]
    d_u_matrix = compute_diversity(list(diversity_values), utility_values, files)
    k = int((percentage / 100) * len(utility_values))
    ans = iter_trunc_pow(d_u_matrix, k)
    print(ans[0])

