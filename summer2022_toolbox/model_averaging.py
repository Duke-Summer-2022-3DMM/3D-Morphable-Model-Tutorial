import numpy as np
from tqdm import tqdm
import pickle

def min_distance(S,S_reg):
    """
    calculate the min distance between all S[:,i] and one S_reg[:,j]
    @param S (np.ndarray(3, n)): Source shape
    @param S_reg (np.ndarray(3, 1)): Target point in reference shape
    @return min_index: the index of the closest point to the target point
    """
    distance = S-S_reg.reshape(3,1)
    distance = distance**2
    distance = distance.sum(axis=0)
    distance = np.sqrt(distance)
    distance = distance.tolist()
    min_dis = np.min(distance)
    min_index = distance.index(min_dis)
    return min_index

def alignment(S,S_reg):
    """
    align S with S_reg
    @param S (np.ndarray(3, n_1)): Source shape
    @param S_reg (np.ndarray(3, n_2)): Reference shape
    @return result: indexes of the closest points in S with S_Reg
            S_Reorder: Reordered shape S that has same dimension with S_Reg and full correspondence to S_Reg
    """
    result = []
    S_reorder = np.zeros((3,S_reg.shape[1]))
    for i in range(S_reg.shape[1]):
        index = min_distance(S,S_reg[:,i])
        result.append(index)
        S_reorder[:,i]=S[:,index]
    return result, S_reorder

def reOrder(sample_X, S_reg):
    """
    align all shapes in sample_X with S_reg
    @param sample_X (np.ndarray(n, 3, n_1)): all shapes
    @param S_reg (np.ndarray(3, n_2)): Reference shape
    @return S_tilde_all: save all indexes of the closest points in each S with S_Reg
            S_reorder_all: save all reordered shape S that has same dimension with S_Reg and full correspondence to S_Reg
    """
    S_tilde_all = []
    S_reorder_all = []
    for i in range(len(sample_X)):
    # for i in range(10):
        index, reorder = alignment(sample_X[i],S_reg)
        S_tilde_all.append(index)
        S_reorder_all.append(reorder)
    return S_tilde_all, S_reorder_all

def getMean(S_reorder):
    """
    calculate the average model
    @param S_reorder (np.ndarray(n, 3, n_2)): all reordered shapes
    @return result: the average model
    """
    result = np.zeros((3,S_reorder[0].shape[1]))
    for i in range(len(S_reorder)):
        result += S_reorder[i]
    result = result/len(S_reorder)
    return result


def train_Mean(sample_X, S_reg, saver_name, n = 5, threshold = 20):
    """
    repeatedly calculate average model
    @param sample_X (np.ndarray(n, 3, n_1)): all shapes
    @param S_reg (np.ndarray(3, n_2)): Reference shape
    @param saver_name (str): filename to save results in pickle
    @param n (int): training iterations
    @param threshold (int): threshold to end iteration
    @return: the average model after n iterations
    """
    model_mean = open(saver_name, 'wb')
    result_all = []

    for i in tqdm(range(n)):
        S_tilde, S_reorder = reOrder(sample_X, S_reg)
        S_avg = getMean(S_reorder)
        MSE = ((S_avg.flatten('F') - S_reg.flatten('F'))**2).mean()
        if (MSE < threshold):
            break
        S_reg = S_avg
        result_all.append(S_reg)

    pickle.dump(result_all, model_mean)
    return S_reg


def continue_train_Mean(sample_X, n, saver_name):
    """
    continue to calculate average model repeatedly
    @param sample_X (np.ndarray(n, 3, n_1)): all shapes
    @param S_reg (np.ndarray(3, n_2)): Reference shape
    @param n (int): training iterations
    @param saver_name (str): filename to save results in pickle
    @return: the average model after n iterations
    """
    f1 = open(saver_name, 'rb')
    result_all = pickle.load(f1)
    S_reg = result_all[-1]

    plane_mean = open(saver_name, 'wb')

    remaining_n = n - len(result_all)
    print("Remaining sessions: %d" % (remaining_n))
    for i in tqdm(range(remaining_n)):
        S_tilde, S_reorder = reOrder(sample_X, S_reg)
    S_reg = getMean(S_reorder)
    result_all.append(S_reg)

    pickle.dump(result_all, plane_mean)
    return S_reg