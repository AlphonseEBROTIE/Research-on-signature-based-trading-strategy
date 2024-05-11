# This .py contains useful functions to build our strats.
# Comments :  english


import numpy as np
import torch
import signatory
import itertools
import time
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List


sns.set_style("white")

# function execution_dur : measures a function f_eval execution duration and returns value of f_eval(args)

def execution_dur(f_eval):
    """
    Decorator to time a function, displaying the time it took to run it iff it took more than 1 second.
    """

    def ex_time(*args, **kwargs):
        start = time.time() ; value = f_eval(*args, **kwargs)
        end = time.time() ; dur = end - start
        if dur > 1:
            print(f"function {f_eval.__name__} runtime is {dur:.2f}s")
        return value

    return ex_time


# word processing

# function from_word_to_int : go from word to integer , used for indexing signature objects 

def from_word_to_int(word: List[int], d: int) -> int:
    """
    We define our alphabet as in the paper A_tilde={0, 1, ..., d-1}
    a word w = list of integers with integer n representing (n-1)-th letter of A
    To stay coherent with paper , notations are similar 
    """
    k = len(word)  # the k-th signature term
    add = sum(d**i for i in range(k))  # represent |W_{d}^{k}| , in the paper
    S = 0
    for i, letter in enumerate(word):
        S += letter * d ** (k - 1 - i)
    return add + S


############
def get_words_len_k(k: int, dim_Z: int) -> np.ndarray:
    """
    Let Z = market_factor_process.
    For letters in {0,...,dim(Z)-1} , this method offer all possible words of length k. 
    """
    A_tilde = list(range(dim_Z))
    return list(list(r) for r in itertools.product(A_tilde, repeat=k)) 

    
def get_words_with_length_leq_k(k: int, dim_Z: int) -> np.ndarray:
    """
    Let Z = market_factor_process.
    For letters in {0,...,dim(Z)-1} , this method offer all possible words of length less or equal to k. 
    """
    words= []
    for i in range(k + 1):
        w_len_i = get_words_len_k(i, dim_Z)
        words.append(w_len_i)
        
    w_list = [word for sublist in words for word in sublist]
    return w_list


def get_num_w_k(k: int, dim_Z: int) -> int:
    """
    returns the number of words of length k with letters in A_tilde.
    """
    return dim_Z**k


def get_num_of_w_leq_k(k: int, dim_Z: int) -> int:
    """
    returns the number of words of length less or equal to k with letters in A_tilde.
    """
    return sum(get_num_w_k(i, dim_Z) for i in range(k + 1))



#@execution_dur
#def compute_lead_lag_transform_Hoff(S_t: torch.Tensor):
#    
#    #X_lead = torch.zeros_like(S_t)
#    #X_lag = torch.zeros_like(S_t)
#    
#    S_extended = S_t.repeat_interleave(2, dim=1) 
#    #times = torch.linspace(0, 1, steps=S_extended.shape[1])
#
#    X_lead = torch.zeros_like(S_extended)
#    X_lag = torch.zeros_like(S_extended)
#    N = S_extended.shape[1] // 2
#    
#    for t in range(1,2*N-3):
#        k = t // 2  # indice de la paire (k, k+1)
#
#        # Calcul de X_lead
#        
#        if t >= 2 * k and t <= 2 * k + 1:  # t est pair, donc t ∈ [2k, 2k+1]
#            X_lead[:,t,:] = S_t[:,k + 1,:]
#        else:  # t est impair
#            if t >=  2 * k + 1 and t <= 2 * k + 1.5:
#                X_lead[:,t,:] = S_t[:,k + 1,:] + 2 * (t - (2 * k + 1)) * (S_t[:,k + 2,:] - S_t[:,k + 1,:])
#            elif t >= 2 * k + 1.5 and t < 2 * k + 2:
#                X_lead[:,t,:] = S_t[:,k + 2,:]
#
#        # Calcul de X_lag
#        
#        if t >= 2 * k and t <= 2 * k + 1.5:
#            X_lag[:,t,:] = S_t[:,k,:]
#        elif t >= 2*k + 1.5 and t < 2 * k + 2:
#            X_lag[:,t,:] = S_t[:,k + 1,:] + 2 * (t - (2 * k + 1.5)) * (S_t[:,k + 1,:] - S_t[:,k,:])
#            
#    LL_transformed = torch.cat((X_lead[:, :-3, :], X_lag[:, :-3, :]), dim=2)
#    return LL_transformed #2N points

@execution_dur
def Naive_LL(data: torch.Tensor) -> torch.Tensor:
    
    path_d = data.repeat_interleave(2, dim=1)  
    lead = path_d[:, 1:, :]  
    lag = path_d[:, :-1, :]  

    LL_ok = torch.cat((lead, lag), dim=2)
    return LL_ok


@execution_dur
def get_signature(
    batch_X: torch.Tensor, depth: int, with_batch: bool = False
) -> torch.Tensor:
    """
    Computes the signature of a batch of paths, using package signatory 
    (https://pypi.org/project/signatory/#description)
    """
    if depth == 0:
        if with_batch:
            return torch.ones(1)
        else:
            return torch.ones(batch_X.shape[0], 1)
    if with_batch:
        res_sig = signatory.signature(
            batch_X.unsqueeze(0), depth, scalar_term=True
        ).squeeze(0)
    else:
        res_sig = signatory.signature(batch_X, depth, scalar_term=True)

    return res_sig


def get_shuffle_product(w1, w2):
    """
    Given two words w and v, return the shuffle product w ⧢v.
    """
    
    
    # some useful points (similar to report notations):
    # ua ⧢ vb = (u ⧢ vb)a + (ua ⧢ v)b
    # w1 = ua, w2 = vb
    
    if len(w1) == 0:
        return w2
    if len(w2) == 0:
        return w1

    if len(w1) == 1:
        return [w2[:k] + w1 + w2[k:] for k in range(len(w2) + 1)]
    elif len(w2) == 1:
        return [w1[:k] + w2 + w1[k:] for k in range(len(w1) + 1)]

    else:

        u, a = w1[:-1], w1[-1]
        v, b = w2[:-1], w2[-1]

        shuffle1 = get_shuffle_product(u, w2)  
        term1 = [word + [a] for word in shuffle1]  

        shuffle2 = get_shuffle_product(w1, v)  
        term2 = [word + [b] for word in shuffle2]
        res=term1+term2
        return res  
    
    
def plot_cum_pnl(pnl_cumul: torch.tensor) -> None:
    """
    Will be useful when we'll compute the cumulative PnL of our trading strategy.
    """
    plt.figure(figsize=(8, 6))
    m = pnl_cumul.shape[1] # m = number of asset
    T = pnl_cumul.shape[0]  # number of time steps
    for i in range(m):
        plt.plot(np.arange(T),pnl_cumul[:, i],label=f"cumulative PnL from trading on asset {i}")
    cum_pnl = torch.sum(pnl_cumul, axis=1)
    plt.plot(np.arange(T), cum_pnl, label="cumulative PnL of trading all assets.")
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("cumulative PnL")
    plt.show()
    

def print_signature(flat: torch.tensor, dim_Z: int, depth: int) -> None:
    """
    Returns a kind form of signature 
    Will be useful when we will print functionals l_m , m=1,...,d
    
    """
    N = flat.shape[0]
    s = 0
    for k in range(depth + 1):
        s += dim_Z**k
    assert N == s, "just to make sure the dimensions fit."

    for k in range(depth + 1):
        print(f"Signature level {k}:")
        init_index = get_num_of_w_leq_k(k - 1, dim_Z)
        level_num_k = flat[init_index : init_index + get_num_w_k(k, dim_Z)]
        print(level_num_k)
