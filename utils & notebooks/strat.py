# Foundations for designing our signature-based trading strategy
# In line with the paper and report

import useful
import numpy as np
import torch
import signatory
import itertools
from useful import execution_dur
from time import time
from typing import List


class Signature_Trading(object):

    """
    Here is the implementation of the Signature Trading strategy as defined in the paper , 
    ''Signature Trading:A Path-Dependent Extension of the Mean-Variance Framework with Exogenous Signals'' by Futter et al.
    
    our main paper
    
    Arguments:
    
    - depth (int) : truncation level for the signature
    - delta (float) : the maximum risk the trader is willing to take or the risk aversion parameter     
    """
# Notations conform to the original paper

    def __init__(self, depth: int, delta: float) -> None:
        
        self.depth = depth
        self.delta = delta
        
        # the following elements are initialized here but will be updated during fit :
        
        self.l = None  # functionals l_m for m = 1, ..., d
        self.d = None  # number of tradable assets as defined in the paper (will be set when fitting)
        self.N = None  # number of non-tradable factors (will be set when fitting)
        self.Z_dimension = None  # dimension of Z_{t} = (t, X_t, f_t), egal to d + N + 1 in fitting phase
        # self.Z_dimension = number of letters in alphabet= A_tilde
        
        self.mu_sig = None
        self.sigma_sig = None

    def f(self, m: int) -> List[int]:
        """
        Shift operator \mathcal{f} as defined in the paper
        """
        return [self.Z_dimension + m + 1]  

    @execution_dur
    def compute_mu_sig(self, E_ZZ_LL: torch.Tensor) -> None:
        """
        mu^{sig}  vector computation as defined in the paper
        """
        ##
        # 0. mu_sig initialization
        # we want : mu^{sig} = [mu_{1}^{sig}, ..., mu_{d}^{sig}]
        
        mu_i_sig_length = useful.get_num_of_w_leq_k(
            self.depth, self.Z_dimension
        )  
        mu_sig = torch.zeros(
            self.d * mu_i_sig_length
        )  
        
        # 1. stock all words of length <= depth in list 
        
        w = useful.get_words_with_length_leq_k(self.depth, self.Z_dimension)
        w_len = len(w)

        # 2. compute effective mu_sig
        
        for m in range(self.d):  
            fm = self.f(m)  #  f(m) in the paper
            for word in w:
                #  mu_sig[wfm] (see 3.2 in  paper)

                w_shifted = word + fm  # represents the word wf(m) in the paper

                # integer index of word_shifted in our mu_sig
                
                mu_sig_index = m * w_len + useful.from_word_to_int(
                    word, self.Z_dimension 
                )

                # int. index of word_shifted in E_ZZ_LL
                E_ZZ_LL_index = useful.from_word_to_int(
                    w_shifted, 2 * self.Z_dimension
                ) 

                mu_sig[mu_sig_index] = E_ZZ_LL[
                    E_ZZ_LL_index
                ] 
                
        print("mu_sig is ok" )
        self.mu_sig = mu_sig

    # after mu_sig , it's time for sigma_{sig}
    
    @execution_dur
    def compute_sigma_sig(self, E_ZZ_LL: torch.Tensor) -> None:
        """
        This method computes the mu_sig matrix as defined in the paper.
        Note that E_ZZ_LL must be truncated to level >=2*(self.depth+1) in order for this method to work. (see (*))
        """

        # same reasoning, we follow the paper
        
        sigma_sig_length = self.d * useful.get_num_of_w_leq_k(self.depth, self.Z_dimension)
        
        sigma_sig = torch.zeros(
            (sigma_sig_length, sigma_sig_length)
        )  
        
        all_words = useful.get_words_with_length_leq_k(self.depth, self.Z_dimension)
        words_len = len(all_words)

        for m in range(self.d):  
            fm = self.f(m)  # f(m) in the paper
            for w in all_words:
                wfm = w + fm  # wf(m) in the paper
                for n in range(self.d):  # iterate over n
                    fn = self.f(n)  #  f(n) in the paper
                    for v in all_words:
                        
                        # now the compute sigma_sig[wfm, vfn] (see 3.3 in paper)
                        vfn = v + fn  
                        term1 = 0
                        shuffled_words = useful.get_shuffle_product(wfm, vfn)
                        for word in shuffled_words:
                            E_ZZ_LL_index = useful.from_word_to_int(word, 2 * self.Z_dimension)
                            term1 += E_ZZ_LL[E_ZZ_LL_index]

                        # computing the right term, which is a product of two values
                        index_E_ZZ_LL_wfm = useful.from_word_to_int(wfm, 2 * self.Z_dimension)
                        index_E_ZZ_LL_vfn = useful.from_word_to_int(vfn, 2 * self.Z_dimension)
                        term2 = (
                            E_ZZ_LL[index_E_ZZ_LL_wfm] * E_ZZ_LL[index_E_ZZ_LL_vfn]
                        )

                        # computing the final value
                        sigma_sig_value = term1 - term2
                        
                        index_i = m * words_len + useful.from_word_to_int(w, self.Z_dimension)  
                        index_j = n * words_len + useful.from_word_to_int(
                            v, self.Z_dimension
                        )  
                        ind = (index_i, index_j)
                        sigma_sig[ind] = sigma_sig_value

        print("sigma_sig is ok" )
        self.sigma_sig = sigma_sig

    def get_l(self) -> None:
        """
        Computes l_{m}=[l_{1},...,l_{d}] , m = 1, ..., d 
        
        """
        self.l = []

        inv_sigma_sig = torch.inverse(self.sigma_sig)
        inv_sigma_sig_times_vector = torch.matmul(inv_sigma_sig, self.mu_sig)

        all_words = useful.get_words_with_length_leq_k(self.depth, self.Z_dimension)
        all_words_len = len(all_words)

        for m in range(self.d):  # iterate over m

            l_m = torch.zeros(all_words_len)
            for w in all_words:
                fm = self.f(m)  
                wfm = w + fm  
                l_m_index = useful.from_word_to_int(w, self.Z_dimension)
                inv_sigma_sig_times_vector_index = m * all_words_len + useful.from_word_to_int(
                    w, self.Z_dimension
                )

                l_m[l_m_index] = inv_sigma_sig_times_vector[ inv_sigma_sig_times_vector_index] / (2 * self.lambda_)

            self.l.append(l_m)

    def main_fitting_process(self, X: torch.Tensor, f: torch.Tensor) -> None:
        """
        This technique tailors the trading strategy to data 

        Arguments:
        - X : tradable asset's price paths
        - f : non-tradable factor's paths or exogenous signal
        """
        # we verify that length(X)=length(f)
        assert X.shape[0] == f.shape[0]
        
        assert X.shape[1] == f.shape[1]

        # 0. retrieve dimensions
        M = X.shape[0]  # length of paths 
        T = X.shape[1]  # length of time steps for each path
        d = X.shape[2]  # dim. of the price process paths X=(X^{1},...,X^{d})
        N = f.shape[2]  # exogenous signal dim.

        self.d = d
        self.N = N
        self.Z_dimension = d + N + 1

        # Z_t = (t, X_t, f_t) 
        Z = torch.zeros((M, T, self.Z_dimension))
        # we add time
        Z[:, :, 0] = torch.arange(T)  
        # then price process X_{t}
        Z[:, :, 1 : d + 1] = X
        # and exogenous signal f_{t}
        Z[:, :, d + 1 :] = f
        
        # Lead-lag transform
        
        Z_LL = useful.Naive_LL(Z)
        
        # Computes signature after lead lag transformation
        ZZ_LL = useful.get_signature(Z_LL, 2 * (self.depth + 1))

        # Compute E[ZZ^^LL_t]
        
        E_ZZ_LL = torch.mean(ZZ_LL, axis=0)

        # Compute  mu_sig  and sigma_sig
        
        self.compute_mu_sig(E_ZZ_LL)
        self.compute_sigma_sig(E_ZZ_LL)

        # Initialization of lambda the variance-scaling (see paper btw 3.1 and 3.2)
        self.get_lambda()
        # and func. calculation
        
        self.get_l()
        for i, func in enumerate(self.l):
            print( "l_" + str(i + 1))
            useful.print_signature(func.flatten(), self.Z_dimension, self.depth)
            
        print("fitting process is ok")

    @execution_dur
    def get_lambda(self) -> float:
        """
        Computes lambda according to paper , btw (3.1) and (3.2)
        """
        inv_sigma_sig = torch.inverse(self.sigma_sig)
        inv_sigma_sig_times_vector = torch.matmul(inv_sigma_sig, self.mu_sig)

        all_words = useful.get_words_with_length_leq_k(self.depth, self.Z_dimension)
        all_words_len = len(all_words)

        S = 0  
        for m in range(self.d): 
            for n in range(self.d): 
                for w in all_words:
                    ind_wfm = m * all_words_len + useful.from_word_to_int(w, self.Z_dimension) 
                    for v in all_words: 
                        ind_vfn = n * all_words_len + useful.from_word_to_int(v, self.Z_dimension)
                        
                        wfm= inv_sigma_sig_times_vector[ind_wfm]  
                        vfn= inv_sigma_sig_times_vector[ind_vfn]  
                        sigma_sig_ind = (ind_wfm, ind_vfn)
                        sigma_sig_lbd = self.sigma_sig[sigma_sig_ind] 

                        s_incr = wfm * vfn * sigma_sig_lbd
                        S += s_incr

        print("lambda is ok")
        
        self.lambda_ = 0.5 * np.sqrt(S / self.delta)

    def sig_trader(self, X: torch.Tensor, f: torch.Tensor, minimum_steps: int = 5) -> torch.Tensor:
        """
        Use model for trading after fitting 
        Arguments:
        - X  : tradable asset's price paths
        - f: exogenous signal
        - minimum_steps : minimum number of time steps to consider before trading

        Returns: 
        position at time t : xi_{t}^{m} 
        """
        assert (
            self.l is not None
        ), "The model must be fitted before using the sig-trader "

        assert X.shape[0] == f.shape[0]

        T = X.shape[0]  
        d = X.shape[1]  
        N = f.shape[1]  
        
        #Z_t = (t, X_t, f_t)
        
        Z = torch.zeros((T, d + N + 1)) ; Z[:, 0] = torch.arange(T) 
        Z[:, 1 : d + 1] = X ; Z[:, d + 1 :] = f

        # we initialize  xi as in the paper
        
        xi = torch.zeros((T, self.d))  
        
        #  xi_t for t = minimum_steps, ..., T-1
        
        for t in range(minimum_steps, T):
            Z_t = Z[:t, :]  # we only look at information up to now
            ZZ_t = useful.get_signature(Z_t, self.depth, no_batch=True)
            for m in range(self.d):
                xi[t, m] = torch.dot(self.l[m], ZZ_t)

        return xi

    def get_PnL(self, X: torch.Tensor, xi: torch.Tensor) -> torch.tensor:
        """
        Computes the PnL of the trading strategy 
        
        Arguments:
        - X :price paths
        - xi : trading strategy
        
        """
        assert X.shape[0] == xi.shape[0], "xi steps must be = to X steps "
        assert X.shape[1] == xi.shape[1], "dim(xi) must be = dim(X)"

        T = X.shape[0]
        d = X.shape[1]

        # return PnL
        day_pnl = xi[:-1, :] * (X[1:, :] - X[:-1, :])

        return day_pnl




################ Class for prices paths : Markowitz

class Get_data_Markowitz:
    
    def __init__(self, d=10, n_trading_days=252, N_years=1, mu_mean=0.25, sigma_mean=0.2,t0_price=100,batch_len=1000):
        
        self.d ,self.n_trading_days= d,n_trading_days
        self.dt = 1 / n_trading_days
        self.len_points = int(N_years / self.dt)
        self.t0_prices = t0_price * torch.ones(self.d)
        self.mu_mean,self.sigma_mean=mu_mean,sigma_mean
        self.batch_len=batch_len
        self.henkel = 0.05
        
        self.mu = self.simu_mu()
        self.cov = self.simu_cov()
        
    def simu_mu(self):
        """Get mu"""
        mu_annual = self.mu_mean * torch.rand(self.d) + (self.mu_mean/4)
        return mu_annual * self.dt

    # we'll use Hankel cov (with some modifications)
    
    def simu_cov(self):
        """Generate a covariance matrix inspired by Hankel structure but adapted for financial modeling."""
        # Generate annual sigmas for each asset
        sigmas_annual = self.sigma_mean * torch.rand(self.d) + (self.sigma_mean/4)
        
        # Adjust by the square root of the time step to get frequency sigmas
        
        sigmas_freq = sigmas_annual * np.sqrt(self.dt)
        cov = torch.zeros((self.d, self.d))
        
        # Fill the matrix with correlations that change linearly from the main diagonal
        
        for i in range(self.d):
            for j in range(self.d):
                # Calculate a correlation that diminishes or increases linearly from the diagonal
                rho = max(1-self.henkel*self.d,np.exp(-np.abs(i - j)*self.henkel))
                cov[i, j] = rho * sigmas_freq[i] * sigmas_freq[j]
                cov[j, i] = cov[i, j]  # Ensures the matrix is symmetric
        
        return cov

    def simulate(self, batch_len):
        
        """Simulate batch of price paths."""
        
        returns_subset= torch.distributions.MultivariateNormal(self.mu, self.cov).sample((batch_len, self.len_points))
        batch_prices = torch.zeros((batch_len, self.len_points + 1, self.d))
        for i in range(batch_len):
            prices = torch.zeros((self.len_points + 1, self.d))
            prices[0] = self.t0_prices
            for j in range(self.len_points):
                prices[j+1] = prices[j] * (1 + returns_subset[i][j])
            batch_prices[i]=prices

        return batch_prices,returns_subset
    

    def param(self):
        return self.mu,self.cov,self.n_trading_days,self.len_points,self.batch_len