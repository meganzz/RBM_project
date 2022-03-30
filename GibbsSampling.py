import numpy as np

class GibbsSampler():
    #Can be for Boltzmann Machine
    def __init__(self, W, visible_bias, burn_in, n_samples, obj_func, beta=1, ising_form=True, update_func=None, binary=True):
        self.W = W
        self.visible_bias = visible_bias
        self.visible_size = visible_bias.size
        self.burn_in = burn_in
        self.n_samples = n_samples
        self.obj_func = obj_func
        self.beta = beta
        self.binary = binary
        self.p_func = np.tanh
        self.low, self.high = -1, 1
        self.ising_form = ising_form
        self.update_func = update_func
        if binary:
            self.p_func = lambda x: 1.0/(1.0+np.exp(-x))
            self.low, self.high = 0, 1

    def get_mle_estimate(self, dist_lst, n_freq):
        #takes the most frequent n_freq configurations, returns the one with lowest value of obj_func
        labels, freq = np.unique(dist_lst, return_counts=True, axis=0)
        max_idx = (-freq).argsort()[:n_freq]
        min_energy_idx = np.argmin([self.obj_func(labels[i]) for i in max_idx])
        return labels[max_idx[min_energy_idx]]

    def sample(self, v):
        v_new = np.copy(v)
        for i in range(v.size):
            p_v = 0
            if self.ising_form:
                p_v = self.p_func(self.beta*(self.W[i].dot(v_new) - self.W[i,i]*v_new[i] + self.visible_bias[i]))
            else:
                p_v = self.update_func(v_new, i)
            v_new[i] = int(np.random.uniform(self.low, self.high) < p_v)
            if not self.binary:
                v_new[i] = 2*v_new[i]-1
        return v_new

    def run_sampling(self, n_freq, v_init=None, visible_bits=None):
        #obj_func - function to minimize
        #n_freq - number of configurations to evaluate once sampling is done
        if visible_bits is None:
            visible_bits = self.visible_size
        v = v_init
        dist_lst = []
        if v_init is None:
            v = np.random.choice([self.low, self.high], self.visible_size)
        for i in range(self.burn_in):
            v = self.sample(v)
        for i in range(self.n_samples):
            v = self.sample(v)
            if not self.binary:
                dist_lst.append((1-v[:visible_bits])//2)
            else:
                dist_lst.append(v[:visible_bits])
        return dist_lst, self.get_mle_estimate(dist_lst, n_freq)

    def hit_engine(self, v_init=None, visible_bits=None):
        dist_lst = []
        if visible_bits is None:
            visible_bits = self.visible_size
        v = v_init
        if v_init is None:
            v = np.random.choice([self.low, self.high], self.visible_size)
        for i in range(self.burn_in):
            v = self.sample(v)
        max_code = v[:visible_bits]
        max_code_obj = self.obj_func(max_code)
        for i in range(self.n_samples):
            v = self.sample(v)
            dist_lst.append(v)
            v_obj = self.obj_func(v[:visible_bits])
            if v_obj < max_code_obj:
                max_code = v[:visible_bits]
                max_code_obj = v_obj
        return dist_lst, max_code


class BlockGibbsSampler(GibbsSampler):
    #For RBM
    def __init__(self, W, visible_bias, hidden_bias, burn_in, n_samples, binary=True):
        super().__init__(W, visible_bias, burn_in, n_samples, binary)
        self.hidden_bias = hidden_bias
        self.hidden_size = hidden_bias.size

    def sample(self, v, h, clamp_vals=None):
        #takes an individual sample
        #clamp_vals is a list of tuples (index, value) where v[index] = val
        p_h = self.p_func(self.W.T.dot(v)+self.hidden_bias)
        h_new = np.less(np.random.uniform(self.low, self.high, size=self.hidden_size), p_h).astype(int)
        p_v = self.p_func(self.W.dot(h_new)+self.visible_bias)
        v_new = np.less(np.random.uniform(self.low, self.high, size=self.visible_size), p_v).astype(int)
        if clamp_vals is not None:
            for clamp in clamp_vals:
                v_new[clamp[0]] = clamp[1]
        return v_new, h_new

    def run_sampling(self, v_init=None, h_init=None, clamp_vals=None):
        #takes all of the samples
        #clamp_vals is a list of tuples (index, value) where v[index] = val
        v, h = v_init, h_init
        dist_lst = []
        if v_init is None:
            v = np.random.choice([self.low, self.high], self.visible_size)
        if clamp_vals is not None:
            for clamp in clamp_vals:
                v[clamp[0]] = clamp[1]
        for i in range(self.burn_in):
            v, h = self.sample(v, h, clamp_vals)
        for i in range(self.n_samples):
            v, h = self.sample(v, h, clamp_vals)
            dist_lst.append(v)
        return dist_lst
