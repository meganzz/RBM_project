import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

#UTILITY
def plot_graphs(dist_lst, bits, title, xlabel):
    #plots a graph after obtaining samples from sampling
    v_bin = [int(np.array2string(a, separator="")[1:-1], 2) for a in dist_lst]
    #plot probability distribution from Gibbs Sampling
    plt.figure(1)
    total = len(v_bin)
    labels, freq = np.unique(v_bin, return_counts=True)
    freq = freq/total

    fig, ax = plt.subplots()
    ax.bar(labels, freq)
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:"+bits+"b}"))
    ax.xaxis.set_ticks(np.arange(0, 2**int(bits), 1))
    ax.set_xlabel(xlabel)
    ax.set_ylim([0, 1])
    ax.set_ylabel("Probability")
    plt.title(title)
    plt.show()

class GibbsSampler():
    #Can be for Boltzmann Machine
    def __init__(self, W, visible_bias, burn_in, n_samples, binary=True):
        self.W = W
        self.visible_bias = visible_bias
        self.visible_size = visible_bias.size
        self.burn_in = burn_in
        self.n_samples = n_samples
        self.binary = binary
        self.p_func = np.tanh
        self.low, self.high = -1, 1
        if binary:
            self.p_func = lambda x: 1.0/(1.0+np.exp(-x))
            self.low, self.high = 0, 1
    def sample(self, v):
        v_new = np.copy(v)
        for i in range(v.size):
            p_v = self.p_func(self.W[i].dot(v_new) - self.W[i,i]*v_new[i] + self.visible_bias[i])
            #print(p_v)
            v_new[i] = int(random.uniform(self.low, self.high) < p_v)
            if not self.binary:
                v_new[i] = 1-2*v_new[i]
        return v_new
    def run_sampling(self, v_init=None, visible_bits=None):
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
            dist_lst.append(v[:visible_bits])
        return dist_lst

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

"""
#This is for AND:
#weights and biases
W = np.array([[-9, -12, 4], [-9, 4, -12], [-1, -10, -10]])
b = np.array([6, 6, 4])
c = np.array([4, 6, 6])
and_sampler = BlockGibbsSampler(W, b, c, 50, 100000)
and_dist_lst = and_sampler.run_sampling()
plot_graphs(and_dist_lst, "03", "AND: No Clamp", "ABC")
"""

"""
#This is for AND with general boltzmann machine:
#weights and biases obtained from Wikipedia example
W = np.array([
    [0, 0, 0, -9, -12, 4],
    [0, 0, 0, -9, 4, -12],
    [0, 0, 0, -1, -10, -10],
    [-9, -9, -1, 0, 0, 0],
    [-12, 4, -10, 0, 0, 0],
    [4, -12, -10, 0, 0, 0],

])
b = np.array([6, 6, 4, 4, 6, 6])
and_sampler = GibbsSampler(W, b, 50, 1000000)
and_dist_lst = and_sampler.run_sampling(visible_bits=3)
#print(and_dist_lst)
plot_graphs(and_dist_lst, "03", "AND: No Clamp", "ABC")
"""

#This is for LDPC
