import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from pyldpc import make_ldpc, encode, decode

#UTILITY
def plot_graphs(dist_lst, bits, title, xlabel):
    #plots a graph after obtaining samples from sampling
    v_bin = [int(np.array2string(a, separator="")[1:-1], 2) for a in dist_lst]
    #plot probability distribution from Gibbs Sampling
    total = len(v_bin)
    labels, freq = np.unique(v_bin, return_counts=True)
    freq = freq/total

    fig, ax = plt.subplots()
    ax.bar(labels, freq)
    #ax.xaxis.set_major_formatter(StrMethodFormatter("{x:"+bits+"b}"))
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
                v_new[i] = 2*v_new[i]-1
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
            else:
                dist_lst.append(v[:visible_bits])
        return dist_lst, v

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

class LDPC():
    #initializes LDPC weights and biases for Gibbs Sampling
    def __init__(self, H, G, message_len, parity_len, h_km, h, received_signal):
        #H - parity check matrix, G - generator matrix
        #message_len - length of message, parity_len - number of parity bits
        #h_km - hyperparameter associated with penalty terms
        #h - another hyperparameter associated with parity check
        self.H_matrix = H
        self.G_matrix = G
        self.message_len = message_len
        self.parity_len = parity_len
        self.h_km = h_km
        self.h = h
        self.calc_w_b(received_signal)

    def calc_w_b(self, received_signal):
        #calculates biases
        H = self.H_matrix
        h_km = self.h_km
        h = self.h
        n = self.message_len
        #indices of nonzero elements of H (indexed by H)
        nonzero_row, nonzero_col = np.nonzero(H)
        #indices of last nonzero element per row (indexed by p)
        p_last_idx = np.nonzero(nonzero_row - np.roll(nonzero_row, -1))[0]
        #indices of nonzero elements, split by row (indexed by H)
        nonzero_col_by_row = np.split(nonzero_col, p_last_idx[:-1]+1)
        #indices of first nonzero element per row (indexed by H)
        first_nonzero = np.argmax(H, axis=1)
        #indices of first nonzero element per row (indexed by p)
        p_first_idx = np.concatenate([np.array([0]), p_last_idx[:-1]+1])
        #indices of last nonzero element in row (indexed by H)
        last_nonzero = [nonzero[-1] for nonzero in nonzero_col_by_row]

        #number of p spins
        p_len = nonzero_row.size - H.shape[0]
        #constructs visible bias:
        b_sigma = -received_signal + 0.5 + 0.5*h_km*np.sum(H, axis=0)
        b_p = h_km*np.ones(p_len)
        for idx in range(len(p_last_idx)):
            b_p[p_last_idx[idx] - idx - 1] = 0.5*h_km+0.5*h

        #number of a spins
        a_len = p_len
        b_a = h_km*np.ones(a_len)

        self.b = np.concatenate([b_sigma, b_p, b_a])

        #calculate weight matrix
        total_len = p_len + a_len + n
        #print(total_len)
        #weights for sigmas
        w_sigma = np.zeros((n, total_len))
        for i in range(self.parity_len):
            #first index
            new_w = np.zeros(total_len)
            start_i = p_first_idx[i]
            row = nonzero_col_by_row[i]
            new_w[row[1]] = -0.5*h_km
            new_w[n + start_i - i] = -0.5*h_km
            new_w[n + p_len + start_i - i] = h_km
            w_sigma[nonzero_col_by_row[i][0]] += new_w
            #second index
            new_w[row[1]] = 0
            new_w[first_nonzero[i]] = -0.5*h_km
            w_sigma[nonzero_col_by_row[i][1]] += new_w
            for j in range(2, row.size):
                new_w = np.zeros(total_len)
                new_w[n + p_first_idx[i] - i + j - 2] = -0.5*h_km
                new_w[n + p_first_idx[i] - i + j - 1] = -0.5*h_km
                new_w[n + p_len + p_first_idx[i] - i + j - 1] = h_km
                w_sigma[nonzero_col_by_row[i][j]] += new_w

        w_p = np.zeros((p_len, total_len))
        for i in range(self.parity_len):
            start_i = p_first_idx[i]
            row = nonzero_col_by_row[i]
            w_p[start_i - i, row[0]] = -0.5*h_km
            w_p[start_i - i, row[1]] = -0.5*h_km
            w_p[start_i - i, row[2]] = -0.5*h_km
            w_p[start_i - i, n + start_i - i] = -0.5*h_km
            w_p[start_i - i, n + p_len + start_i - i] = h_km
            w_p[start_i - i, n + p_len + start_i - i + 1] = h_km
            for j in range(2, row.size - 1):
                w_p[start_i - i + j - 1, row[j-1]] = -0.5*h_km
                w_p[start_i - i + j - 1, row[j]] = -0.5*h_km
                w_p[start_i - i + j - 1, n + start_i - i + j - 2] = -0.5*h_km
                w_p[start_i - i + j - 1, n + start_i - i + j] = -0.5*h_km
                w_p[start_i - i + j - 1, n + p_len + start_i - i + j - 1] = h_km
                w_p[start_i - i + j - 1, n + p_len + start_i - i + j] = h_km
            w_p[start_i - i + row.size - 2, row[-1]] = -0.5*h_km
            w_p[start_i - i + row.size - 2, n + start_i - i + row.size - 2] = -0.5*h_km
            w_p[start_i - i + row.size - 2, n + p_len + start_i - i + row.size - 2] = h_km

        w_a = np.zeros((a_len, total_len))
        for i in range(self.parity_len):
            start_i = p_first_idx[i]
            row = nonzero_col_by_row[i]
            w_a[start_i - i, row[0]] = h_km
            w_a[start_i - i, row[1]] = h_km
            w_a[start_i - i, n + start_i - i] = h_km
            w_a[start_i - i, n + p_len + start_i - i] = -2*h_km
            for j in range(2, row.size):
                w_a[start_i - i + j - 1, row[j]] = h_km
                w_a[start_i - i + j - 1, n + start_i - i + j - 2] = h_km
                w_a[start_i - i + j - 1, n + start_i - i + j - 1] = h_km
                w_a[start_i - i + j - 1, n + p_len + start_i - i + j - 1] = -2*h_km
        self.W = np.vstack([w_sigma, w_p, w_a])

"""
n_code = 6
k = 3
h_km = 0.5 #from paper
h = 0.3 #from paper
H = np.array([
    [1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 1],
    [1, 0, 0, 1, 1, 0]
])
G = np.array([
    [1, 0, 0, 1, 0, 1],
    [0, 1, 0, 1, 1, 1],
    [0, 0, 1, 1, 1, 0]
])
m = np.array([1, 0, 1])
r = np.mod(G.T.dot(m), 2)
print("should get", r)
ldpc = LDPC(H, G, n_code, k, h_km, h, r)
"""
"""
W = np.array([
    [0,-0.5*h_km,0,-0.5*h_km,0,0,   -0.5*h_km,0,0,   0,0,  -0.5*h_km,0,  h_km,0,0,  0,0,  h_km,0],
    [-0.5*h_km,0,0,0,0,0,   -0.5*h_km,0,0,   0,0,  0,0,  h_km,0,0,  0,0,  0,0],
    [0,0,0,-0.5*h_km,0,0,   -0.5*h_km,-0.5*h_km,0,   -0.5*h_km,0,  0,0,  0,h_km,0,  h_km,0,  0,0],
    [-0.5*h_km,0,-0.5*h_km,0,0,0,   0,-0.5*h_km,-0.5*h_km,   -0.5*h_km,0,  -0.5*h_km,0,  0,0,h_km,  h_km,0,  h_km,0],
    [0,0,0,0,0,0,   0,0,0,   0,0,  -0.5*h_km,-0.5*h_km,  0,0,0,  0,0,  0,h_km],
    [0,0,0,0,0,0,   0,0,0,   -0.5*h_km,-0.5*h_km,  0,0,  0,0,0,  0,h_km,  0,0],

    [-0.5*h_km,-0.5*h_km,-0.5*h_km,0,0,0,   -0.5*h_km,0,0,   0,0,   0,0,   h_km,h_km,0,  0,0,  0,0],
    [0,-0.5*h_km,-0.5*h_km,0,0,0,   -0.5*h_km,0,-0.5*h_km,   0,0,   0,0,   0,h_km,h_km,  0,0,  0,0],
])
"""
n_code = 8 #length of total bit string
h_km = 0.5#0.5 #0.5 #from paper
h = 0.3#0.3 #0.3 #from paper
w_r = 4 #from paper
w_c = 2 #from paper

blocks = 40
snr_range = 6
bp_ber = np.zeros(snr_range)
mle_ber = np.zeros(snr_range)
for _ in range(blocks):
    H, G = make_ldpc(n_code, w_c, w_r, systematic=True)
    n_message = G.shape[1] #number of message bits (not including parity bits)
    k = H.shape[0] #number of rows of parity check

    m = np.random.choice([0, 1], n_message)
    r = np.mod(G.dot(m), 2)
    for snr in range(snr_range):
        r_noise = encode(G, m, snr=snr)
        r_decode_bp = decode(H, r_noise, 20)

        ldpc = LDPC(H, G, n_code, k, h_km, h, (1-r_noise)/2)

        ldpc_sampler = GibbsSampler(ldpc.W, ldpc.b, 100000, 100000, binary=False)
        ldpc_dist_lst, v = ldpc_sampler.run_sampling(visible_bits=n_code)
        #plot_graphs(ldpc_dist_lst, "06", "LDPC: " + np.array2string(r, precision=1, separator='')[1:-1], "Message")
        #v_bin = [int(np.array2string(a, separator="")[1:-1], 2) for a in ldpc_dist_lst]
        total = len(ldpc_dist_lst)
        labels, freq = np.unique(ldpc_dist_lst, return_counts=True, axis=0)
        freq = freq/total
        mle_result = labels[np.argmax(freq)]
        bp_ber[snr] += (n_code - np.sum(np.equal(r_decode_bp, r)))/(blocks*n_code)
        mle_ber[snr] += (n_code - np.sum(np.equal(mle_result, r)))/(blocks*n_code)
        # print("mle result", mle_result)
        # print("original", r)
        # print("bp result", r_decode_bp)

fig, ax = plt.subplots()
plt.plot(range(snr_range), bp_ber, color='red', label='bp')
plt.plot(range(snr_range), mle_ber, color='blue', label='parity_check')
ax.set_xlabel("SNR")
ax.set_ylabel("BER")
ax.set_yscale('log')
plt.title("ldpc_8_4_2")
plt.legend(loc='best')
plt.savefig('ldpc_8_4_2_paper.png')
