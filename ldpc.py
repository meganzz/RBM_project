import numpy as np

class LDPC():
    #initializes LDPC weights and biases for Gibbs Sampling
    def __init__(self, H, message_len, parity_len, h_km, h, h_mod2, received_signal, w_c, beta=1):
        #H - parity check matrix, G - generator matrix
        #message_len - length of message, parity_len - number of parity bits
        #h_km - hyperparameter associated with penalty terms
        #h - another hyperparameter associated with parity check
        #w_c - number of nonzero elements per column of H
        #beta - inverse temperature for Gibbs Sampling
        self.H_matrix = H
        self.message_len = message_len
        self.parity_len = parity_len
        self.h_km = h_km
        self.h = h
        self.h_mod2 = h_mod2 #for the objective function
        self.r = received_signal
        self.H_spin_rows = [H[np.nonzero(H[:,i])] for i in range(message_len)]
        self.w_c = w_c
        self.beta = beta
        self.calc_w_b()

    def obj_func(self, spin_config):
        #objective function for LDPC
        if spin_config.ndim > 1:
            return np.linalg.norm(self.r - spin_config, axis=1)**2 + self.h_mod2*np.sum(np.mod(self.H_matrix.dot(spin_config.T), 2), axis=0)
        else:
            return np.linalg.norm(self.r - spin_config)**2 + self.h_mod2*np.sum(np.mod(self.H_matrix.dot(spin_config), 2))

    def calc_w_b(self):
        #calculates biases
        H = self.H_matrix
        h_km = self.h_km
        h = self.h
        n = self.message_len
        received_signal = self.r
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
            w_p[start_i - i, n + start_i - i + 1] = -0.5*h_km
            w_p[start_i - i, n + p_len + start_i - i] = h_km
            w_p[start_i - i, n + p_len + start_i - i + 1] = h_km
            for j in range(2, row.size - 1):
                w_p[start_i - i + j - 1, row[j]] = -0.5*h_km
                w_p[start_i - i + j - 1, row[j+1]] = -0.5*h_km
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

    def update_func(self, spin_config, i):
        #for mod2 formulation
        h = self.h_mod2
        b = 2*self.r - 1
        H_spin_rows = self.H_spin_rows
        spin = spin_config[i]
        E_v = b[i] + h*self.w_c*(2*spin - 1) + 2*h*(1-spin)*np.sum(np.mod(H_spin_rows[i].dot(spin_config), 2))
        p_v = 1.0/(1.0+np.exp(-self.beta*E_v))
        return p_v
