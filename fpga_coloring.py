import argparse
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import utils
import ldpc
from pyldpc import encode

n_code = 8 #length of total bit string
h_km = 0.25 #0.5 #from paper
h = 0.15 #0.015 #0.3 #from paper
h_mod2 = 0.15 #for mod2 formulation
w_r = 4#8 #from paper
w_c = 2#4 #from paper
beta = 2 #inverse temperature
burn_in = 10000
n_samples = 10000
precision_bits = 10
p_index = 5

def gen_ising_mem(snr, ising_form):
	#generate and save .mem files for the given formulation
	#ising_form is true when it is ising and false when mod2
	H, G, m = utils.gen_matrices(n_code, w_c, w_r)
	print("G:", G)
	np.savetxt("ldpc_"+str(n_code)+"_G.txt", G, fmt="%d")
	print("H:", H)
	utils.bin_to_file(np.flip(H, axis=0).flatten(), "ldpc_"+str(n_code)+"_H.mem")
	r = np.mod(G.dot(m), 2)
	print("original:", r)
	r_noise = (1-encode(G, m, snr=snr))/2
	print("received:", r_noise)
	utils.float_to_fix(r_noise, precision_bits, p_index, "ldpc_"+str(n_code)+"_received.mem")
	ldpc_obj = ldpc.LDPC(H, n_code, H.shape[0], h_km, h, h_mod2, r_noise, w_c, beta=beta)
	if ising_form:
		b = ldpc_obj.b
		W = ldpc_obj.W
		total = W.shape[0]
		edge_lst = np.transpose(np.nonzero(W))
		g = nx.Graph()
		g.add_nodes_from(range(total))
		g.add_edges_from(edge_lst)
		color_dict = nx.coloring.greedy_color(g, strategy="saturation_largest_first")
		groups = [[] for _ in range(len(set(color_dict.values())))] #each list represents 1 color
		#add spins to the list they are colored
		for i in range(total):
			c = color_dict[i]
			groups[c].append(i)
		print("h in fixed point:", np.binary_repr(np.around(2**p_index*h, decimals=0).astype(int), precision_bits))
		print("h_km in fixed point:", np.binary_repr(np.around(2**p_index*h_km, decimals=0).astype(int), precision_bits))
		print("h_mod2 in fixed point:", np.binary_repr(np.around(2**p_index*h_mod2, decimals=0).astype(int), precision_bits))
		print("Colorings:", groups)
		#the order to shuffle the spins to form blocks of the same color
		group_order = np.concatenate(groups).astype(int).tolist()
		print("group order indices: ", group_order)
		np.savetxt("ldpc_"+str(n_code)+"_order.txt", group_order, fmt='%d')
		np.savetxt("ldpc_"+str(n_code)+"_group_size.txt", [len(g) for g in groups], fmt='%d')
		#the indices where the sigma spins have been shuffled to
		print("Sigma indices:", [group_order.index(i) for i in range(8)])
		W = W[group_order][:, group_order]
		b = b[group_order]
		utils.float_to_fix(np.flip(W, axis=0).flatten(), precision_bits, p_index, "ldpc_"+str(n_code)+"_weights.mem")
		utils.float_to_fix(b, precision_bits, p_index, "ldpc_"+str(n_code)+"_bias.mem")
	else:
		#TODO: finish implementing for mod2
		return

def read_results():
	#reads in the results, then plots dist
	#print(np.loadtxt("basic_dist_lst.txt", dtype=int)[:10])
	dist_lst = []
	with open("basic_dist_lst.txt", 'r') as f:
		for line in f:
			dist_lst.append(int(line[:-1][::-1], 2))
	labels, freq = np.unique(dist_lst, return_counts=True)
	freq = freq/len(dist_lst)
	plt.stem(labels, freq)
	plt.title("FPGA: Probability")
	plt.show()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	help_str = "1 - generate .mem files"
	parser.add_argument("--mode", "-m", type=int, help=help_str, required=True)
	args = parser.parse_args()
	if args.mode == 1:
		#generate the .mem files
		gen_ising_mem(8, ising_form=True)
	if args.mode == 2:
		read_results()
