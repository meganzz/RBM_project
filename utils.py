#UTILITY
import numpy as np
import matplotlib.pyplot as plt
from pyldpc import make_ldpc

def int_to_arr(num, b):
    #converts int to an array of bits
    f = '{0:0'+str(b)+'b}'
    return [int(x) for x in f.format(num)]

def plot_ber(ber, labels, colors, title, fname):
    #plots ber for data on semilog scale with labels and color on legend
    #source: https://stackoverflow.com/questions/24535848/drawing-log-linear-plot-on-a-square-plot-area-in-matplotlib
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_yscale("log")
    for i in range(len(labels)):
        plt.plot(range(6), ber[i], color=colors[i], label=labels[i])
    plt.ylim(0.0002, 0.2)
    plt.xlim(0, 5)

    # now get the figure size in real coordinates:
    fwidth = fig.get_figwidth()
    fheight = fig.get_figheight()

    # get the axis size and position in relative coordinates
    # this gives a BBox object
    bb = ax.get_position()

    # calculate them into real world coordinates
    axwidth = fwidth * (bb.x1 - bb.x0)
    axheight = fheight * (bb.y1 - bb.y0)

    # if the axis is wider than tall, then it has to be narrowe
    if axwidth > axheight:
        # calculate the narrowing relative to the figure
        narrow_by = (axwidth - axheight) / fwidth
        # move bounding box edges inwards the same amount to give the correct width
        bb.x0 += narrow_by / 2
        bb.x1 -= narrow_by / 2
    # else if the axis is taller than wide, make it vertically smaller
    # works the same as above
    elif axheight > axwidth:
        shrink_by = (axheight - axwidth) / fheight
        bb.y0 += shrink_by / 2
        bb.y1 -= shrink_by / 2

    ax.set_position(bb)

    ax.set_xlabel("Scaled SNR")
    ax.set_ylabel("BER")
    ax.set_yscale('log')
    plt.grid(visible=True, which='major', axis='both')
    plt.grid(visible=True, which='minor', axis='y', ls='--')
    plt.title(title)
    plt.legend(loc='best')
    plt.savefig(fname)
    plt.close()

def gen_matrices(n_code, w_c, w_r):
    #generates parity-check matrices and message
    H, G = make_ldpc(n_code, w_c, w_r, systematic=True)
    n_message = G.shape[1] #number of message bits (not including parity bits)
    m = np.random.choice([0, 1], n_message)
    return H, G, m

def bin_to_file(arr, fname):
	#saves binary arr to fname
	output = np.array2string(arr)[1:-1]
	with open(fname, "w") as f:
		f.write(output)

def float_to_fix(arr, precision_bits, p_index, fname):
    #converts floating point to fixed point of length precision_bits and p_index bits below the fixed point
    #writes data to fname
    output = np.around(2**p_index*arr, decimals=0).astype(int)
    output = [np.binary_repr(o, precision_bits) + "\n" for o in output]
    with open(fname, "w") as f:
        f.writelines(output)
