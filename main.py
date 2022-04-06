import argparse
import numpy as np
import matplotlib.pyplot as plt
from pyldpc import make_ldpc, encode, decode
import ldpc
import GibbsSampling

n_code = 8#32 #length of total bit string
n_freq = 20 #how many configurations to evaluate
h_km = 0.25#0.025 #0.5 #from paper
h = 0.15#0.015 #0.015 #0.3 #from paper
h_mod2 = 0.15 #for mod2 formulation
w_r = 4#8 #from paper
w_c = 2#4 #from paper
beta = 1#2 #inverse temperature
burn_in = 10000
n_samples = 10000

blocks = 1#60#5#100#5#300#50
snr_range = np.linspace(1.5, 9.5, 6)

#UTILITY
def int_to_arr(num, b):
    #converts
    f = '{0:0'+str(b)+'b}'
    return [int(x) for x in f.format(num)]

def plot_ber(bp_ber, mle_ber, title, fname):
    #plots ber on semilog scale
    #source: https://stackoverflow.com/questions/24535848/drawing-log-linear-plot-on-a-square-plot-area-in-matplotlib
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_yscale("log")
    plt.plot(range(6), bp_ber, color='red', label='bp')
    plt.plot(range(6), mle_ber, color='blue', label='parity_check')
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

def gen_matrices():
    #generates parity-check matrices and message
    H, G = make_ldpc(n_code, w_c, w_r, systematic=True)
    n_message = G.shape[1] #number of message bits (not including parity bits)
    m = np.random.choice([0, 1], n_message)
    return H, G, m

def setup_gibbs(ising_form, H, r_noise):
    #setup objects for Gibbs sampling, returns sampling results
    k = H.shape[0] #number of rows of parity check
    ldpc_obj = ldpc.LDPC(H, n_code, k, h_km, h, h_mod2, (1-r_noise)/2, w_c, beta=beta)
    if ising_form:
        b = -2*(ldpc_obj.b - np.sum(ldpc_obj.W, axis=1))
        W = -4*ldpc_obj.W
        ldpc_sampler = GibbsSampling.GibbsSampler(W, b, burn_in=burn_in, n_samples=n_samples, obj_func=ldpc_obj.obj_func, beta=beta)
        return ldpc_sampler
    else:
        ldpc_sampler = GibbsSampling.GibbsSampler(None, -r_noise, burn_in=burn_in, n_samples=n_samples, obj_func=ldpc_obj.obj_func, beta=beta, ising_form=False, update_func=ldpc_obj.update_func)
        return ldpc_sampler

def run_gibbs(ising_form, override):
    #run Gibbs sampling over several blocks
    bp_ber = np.zeros(6)
    mle_ber = np.zeros(6)
    for _ in range(blocks):
        for i in range(1):
            H, G, m = gen_matrices()
            r = np.mod(G.dot(m), 2)
            r_noise = encode(G, m, snr=snr_range[i])
            r_decode_bp = decode(H, r_noise, 20)
            ldpc_sampler = setup_gibbs(ising_form, H, r_noise)
            dist_lst, mle_result = ldpc_sampler.hit_engine(visible_bits=n_code)
            trunc_labels = np.unique(np.array(dist_lst)[:,:n_code], axis=0)
            energy = ldpc_sampler.obj_func(trunc_labels)
            top_energy_idx = energy.argsort()[0]
            mle_result = trunc_labels[top_energy_idx]

            bp_ber[i] += (n_code - np.sum(np.equal(r_decode_bp, r)))
            mle_ber[i] += (n_code - np.sum(np.equal(mle_result, r)))
    bp_ber = bp_ber/(blocks*n_code)
    mle_ber = mle_ber/(blocks*n_code)
    print("bp: ", bp_ber)
    print("parity_check: ", mle_ber)
    if override:
        ldpc_name = "ldpc_"+str(n_code)+"_"+str(w_c)+"_"+str(w_r)
        title, fname = None, None
        if ising_form:
            title = ldpc_name + " original formulation"
            fname = ldpc_name + "_original.png"
        else:
            title = ldpc_name + " mod2 formulation"
            fname = ldpc_name + "_mod2.png"
        plot_ber(bp_ber, mle_ber, title, fname)

def prob_dist(override):
    #gets distribution
    for block in range(blocks):
        H, G, m = gen_matrices()
        r = np.mod(G.dot(m), 2)
        for i in range(6):
            r_noise = encode(G, m, snr=snr_range[i])
            ldpc_sampler = setup_gibbs(True, H, r_noise)
            dist_lst, _ = ldpc_sampler.hit_engine(visible_bits=n_code)
            W = ldpc_sampler.W
            b = ldpc_sampler.visible_bias

            #original formulation
            #frequency
            labels, freq = np.unique(dist_lst, axis=0, return_counts=True)
            v_bin = [int(np.array2string(a, separator="", max_line_width=1000)[1:-1], 2) for a in labels]
            v_bin = np.array(v_bin)
            top_freq_idx = (-freq).argsort()[:n_freq]

            #log probability
            labels = np.array(labels)
            obj = np.sum(np.multiply(labels.T, np.dot(W, labels.T)), axis=0) + np.dot(b, labels.T)
            top_obj_idx = (-obj).argsort()[:n_freq]

            #energy
            trunc_labels = np.unique(labels[:,:n_code], axis=0)
            energy = ldpc_sampler.obj_func(trunc_labels)
            top_energy_idx = energy.argsort()[:n_freq]
            scaling = 2**(len(b)-n_code)
            v_energy = [int(np.array2string(a, separator="", max_line_width=1000)[1:-1], 2) for a in trunc_labels[top_energy_idx]]
            areas_v = []
            for j in range(len(v_energy)):
                v = v_energy[j]
                v_en = energy[top_energy_idx[j]]
                areas_v.append([[v, v + (2**(len(b)-n_code) - 1)/ scaling], [v_en, v_en]])

            true_code = int(np.array2string(r, separator="")[1:-1], 2)
            fig, axs = plt.subplots(3, 1, sharex=True)
            fig.set_size_inches(11, 9)

            #graphing
            markerline0a, stemlines0a, baseline0a = axs[0].stem(v_bin[top_freq_idx[0]] / scaling, freq[top_freq_idx[0]], markerfmt='ro', basefmt=" ")
            plt.setp(stemlines0a, 'color', plt.getp(markerline0a,'color'))
            markerline0, stemlines0, baseline0 = axs[0].stem(v_bin[top_freq_idx[1:]] / scaling, freq[top_freq_idx[1:]], markerfmt='bo', basefmt=" ")
            plt.setp(stemlines0, 'color', plt.getp(markerline0,'color'))
            axs[0].set_ylabel("Count")
            axs[0].axvline(x=true_code, color = "orange", linestyle='--')

            markerline1a, stemlines1a, baseline1a = axs[1].stem(v_bin[top_obj_idx[0]] / scaling, obj[top_obj_idx[0]], markerfmt='ro', basefmt=" ")
            plt.setp(stemlines1a, 'color', plt.getp(markerline1a,'color'))
            markerline1, stemlines1, baseline1 = axs[1].stem(v_bin[top_obj_idx[1:]] / scaling, obj[top_obj_idx[1:]].astype(float), markerfmt='go', basefmt=" ")
            plt.setp(stemlines1, 'color', plt.getp(markerline1,'color'))
            axs[1].set_ylabel("Log Probability")
            axs[1].axvline(x=true_code, color = "orange", linestyle='--')

            axs[2].fill_between(areas_v[0][0], 0, areas_v[0][1], color="red")
            for pair in areas_v[1:]:
                v, v_en = pair[0], pair[1]
                axs[2].fill_between(v, 0, v_en, color="m")
            axs[2].set_ylabel("Energy")
            axs[2].axvline(x=true_code, color = "orange", linestyle='--')
            axs[2].set_xlabel("codeword")
            axs[0].set_title("Ising " + str(n_code) + "-bit: SNR Level " + str(i))
            print("ising" + str(block) + str(i))
            print("probability: ", n_code - np.sum(np.equal(labels[top_obj_idx[0]][:n_code], r)))
            print("energy: ", n_code - np.sum(np.equal(trunc_labels[top_energy_idx[0]][:n_code], r)))
            if override:
                plt.savefig('prob_dist_ising_'+str(n_code)+'_mle_'+str(block)+'_'+str(i)+'.png')
                plt.close()

            #mod2 formulation
            ldpc_sampler = setup_gibbs(False, H, r_noise)
            dist_lst, _ = ldpc_sampler.hit_engine(visible_bits=n_code)
            #frequency
            labels, freq = np.unique(dist_lst, axis=0, return_counts=True)
            v_bin = [int(np.array2string(a, separator="")[1:-1], 2) for a in labels]
            v_bin = np.array(v_bin)
            top_freq_idx = (-freq).argsort()[:n_freq]

            labels = np.array(labels)
            #energy
            energy = ldpc_sampler.obj_func(labels)
            top_energy_idx = energy.argsort()[:n_freq]
            v_energy = [int(np.array2string(a, separator="")[1:-1], 2) for a in labels[top_energy_idx]]

            true_code = int(np.array2string(r, separator="")[1:-1], 2)
            fig, axs = plt.subplots(2, 1, sharex=True)
            fig.set_size_inches(11, 6)
            markerline0a, stemlines0a, baseline0a = axs[0].stem(v_bin[top_freq_idx[0]], freq[top_freq_idx[0]], markerfmt='ro', basefmt=" ")
            plt.setp(stemlines0a, 'color', plt.getp(markerline0a,'color'))
            markerline0, stemlines0, baseline0 = axs[0].stem(v_bin[top_freq_idx[1:]], freq[top_freq_idx[1:]], markerfmt='bo', basefmt=" ")
            plt.setp(stemlines0, 'color', plt.getp(markerline0,'color'))
            axs[0].set_ylabel("Count")
            axs[0].axvline(x=true_code, color = "orange", linestyle='--')

            markerline1a, stemlines1a, baseline1a = axs[1].stem(v_bin[top_energy_idx[0]], energy[top_energy_idx[0]], markerfmt='ro', basefmt=" ")
            plt.setp(stemlines1a, 'color', plt.getp(markerline1a,'color'))
            markerline1, stemlines1, baseline1 = axs[1].stem(v_bin[top_energy_idx[1:]], energy[top_energy_idx[1:]], markerfmt='mo', basefmt=" ")
            plt.setp(stemlines1, 'color', plt.getp(markerline1,'color'))
            axs[1].set_ylabel("Energy")
            axs[1].axvline(x=true_code, color = "orange", linestyle='--')

            axs[1].set_xlabel("codeword")
            #axs[0].set_title("Transmitted: "+str(r)+", Received: "+str(r_noise))
            axs[0].set_title("mod2 " + str(n_code) + "-bit: SNR Level " + str(i))
            print("mod2" + str(block) + str(i))
            print("energy: ", n_code - np.sum(np.equal(labels[top_energy_idx[0]][:n_code], r)))
            print()
            if override:
                plt.savefig('prob_dist_mod2_'+str(n_code)+'_mle_'+str(block)+'_'+str(i)+'.png')
                plt.close()
            #plt.show()

def check_formulation():
    #checks formulation for a given example
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
    m = np.array([0, 1, 1])
    r = np.mod(G.T.dot(m), 2) + np.array([.1,-.1,.05,.04,-.05,-.06])
    b_sigma = -r+0.5+0.5*h_km*(np.sum(H, axis=0))
    b_p = np.array([h_km, h_km, 0.5*h_km + 0.5*h, h_km, 0.5*h_km + 0.5*h, h_km, 0.5*h_km + 0.5*h])
    b_a = -h_km*np.ones(7)
    b = np.concatenate([b_sigma, b_p, b_a])
    W = np.array([
        [0,-0.5*h_km,0,-0.5*h_km,0,0,  -0.5*h_km,0,0,0,0,-0.5*h_km,0,  h_km,0,0,0,0,h_km,0],
        [-0.5*h_km,0,0,0,0,0,  -0.5*h_km,0,0,0,0,0,0,  h_km,0,0,0,0,0,0],
        [0,0,0,-0.5*h_km,0,0,  -0.5*h_km,-0.5*h_km,0,-0.5*h_km,0,0,0,  0,h_km,0,h_km,0,0,0],
        [-0.5*h_km,0,-0.5*h_km,0,0,0,  0,-0.5*h_km,-0.5*h_km,-0.5*h_km,0,-0.5*h_km,0,  0,0,h_km,h_km,0,h_km,0],
        [0,0,0,0,0,0,  0,0,0,0,0,-0.5*h_km,-0.5*h_km,  0,0,0,0,0,0,h_km],
        [0,0,0,0,0,0,  0,0,0,-0.5*h_km,-0.5*h_km,0,0,  0,0,0,0,h_km,0,0],

        [-0.5*h_km,-0.5*h_km,-0.5*h_km,0,0,0,  0,-0.5*h_km,0,0,0,0,0,  h_km,h_km,0,0,0,0,0],
        [0,0,-0.5*h_km,-0.5*h_km,0,0,  -0.5*h_km,0,-0.5*h_km,0,0,0,0,  0,h_km,h_km,0,0,0,0],
        [0,0,0,-0.5*h_km,0,0,  0,-0.5*h_km,0,0,0,0,0,  0,0,h_km,0,0,0,0],

        [0,0,-0.5*h_km,-0.5*h_km,0,-0.5*h_km,  0,0,0,0,-0.5*h_km,0,0,  0,0,0,h_km,h_km,0,0],
        [0,0,0,0,0,-0.5*h_km,  0,0,0,-0.5*h_km,0,0,0,  0,0,0,0,h_km,0,0],

        [-0.5*h_km,0,0,-0.5*h_km,-0.5*h_km,0,  0,0,0,0,0,0,-0.5*h_km,  0,0,0,0,0,h_km,h_km],
        [0,0,0,0,-0.5*h_km,0,  0,0,0,0,0,-0.5*h_km,0,  0,0,0,0,0,0,h_km],

        [h_km,h_km,0,0,0,0,  h_km,0,0,0,0,0,0,  -2*h_km,0,0,0,0,0,0],
        [0,0,h_km,0,0,0,  h_km,h_km,0,0,0,0,0,  0,-2*h_km,0,0,0,0,0],
        [0,0,0,h_km,0,0,  0,h_km,h_km,0,0,0,0,  0,0,-2*h_km,0,0,0,0],

        [0,0,h_km,h_km,0,0,  0,0,0,h_km,0,0,0,  0,0,0,-2*h_km,0,0,0],
        [0,0,0,0,0,h_km,  0,0,0,h_km,h_km,0,0,  0,0,0,0,-2*h_km,0,0],

        [h_km,0,0,h_km,0,0,  0,0,0,0,0,h_km,0,  0,0,0,0,0,-2*h_km,0],
        [0,0,0,0,h_km,0,  0,0,0,0,0,h_km,h_km,  0,0,0,0,0,0,-2*h_km]
    ])
    b += np.diag(W)
    np.fill_diagonal(W, 0)
    W = 0.5*W
    assert np.array_equal(W, W.T), f"not symmetric"
    ldpc_obj = ldpc.LDPC(H, 6, 3, h_km, h, h_mod2, r, w_c)
    assert np.array_equal(ldpc_obj.b, b), f"bias different"
    assert np.array_equal(ldpc_obj.W, W), f"weights different"

def ber_dist(override):
    #BER distribution according to distributions in prob_dist
    freq_ber = [[] for _ in range(6)]
    obj_ber = [[] for _ in range(6)]
    energy_ber = [[] for _ in range(6)]
    mod2_freq_ber = [[] for _ in range(6)]
    mod2_energy_ber = [[] for _ in range(6)]
    for block in range(blocks):
        H, G, m = gen_matrices()
        r = np.mod(G.dot(m), 2)
        for i in range(6):
            r_noise = encode(G, m, snr=snr_range[i])
            ldpc_sampler = setup_gibbs(True, H, r_noise)
            dist_lst, _ = ldpc_sampler.hit_engine(visible_bits=n_code)
            W = ldpc_sampler.W
            b = ldpc_sampler.visible_bias

            #original formulation
            #frequency
            labels, freq = np.unique(dist_lst, axis=0, return_counts=True)
            top_freq_idx = (-freq).argsort()[0]
            freq_ber[i].append((n_code - np.sum(np.equal(labels[top_freq_idx][:n_code], r)))/n_code)

            #unnormalized log probability
            obj = -(np.sum(np.multiply(labels.T, np.dot(W, labels.T)), axis=0) + np.dot(b, labels.T))
            top_obj_idx = obj.argsort()[0]
            obj_ber[i].append((n_code - np.sum(np.equal(labels[top_obj_idx][:n_code], r)))/n_code)

            #energy
            trunc_labels = np.unique(labels[:,:n_code], axis=0)
            energy = ldpc_sampler.obj_func(trunc_labels)
            top_energy_idx = energy.argsort()[0]
            energy_ber[i].append((n_code - np.sum(np.equal(labels[top_energy_idx][:n_code], r)))/n_code)

            #mod2
            ldpc_sampler = setup_gibbs(False, H, r_noise)
            dist_lst, _ = ldpc_sampler.hit_engine(visible_bits=n_code)
            #frequency
            labels, freq = np.unique(dist_lst, axis=0, return_counts=True)
            top_freq_idx = (-freq).argsort()[0]
            mod2_freq_ber[i].append((n_code - np.sum(np.equal(labels[top_freq_idx], r)))/n_code)

            #energy
            energy = ldpc_sampler.obj_func(labels)
            top_energy_idx = energy.argsort()[0]
            mod2_energy_ber[i].append((n_code - np.sum(np.equal(labels[top_energy_idx], r)))/n_code)

    bins = np.linspace(0, 1, num=n_code+1)
    for i in range(6):
        fig, axs = plt.subplots(3, 1, sharex=True)
        fig.set_size_inches(11, 9)
        axs[0].hist(freq_ber[i], bins=bins, color="blue")
        axs[0].set_title("Count")
        axs[1].hist(obj_ber[i], bins=bins, color="green")
        axs[1].set_title("Log Probability")
        axs[2].hist(energy_ber[i], bins=bins, color="magenta")
        axs[2].set_title("Energy")
        plt.xlim(0,1)
        fig.suptitle("BER for Ising " + str(n_code) + "-bit: SNR Level " + str(i))
        if override:
            plt.savefig('ber_dist_ising_'+str(n_code)+'_mle_'+str(block)+'_'+str(i)+'.png')
            plt.close()

        fig, axs = plt.subplots(2, 1, sharex=True)
        fig.set_size_inches(11, 6)
        axs[0].hist(mod2_freq_ber[i], bins=bins, color="blue")
        axs[0].set_title("Count")
        axs[1].hist(mod2_energy_ber[i], bins=bins, color="magenta")
        axs[1].set_title("Energy")
        plt.xlim(0,1)
        fig.suptitle("BER for mod2 " + str(n_code) + "-bit: SNR Level " + str(i))
        if override:
            plt.savefig('ber_dist_mod2_'+str(n_code)+'_mle_'+str(block)+'_'+str(i)+'.png')
            plt.close()

def test_self_loop():
    #determine how to handle self loops
    W = np.array([
        [-1, 2, 1.5],
        [2, 0, -2],
        [1.5, -2, -1.5]
    ])
    b = np.array([0.5, -2, 1])

    all_codes = np.array([int_to_arr(i, 3) for i in range(8)])
    neg_energy = np.sum(np.multiply(all_codes.T, np.dot(W, all_codes.T)), axis=0) + np.dot(b, all_codes.T)
    p_codes = np.exp(neg_energy)
    p_codes = p_codes / np.sum(p_codes)
    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
    axs[0].stem(p_codes)
    axs[0].set_title("Probability")

    b += np.diag(W)
    np.fill_diagonal(W, 0)
    sampler = GibbsSampling.GibbsSampler(W, b, burn_in=100000, n_samples=100000, obj_func=None)
    dist_lst, _ = sampler.hit_engine()
    labels, freq = np.unique(dist_lst, axis=0, return_counts=True)
    labels = [int(np.array2string(a, separator="")[1:-1], 2) for a in labels]
    freq = freq/len(dist_lst)
    all_freq = [freq[i] if i in labels else 0 for i in range(8)]
    axs[1].stem(all_freq)
    axs[1].set_title("Move diagonal entries to bias")
    plt.show()

if __name__ == "__main__":
    #parsing code from hw1 template
    parser = argparse.ArgumentParser()
    help_str = "1 - run Gibbs sampling with original formulation\n2 - run Gibbs sampling with mod2 formulation\n3 - get distribution of both methods"
    parser.add_argument("--mode", "-m", type=int, help=help_str, required=True)
    parser.add_argument("--override", "-o", type=int, default=0, help="1 - override graph, 0 - no graph\nFor modes 1 and 2")
    args = parser.parse_args()
    if args.mode == 1:
        #Gibbs sampling with original formulation
        run_gibbs(True, args.override)
    elif args.mode == 2:
        #Gibbs sampling with mod2 formulation
        run_gibbs(False, args.override)
    elif args.mode == 3:
        #distribution of both methods
        prob_dist(args.override)
    elif args.mode == 4:
        #check formulation
        check_formulation()
    elif args.mode == 5:
        #ber distribution
        ber_dist(args.override)
    elif args.mode == 6:
        #determine self loop
        test_self_loop()

#distribution of bers for mod2 and probs for different snrs hist
#check with 8 full - 1 sample
