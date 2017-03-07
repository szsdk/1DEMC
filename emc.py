import re
import numpy as np
import matplotlib.pylab as plt
import itertools
import warnings
import logging
FORMAT = '%(asctime)s %(levelname)s: %(message)s'
from emc_core import EMC, simulate


import argparse
import configparser
import glob
import os
parser = argparse.ArgumentParser()
parser.add_argument("-c","--config", help="the config file")
group = parser.add_mutually_exclusive_group()
group.add_argument("-s","--simulate", help="force to simulate", action="store_true")
group.add_argument("-r","--recover", help="recover from data", action="store_true")
parser.add_argument("--show", help="plot iterations", action="store_true")
parser.add_argument("--logfile", help="output log to run.log", action="store_true")
parser.add_argument("-i","--iteration", help="the iteration number", type=int, default=10)
parser.add_argument("--savestep", help="save data every n step", type=int, default=1)
parser.add_argument("--seed", help="the iteration number", type=int)
args = parser.parse_args()
# if not args.config:
config = configparser.ConfigParser()

if args.config:
    config.read(args.config)
    S = config.getint("make_data","detsize")
    N = config.getint("make_data","num_data")
    flu = config.getfloat("make_data","fluence")
else:
    args.config = "config_tmp.ini"
    config.add_section("make_data")
    S = 100
    N = 2000
    flu = 2000
    config.set("make_data", "detsize", str(S))
    config.set("make_data", "num_data", str(N))
    config.set("make_data", "fluence", str(flu))
    config.set("make_data", "intens", "(np.sin(x)**2)*(1+np.cos(x * 7.3))")
    # config.set("make_data", "intens", "(np.sin(0.5*x)**2)")

if config.has_option("output", "main_folder"):
    main_folder = config["output" ]["main_folder"]
else:
    main_folder = os.path.dirname(os.path.abspath(args.config))

if args.logfile:
    logging.basicConfig(
        filename=f"{main_folder}/run.log",
        format=FORMAT,level=logging.DEBUG)
else:
    logging.basicConfig(
            format=FORMAT,level=logging.DEBUG)

print(f"main_folder: {main_folder}")
if not config.has_section("output"):
    config.add_section("output")
if config.has_option("output", "inten_folder"):
    inten_folder = config["output"]["inten_folder"]
else:
    inten_folder = main_folder + "/output"
    config.set("output", "inten_folder", inten_folder)

if config.has_option("output", "probabilities_folder"):
    prob_folder = config["output"]["probabilities_folder"]
else:
    prob_folder = main_folder +  "/probabilities"
    config.set("output", "probabilities_folder", prob_folder)

for i in [inten_folder, prob_folder]:
    if not os.path.isdir(i):
        os.makedirs(i)
        logging.info("make new folder: %s", i)
    # print(inten_folder)
# with open(args.config, "w") as cfgfile:
    # config.write(cfgfile)

if not os.path.exists(main_folder + "/intensities.npy") or \
   not os.path.exists(main_folder + "/true_oriens.npy") or \
   not os.path.exists(main_folder + "/patterns.npy") or \
    args.simulate:
    logging.info("Do simulation")
    patterns, true_intens, true_oriens = simulate(N, S, flu, config["make_data"]["intens"])
    np.save(main_folder + "/intensities" , true_intens)
    np.save(main_folder + "/true_oriens", true_oriens)
    np.save(main_folder + "/patterns", patterns)
else:
    logging.info("read patterns from file")
    true_intens = np.load(main_folder + "/intensities.npy")
    true_oriens = np.load(main_folder + "/true_oriens.npy")
    patterns = np.load(main_folder + "/patterns.npy")

start = -1
if args.recover:
    try:
        prog = re.compile(r"(\d+)\.npy$")
        start_tmp = max([int(prog.search(i).group(1)) for i in glob.glob(inten_folder + "/intens*")])
        recon = np.load(inten_folder + "/intens_%07d.npy" % start_tmp)
        start = start_tmp + 1
        logging.info("start from %d", start)
    except:
        pass
if args.seed:
    logging.info("fix random seed: %d", args.seed)
    np.random.seed(args.seed)
if start <0 or not args.recover:
    recon = np.random.rand(patterns.shape[1])*2*np.average(patterns)
    start = 1
    for i in itertools.chain(
            glob.glob(inten_folder+"/*"),
            glob.glob(prob_folder +"/*")):
        os.remove(i)
    logging.info("Start randomly")


def save_strategy(g):
    def ss(recon):
        i = start
        for j in g(recon):
            if (i-start+1) % args.savestep == 0:
                np.save(inten_folder + "/intens_%07d" % i, j[0])
                np.save(prob_folder + "/prob_%07d" % i, j[1])
            i += 1
            yield j
    return ss


@save_strategy
def get_iter(r):
    recon = r.copy()
    fluence = np.average(np.sum(patterns, axis=1))
    logging.info("Size: %d, Number of patterns: %d, Total intensities: %f, Photons per patterns: %f", S, N, flu, np.mean(np.sum(patterns, axis=1)))
    i = start
    while True:
        recon_next, prob = EMC(recon, patterns, fluence)
        logging.info("%07dth iter, recon change: %.3E", i, np.mean((recon_next-recon)**2))
        recon = recon_next
        i += 1
        yield recon, prob

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def update_intens_plot(o):
    recon = o[0]
    n_min = 0
    val = 1e100
    for i in range(len(recon)):
        val_tmp = np.sum((np.roll(recon, i)-true_intens)**2)
        if val_tmp < val:
            n_min = i
            val = val_tmp
    line.set_ydata(np.roll(recon, n_min))
    tmp_prob = np.sqrt(o[1][:S*4]).T
    tmp_prob -= np.min(tmp_prob)
    tmp_prob /= np.max(tmp_prob)
    prob_array.set_array(tmp_prob)

    temp_x = np.arange(patterns.shape[1])-0.5
    heights = []
    for i in range(n_sample):
        roll = np.argmax(o[1][i]) + n_min
        heights.append(np.roll(patterns[i], roll))
        # example_patterns[i].(ax[0].bar(temp_x, height, bottom=bottom))
    bottom = np.zeros(patterns.shape[1])
    heights = np.array(heights)
    fact = max(recon)/max(np.sum(heights,axis=0))
    heights *= fact
    for i in range(n_sample):
        for rect, h, b in zip(example_patterns[i], heights[i], bottom):
            rect.set_height(h)
            rect.set_y(b)
            # example_patterns[i].set_bottom(bottom)
        bottom += heights[i]
    return line, prob_array

# Init only required for blitting to give a clean slate.
def init_intens_plot():
    # line.set_ydata(np.ma.array(recon, mask=True))
    line.set_ydata(recon)
    return line,

anim_running = True
def onClick(event):
    global anim_running
    if anim_running:
        anim.event_source.stop()
        anim_running = False
    else:
        anim.event_source.start()
        anim_running = True

if args.show:
    fig, ax = plt.subplots(2,1,figsize=(10,6))
    fig.canvas.mpl_connect('button_press_event', onClick)
    line, = ax[0].plot(recon)
    prob_array = ax[1].imshow(np.random.rand(S,S*4), animated=True)
    ax[1].set_xlabel("patterns")
    ax[1].set_ylabel("probability")
    ax[0].plot(true_intens)
    ax[0].set_xlabel("detector")
    ax[0].set_ylabel("intens")
    plt.tight_layout()
    example_patterns = []
    n_sample = 50
    temp_x = np.arange(patterns.shape[1])-0.5
    fact = max(recon)/max(np.sum(patterns[:n_sample],axis=0))
    bottom = np.zeros(patterns.shape[1])
    color_sample = np.array(list(map(
        matplotlib.cm.get_cmap('viridis'),np.linspace(0,1,n_sample))))
    for i in range(n_sample):
        example_patterns.append(ax[0].bar(temp_x, patterns[i]*fact, bottom=bottom, color=color_sample[i]))
        bottom += patterns[i]*fact

    ax[0].set_ylim([0, np.max(true_intens)*1.2])

    anim = animation.FuncAnimation(fig, update_intens_plot,frames=get_iter(recon), init_func=init_intens_plot,
                                  interval=25)
    plt.show()
else:
    for i, (recon, prob) in enumerate(itertools.islice(get_iter(recon), args.iteration), start=start):
        pass
    np.save(inten_folder + "/intens_%07d" % i, recon)
    np.save(prob_folder + "/prob_%07d" % i, prob)
