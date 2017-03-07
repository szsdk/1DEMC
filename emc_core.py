import numpy as np
import os
import logging

def EMC(recon_in, patterns, fluence):
    recon = recon_in.copy()
    recon[recon==0] = 1e-100
    S = recon.shape[0]
    N = patterns.shape[0]
    recons = np.zeros([S,S])
    for i in range(S):
        recons[i] = np.roll(recon, -i)

    recons = np.log(recons)
    prob = np.dot(patterns, recons.T)
    
    with np.errstate(divide='raise'):
        prob = np.exp(prob - np.max(prob, axis=1).reshape(N,1))
        prob /= np.sum(prob, axis=1).reshape(N,1)

    new_patterns = np.zeros([S,S])
    probn = np.sum(prob, axis=0)

    for s in range(S):
        new_patterns[s] = np.sum(patterns * prob[:,s].reshape(N,1), axis=0)/ np.sum(prob[:,s])
        new_patterns[s] = np.roll(new_patterns[s], s)

    new_recon = np.average(new_patterns, axis=0)
    new_recon = new_recon / np.sum(new_recon) * fluence
    return new_recon, prob

def simulate(N, S, flu, intens):
    # true_intens = (np.sin(x)**2)*(1+np.cos(x * 7.4)) * flu / S
    # x = np.linspace(0, 2*np.pi, S)
    if os.path.isfile(intens):
        true_intens = np.load(intens)
        if not true_intens.shape[0] ==  S:
            raise Exception(" The length of intens file should match S.")
        logging.info("read intens from file %s", intens)
    else:
        x = np.linspace(0, 2*np.pi, S)
        true_intens = eval(intens)
        logging.info("generate the intens by %s", intens)
    true_intens = true_intens / np.sum(true_intens) * flu / S
    patterns = np.zeros([N,S])
    for i in range(S):
        patterns[:,i] = np.random.poisson(true_intens[i], N)

    true_oriens = np.random.randint(S, size=N)
    for i,o in enumerate(true_oriens):
        patterns[i] = np.roll(patterns[i], -o)

    return patterns, true_intens, true_oriens
