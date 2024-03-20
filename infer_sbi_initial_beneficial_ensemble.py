import random
import argparse
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal
import scipy.stats
import scipy.optimize
import seaborn as sns
import os
from PyPDF2 import PdfFileMerger
from scipy.special import logsumexp
import pyabc.visualization
import pickle

from cnv_simulation_initial_beneficial import CNVsimulator_simpleWF


import sbi.utils as utils
from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble as Ensemble

from sbi.inference.base import infer
from sbi.inference import SNPE, prepare_for_sbi
import torch

import arviz as az
from sbi.utils import MultipleIndependent
from sbi import analysis as analysis


#### arguments ####
parser = argparse.ArgumentParser()
parser.add_argument('-m', "--model")
parser.add_argument('-pd', "--presimulated_data")
parser.add_argument('-pt', "--presimulated_theta")
parser.add_argument('-s', "--seed")
parser.add_argument('-g', "--generation_file")
parser.add_argument('-e', "--ensemble_size")
args = parser.parse_args()

argseed = int(args.seed)
random.seed(int(argseed))
presim_data = str(args.presimulated_data)
presim_theta = str(args.presimulated_theta)
EvoModel = str(args.model)
g_file = str(args.generation_file)
ensemble_size = int(args.ensemble_size)

#####other parameters needed for model #####
# pop size, fitness SNVs, mutation rate SNVs, number of generations
N = 3.3e8
reps=1
generation=np.genfromtxt(g_file,delimiter=',', skip_header=1,dtype="int64")

#### prior ####
prior_min = np.log10(np.array([1e-2,1e-7,1e-8]))
prior_max = np.log10(np.array([1,0.5,1]))
prior = utils.BoxUniform(low=torch.tensor(prior_min), 
                         high=torch.tensor(prior_max))

posterior_list = []

#### sbi simulator ####
def CNVsimulator(cnv_params):
    cnv_params = np.asarray(torch.squeeze(cnv_params,0))
    reps = 1
    if EvoModel == "WF":
        states = CNVsimulator_simpleWF(reps = reps, N=N, generation=generation, seed=None, parameters=cnv_params)
    return states

#make sure simulator and prior adhere to sbi requirements
simulator, prior = prepare_for_sbi(CNVsimulator, prior)

#### get presimulated data ####
theta_presimulated = torch.tensor(np.genfromtxt(presim_theta,delimiter=',')).type('torch.FloatTensor')
x_presimulated = torch.tensor(np.genfromtxt(presim_data,delimiter=',')).type('torch.FloatTensor')

# Resources of each network in the ensemble
stop_after_epochs = 100


def train(i):
    #### run inference ####
    if i=='ensemble':
        inference = Ensemble(posterior_list)
        posterior = inference
    else:
        inference = SNPE(prior, density_estimator='maf')
        density_estimator = inference.append_simulations(theta_presimulated, x_presimulated).train(stop_after_epochs=stop_after_epochs)
        posterior = inference.build_posterior(density_estimator)
        posterior_list.append(posterior)


    #### save posterior ####
    ending = f'Chuong_ensemble{stop_after_epochs}_{i}'
    with open(f"posteriors/ensemble/{ending}.pkl", "wb") as handle:
        pickle.dump(posterior, handle)

    #### Get Training and Validation losses ####
    if i != 'ensemble':
        dict = {'train loss':np.array(inference.summary['training_log_probs']), 'validation loss':np.array(inference.summary['validation_log_probs'])}
        losses = pd.DataFrame(dict)
        losses.to_csv(f'losses/ensemble/losses_{stop_after_epochs}_epochs_{i}.csv', index = False)

for i in range(ensemble_size):
    train(str(i))

train('ensemble')
