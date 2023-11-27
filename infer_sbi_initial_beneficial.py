 # -*- coding: utf-8 -*-
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
parser.add_argument('-n', "--name")
args = parser.parse_args()

argseed = int(args.seed)
random.seed(int(argseed))
presim_data = str(args.presimulated_data)
presim_theta = str(args.presimulated_theta)
EvoModel = str(args.model)
g_file = str(args.generation_file)
name = str(args.name)

#####other parameters needed for model #####
# pop size, fitness SNVs, mutation rate SNVs, number of generations
N = 3.3e8
reps=1
generation=np.genfromtxt(g_file,delimiter=',', skip_header=1,dtype="int64")

#### prior ####
prior_min = np.log10(np.array([1e-3,1e-9,1e-8]))
prior_max = np.log10(np.array([1,0.5,1e-2]))
prior = utils.BoxUniform(low=torch.tensor(prior_min), 
                         high=torch.tensor(prior_max))


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
theta_presimulated = torch.tensor(np.genfromtxt('presimulated_data/'+presim_theta,delimiter=',')).type('torch.FloatTensor')
x_presimulated = torch.tensor(np.genfromtxt('presimulated_data/'+presim_data,delimiter=',')).type('torch.FloatTensor')

# Training stops after 100 unimproved epochs
stop_after_epochs = 100

#### run inference ####
inference = SNPE(prior, density_estimator='maf')
density_estimator = inference.append_simulations(theta_presimulated, x_presimulated).train(stop_after_epochs=stop_after_epochs)
posterior = inference.build_posterior(density_estimator)

#### save posterior ####
with open(f"posteriors/posterior_{name}_{stop_after_epochs}.pkl", "wb") as handle:
    pickle.dump(posterior, handle)

#### Get Training and Validation losses ####
dict = {'train loss':np.array(inference.summary['training_log_probs']), 'validation loss':np.array(inference.summary['validation_log_probs'])}
losses = pd.DataFrame(dict)
losses.to_csv(f'losses/losses_{stop_after_epochs}_epochs_{name}.csv', index = False)


