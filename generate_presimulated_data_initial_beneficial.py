# -*- coding: utf-8 -*-

import random
import argparse

import numpy as np
from numpy.random import normal
import os

from cnv_simulation_initial_beneficial import CNVsimulator_simpleWF

import sbi.utils as utils
from sbi.inference.base import infer
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
import torch

parser = argparse.ArgumentParser()
parser.add_argument('-n', "--name")
parser.add_argument('-p', "--presimulate")
parser.add_argument('-m', "--model")
parser.add_argument('-g', '--generation_file')
args = parser.parse_args()
name = str(args.name)
n_presim = int(args.presimulate)
EvoModel = str(args.model)
g_file = str(args.generation_file)

#####other parameters needed for model #####
# pop size, fitness SNVs, mutation rate SNVs, number of generations
N = 3.3e8
reps=1
generation=np.genfromtxt(g_file,delimiter=',', skip_header=1,dtype="int64")

#### prior ####
prior_min = np.log10(np.array([1e-2,1e-7,1e-8]))
prior_max = np.log10(np.array([1,1e-2,1e-2]))
prior = utils.BoxUniform(low=torch.tensor(prior_min), 
                         high=torch.tensor(prior_max))


#### sbi simulator ####
def CNVsimulator(cnv_params):
    cnv_params = np.asarray(torch.squeeze(cnv_params,0))
    reps = 1
    if EvoModel == "WF":
        states = CNVsimulator_simpleWF(reps = reps, N=N, generation=generation, seed=None, parameters=cnv_params)
    return states[0]

simulator, prior = prepare_for_sbi(CNVsimulator, prior)

theta_presimulated, x_presimulated = simulate_for_sbi(simulator, proposal=prior, num_simulations=n_presim, num_workers=1)


#save presimulated thetas and data to csvs
np.savetxt('presimulated_data/' + EvoModel+"_presimulated_theta_"+str(n_presim)+ "_" + name +".csv", theta_presimulated.numpy(), delimiter=',')
np.savetxt('presimulated_data/' + EvoModel+"_presimulated_data_"+str(n_presim)+"_" + name +".csv", x_presimulated.numpy(), delimiter=',')
