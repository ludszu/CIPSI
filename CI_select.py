#%%
import numpy as np
from matplotlib import pyplot as plt
import functions.CIPSI as CI
import functions.LCombination as LCombination

#%% PROBLEM SETUP

Neltot=2 # total number of electrons
Sz=0 # total Sz of the system
nShell=4 # number of SP energy shells to consider (2D Qoscillator like)

NelU=int(Neltot/2+Sz) # number of electrons U
NelD=int(Neltot/2-Sz) # number of electrons D
# number of SP energy levels follow from shells
NeneU=np.sum(range(1,nShell+1))*2
NeneD=np.sum(range(1,nShell+1))*2
# set maximum number of excitations for full CI
maxX=Neltot

Eshift=0 # shift SP energies
screen=1#2.5 # dielectric constant
folder="R200V300" # working folder
nEne=6 # number of eigenenergies sought from sparse diagonalisation

#%% PARAMETERS

# input files
files=np.empty((5,1),dtype='S17')
files[0]=folder+"\EmaU.dat" # SP energies UP
files[1]=folder+"\EmaD.dat" # SP energies DN
files[2]=folder+"\VSOU.dat" # CME for spin UU
files[3]=folder+"\VSOD.dat" # CME for spin DD
files[4]=folder+"\VSUD.dat" # CME for spin UD
files2=np.empty((2,1),dtype='S17')
files2[0]=folder+"\Lfil.dat" # angular momentum
files2[1]=folder+"\Vfil.dat" # valley index

maxiter=10 # max CIPSI interation number
# threshold for the perturbation theory criterion for new configurations
thresh=1e-4 
# number of anticipated non0 elements in each row of Hamiltonian
sparseLeng=100

# initial state
cu=np.array([0])
cd=np.array([0])
c0=CI.Configuration.fromSecQ(cu,cd,NeneU,NeneD)
cu=np.array([1])
cd=np.array([0])
c1=CI.Configuration.fromSecQ(cu,cd,NeneU,NeneD)
cu=np.array([2])
cd=np.array([0])
c2=CI.Configuration.fromSecQ(cu,cd,NeneU,NeneD)
cu=np.array([4])
cd=np.array([0])
c3=CI.Configuration.fromSecQ(cu,cd,NeneU,NeneD)
cu=np.array([5])
cd=np.array([0])
c4=CI.Configuration.fromSecQ(cu,cd,NeneU,NeneD)
start=CI.LCombination(np.ones(5),[c0,c1,c2,c3,c4])

paramsCI=np.array([Neltot,Sz,NeneU,NeneD,maxX])
params=np.array([Eshift,screen,maxiter,thresh,sparseLeng,nEne])
CI1=CI.CIPSI(files,files2,paramsCI,params,start)

#%%
# MAIN CIPSI LOOP
CI1.iterate()
v=CI1.vFin
e=CI1.eFin
mix=CI1.mix



# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 10:26:25 2021

@author: ludka
"""

