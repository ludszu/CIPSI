import numpy as np
import random
from functions.Configuration import Configuration

        
# LINEAR COMBINATION OF CONFIGURATIONS
class LCombination(object):
    def __init__(self,coefs,configs):
        '''
        INPUTS: - coefficients of the linear combination
                - configurations of the linear combination
        '''        
        
        # check length of inputs
        if len(coefs) != len(configs):
            raise ValueError('Must provide a equal number of coefs and configs.')
        
        # check if all configurations preserve basic parameters
        c1=configs[0]
        params=[c1.Unel,c1.Dnel,c1.Unen,c1.Dnen,c1.Sz]
        for cfg in configs:
            parami=[cfg.Unel,cfg.Dnel,cfg.Unen,cfg.Dnen,cfg.Sz]
            if not (parami==params):
                if (cfg.Neltot!=0):
                    raise ValueError('NelU,NelD,NeneU,NeneD,Sz must be conserved.')
        
        
        self.coefs=np.array(coefs).astype(complex) # coefficients
        self.configs=configs # configurations
        self.nterms=len(coefs) # number of terms
        self.Unel=c1.Unel # number of electrons spin U
        self.Dnel=c1.Dnel # number of electrons spin D
        self.Neltot=self.Unel+self.Dnel # total number of electrons
        self.Sz=c1.Sz # total Sz
        self.Unen=c1.Unen # number of SP energy levels spin U
        self.Dnen=c1.Dnen # number of SP energy levels spin D
        self.QNums=None # initialise quantum numbers
    
    @classmethod
    def fromMatrixSecQ(cls,matrix,NelU,NelD,NenU,NenD,coefs=None):
        '''
        INPUTS: - matrix of configurations as rows
                - number of electrons of each spin
                - number of SP energy levels of each spin
                - coefficients (if None, they'll be ones)
        '''
        
        N=matrix.shape[0]
        confs=[]
        for i in range(0,N):
            row=matrix[i,:]
            confU=row[0:NelU]
            confD=row[NelU:]
            conf=Configuration.fromSecQ(confU,confD,NenU,NenD)
            confs.append(conf)
            
        if(coefs is None):
            coefs=np.ones(N)
        
        return cls(coefs,confs)
    
    def __str__(self):        
        out=str(self.nterms)+" configs for "+str(self.Neltot)+" electrons on "
        out+=str(self.Unen)+ "U, "+str(self.Dnen)+"D energies with Sz="+str(self.Sz)+":" 
        i=0
        for coef,cfg in zip(self.coefs,self.configs):
            if (self.QNums is not None):
                l=self.QNums[i,0]
                v=self.QNums[i,1]
                out+='\n' + f'{str(cfg)}, coef: ({np.real(coef):+.5f},{np.imag(coef):+.5f})  L: {int(l)}, V: {int(v)} '
            else:
                out+='\n' + f'{str(cfg)}, coef: ({np.real(coef):+.5f},{np.imag(coef):+.5f})'
            i+=1
            
        return out
                
    def __add__(self,other):
        # concatenates a linear combination with another
        if not isinstance(other,LCombination):
            raise ValueError('Can only add with another LCombination')
          
        params=[self.Unel,self.Dnel,self.Unen,self.Dnen,self.Sz]
        params2=[other.Unel,other.Dnel,other.Unen,other.Dnen,other.Sz]
        if not (params2==params):
            raise ValueError('Can only add with LCombination of equal params')   
            
        new_coefs = np.concatenate((self.coefs,other.coefs))
        new_configs = np.concatenate((self.configs,other.configs))
        
        return self.__class__(new_coefs, new_configs)
    
    def copy(self):
        return self.__class__(self.coefs, self.configs)
    
    def setDiff(self,other):
        # find set difference of a linear combination with another
        if not isinstance(other,LCombination):
            raise ValueError('Can only find difference with another LCombination')
            
        selfRow=set(map(tuple, self.to_matrix_2ndQ()))
        otherRow=set(map(tuple, other.to_matrix_2ndQ()))
        
        sdiff=selfRow.difference(otherRow)
        result=np.array(list(sdiff))
        
        return result
      
    def combine(self):
        # "reduces" a linear combination to only unique configurations
        
        new_coefs = new_confs = None

        matrix=self.to_matrix()
        n0 = self.nterms
        
        unique, unique_inds, unique_inv = np.unique(matrix, \
                return_index = True, return_inverse = True, axis = 0)
        n = unique.shape[0]
        
        NenU=self.Unen
        new_confs = np.zeros(n,dtype = Configuration)
        
        for i in range(0,n):
            new_confs[i] = Configuration.from_vector(unique[i,:],NenU)
        
        new_coefs = np.zeros(n,dtype = complex)
        
        for i in range(0,n0):
            ind = unique_inv[i]
            val = self.coefs[i]
            new_coefs[ind] += val
        
        
        return self.__class__(new_coefs, new_confs)  
    
    def sort(self):
        # sorts the configurations in a linear combination using binary numbers
        order = None
        matrix=self.to_matrix()
        
        base = 2**np.arange(matrix.shape[1])
        decimals=matrix.astype(int) @ base
        order = np.argsort(decimals)
        
        new_coefs = self.coefs[order]
        new_configs = self.configs[order]

        return self.__class__(new_coefs, new_configs)
    
    def to_matrix(self):
        # translated a linear combination to a matrix using 1st quantisation
        mat = np.zeros((self.nterms, self.Unen+self.Dnen), dtype=int)

        n = self.nterms
        for i in range(0,n):
            mat[i,:] = self.configs[i].to_vector()
        
        return mat
    
    def to_matrix_2ndQ(self):
        # translated a linear combination to a matrix using 2nd quantisation
        mat = np.zeros((self.nterms, self.Unel+self.Dnel), dtype=int)

        n = self.nterms
        for i in range(0,n):
            mat[i,:] = self.configs[i].to_vector_2ndQ()
        
        return mat

    def calculateQNums(self,Lvec,Vvec):
        # calculates Qnumbers
        QNums=np.zeros((self.nterms,2))
        i=0
        for cfg in self.configs:
            cfg.calculateQNums(Lvec,Vvec)
            QNums[i,:]=cfg.QNums
            i+=1
            
        self.QNums=QNums
        
    def shuffle(self):
        # shuffles the linear combination randomly
        inds = np.arange(1,self.nterms)
        random.shuffle(inds)
        self.configs[1:self.nterms]=self.configs[inds]
        self.coefs[1:self.nterms]=self.coefs[inds]
        return
        

        
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 13:11:57 2021

@author: ludka
"""
