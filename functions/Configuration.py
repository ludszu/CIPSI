
import numpy as np
        
class Configuration(object):
    def __init__(self,secQformU,secQformD,firQformU,firQformD):
        
        '''
        INPUTS: - secQform - second quantised form e.g. (1,3,5)
                - firQform - first quantised form e.g. (1,0,1,0,1)        
        '''
        
        # number of occupied states has to equal number of electrons for each spin
        if ((np.sum(firQformU==1)==len(secQformU)) and (np.sum(firQformD==1)==len(secQformD))):
            self.Unumbers=secQformU.astype(int) # configuration e.g. (1,3,5) U
            self.Ufilled=firQformU.astype(int) # occupiations e.g. (1,0,1,0,1) U 
            self.Dnumbers=secQformD.astype(int) # configuration e.g. (1,3,5) D
            self.Dfilled=firQformD.astype(int) # occupiations e.g. (1,0,1,0,1) D
            self.Unel=np.sum(firQformU==1) # number of electrons U
            self.Dnel=np.sum(firQformD==1) # number of electrons D
            self.Neltot=self.Unel+self.Dnel # total number of electrons
            self.Sz=(self.Unel-self.Dnel)/2 # total Sz
            self.Unen=len(firQformU) # number of SP energy levels spin U
            self.Dnen=len(firQformD) # number of SP energy levels spin D
            self.QNums=[] # vector of quantum numbers
        else:
            raise ValueError("error in 1st and 2nd quantisation form")
        
    @classmethod
    def fromSecQ(cls,confU,confD,NenU,NenD):
        # makes first quantised form and creates configuration
        fillU=np.zeros(NenU,dtype=int)
        fillD=np.zeros(NenD,dtype=int)
        fillU[confU]=1
        fillD[confD]=1
                      
        return cls(confU,confD,fillU,fillD)
        
    @classmethod
    def fromFirQ(cls,fillU,fillD):
        # makes second quantised form and creates configuration
        NU=len(fillU)
        ND=len(fillD)
        numbersU=np.arange(0,NU)
        numbersD=np.arange(0,ND)
        confU=numbersU[fillU==1]
        confD=numbersD[fillD==1]
                      
        return cls(confU,confD,fillU,fillD)
    
    def __str__(self):
        out="U: "+str(self.Unumbers)+", D:"+ str(self.Dnumbers)
        return out
        
    def __eq__(self, other):
        isit=isinstance(other, Configuration)
        if (isit):
            iff=(self.Ufilled==other.Ufilled).all() and (self.Dfilled==other.Dfilled).all()
        return iff
            
    def copy(self):
        return self.__class__(self.Unumbers,self.Dnumbers,self.Ufilled,self.Dfilled)   
    
    def to_vector(self):
        # returns a vector form of 1st quantised form of configuration
        NenU=self.Unen
        NenD=self.Dnen
        vec=np.zeros(NenU+NenD,dtype=int)
        vec[0:NenU]=self.Ufilled
        vec[NenU:]=self.Dfilled
        return vec
    
    def to_vector_2ndQ(self):
        # returns a vector form of 2nd quantised form of configuration
        NelU=self.Unel
        NelD=self.Dnel
        vec=np.zeros(NelU+NelD,dtype=int)
        vec[0:NelU]=self.Unumbers
        vec[NelU:]=self.Dnumbers
        return vec
    
    @classmethod
    def from_vector(cls,vec,NenU):
        # uses 1st quantised form vector to build a configuration
        u01=vec[0:NenU]
        d01=vec[NenU:]
        
        return cls.fromFirQ(u01,d01)
        
    def calculateQNums(self,Lvec,Vvec):
        # calculates qnumbers of a configuration: 
        # angular momentum and valley index
        QNums=np.zeros(2)
        QNums[0]=np.sum(Lvec[self.Unumbers])-np.sum(Lvec[self.Dnumbers])
        QNums[1]=np.sum(Vvec[self.Unumbers])-np.sum(Vvec[self.Dnumbers])
            
        self.QNums=QNums
        
    def moveElectron(self,xor,t):
        '''
        modifies configuration by moving one electron
        INPUTS: - xor - vector used for a XOR operation
                - t - spin
        OUTPUTS: - configuration
                 - is the move valid?
        '''
        
        xoru=xor[0:self.Unen]
        xord=xor[0:self.Dnen]
        
        # spin UP
        if(t>-1):
            # if more or less than 2 positions in xor are non0 - 
            # something wrong. copy the initial configuration
            if(np.sum(xoru)!=2):
                ifValid=False
                conf=self.copy()
            else:
                new=np.bitwise_xor(self.Ufilled,xoru)
                if1=np.sum(new)==np.sum(self.Ufilled)
                if(if1):
                    conf=Configuration.fromFirQ(new,self.Dfilled)
                    ifValid=True
                else:
                    ifValid=False
                    conf=self.copy()
        # spin DN        
        elif(t==-1):
            # if more or less than 2 positions in xor are non0 - 
            # something wrong. copy the initial configuration
            if(np.sum(xord)!=2):
                ifValid=False
                conf=self.copy()
            else:
                new=np.bitwise_xor(self.Dfilled,xord)
                if1=np.sum(new)==np.sum(self.Dfilled)
                if(if1):
                    conf=Configuration.fromFirQ(self.Ufilled,new)
                    ifValid=True
                else:
                    ifValid=False
                    conf=self.copy()
                    
        else:
            conf=self.copy()
            ifValid=False
            
        return conf,ifValid
            
        
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 13:11:57 2021

@author: ludka
"""
