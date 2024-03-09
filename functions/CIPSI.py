
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as la
from scipy import interpolate as itp
from scipy import sparse as sp
from scipy import special as sc
import scipy.sparse.linalg as spla
import random
from functions.LCombination import LCombination
from functions.Configuration import Configuration


def combs(n,k,nX):
    # creates k-electron combinations on n SP levels, 
    # for maximum nX of excitations
    hm=sc.comb(n,k,exact=True)
    co=np.zeros((hm,k),dtype=int)
    tags=np.ones(hm,dtype=int)
    Xs=np.zeros(hm,dtype=int)
    
    for i in range(1,hm+1):
        ix=i-1
        kx=k
        for s in range(1,n+1):
            if (kx==0):
                break
            else:
                t=sc.comb(n-s,kx-1,exact=True)
                if (ix<t):
                    co[i-1,k-kx]=s
                    kx=kx-1
                else:
                    ix=ix-t
        if (k>nX):
            if (co[i-1,k-nX-1]>k): tags[i-1]=0
        
        Xs[i-1]=sum(co[i-1,:]>k)
        
    return co,tags,Xs
    
    
    
class CIPSI:
    def __init__(self,files,files2,paramsCI,params,initial=None):
        '''
        INPUTS: files: string array (fixed length strings) (5,1):
                        - SP energies U
                        - SP energies D
                        - CME tensor for spin UU
                        - CME tensor for spin DD
                        - CME tensor for spin UD
                files2: files for quantum numbers: angular momentum and valley index
                paramsCI: [Neltotal,Sz,NESPU,NESPD,maxX]
                        - total electron number
                        - total Sz of the system
                        - number of SP energy levels spin U
                        - number of SP energy levels spin D
                params: [Eshift,screen,nEne*,sparseLeng*,omega*] *optional or used for sparse diagonalisation
                        - SP energy shift (if negative, shifts by EmatU[0])
                        - dielectric constant
                        - number of energies to obtain from sparse matrix diagonalisation
                        - estimated number of nonzero elements in the row of Hamiltonian matrix
                        - unitless oscillator frequency for a 2D Qwell problem
                initial: starting linear combination of configurations (optional)
        '''
    
        self.Neltotal=int(paramsCI[0]) # total electron number
        self.Sz=paramsCI[1] # total Sz of electron system	
        self.NESPU=int(paramsCI[2]) # number of SP energy levels spin U
        self.NESPD=int(paramsCI[3]) # number of SP energy levels spin D
        self.NeltotalU=int(self.Neltotal/2+self.Sz) # total spin U electron number 
        self.NeltotalD=int(self.Neltotal/2-self.Sz) # total spin D electron number 
        
        print("Neltot U: ",self.NeltotalU)
        print("Neltot D: ",self.NeltotalD)
        print("NESPU: ",self.NESPU)
        print("NESPD: ",self.NESPD)
        
        self.NstVU=self.NeltotalU # number of filled VB SP states spin U
        self.NstVD=self.NeltotalD # number of filled VB SP states spin D
        self.NstCU=self.NESPU-self.NstVU # number of empty CB SP states spin U
        self.NstCD=self.NESPD-self.NstVD # number of empty CB SP states spin D
        self.NelU=self.NstVU # number of electrons promoted up spin U
        self.NelD=self.NstVD # number of electrons promoted up spin D
        
        self.VmatUU=[] # Coulomb matrix element tensor spin UU
        self.VmatDD=[] # Coulomb matrix element tensor spin DD
        self.VmatUD=[] # Coulomb matrix element tensor spin UD
        self.EmatU=[] # SP energies spin U
        self.EmatD=[] # SP energies spin D

        self.noConfU=[] # number of possible spin U configurations
        self.noConfD=[] # number of possible spin D configurations
        self.allConfs=[] # all configurations U&D
        
        self.Eshift=params[0] # SP energy shift (if negative, shifts by EmatU[0])
        self.screen=params[1] # dielectric constant
        self.maxiter=int(params[2]) # maximum number of CIPSI iterations
        # matrix element threshold for ignoring unimportant configurations
        self.thresh=params[3] 
        # estimated number of nonzero elements in the row of Hamiltonian matrix
        self.sparseLeng=int(params[4])
        # number of eigenenergies sought from diagonalisation
        self.nEne=int(params[5]) 
        
        self.ham=[] # current Hamiltonian
        self.mix=[] # current linear combination of configurations
        self.eFin=[] # final eigenenergies
        self.vFin=[] # final eigenvectors
        
        self.initial=initial # starting linear combination of configurations
        if (initial is not None):
            self.initSize=initial.nterms # starting configuration basis size
        else:
            self.initSize=0
        
        
        self.loadInputs(files,files2,self.Eshift)
        self.countConfigs()
    
    def iterate(self):
        # MAIN CIPSI LOOP
        
        maxiter=self.maxiter
        # if there is no initial LCombination, use the GS
        if (self.initial is None):
            GSu=np.arange(0,self.NelU)
            GSd=np.arange(0,self.NelD)
            GS=Configuration.fromSecQ(GSu,GSd,self.NESPU,self.NESPD)
            mix=LCombination([1],[GS])
        else:
            mix=self.initial
        
        # make an diagonalise Hamiltonian for the initial LCombination
        ham=self.makeCIham(mix)
        if (ham.shape[0]>(self.nEne+1)):
            e,v=spla.eigsh(ham.tocsr(),k=self.nEne,sigma=0)
        else:
            e,v=la.eig(ham.toarray())
                    
        en=np.sort(e)       
        mix.calculateQNums(self.Lvec,self.Vvec)
        print('init:',mix)
        
        # MAIN LOOP
        for i in range(0,maxiter):
            print("iter: ",i,", eGS:",np.real(en[0]),", selected mix:",mix)
            
            # find all connecting configurations by moving up to 2 electrons
            # we obtain off diagonal elements between new and old configurations
            new,offDiags=self.connect(mix,np.real(en[0]))
            
            # check if there are any new connected configurations to add
            if (new!=None):
                print("    selected ",new.nterms,"new configurations")
                
                # make and diagonalise Hamiltonian for a new expanded LCombination
                ham=self.makeNewCIham(mix,new,offDiags)
                if (ham.shape[0]>(self.nEne+1)):
                    e,v=spla.eigsh(ham.tocsr(),k=self.nEne,sigma=0)
                else:
                    e,v=la.eig(ham.toarray())
                    
                # replace old LCombination with a new one
                mix=mix+new
                mix.calculateQNums(self.Lvec,self.Vvec)
                order=np.argsort(e)
                en=e[order]
                w=v[:,order]
                mix.coefs=w[:,0]
                
                print('iter: ', i,', GS ENERGY: ',np.real(en[0]))
                
                if (i==(maxiter-1)):
                    print("MAXITER REACHED")
            
            else:
                print("NO NEW CONFIGS, DIM:",ham.shape[0], 'FOR I:', i)
                self.ham=ham
                break
            
        
        self.eFin=e
        self.vFin=w
        self.mix=mix
        mix.calculateQNums(self.Lvec,self.Vvec)
        QNums=mix.QNums
        print('Eigenvalues: ',np.real(en)) 
        print('GS ENERGY: ',np.real(en[0]))   
        print("END OF LOOP")
         
        return
        
    def loadInputs(self,files,files2,Eshift):
        # INPUTS: as in __init__        
        
        filEU=str(files[0],encoding='ascii')
        filED=str(files[1],encoding='ascii')
        filVUU=str(files[2],encoding='ascii')
        filVDD=str(files[3],encoding='ascii')
        filVUD=str(files[4],encoding='ascii')
        filL=str(files2[0],encoding='ascii')
        filV=str(files2[1],encoding='ascii')
        
        # READ SP ENERGIES
        EmatU=np.loadtxt(filEU)
        EmatD=np.loadtxt(filED)
        if (Eshift<0):
            Eshift=EmatU[0]
            
        EmatU=EmatU-Eshift
        EmatD=EmatD-Eshift
        
        Lvec=np.loadtxt(filL)
        Vvec=np.loadtxt(filV)
        
        # READ CME for 2 electrons of spin UU, DD, UD
        fuu=open(filVUU, "r")
        fdd=open(filVDD, "r")
        fud=open(filVUD, "r")
                
        luu=fuu.readlines()
        ldd=fdd.readlines()
        lud=fud.readlines()
        
        vecNEU=np.array([self.NESPU,self.NESPU,self.NESPU,self.NESPU])
        vecNED=np.array([self.NESPD,self.NESPD,self.NESPD,self.NESPD])
        vecNE=np.array([self.NESPU,self.NESPD,self.NESPD,self.NESPU])
        
        VmatUU=self.insertVelem(luu,vecNEU,True)
        VmatDD=self.insertVelem(ldd,vecNED,True)
        VmatUD=self.insertVelem(lud,vecNE,False)
               
        self.VmatUU=VmatUU
        self.VmatDD=VmatDD
        self.VmatUD=VmatUD
        self.EmatU=EmatU
        self.EmatD=EmatD
        self.Lvec=Lvec
        self.Vvec=Vvec
        
        print("inputs loaded")
        
    def insertVelem(self,lines,noVec,ifSameSpin):
        '''
        reading CME tensor for one spin pair
        INPUTS: - lines from open file
                - dimensions in 4D
                - if spin polarised?
        OUTPUT: CME tensor as a dictionary
        '''
        Vmat=dict()
        for line in lines:
            ix=0
            ii=np.zeros(4,dtype=np.int8)
            y=[x for x in line.strip().split(' ') if x]
            string1=''
            for x in y:
                if (ix<4):
                    ii[ix]=int(x)-1
                    string1=string1+str(int(x)-1).zfill(2)
                elif(ix==4):                    
                    Vr=float(x)
                elif(ix==5): 
                    Vi=float(x)
                ix=ix+1
            
            string2="".join([str(a).zfill(2) for a in ii[[1,0,3,2]]])
            if (all(ii<noVec)):
                Vmat[string1]=(Vr+Vi*1j)/self.screen
                # insert a flipped element if spins are the same
                if (ifSameSpin):
                    Vmat[string2]=(Vr+Vi*1j)/self.screen
                
        return Vmat
    
    def countConfigs(self):
        # count total number of possible configurations
        if ((self.NstVU<=self.NeltotalU) and (self.NstVD<=self.NeltotalD)):
            NU=self.NstVU+self.NstCU
            ND=self.NstVD+self.NstCD
            
            if (self.NelU>0): confU,tagU,XsU=combs(NU,self.NelU,self.Neltotal)
            if (self.NelD>0): confD,tagD,XsD=combs(ND,self.NelD,self.Neltotal)

            self.noConfU=int(sum(tagU)) if (self.NelU>0) else 0
            self.noConfD=int(sum(tagD)) if (self.NelD>0) else 0
            print("Total number of configs U,D: ",self.noConfU,self.noConfD)
            
    def makeConfigs(self):
        # number of filled states has to be lower or equal to number of total electrons
        if ((self.NstVU<=self.NeltotalU) and (self.NstVD<=self.NeltotalD)):
            NU=self.NstVU+self.NstCU
            ND=self.NstVD+self.NstCD
            
            # create all combinations for spins U and D separately
            if (self.NelU>0): confU,tagU,XsU=combs(NU,self.NelU,self.NelU)
            if (self.NelD>0): confD,tagD,XsD=combs(ND,self.NelD,self.NelD)

            if (self.NelU>0): sU=confU.shape[0]
            if (self.NelD>0): sD=confD.shape[0]
            
            self.noConfU=int(sum(tagU)) if (self.NelU>0) else 0
            self.noConfD=int(sum(tagD)) if (self.NelD>0) else 0

            if (self.NelU>0): print("# confu",self.noConfU)
            if (self.NelD>0): print("# confd",self.noConfD)


            if (self.NelU>0):
                tempU=np.zeros((sU,self.NelU+1),dtype=int)
                indU=np.zeros(self.noConfU,dtype=int)
                tempU[:,0:self.NelU]=confU
                tempU[:,self.NelU]=XsU
                nums=np.arange(0,sU)
                indU=nums[tagU==1]
                tempU2=tempU[indU,:]
                ix=np.lexsort([np.zeros(self.noConfU),tempU2[:,self.NelU]])
                confUtru=tempU2[ix,:]
            if (self.NelD>0):
                tempD=np.zeros((sD,self.NelD+1),dtype=int)
                indD=np.zeros(self.noConfD,dtype=int)
                tempD[:,0:self.NelD]=confD
                tempD[:,self.NelD]=XsD
                nums=np.arange(0,sD)
                indD=nums[tagD==1]
                tempD2=tempD[indD,:]
                ix=np.lexsort([np.zeros(self.noConfD),tempD2[:,self.NelD]])
                confDtru=tempD2[ix,:]

            if (self.NelU>0): configsU=confUtru[:,0:self.NelU]-1
            if (self.NelD>0): configsD=confDtru[:,0:self.NelD]-1

            if (self.NelU>0): print("configsU:",configsU)
            if (self.NelD>0): print("configsD:",configsD)
            
            confList=[]
            for i in range(0,self.noConfU):
                for j in range(0,self.noConfD):
                    cu=configsU[i,:]
                    cd=configsD[j,:]
                    newConf=Configuration.fromSecQ(cu,cd,NU,ND)
                    confList.extend([newConf])
            self.allConfs=LCombination(np.ones(len(confList)),confList)
            print("allConfs:",self.allConfs)
        
    def makeNewCIham(self,oldMix,newMix,offDiags):
        '''
        makes a Hamiltonian for a bigger new Lcombination
        INPUTS: - old LCombination
                - new LCombination, connected by max 2 electron changes
                - off-diagonal elements connecting the new to old
        OUTPUT: new big Hamiltonian
        '''
        
        n=offDiags.shape[0]
        m=offDiags.shape[1]
        siz=n+m
        bigHam=sp.lil_matrix((siz,siz),dtype=complex)       
        
        # obtain two blocks of the new bigger Hamiltonian
        oldHam=self.makeCIham(oldMix)
        newHam=self.makeCIham(newMix)
                
        # fill off-diagonal blocks of the new bigger Hamiltonian
        bigHam[0:n,0:n]=oldHam
        bigHam[n:,n:]=newHam
        bigHam[0:n,n:]=offDiags
        bigHam[n:,0:n]=np.transpose(offDiags.conjugate())
        
        return bigHam
        
    def makeCIham(self,mix):
        '''
        makes a Hamiltonian (CSR sparse format) for a given Lcombination
        INPUT: mix -given LCombination
        OUTPUT: Hamiltonian matrix
        '''
        
        leng=self.sparseLeng
        N=mix.nterms
        
        hamval=np.zeros((N,leng),dtype=complex)
        hampos=-1*np.ones((N,leng),dtype=int)
        counts=np.zeros(N,dtype=int)
        
        # loop over all configurations in the mix
        i=0
        for co1 in mix.configs:
            j=0
            for co2 in mix.configs:
                # compare configurations
                ileu,iled,diffu,diffd=CIPSI.compareConfigs(co1,co2)
                ile=ileu+iled                
                
                # DIAGONAL TERM
                if (ile==0):
                    Ediag=self.diagSPCo(co1) # SP energy
                    # self energy and vertex correction
                    Vdiag=self.diagCo(co1)
                    
                    hampos[i,counts[i]]=j
                    hamval[i,counts[i]]=Ediag+Vdiag
                    counts[i]=counts[i]+1

                # OFF DIAGONAL TERM - 1 ELECTRON DIFFERENCE
                elif (ile==1):
                    
                    if (ileu==1):
                        UorD=True
                        Voff1=self.offDiagOneCo(co1,diffu,UorD)
                    elif (iled==1):
                        UorD=False
                        Voff1=self.offDiagOneCo(co1,diffd,UorD)
                    else:
                        Voff1=0+0j
                        
                    if (abs(Voff1)>1e-8):
                        hampos[i,counts[i]]=j
                        hamval[i,counts[i]]=Voff1
                        counts[i]=counts[i]+1
                        
                # OFF DIAGONAL TERM - 2 ELECTRON DIFFERENCE
                elif (ile==2):
                    
                    if (ileu==2):
                        UD=1
                        Voff2=self.offDiagTwoCo(co1,diffu,diffd,UD)                        
                    elif (iled==2):
                        UD=-1
                        Voff2=self.offDiagTwoCo(co1,diffu,diffd,UD)
                    elif (ileu==1 and iled==1):
                        UD=0
                        Voff2=self.offDiagTwoCo(co1,diffu,diffd,UD)
                    else:
                        Voff2=0+0j
                        
                    if (abs(Voff2)>1e-8):                        
                        hampos[i,counts[i]]=j
                        hamval[i,counts[i]]=Voff2
                        counts[i]=counts[i]+1
                        
                j+=1
            i+=1                
                    
        
        # build the CSR sparse format
        nnz=np.sum(hampos>-1)
        indptr=np.zeros(N+1,dtype=int)
        indic=np.zeros(nnz,dtype=int)
        data=np.zeros(nnz,dtype=complex)
                
        indptr[0]=0
        inds=np.where((hampos==-1).any(axis=1),(hampos==-1).argmax(axis=1),-1)
        last=0
        for i in range(0,N):
            last=last+inds[i]
            indptr[i+1]=last
            indic[indptr[i]:indptr[i+1]]=hampos[i,0:inds[i]]
            data[indptr[i]:indptr[i+1]]=hamval[i,0:inds[i]]
            
        ham=sp.csr_matrix((data,indic,indptr)).tolil()
        return   ham
        
    def connect(self,mix,eigval):
        '''
        finds all configurations connected to the old configurations 
        by maximum 2 electron changes (with a large perturbation)
        also calculates the offdiagonal matrix elements between 
        the old and the new
        
        INPUT: mix - initial LCombination and its eigenvalues
        OUTPUT: new set of configurations and offdiagonal matrix elements
        '''
        
        # setup
        thresh=self.thresh
        N=mix.nterms
        configs=mix.configs
        coefs=mix.coefs
        # find new set of connected configurations
        newMix=CIPSI.makeConnectedMix(mix)
        M=newMix.nterms
        newConfigs=newMix.configs
        
        elements=np.zeros(M,dtype=complex)
        offDiags=np.zeros((N,M),dtype=complex)
        
        # loop over new configurations
        i=0
        for newCo in newConfigs:  
            # DIAGONAL ELEMENTS
            Ediag=self.diagSPCo(newCo) # SP energy
            Vdiag=self.diagCo(newCo) # self energy and vertex correction
            ene=np.real(Ediag+Vdiag) # energy of a configuration
            
            # loop over old configurations
            j=0
            for co,coef in zip(configs,coefs):
                # compare old and new configuration
                ileu,iled,diffu,diffd=CIPSI.compareConfigs(co,newCo)
                ile=ileu+iled
                
                # check if they are indeed connected by max 2 electron changes
                if(ile<3):
                    
                    # # OFF DIAGONAL TERM - 1 ELECTRON DIFFERENCE
                    if (ile==1):
                        if (ileu==1):
                            UorD=True
                            Voff=self.offDiagOneCo(co,diffu,UorD)
                        elif (iled==1):
                            UorD=False
                            Voff=self.offDiagOneCo(co,diffd,UorD)
                        else:
                            Voff=0+0j                        
                        
                    # # OFF DIAGONAL TERM - 2 ELECTRON DIFFERENCE
                    elif (ile==2):
                        
                        if (ileu==2):
                            UD=1
                            Voff=self.offDiagTwoCo(co,diffu,diffd,UD)                        
                        elif (iled==2):
                            UD=-1
                            Voff=self.offDiagTwoCo(co,diffu,diffd,UD)
                        elif (ileu==1 and iled==1):
                            UD=0
                            Voff=self.offDiagTwoCo(co,diffu,diffd,UD)
                        else:
                            Voff=0+0j
                        
                    elif(ile==0):
                        Voff=0+0j
                        elements[i]=0+0j
                        break
                    
                else:
                    Voff=0+0j
                    
                offDiags[j,i]=Voff
                elements[i]+=Voff*coef
                j+=1
            
            elements[i]=elements[i]/(eigval-ene) # build the perturbation criterion
            i+=1
        
        # check which new configurations pass the perturbation theory criterion
        which=abs(elements)>thresh
        offDiagsSelected=offDiags[:,which]
        selected=newConfigs[which]
        selectedElems=elements[which]
        nn=len(selected)
        
        # return new configurations if any were selected
        if(nn>0):
            selectedMix=LCombination(selectedElems,selected)
            return selectedMix,offDiagsSelected
        else:
            return None,None
    
    def diagCo(self,co):
        '''
        Calculates diagonal Hamiltonian matrix element due to Coulomb interaction 
        it's self energy and vertex correction part
        INPUTS: configuration
        OUTPUT: Hamiltonian matrix element
        '''
        V=0+0j
        coU=co.Unumbers
        coD=co.Dnumbers
        for i in range(0,self.NelU):
            for j in range(0,self.NelU):
                if (i!=j):
                    vec1=[coU[i],coU[j],coU[j],coU[i]]
                    vec2=[coU[i],coU[j],coU[i],coU[j]]
                    string1="".join([str(a).zfill(2) for a in vec1])
                    string2="".join([str(a).zfill(2) for a in vec2])
                    Vd=self.VmatUU.get(string1,0+0j)
                    Vx=self.VmatUU.get(string2,0+0j)
                    V=V+(Vd-Vx)/2
            for j in range(0,self.NelD):
                vec=[coU[i],coD[j],coD[j],coU[i]]
                string="".join([str(a).zfill(2) for a in vec])
                Vd=self.VmatUD.get(string,0+0j)
                V=V+Vd
        
        
        for i in range(0,self.NelD):
            for j in range(0,self.NelD):
                if (i!=j):
                    vec1=[coD[i],coD[j],coD[j],coD[i]]
                    vec2=[coD[i],coD[j],coD[i],coD[j]]
                    string1="".join([str(a).zfill(2) for a in vec1])
                    string2="".join([str(a).zfill(2) for a in vec2])
                    Vd=self.VmatDD.get(string1,0+0j)
                    Vx=self.VmatDD.get(string2,0+0j)
                    V=V+(Vd-Vx)/2            
        return V
    
    def diagSPCo(self,co):
        '''
        Calculates diagonal SP Hamiltonian matrix element
        INPUTS: configuration
        OUTPUT: Hamiltonian matrix element
        '''
        Eu=0
        Ed=0
        coU=co.Unumbers
        coD=co.Dnumbers
        
        if (len(coU)>0): Eu=sum(self.EmatU[coU])
        if (len(coD)>0): Ed=sum(self.EmatD[coD])
        
        E=Eu+Ed
        return E
    
    def offDiagOneCo(self,co,diff,UorD):
        '''
        calculates the Hamiltonian matrix element 
        for configurations differing by 1 electron state
        INPUTS: - left configuration (bra), 
                - difference between initial and final configuration, 
                - spin U or D
        OUTPUTS: Hamiltonian matrix element
        '''
        
        il=diff[0,0]
        ir=diff[1,0]
        
        V=0+0j
        coU=co.Unumbers
        coD=co.Dnumbers
        
        # working out the occupied states in the bra and ket for spin U
        ilel=sum(coU>il)
        iler=sum(np.logical_and(coU>ir,coU!=il))
        # sign factor based on the odd or even number of operator swaps 
        # in the bra and ket 
        if ((ilel%2)==(iler%2)):
            facu=1
        else:
            facu=-1
        
        # working out the occupied states in the bra and ket for spin U
        ilel=sum(coD>il)
        iler=sum(np.logical_and(coD>ir,coD!=il))
        # sign factor based on the odd or even number of operator swaps 
        # in the bra and ket 
        if ((ilel%2)==(iler%2)):
            facd=1
        else:
            facd=-1
        
        # for 1 electron difference must sum over all unchanged occupied states
        for i in range(0,self.NelU):
            if (UorD and (coU[i]!=il)):
                vec1=[coU[i],il,ir,coU[i]]
                vec2=[coU[i],il,coU[i],ir]
                string1="".join([str(a).zfill(2) for a in vec1])
                string2="".join([str(a).zfill(2) for a in vec2])
                Vd=self.VmatUU.get(string1,0+0j)
                Vx=self.VmatUU.get(string2,0+0j)
                V=V+facu*(Vd-Vx)
            elif (not UorD):
                vec=[coU[i],il,ir,coU[i]]
                string="".join([str(a).zfill(2) for a in vec])
                Vd=self.VmatUD.get(string,0+0j)
                V=V+facd*Vd
        
        
        for i in range(0,self.NelD):
            if ((not UorD) and (coD[i]!=il)):
                vec1=[coD[i],il,ir,coD[i]]
                vec2=[coD[i],il,coD[i],ir]
                string1="".join([str(a).zfill(2) for a in vec1])
                string2="".join([str(a).zfill(2) for a in vec2])
                Vd=self.VmatDD.get(string1,0+0j)
                Vx=self.VmatDD.get(string2,0+0j)
                V=V+facd*(Vd-Vx)
            elif (UorD):
                vec=[il,coD[i],coD[i],ir]
                string="".join([str(a).zfill(2) for a in vec])
                Vd=self.VmatUD.get(string,0+0j)
                V=V+facu*Vd
                
        return V
    
    def offDiagTwoCo(self,co,diffU,diffD,UorD):
        '''
        calculates the Hamiltonian matrix element 
        for configurations differing by 2 electron states
        INPUTS: - left configuration (bra), 
                - difference between initial and final configuration, 
                - spin U or D
        OUTPUTS: Hamiltonian matrix element
        '''
        
        V=0+0j
        coU=co.Unumbers
        coD=co.Dnumbers
        
        if (UorD==-1):
            # sorting indices to work out the sign
            diffl=np.sort(diffD[0,:])
            diffr=np.sort(diffD[1,:])
            
            vec1=[diffl[0],diffl[1],diffr[1],diffr[0]]
            vec2=[diffl[0],diffl[1],diffr[0],diffr[1]]
            string1="".join([str(a).zfill(2) for a in vec1])
            string2="".join([str(a).zfill(2) for a in vec2])
                
            Vd=self.VmatDD.get(string1,0+0j)
            Vx=self.VmatDD.get(string2,0+0j)
            
            # working out the occupied states in the bra and ket
            if0=(coD!=diffl[0]) & (coD!=diffl[1])
            if1=if0 & (coD>diffl[0]) & (coD<diffl[1])
            if2=if0 & (coD>diffr[0]) & (coD<diffr[1])
            
            ilel=sum(if1)
            iler=sum(if2)
            
            # sign factor based on the odd or even number 
            # of operator swaps in the bra and ket 
            if ((ilel%2)==(iler%2)):
                fac=1
            else:
                fac=-1
                
            V=fac*(Vd-Vx)
            
        elif (UorD==1):
            # sorting indices to work out the sign
            diffl=np.sort(diffU[0,:])
            diffr=np.sort(diffU[1,:])
            
            vec1=[diffl[0],diffl[1],diffr[1],diffr[0]]
            vec2=[diffl[0],diffl[1],diffr[0],diffr[1]]
            string1="".join([str(a).zfill(2) for a in vec1])
            string2="".join([str(a).zfill(2) for a in vec2])
                
            Vd=self.VmatUU.get(string1,0+0j)
            Vx=self.VmatUU.get(string2,0+0j)
            
            # working out the occupied states in the bra and ket
            if0=(coU!=diffl[0]) & (coU!=diffl[1])
            if1=if0 & (coU>diffl[0]) & (coU<diffl[1])
            if2=if0 & (coU>diffr[0]) & (coU<diffr[1])
            
            ilel=sum(if1)
            iler=sum(if2)
            
            # sign factor based on the odd or even number 
            # of operator swaps in the bra and ket 
            if ((ilel%2)==(iler%2)):
                fac=1
            else:
                fac=-1
                
            V=fac*(Vd-Vx)
            
        elif (UorD==0):
            diffl=np.array([diffU[0,0],diffD[0,0]])
            diffr=np.array([diffU[1,0],diffD[1,0]])
            
            vec=[diffl[0],diffl[1],diffr[1],diffr[0]]
            string="".join([str(a).zfill(2) for a in vec])
                
            Vd=self.VmatUD.get(string,0+0j)
            
            # working out the occupied states in the bra and ket
            if1=(coU>diffl[0]) & (coU!=diffl[0])
            if2=(coD<diffl[1]) & (coD!=diffl[1])
            if3=(coU>diffr[0]) & (coU!=diffl[0])
            if4=(coD<diffr[1]) & (coD!=diffl[1])
            
            ilel=sum(if1)+sum(if2)
            iler=sum(if3)+sum(if4)
        
            # sign factor based on the odd or even number 
            # of operator swaps in the bra and ket 
            if ((ilel%2)==(iler%2)):
                fac=1
            else:
                fac=-1
                
            V=fac*Vd
        
        
        return V
    
    @staticmethod
    def compareConfigs(c1,c2):
        '''
        conpares two electron configurations: 
        finds number of electrons on different SP states 
        and indices of these states
        IPUTS: two configurations
        OUTPUTS:- ile - number of different electron states
                - diff - different electron states (size varies based on ile)
        '''
        
        nu=c1.Unen
        nd=c1.Dnen
        
        codiffu=np.zeros(nu,dtype=int)
        codiffd=np.zeros(nd,dtype=int)
        codiffu=c1.Ufilled-c2.Ufilled
        codiffd=c1.Dfilled-c2.Dfilled
        ileu=int(sum(codiffu!=0)/2)
        iled=int(sum(codiffd!=0)/2)
        
        diffu=CIPSI.diffElectrons(ileu,codiffu)
        diffd=CIPSI.diffElectrons(iled,codiffd)
        
        return ileu,iled,diffu,diffd
    
    @staticmethod
    def diffElectrons(ile,codiff):
        # make a matrix of electrons states different in two configurations
        n=len(codiff)
        if (ile==0):
            diff=np.zeros((1,1),dtype=int)
        elif (ile==1):
            diff=np.zeros((2,1),dtype=int)
            nums=np.arange(0,n)
            diff[0,:]=nums[codiff>0]
            diff[1,:]=nums[codiff<0]
        elif(ile==2):
            diff=np.zeros((2,2),dtype=int)
            nums=np.arange(0,n)
            diff[0,:]=nums[codiff>0]
            diff[1,:]=nums[codiff<0]
        else:
            diff=np.zeros((1,1),dtype=int)
                
        return diff
    
    @staticmethod
    def makeConnectedMix(mix):
        # create a set of configurations connected 
        # to the existing LCombination by max 2 electron changes
        
        configs=mix.configs        
        allNew=[]
        
        for cfg in configs:
            newConfs=CIPSI.makeConnected_1(cfg) # one electron change
            allNew.extend(newConfs)
            newConfs=CIPSI.makeConnected_2(cfg) # two electron changes
            allNew.extend(newConfs)
        
        n=len(allNew)
        new=LCombination(np.ones(n),allNew)
        new2=new.combine()            
        
        return new2
        
    @staticmethod
    def makeConnected_1(conf):
        # find a set of configurations connected to the existing 
        # configuration by 1 electron change
        
        NU=conf.Unel
        ND=conf.Dnel
        cU=conf.Unumbers
        cD=conf.Dnumbers
        c01u=conf.Ufilled
        c01d=conf.Dfilled
        NenU=conf.Unen
        NenD=conf.Dnen
        # find all spots where an electron can be moved
        numsU=np.arange(0,NenU)
        numsD=np.arange(0,NenD)
        otherU=numsU[~c01u.astype(bool)]
        otherD=numsD[~c01d.astype(bool)]
        
        allNew=[]
        
        # SPIN U
        # loop over all existing electrons
        for i in range(0,NU):            
            # remove an electron from configuration
            short=np.delete(cU,i)
            
            # loop over all spaces where an electron can be moved
            for num in otherU:
                # add an electron to a free state
                new=np.append(short,num)
                new01=np.zeros(NenU)
                new01[new]=1
                newConf=Configuration.fromFirQ(new01,c01d)
                allNew.append(newConf)
        
        # SPIN DN
        # loop over all existing electrons
        for i in range(0,ND):            
            # remove an electron from configuration
            short=np.delete(cD,i)
            
            # loop over all spaces where an electron can be moved
            for num in otherD:
                # add an electron to a free state
                new=np.append(short,num)
                new01=np.zeros(NenD)
                new01[new]=1
                newConf=Configuration.fromFirQ(c01u,new01)
                allNew.append(newConf)
            
        return allNew
        
    @staticmethod
    def makeConnected_2(conf):
        # find a set of configurations connected to the existing 
        # configuration by 2 electron changes
        
        NU=conf.Unel
        ND=conf.Dnel
        cU=conf.Unumbers
        cD=conf.Dnumbers
        c01u=conf.Ufilled
        c01d=conf.Dfilled
        NenU=conf.Unen
        NenD=conf.Dnen
        # find all spots where an electron can be moved
        numsU=np.arange(0,NenU)
        numsD=np.arange(0,NenD)
        otherU=numsU[~c01u.astype(bool)]
        otherD=numsD[~c01d.astype(bool)]
        
        allNew=[]
        
        # SPIN U - if there's min 2 electrons and min 4 SP states
        if(NU>1 and NenU>3):
            # loop over all existing electrons
            for i in range(0,NU):            
                # remove electron 1 from configuration
                short0=np.delete(cU,i)
                # loop over all remaining electrons
                for j in range(0,NU-1):
                    # remove electron 2 from configuration
                    short=np.delete(short0,j)
                
                    # loop over all spaces where an electron can be moved
                    for k in range(0,len(otherU)):
                        num=otherU[k]
                        new0=np.append(short,num)
                        other=np.delete(otherU,k)
                        # loop over all spaces where an electron can be moved
                        # apart from k
                        for num2 in other:
                            new=np.append(new0,num2)                        
                            new01=np.zeros(NenU)
                            new01[new]=1
                            newConf=Configuration.fromFirQ(new01,c01d)
                            allNew.append(newConf)
               
        # SPIN DN - if there's min 2 electrons and min 4 SP states
        if(ND>1 and NenD>3):
            # loop over all existing electrons
            for i in range(0,ND):            
                # remove electron 1 from configuration
                short0=np.delete(cD,i)
                # loop over all remaining electrons
                for j in range(0,ND-1):
                    # remove electron 2 from configuration
                    short=np.delete(short0,j)
                
                    # loop over all spaces where an electron can be moved
                    for num in otherD:
                        new0=np.append(short,num)
                        other=np.delete(otherD,num)
                        # loop over all spaces where an electron can be moved
                        # apart from num
                        for num2 in other:
                            new=np.append(new0,num2)
                        
                            new01=np.zeros(NenD)
                            new01[new]=1
                            newConf=Configuration.fromFirQ(new01,c01d)
                            allNew.append(newConf)
        
        # SPIN UP&DN - if there're more SP states than electrons
        if(NenU>NU and NenD>ND):
            # loop over all existing electrons U
            for i in range(0,NU):            
                # remove electron 1 from configuration
                shortU=np.delete(cU,i)
            
                # loop over all states where an electron U can be moved
                for numu in otherU:
                    newU=np.append(shortU,numu)
                    new01u=np.zeros(NenU)
                    new01u[newU]=1
                    
                    # loop over all existing electrons D
                    for j in range(0,ND):
                        shortD=np.delete(cD,j)
                
                        # loop over all states where an electron D can be moved
                        for numd in otherD:
                            newD=np.append(shortD,numd)
                            new01d=np.zeros(NenD)
                            new01d[newD]=1
                    
                            newConf=Configuration.fromFirQ(new01u,new01d)
                            allNew.append(newConf)
            
        return allNew
        
        
        
        
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 13:11:57 2021

@author: ludka
"""

