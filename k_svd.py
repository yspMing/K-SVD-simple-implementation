import numpy as np
import random
from scipy import linalg

class ksvd(object):

    def __init__(self,words,iteration,stopFlag='solidError',errGoal=10,atomNumber=20,preserveDC=False):
        self.K=words
        self.iteration=iteration
        self.stopFlag=stopFlag  # This parameter determines how the sparse coding period converge
        self.errGoal=errGoal
        self.atomNumber=atomNumber
        self.preserveDC=preserveDC

    def dictInitialization(self,inputMat,givenMat):
        assert len(inputMat.shape)==2,'Input matrix shoule have dimension equal to 2'
        if inputMat.shape[1]<self.K:
            print("Sample size is smaller than the dictionary size")
            self.D=inputMat
            return
        if givenMat:
            assert givenMat.shape[1]==self.K,'Dictionary initializaiton dimension not match'
            self.D=givenMat
        else:
            self.D=inputMat[:,:self.K]

        norm=1/(np.sqrt(np.sum(self.D*self.D,0)))
        self.D=np.dot(self.D,np.diag(norm))
        self.D=self.D*np.tile(np.sign(self.D[0,:]),[self.D.shape[0],1])

    def constructDictionary(self,inputMat,givenMat=None):
        self.dictInitialization(inputMat,givenMat)
        print("dictionary initializaiton done")
        for iterNum in range(self.iteration):
            if self.stopFlag=='limitedAtoms':
                # This part remains unfinished
                return
            elif self.stopFlag=='solidError':
                coef=self.OMP(self.D,inputMat,self.errGoal)
            else:
                print("invalid stop flag")
                return
            replacedVector=0
            sequence=list(range(self.D.shape[1]))
            random.shuffle(sequence)
            for j in sequence:
                updateWord,coef,addVector=self.updateDictionary(inputMat,self.D,j,coef)
                self.D[:,j]=updateWord
                replacedVector+=addVector
            nonZeroCoef=np.nonzero(coef)
            nonZeroRatio=len(nonZeroCoef[0])/(coef.shape[1]*coef.shape[0])
            print("iter:%3d, average coefficient ratio is%f "%(iterNum,nonZeroRatio))
            self.D=self.cleanDictionary(inputMat,self.D,coef)
        return self.D

    def OMP(self,dictionary,data,errorGoal,showFlag=True):
        """
        constructing the sparse representation using Orthogonal Matching Pursuit algorithm
        :param dictionary: Given dictionary with dimension [n,K]
        :param data: Given data to be represented with dimension [n,N]
        :param errorGoal:
        :return:
        """
        n,N=data.shape
        errThresh=errorGoal*errorGoal*n
        maxNumCoef=n/2
        coef=np.zeros([self.K,N])
        for i in range(N):
            y=data[:,i]
            residual=y
            index=[]
            atomUsed=0
            currentRes=np.sum(residual*residual)
            if showFlag:
                if i%(int(N/5))==int(N/5)-1:
                    print('done with the {}% data coding'.format(int((i+1)*100/N)))

            while currentRes>errThresh and atomUsed<maxNumCoef:
                atomUsed+=1
                proj=np.dot(np.transpose(dictionary,[1,0]),residual)  # proj in shape [K,1]
                maxIndex=np.argmax(proj,axis=0)
                index.append(maxIndex)  # index in shape (atoms,)
                expression=np.dot((linalg.pinv(dictionary[:,index])),y)   # atoms*n ,n*1 -->atoms*1
                residual=y-np.dot(dictionary[:,index],expression)
                currentRes=np.sum(residual*residual)
            if len(index)>0:
                coef[index,i]=expression
        return coef

    def updateDictionary(self,data,dictionary,wordToUpdate,coefMatrix,):
        """
        update one atom in the dictionary
        return:
        1)the updated word for the item in dictionary,
        2)new coefMatrix. this is done since only nonZeroEntry is updated and we wrap this step
        3)addVector or not. IF the atom selected is used by non data, we need to delete this atom and add new one
        """
        nonZeroEntry=np.nonzero(coefMatrix[wordToUpdate,:])
        nonZeroEntry=nonZeroEntry[0]
        if len(nonZeroEntry)<1:  #the word to be updated isn't used any data
            addVector=1
            errorMat=data-np.dot(dictionary,coefMatrix)
            selectAtom=data[:,np.argmax(np.sum(errorMat,0))]
            # normalization
            selectAtom=selectAtom/np.sqrt(np.sum(selectAtom*selectAtom))
            selectAtom=selectAtom*(np.sign(selectAtom[0]))
            return selectAtom,coefMatrix,addVector
        addVector=0
        tmpCoefMatrix=coefMatrix[:,nonZeroEntry]
        tmpCoefMatrix[wordToUpdate,:]=0
        errorMat=data[:,nonZeroEntry]-np.dot(dictionary,tmpCoefMatrix)
        U,S,V=np.linalg.svd(errorMat)  # V refers to V' in svd
        updateWord=U[:,0]
        coefMatrix[wordToUpdate,nonZeroEntry]=S[0]*V[0,:]  # first row of V'
        return updateWord,coefMatrix,addVector

    def cleanDictionary(self,data,dictionary,coefMatrix):
        T1=3
        T2=0.99
        error=np.sum(np.square(data-np.dot(dictionary,coefMatrix)),0)
        G=np.dot(np.transpose(dictionary,[1,0]),dictionary)
        G=G-np.diag(np.diag(G))
        for j in range(dictionary.shape[1]):
            if np.max(G[j,:])>T2 or np.sum(np.abs(coefMatrix[j,:])>1e-7)<=T1:
                index=np.argmax(error)
                error[index]=0
                dictionary[:,j]=data[:,index]/np.sqrt(np.sum(data[:,index]*data[:,index]))
                G = np.dot(np.transpose(dictionary, [1, 0]), dictionary)
                G = G - np.diag(np.diag(G))
        return dictionary