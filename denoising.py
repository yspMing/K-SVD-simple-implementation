"""
a test example using the k-svd method
"""

import cv2
import numpy as np
import k_svd

def im2col(image,blockSize,step):
    M,N=image.shape
    rowNumber=int((M-blockSize)/step)+1
    colNumber=int((N-blockSize)/step)+1
    rows=[i*step for i in range(rowNumber)]
    cols=[i*step for i in range(colNumber)]
    if (rowNumber-1)*step+blockSize<M:
        rows.append(M-blockSize)
    if (colNumber-1)*step+blockSize<N:
        cols.append(N-blockSize)
    repmat=np.array([[image[i:i+blockSize,j:j+blockSize] for j in cols] for i in rows])
    repmat=np.reshape(repmat,[len(rows),len(cols),blockSize*blockSize])
    repmat=np.reshape(repmat,[len(rows)*len(cols),blockSize*blockSize])
    return repmat,rows,cols

def main():

    sigma=25
    blockSize=8
    step=1
    maxBlockToTrain=65000
    maxBlockToConsider=260000

    image=cv2.imread('lena.bmp',0)
    noisy=image.astype('float')+sigma*np.random.randn(*image.shape)

    dataMatrix,_,_=im2col(noisy,blockSize,step)
    np.random.shuffle(dataMatrix)
    dataMatrix=np.transpose(dataMatrix,[1,0])

    if dataMatrix.shape[1]>maxBlockToTrain:
        dataMatrix=dataMatrix[:,:maxBlockToTrain]  # shape [n,N]

    #subtract the DC value from the original signal
    mean=np.sum(dataMatrix,0)/dataMatrix.shape[0]
    dataMatrix=dataMatrix-np.tile(mean,[dataMatrix.shape[0],1])
    #construct the k-svd object to do the sparse coding
    ksvd=k_svd.ksvd(words=256,iteration=10,errGoal=sigma*1.15)
    dictionary=ksvd.constructDictionary(dataMatrix)
    print("finish dictionary training")

    #denoising the image using the resulted dictionary
    while ((image.shape[0]-blockSize)/step+1)*((image.shape[1]-blockSize)/step+1)>maxBlockToConsider:
        step+=1

    dataMatrix,rowIndex,colIndex=im2col(noisy,blockSize,step)
    dataMatrix=np.transpose(dataMatrix,[1,0])
    n,N=dataMatrix.shape
    processstep=10000
    maxStep=N//processstep
    if N%processstep:
       maxStep+=1
    for i in range(maxStep):
        maxColumn=np.minimum((i+1)*processstep,N)
        mean=np.sum(dataMatrix[:,i*processstep:maxColumn],0)/n
        dataMatrix[:,i*processstep:maxColumn]-=np.tile(mean,[n,1])
        coef=ksvd.OMP(dictionary,dataMatrix[:,i*processstep:maxColumn],sigma*1.15,showFlag=False)
        dataMatrix[:,i*processstep:maxColumn]=np.dot(dictionary,coef)+np.tile(mean,[n,1])


    imageOut=np.zeros(image.shape)
    weight=np.zeros(image.shape)
    for i,r in enumerate(rowIndex):
        for j,c in enumerate(colIndex):
            block=np.reshape(dataMatrix[:,i*len(rowIndex)+j],[blockSize,blockSize])
            imageOut[r:r+blockSize,c:c+blockSize]+=block
            weight[r:r+blockSize,c:c+blockSize]+=1
    denoised=(imageOut/weight).astype(np.uint8)

    cv2.imshow("origin",image)
    cv2.imshow('noisy',noisy.astype(np.uint8))
    cv2.imshow('denoised',denoised)
    cv2.waitKey(10000)


if __name__=="__main__":
    main()
