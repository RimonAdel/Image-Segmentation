from scipy.spatial import distance_matrix
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
import scipy.io
from PIL import Image
import math
from skimage import  color
from skimage.future import graph

def readImagesPath(path):
    dataName = path
    dataDir = "" + dataName
    trainData = []
    for ImageName in os.listdir(dataDir):
        ImagePath = os.path.join(dataDir, ImageName)
        trainData += [ImagePath]
    # return all images paths
    return trainData
# **************************************************************************************************
def imgRGBread(images):
    rgbImages = []  # 3D images
    vectorizedImages = []
    for i in range(0, len(images)):
        image = mpimg.imread(images[i])
        rgbImages.append(image)
        # Blur image to reduce the edge content and makes the transition form one color to the other very smooth.
        # Check video:https://www.youtube.com/watch?list=PLQVvvaa0QuDdttJXlLtAJxJetJcqmqlQq&v=sARklx6sgDk
        image = cv2.GaussianBlur(rgbImages[i], (7, 7), 0)

        # convert image to (M*N)*3 Matrix
        vectorizedImages.append(image.reshape(-1, 3))
    return rgbImages, vectorizedImages
# **************************************************************************************************
def imgRGBreadOneImage(imagePath):
    rgbImages = []  # 3D images
    vectorizedImage = []
    image = mpimg.imread(imagePath)
    rgbImages.append(image)
    # Blur image to reduce the edge content and makes the transition form one color to the other very smooth.
    # Check video:https://www.youtube.com/watch?list=PLQVvvaa0QuDdttJXlLtAJxJetJcqmqlQq&v=sARklx6sgDk
    image = cv2.GaussianBlur(image, (7, 7), 0)

    # convert image to (M*N)*3 Matrix
    vectorizedImage.append(image.reshape(-1, 3))
    return rgbImages, vectorizedImage
# **************************************************************************************************
def kmeans(dataSet, k):
    # I/p one of the images of vectorized Images list
    numOfPoints = len(dataSet)
    # k random initial points
    randomIndeces = np.random.choice(numOfPoints, k, replace=False)
    centers = []
    for i in range(0, len(randomIndeces)):
        centers.append(dataSet[randomIndeces[i]])
    centersOld = [0] * k
    clusterAssignment = [0] * len(dataSet)
    start = 0
    while (1):
        flag = 0
        if start != 0:
            for i in range(0, k):
                if centersOld[i] != centers[i]:
                    centersOld[i] = centers[i]
                else:
                    flag += 1
        start = 1
        if flag == k:
            return (centers, clusterAssignment)
            # distance between points and centers matrix
        distMatrix = distance_matrix(dataSet, centers, p=2)

        for i in range(0, numOfPoints):
            # closest center
            d = distMatrix[i]
            closestCenter = (np.where(d == np.min(d)))[0][0]
            # associate point to closest center
            clusterAssignment[i] = closestCenter

        # new centers
        for i in range(0, k):
            sumX = 0
            sumY = 0
            sumZ = 0
            count = 0
            for j in range(0, numOfPoints):
                if (clusterAssignment[j] == i):
                    sumX += (dataSet[j])[0]
                    sumY += (dataSet[j])[1]
                    sumZ += (dataSet[j])[2]
                    count += 1
            centers[i] = (sumX / count, sumY / count, sumZ / count)
    return (centers, clusterAssignment)
# **************************************************************************************************
def __extractGrondTruthMatrix(mat):
    _groundTruthLabelVectorList = []
    _groundTruthMatrixes = []
    for _groundTruthMatrix in mat["groundTruth"][0]:
        _groundTruthMatrix = _groundTruthMatrix[0][0][0]
        _groundTruthMatrixes.append(_groundTruthMatrix)
        tempList = []
        for row in _groundTruthMatrix:
            tempList.extend(row.tolist())
        _groundTruthLabelVectorList.append(tempList)

    return _groundTruthMatrixes,_groundTruthLabelVectorList
# **************************************************************************************************
def __getGroundTruthLabels(groundTruthMatrix,image):
    _labelsDict = {}
    i = -1
    for row in groundTruthMatrix:
        i += 1
        j = -1
        for key in row:
            j += 1
            if key not in _labelsDict:
                ima = image[i][j]
                _labelsDict.update({key:[ima[2],ima[1],ima[0]]})
    return _labelsDict
# **************************************************************************************************
def getGroundTruthLabelsAndGenerateImage(matPath,imagePath):

    image = cv2.imread(imagePath)
    mat = scipy.io.loadmat(matPath)
    _groundTruthMatrixes,_groundTruthLabelVectorList = __extractGrondTruthMatrix(mat)
    for z in range(len(_groundTruthMatrixes)):
        groundTruthMatrix = _groundTruthMatrixes[z]
        labelsDict = __getGroundTruthLabels(groundTruthMatrix,image)
        rowsNumber  = len(groundTruthMatrix)
        colsNumber = len(groundTruthMatrix[0])
        rgbArray = np.zeros((rowsNumber, colsNumber, 3), 'uint8')
        for i in range(rowsNumber):
            for j in range(colsNumber):
                rgbArray[i][j] = labelsDict[groundTruthMatrix[i][j]]
        img = Image.fromarray(rgbArray)
        img.save('groundTruth#'+str(z)+'.jpg')

    return _groundTruthLabelVectorList
# **************************************************************************************************
def purityOfEachClass(labels,groundTruth2,k=3,sorted = True):
    groundTruthLabesNumber = 0
    for i in range(len(groundTruth2)):
        if groundTruthLabesNumber < groundTruth2[i]:
            groundTruthLabesNumber = groundTruth2[i]
    groundTruthLabesNumber +=1
    dataInClusterindexies = []
    for i in range(k):
        dataInClusterindexies.append([])
    for i in range(len(groundTruth2)):
        dataInClusterindexies[labels[i]].append(i)
    listNij = []
    for i in range(k):
        list = [0] * (groundTruthLabesNumber)
        listNij.append(list)

    for i in range(k):
        for j in range(len(dataInClusterindexies[i])):
            listNij[i][groundTruth2[dataInClusterindexies[i][j]]] += 1
    finalListNij = []
    for i in range(k):
        list = [0] * (groundTruthLabesNumber)
        finalListNij.append(list)

    for i in range(k):
        for j in range(groundTruthLabesNumber):
            finalListNij[i][j] = (listNij[i][j],j+1)

    if sorted == True:
        for i in range(k):
            finalListNij[i].sort(reverse=True)

    groundtruthList = [0] * (groundTruthLabesNumber)

    for i in range(k):
        for j in range(groundTruthLabesNumber):
            listNij[i][j] = finalListNij[i][j][0]

    for j in range(groundTruthLabesNumber):
        sum = 0
        for i in range(k):
            sum += finalListNij[i][j][0]
        groundtruthList[j] = sum

    return listNij,groundtruthList,groundTruthLabesNumber
#########################################################################################################################################################
def calculatePurity(labels,groundTruth,k=3):
    listNij,groundtruthList,groundTruthLabesNumber = purityOfEachClass(labels,groundTruth,k)
    sum = 0
    for i in range(k):
        sum += listNij[i][0]
    purity = sum / len(labels)
    return purity
#########################################################################################################################################################
def calculateF_Measure(labels,groundTruth,k=3):
    listNij, groundtruthList,groundTruthLabesNumber = purityOfEachClass(labels,groundTruth, k)
    NumberOfElementsInEachCluster = [0]*k
    for i in range(k):
        for j in range(len(listNij[i])):
            NumberOfElementsInEachCluster[i] += listNij[i][j]
    listF_measure = [0]*k
    for i in range(k):
        if NumberOfElementsInEachCluster[i] == 0:
            listF_measure[i] = 0
        else:
            if i > (len(groundtruthList)-1):
                listF_measure[i] = 2 * listNij[i][0] / (NumberOfElementsInEachCluster[i] + 1)
            else:
                listF_measure[i] = 2*listNij[i][0]/(NumberOfElementsInEachCluster[i]+groundtruthList[i])
    sum = 0
    for i in range(k):
        sum += listF_measure[i]
    f_Measure = sum/k
    return f_Measure
#########################################################################################################################################################
def calculateConditionalEntropy(labels,groundTruth,k=3):
    listNij, groundtruthList,groundTruthLabesNumber = purityOfEachClass(labels,groundTruth, k,sorted=False)
    sizeOfData = len(groundTruth)
    numberOfElementsInEachCluster = [0]*k
    entropyOfEachCluster = [0]*k
    for i in range(k):
        for j in range(groundTruthLabesNumber):
            numberOfElementsInEachCluster[i] += listNij[i][j]
    for i in range(k):
        for j in range(groundTruthLabesNumber):
            if numberOfElementsInEachCluster[i] != 0:
                tempValue = listNij[i][j]/numberOfElementsInEachCluster[i]
            if tempValue != 0:
                entropyOfEachCluster[i] += (-tempValue)*math.log10((tempValue))

    entropy = 0
    for i in range(k):
        entropy += (numberOfElementsInEachCluster[i]/sizeOfData)*entropyOfEachCluster[i]
    return entropy
#########################################################################################################################################################
def normalizedCut(testRGBImage,imagePath,clustersLabels,groundTruthLabelVector,k):
    image = mpimg.imread(imagePath)   
    
    #Compute the Region Adjacency Graph    
    g = graph.rag_mean_color(testRGBImage, np.reshape(clustersLabels,(nrows,ncols)), mode='similarity')
    #Perform Normalized Graph cut on the Region Adjacency Graph.
    labels = graph.cut_normalized(np.reshape(clustersLabels,(nrows,ncols)), g)
    #return image
    out = color.label2rgb(labels, testRGBImage, kind='avg')
    fig, ax = plt.subplots(nrows=2,figsize=(6, 8))
    ax[0].imshow(image)
    ax[1].imshow(out)
    plt.tight_layout()
    plt.show()
#########################################################################################################################################################
if __name__ == '__main__':
    trainImages = readImagesPath("data/images/train")
    graundTruthImages = readImagesPath("data/groundTruth/train") 
    
    
    ######################################    
    #The only values to change in the code
    kValues = [3,5,7,9,11]
    imageIndex=79
    ######################################   
      
    matPath = graundTruthImages[imageIndex]
    imagePath=trainImages[imageIndex]
    
    groundTruthLabelsVectorList = getGroundTruthLabelsAndGenerateImage (matPath,imagePath)
    #rgbImages, vectorizedImages = imgRGBreadOneImage(imagePath)
    rgbImages, vectorizedImages = imgRGBread(trainImages)   
    nrows = len(rgbImages[imageIndex])
    if nrows==321:
        ncols = 481
    else:
        nrows=481
        ncols=321
           
    testRGBImage = rgbImages[imageIndex]
    testImage = vectorizedImages[imageIndex]
    for k in kValues:
        print('K value = ',k)
        finalCenters, clustersLabels = kmeans(testImage, k)
        out = color.label2rgb(np.reshape(clustersLabels,(nrows,ncols)), testRGBImage, kind='avg')
        fig, ax = plt.subplots(nrows=2,figsize=(6, 8))
        ax[0].imshow(mpimg.imread(imagePath))
        ax[1].imshow(out)
        plt.tight_layout()
        plt.show()
        
        bestTruthLabel=0
        condEntropy=0
        fMeasure=0
        lastCondEntropy=0
        lastfMeasure=0
        print("Manually Implemented Kmeans")
        i = -1
        for groundTruthLabelVector in groundTruthLabelsVectorList:
            lastCondEntropy=condEntropy
            lastfMeasure=fMeasure
            i += 1
            condEntropy=calculateConditionalEntropy(clustersLabels,groundTruthLabelVector,k = k)
            fMeasure=calculateF_Measure(clustersLabels,groundTruthLabelVector,k=k)
            print("ConditionalEntropy of segment #",i," = ",condEntropy)
            print("F_Measure of segment #",i," = ",fMeasure,"\n")
            if condEntropy>lastCondEntropy:
                if fMeasure>lastfMeasure:
                    bestTruthLabel=groundTruthLabelVector
                  
        i=-1
        print('*****************************************************************')
        print('Sickit learn KMeans:')
        #usage for sickit learn Kmeans
        testKMeansSickitLearn = KMeans(n_clusters=k).fit(testImage)
        for groundTruthLabelVector in groundTruthLabelsVectorList:
            i += 1
            print("ConditionalEntropy of segment #",i," = ",calculateConditionalEntropy(testKMeansSickitLearn.labels_,groundTruthLabelVector,k=k))
            print("F_Measure of segment #",i," = ",calculateF_Measure(testKMeansSickitLearn.labels_,groundTruthLabelVector,k=k),"\n")
       
        print('*****************************************************************')
        print("Normalized Cut:")
        normalizedCut(testRGBImage,imagePath,clustersLabels,bestTruthLabel,k)     
        print('*****************************************************************')
#........................................................................................................                    