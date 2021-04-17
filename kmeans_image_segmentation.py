import numpy as np
import sys
from PIL import Image
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances
import cv2


# This method returns index of pixel in dataVector array with respect to x-coordinate and y-coordinate provided
def getDataVectorIndex(x,y,imageW):
    return (imageW-1)*y+y+x

def kmeans(image,K,iterations,opt1):
	imageW = image.size[0]
	imageH = image.size[1]
	#Initializing empty Data vector with attribute [red,green,blue,x-coordinate,y-coordinate] for each pixel
	dataVector = np.ndarray(shape=(imageW * imageH, 5), dtype=float)
	#this variable holds meta data which cluster a pixel is currently in
	pixelClusterAppartenance = np.ndarray(shape=(imageW * imageH), dtype=int)
	#Adding image values to dataVector array
	for y in range(0, imageH):
	      for x in range(0, imageW):
	      	xy = (x, y)
	      	rgb = image.getpixel(xy)
	      	dataVector[x + y * imageW, 0] = rgb[0]
	      	dataVector[x + y * imageW, 1] = rgb[1]
	      	dataVector[x + y * imageW, 2] = rgb[2]
	      	dataVector[x + y * imageW, 3] = x
	      	dataVector[x + y * imageW, 4] = y
	#Normalizing the values of our dataVector
	dataVector_scaled = preprocessing.normalize(dataVector)




	#Set centers
	minValue = np.amin(dataVector_scaled)
	maxValue = np.amax(dataVector_scaled)

	centers = np.ndarray(shape=(K,5))
	for index, center in enumerate(centers):
		if(opt1==0):
			centers[index] = np.random.uniform(minValue, maxValue, 5)
		else:
		    Cx=int(input("Enter x for center: "))
		    Cy=int(input("Enter y for center: "))
		    Ri=getDataVectorIndex(Cx, Cy,imageW) 
		    centers[index] = dataVector[Ri]

	if(opt1==1):
		#Normalizing Center points
		centers=preprocessing.normalize(centers)


	for iteration in range(iterations):
		old_centers=centers
		#Setting pixels to their cluster
		for idx, data in enumerate(dataVector_scaled):
			distanceToCenters = np.ndarray(shape=(K))
			for index, center in enumerate(centers):
				distanceToCenters[index] = euclidean_distances(data.reshape(1, -1), center.reshape(1, -1))
			pixelClusterAppartenance[idx] = np.argmin(distanceToCenters)

		#CHecking if cluster is ever empty, If it is the case, we assign random point to it
		clusterToCheck = np.arange(K)
		clustersEmpty = np.in1d(clusterToCheck, pixelClusterAppartenance)
		for index, item in enumerate(clustersEmpty):
			if item == False:
				pixelClusterAppartenance[np.random.randint(len(pixelClusterAppartenance))] = index	

		#Moving centers to the centroid of their cluster
		for i in range(K):
			dataInCenter = []

			for index, item in enumerate(pixelClusterAppartenance):
				if item == i:
					dataInCenter.append(dataVector_scaled[index])
			dataInCenter = np.array(dataInCenter)
			centers[i] = np.mean(dataInCenter, axis=0)
		
		print("Iteration: ", iteration)
		print("Centers: ", centers)
	#setting the pixels on original image to be that of the pixel's cluster's centroid
	for index, item in enumerate(pixelClusterAppartenance):
		dataVector[index][0] = int(round(centers[item][0] * 255))
		dataVector[index][1] = int(round(centers[item][1] * 255))
		dataVector[index][2] = int(round(centers[item][2] * 255))

	#	Saving output image
	image = Image.new("RGB", (imageW, imageH))

	for y in range(imageH):
		for x in range(imageW):
		 	image.putpixel((x, y), (int(dataVector[y * imageW + x][0]), 
		 							int(dataVector[y * imageW + x][1]),
		 							int(dataVector[y * imageW + x][2])))
	return image


if __name__ == '__main__':
	#Introduction
	print("INTRODUCTION\nIMAGE ANALYSIS ASSIGNMENT 3\n")
	print("This assignment is submitted by:")
	print("Rishab Bahal (002279376)")
	print("Praniti Gokhru (002269229)")
	print("Daksh Mediratta (002279338)\n")
	print("Purpose: Implement Image segmentation using K-Means algorithm.\n\n")


	#Taking inputs
	inputName = input("Enter input filename:")
	outputName = input("Enter output filename:")
	colouredImage=int(input("Press '1' for coloured image, '0' for grayscale image: "))
	K = int(input("Enter K:"))
	if K < 3:
		while(K<3):
			print("Error: K has to be greater than 2")
			K = int(input("Enter K:"))
			if(K>=3):
				break;
	iterations = int(input("Maximum iterations: "))
	opt1=int(input("If you want to enter initial points manually press '1' else '0': "))

	print("Processing image, might take some time.")


		

	#Take image input
	image = cv2.imread(inputName)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = Image.fromarray(image)


	image=kmeans(image,K,iterations,opt1)
	image.save(outputName)





