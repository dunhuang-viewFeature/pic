# -*- coding: UTF-8 -*- 

import numpy as np
import cv2 as cv
import math
import os
import scipy.io as scio 
import sys

alpha = math.sqrt(2)
# 聚类的类数
cluster_num = 512;
baseFile = 'Dunhuang660'
# baseFile = 'test'

# 验证文件夹的存在性，若不存在则新建
def checkAndCreateFile(filePath):
	if os.path.exists(filePath) != True:
		os.mkdir(filePath)

# 将center写入文件
def writeCenter(center):
	dataFile = 'outputFile'
	if os.path.exists(dataFile) != True:
		os.mkdir(dataFile)
	scio.savemat(os.path.join(dataFile, 'center-' + str(cluster_num) + '.mat'), {'center': center})  
	return

# 聚类操作：输入为向量，返回聚类结果
def clustering(vector, write = True):
	# 最大迭代次数
	max_iter = 1000
	# 精确值
	accuracy = 1.0
	# 聚类结束标准：收敛或到达最大迭代次数
	criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, max_iter, accuracy)
	# 聚类
	ret,label,center = cv.kmeans(vector, cluster_num, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
	print('compute the center succeed')
	print('ret:' + str(ret))
	if write == True:
		writeCenter(center)
		print('write the center succeed')

# 提取 sift 特征
def sift(img):
	sift = cv.xfeatures2d.SIFT_create()
	# 找到特征点
	kp = sift.detect(img,None)
	# 将图片转到 lab 颜色空间
	LABimg = cv.cvtColor(img, cv.COLOR_BGR2LAB)
	# 提取LAB分量
	(L,A,B) = cv.split(LABimg)
	# 提取特征描述子
	kp, desL = sift.compute(L,kp)
	kp, desA = sift.compute(A,kp)
	kp, desB = sift.compute(B,kp)
	# concat to one vector
	des = np.hstack((desL,desA,desB))
	
	# img=cv.drawKeypoints(img,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	return des


# 获取仿射变换矩阵
def getTransMatrix(t, fi, col, row):
	# 平移中心
	M1 = np.array([
			[1, 0, -col],
			[0, 1, -row],
			[0, 0, 1]
		])
	# 旋转
	M2 = np.array([
			[math.cos(fi), math.sin(fi), 0],
			[-math.sin(fi), math.cos(fi), 0],
			[0, 0, 1]
		])
	# 缩放
	M3 = np.array([
			[1, 0, 0],
			[0, t, 0],
			[0, 0, 1]
		])
	# 平移回来
	M4 = np.array([
			[1, 0, col],
			[0, 1, row],
			[0, 0, 1]
		])
	# 矩阵相乘 不能改变顺序（M2 、M3）可以
	M = np.dot(M2, M1)
	M = np.dot(M3, M)
	M = np.dot(M4, M)
	M = np.array([
			[M[0][0], M[0][1], M[0][2]],
			[M[1][0], M[1][1], M[1][2]]
		])
	return M

# 对图片进行确定经纬的仿射变换
def AffineTrans(img, t, fi):
	shape = img.shape
	rows,cols,ch = shape
	halfRow = int(rows/2)
	halfCol = int(cols/2)

	M = getTransMatrix(t, fi, halfCol, halfRow)
	dst = cv.warpAffine(img,M,(cols,rows))
	return dst

# 角度 --> 弧度
def AngToArc(ang):
	return (float(ang) / 180.0) * math.pi

# 存储图片（辅助）
def writeImg(img, folder, fileName, t = -1, fi = -1):
	writeFolder = 'writeFolder'
	if os.path.exists(writeFolder) != True:
		os.mkdir(writeFolder)
	subFolder = os.path.join(writeFolder, folder.split('/')[1])
	if os.path.exists(subFolder) != True:
		os.mkdir(subFolder)
	subFolder2 = os.path.join(subFolder, fileName)
	if os.path.exists(subFolder2) != True:
		os.mkdir(subFolder2)
	cv.imwrite(os.path.join(subFolder2, str(t) + '-' + str(fi) + '.jpg'), img)

# 存储图片特征
def writeFeature(features, folder, fileName):
	outputFile = 'outputFile'
	checkAndCreateFile(outputFile)
	featureFile = os.path.join(outputFile, 'features')
	checkAndCreateFile(featureFile)
	subFeatureFile = os.path.join(featureFile, folder)
	checkAndCreateFile(subFeatureFile)
	scio.savemat(os.path.join(subFeatureFile, fileName + '.mat'), {'features': features})


# 对单张图片进行仿射变换并得到相应的 features
def AffineTransAndGetFeatures(img, folder = '', fileName = '', saveImg = False):
	flag = 1;
	features = None
	for i in range(0, 5):
		t = math.pow(alpha, i)
		dfi = 72.0 / t
		fi = 0
		while fi <= 180:
			affinedImg = AffineTrans(img, t, AngToArc(fi))
			if saveImg:
				writeImg(affinedImg, folder, fileName, t, fi)
			feature = sift(affinedImg)
			if flag == 1:
				features = feature
				flag = 0
			else:
				features = np.vstack((features, feature))
			fi = fi + dfi
	return features

# 读取所有train img进行仿射变换并且提取特征，进行聚类并最终得到聚类中心
def getCenter(folders):
	flag = 1
	allFeature = None
	outputFile = 'outputFile'
	for i in range(0, len(folders)):
		print('read img in ' + folders[i])
		imgList = []
		folder = folders[i]
		for root, dirs, files in os.walk(os.path.join(baseFile, folder)):
			imgList = files

		for i in range(0, len(imgList)):
			if imgList[i] == '.DS_Store':
				continue
			img = cv.imread(os.path.join(baseFile, folder, imgList[i]))
			print(str(i) + '\nread img:' + imgList[i])
			featurePath = os.path.join(outputFile, 'features', folder, imgList[i] + '.mat')
			if os.path.exists(featurePath) != True:
				imgfeatures = AffineTransAndGetFeatures(img)
				writeFeature(imgfeatures, folder, imgList[i])
				print('write img feature of ' + featurePath)
			else:
				featuresMat = scio.loadmat(featurePath)
				imgfeatures = featuresMat['features']
				print('load img feature of ' + featurePath)
			print('get img feature of ' + imgList[i] + ', shape:' + str(imgfeatures.shape))
			if flag == 1:
				allFeature = imgfeatures
				flag = 0
			else:
				allFeature = np.vstack((allFeature, imgfeatures))
		print('get the features in ' + folder + ': totoal feature shape:' + str(allFeature.shape))
	print('clustering:')
	clustering(allFeature)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# 读取图片并调用聚类操作
def trainCenter():
	print(os.path.exists(baseFile))
	folders = []
	for root, dirs, files in os.walk(baseFile):
		folders = dirs
		break
	trainFolders = []
	for i in range(0, len(folders)):
		folderName = folders[i]
		if folderName == '.DS_Store' or folderName[2:6] == 'test':
			continue
		trainFolders.append(folderName)
		print(trainFolders)
	getCenter(trainFolders)

# 计算一张图片的直方图，返回为 （1*cluster_num）向量
def getHistogramOfImg(img, folder, fileName):
	# get center
	outputFile = 'outputFile'
	centerPath = os.path.join(outputFile, 'center-' + str(cluster_num) + '.mat')
	print('\ntry to get center: ' + centerPath)
	centerMat = scio.loadmat(centerPath)
	centers = centerMat['center']

	# get feature
	featurePath = os.path.join(outputFile, 'features', folder, fileName + '.mat')
	print('try to get ' + featurePath)
	if os.path.exists(featurePath) != True:
		features = AffineTransAndGetFeatures(img)
		writeFeature(features, folder, fileName)
	else:
		featuresMat = scio.loadmat(featurePath)
		features = featuresMat['features']
	feature_num, v = features.shape
	print('feature shape: ' + str(features.shape))

	#计算每一个 feature 与 centers 的距离，找到最近的center，并计算直方图
	histogram = np.zeros(cluster_num, dtype=np.int)
	for i in range(0, feature_num):
		feature = features[i]
		least = sys.maxsize
		label = -1
		for j in range(centers.shape[0]):
			center = centers[j]
			diff = np.square(feature - center)
			diff_sum = diff.sum()
			if (diff_sum < least):
				least = diff_sum
				label = j
		histogram[label] += 1
		# break
	
	print('get histogram of ' + featurePath)
	print(histogram)
	if histogram.sum() != feature_num:
		print('getHistogramOfImg: istogram.sum() != feature_num')
	return histogram

# 获取所有 train 集下的图片或所有 test 集下的图片
# 返回为一个list，包含所有图片的路径（所属文件夹、文件名）以及该图片的 label(1/2/3)
def getAllImgPath(imgType = 'train'):
	print('data basefile:' + baseFile)
	folders = []
	for root, dirs, files in os.walk(baseFile):
		folders = dirs
		break
	subFolders = []
	labels = []
	for i in range(0, len(folders)):
		folderName = folders[i]
		if folderName == '.DS_Store':
			continue
		if imgType == 'train' and folderName[2:6] == 'test':
			continue
		elif imgType == 'test' and folderName[2:6] != 'test':
			continue
		subFolders.append(folderName)
		labels.append(folderName[0])

	print(subFolders)
	print(labels)
	imgList = []
	for i in range(0, len(subFolders)):
		folder = subFolders[i]
		for root, dirs, files in os.walk(os.path.join(baseFile,folder)):
			for j in range(0, len(files)):
				if (files[j] != '.DS_Store'):
					imgList.append([labels[i], folder, files[j]])
	return imgList

# 存储所有图片的直方图
def saveHistogram(histograms, labels, Type):
	outputFile = 'outputFile'
	checkAndCreateFile(outputFile)

	subFolder = os.path.join(outputFile, 'histograms')
	checkAndCreateFile(subFolder)

	subFolder2 = os.path.join(subFolder, Type)
	checkAndCreateFile(subFolder2)

	# 保存直方图
	scio.savemat(os.path.join(subFolder2, Type + '-histograms-' + str(cluster_num) + '.mat'), {Type + 'Histograms': histograms})
	# 保存对应 label
	scio.savemat(os.path.join(subFolder2, Type + '-labels-' + str(cluster_num) + '.mat'), {Type + 'Labels': labels})
	return

# 计算所有图片的直方图
def getHistogramOfAllImgs(Type):
	imgList = getAllImgPath(Type)
	length = len(imgList)
	# 保存所有图片的直方图
	histograms = np.zeros((length, cluster_num))
	flag = 1
	# 保存所有图片的标号
	labels = []
	for i in range(0, 3):
		print(imgList[i])
		label, folder, fileName = imgList[i]
		imgPath = os.path.join(baseFile, folder, fileName)
		print(imgPath)
		img = cv.imread(imgPath)
		imgHistogram = getHistogramOfImg(img, folder, fileName)
		histograms[i] = imgHistogram
		labels.append(float(label))
	
	print('\nget histograms:')
	print(histograms)
	print('\npicture labels:')
	print(labels)

	saveHistogram(histograms, labels, Type)
	print('save all pictures histograms succeed')


# trainCenter()
getHistogramOfAllImgs('test')
print('done')
