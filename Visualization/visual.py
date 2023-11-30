import numpy as np 
import cv2 as cv

class visual(object):
	def __init__(self):
		self.image = None
		self.image_tf = None

	def contrast(self,image,minvalue=None,maxvalue=None,method=1):
		#I = []
		#I = cv.split(image)
		#self.image = I[0]
		self.image = image
		if(method == 1):
			self.image_tf = self.convert_8u(minvalue,maxvalue)
			return self.image_tf
		elif(method == 2):
			image_conv = self.convert_8u(minvalue,maxvalue)
			hist,bins = np.histogram(image_conv.flatten(),256,[0,256])
			cdf = hist.cumsum()
			cdf_m = np.ma.masked_equal(cdf,0)
			cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
			cdf = np.ma.filled(cdf_m,0).astype('uint8')
			self.image_tf = cdf[image_conv]
			return self.image_tf
		elif(method == 3):
			self.image_tf = self.convert_8u(minvalue,maxvalue)
			self.image_tf = cv.equalizeHist(self.image_tf)
			return self.image_tf
		elif(method == 4):
			self.image_tf = self.convert_8u(minvalue,maxvalue)
			clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
			self.image_tf = clahe.apply(self.image_tf)
			return self.image_tf
		else:
			print("invalid option")
			return -1

	def convert_8u(self,minvalue,maxvalue):
		if(minvalue is None and maxvalue is None):
			minval = self.image.min()
			maxval = self.image.max()
		elif(minvalue is None):
			minval = self.image.min()
			maxval = maxvalue
		elif(maxvalue is None):
			maxval = self.image.max()
			minval = minvalue
		else:
			maxval = maxvalue
			minval = minvalue
		image = self.image - minval
		image = image / maxval * 255
		image = np.uint8(image)
		return image

	def ColorMapMethod(self,id_colormap):
		'''  
		cv::COLORMAP_AUTUMN = 0,
		cv::COLORMAP_BONE = 1,
		cv::COLORMAP_JET = 2,
		cv::COLORMAP_WINTER = 3,
		cv::COLORMAP_RAINBOW = 4,
		cv::COLORMAP_OCEAN = 5,
		cv::COLORMAP_SUMMER = 6,
		cv::COLORMAP_SPRING = 7,
		cv::COLORMAP_COOL = 8,
		cv::COLORMAP_HSV = 9,
		cv::COLORMAP_PINK = 10,
		cv::COLORMAP_HOT = 11,
		cv::COLORMAP_PARULA = 12,
		cv::COLORMAP_MAGMA = 13,
		cv::COLORMAP_INFERNO = 14,
		cv::COLORMAP_PLASMA = 15,
		cv::COLORMAP_VIRIDIS = 16,
		cv::COLORMAP_CIVIDIS = 17,
		cv::COLORMAP_TWILIGHT = 18,
		cv::COLORMAP_TWILIGHT_SHIFTED = 19,
		cv::COLORMAP_TURBO = 20,
		cv::COLORMAP_DEEPGREEN = 21
		'''
		image = cv.applyColorMap(self.image_tf, id_colormap)
		return image