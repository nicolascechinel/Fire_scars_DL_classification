import cupy as np
import cv2 as cv
from osgeo import gdal

class handlingSI(object):
	def __init__(self, path_to_SAR_VV=None):
		self.Bands = []
		self.flag = None
		if(path_to_SAR_VV is None):
			self.path_to_SAR_VV = None
			self.raster = None
			self.SARimage = None
		else:
			self.path_to_SAR_VV = path_to_SAR_VV
			self.raster = gdal.Open(self.path_to_SAR_VV)
			if self.raster.RasterCount == 1:
				self.SARimage = self.raster.ReadAsArray()
				self.flag = 1
			else:
				in_band = self.raster.GetRasterBand(1)
				self.SARimage = in_band.ReadAsArray()
				self.flag = self.raster.RasterCount
				for i in range(self.raster.RasterCount):
					self.Bands.append(self.raster.GetRasterBand(i+1).ReadAsArray())


	def set_raster(self, raster):
		self.raster = raster
		self.flag = self.raster.RasterCount
		if self.flag > 1:
			for i in range(self.raster.RasterCount):
				self.Bands.append(self.raster.GetRasterBand(i+1).ReadAsArray())

	def loadSI(self,path_to_SAR_VV):
		self.path_to_SAR_VV = path_to_SAR_VV
		self.raster = gdal.Open(self.path_to_SAR_VV)
		if self.raster.RasterCount == 1:
			self.SARimage = self.raster.ReadAsArray()
			self.flag= 1
		else:
			in_band = self.raster.GetRasterBand(1)
			self.SARimage = in_band.ReadAsArray()
			self.flag = self.raster.RasterCount
			for i in range(self.raster.RasterCount):
				self.Bands.append(self.raster.GetRasterBand(i+1).ReadAsArray()) 


	def saveSI(self,image,outFileName,NoDataValue = -32768):
		#Este método queda completo para cuando se llama a inferencia
		#Si se hace la validación de las bandas con la imagen que llega y no
		#con la imagen de arriba, es decir, no se debería hacer el if con self.flag
		driver = gdal.GetDriverByName("GTiff")
		print("bandera: ",self.flag)
		print("planos: ",len(self.Bands))
		if self.flag==1:
			outdata = driver.Create(outFileName,image.shape[1], image.shape[0], 1, gdal.GDT_Float32)
			outdata.SetGeoTransform(self.raster.GetGeoTransform())
			outdata.SetProjection(self.raster.GetProjection())
			outdata.GetRasterBand(1).WriteArray(image)
			outdata.GetRasterBand(1).SetNoDataValue(NoDataValue)
			outdata.FlushCache()
		else:			
			outdata = driver.Create(outFileName,image.shape[1], image.shape[0], self.flag, gdal.GDT_Float32)
			outdata.SetGeoTransform(self.raster.GetGeoTransform())
			outdata.SetProjection(self.raster.GetProjection())
			Bandas=cv.split(image)
			print(f"Bandas: {len(Bandas)}; image: {image.shape}")
			print("Bandas en saveSI", Bandas[0].shape)
			for i in range(self.flag):
				outdata.GetRasterBand(i+1).WriteArray(Bandas[i])
				outdata.GetRasterBand(i+1).SetNoDataValue(NoDataValue)
			outdata.FlushCache()

	def saveSI_inference(self,image,outFileName,NoDataValue = -32768):
		driver = gdal.GetDriverByName("GTiff")		
		outdata = driver.Create(outFileName,image.shape[1], image.shape[0], 1, gdal.GDT_Float32)
		outdata.SetGeoTransform(self.raster.GetGeoTransform())
		outdata.SetProjection(self.raster.GetProjection())
		outdata.GetRasterBand(1).WriteArray(image)
		outdata.GetRasterBand(1).SetNoDataValue(NoDataValue)
		outdata.FlushCache()
		


	def saveImage(self,image,outFileName):
		cv.imwrite(outFileName,image)