import csv
import cupy as np

class utils(object):
	def __init__(self,path_to_csv=None):
		if path_to_csv is None:
			self.path_to_csv = None
		else:
			self.path_to_csv = path_to_csv

	def set_path_to_csv(self,path_to_csv):
		self.path_to_csv = path_to_csv

	def write_pointers(self, pointers):
		with open(self.path_to_csv, 'a+', newline='') as file:
			writer = csv.writer(file, delimiter=',')
			writer.writerows(pointers)

	def read_pointers(self):
		pointers = []
		with open(self.path_to_csv, 'r') as file:
			reader = csv.reader(file, delimiter = ',')
			for row in reader:
				pointers.append(row)
		return pointers

	def set_ROI (self,img,punto,kernel):
		fi=punto[1]-kernel
		ff=punto[1]+kernel+1
		ci=punto[0]-kernel
		cf=punto[0]+kernel+1
		ROI = img[fi:ff,ci:cf] #building roi (rows(y) cols(x))
		return ROI

	def printProgressBar (self, iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
		"""
		Call in a loop to create terminal progress bar 
		@params:
			iteration   - Required  : current iteration (Int)
			total       - Required  : total iterations (Int)
			prefix      - Optional  : prefix string (Str)
			suffix      - Optional  : suffix string (Str)
			decimals    - Optional  : positive number of decimals in percent complete (Int)
			length      - Optional  : character length of bar (Int)
			fill        - Optional  : bar fill character (Str)
			printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
		"""
		percent = ("{0:"+ "f}").format(1 * (iteration / float(total)))
		filledLength = int(length * iteration // total)
		bar = fill * filledLength + '-' * (length - filledLength)
		print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
		# Print New Line on Complete
		if iteration == total: 
			print()