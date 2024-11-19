from pylsd import lsd
import cv2
import json
import csv
from pathlib import Path

import os
import os.path as osp
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob

def filter_length(segs, min_line_length=10):
    lengths = LA.norm(segs[:,2:4] - segs[:,:2], axis=1)
    segs = segs[lengths > min_line_length]
    return segs[:,:4]

def main_hlw(file, folder):
	i = 0
	df = pd.read_csv(file, header=None)
	names = df.to_numpy()[:, 0]

	# For HLW dataset
	for name in names:
		file_path = osp.join(folder, name)
		csv_path = file_path.split('.jpg')[0] + '_line.csv'

		image = cv2.imread(file_path)
		img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		org_segs = lsd(img, scale=0.5)
		org_segs = filter_length(org_segs)
		with open(csv_path, 'w', newline='') as csvfile:
			csvwriter = csv.writer(csvfile)
			for line in org_segs:
				x1, y1, x2, y2 = line
				csvwriter.writerow([x1, y1, x2, y2])
		print(f"Done writing {csv_path}. Image {i}")
		i += 1

def main_holicity(file, folder):
	i = 0
	df = pd.read_csv(file, header=None)
	names = df.to_numpy()[:, 0]
	# Create folders
	# For Holicity
	for name in names:
		Path(osp.join(folder, 'line', name.split('/')[0])).mkdir(parents=True, exist_ok=True)
		file_path = osp.join(folder, 'image', name) + '.jpg'
		files = glob(osp.join(folder, 'image', name)+ '*')
		for file_path in files:
			file_name = file_path.split('image/')[-1].split('.jpg')[0]
			file_name = file_name.split('_imag')[0]
			csv_path = osp.join(folder, 'line', file_name) + '_line.csv'
			image = cv2.imread(file_path)
			img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			org_segs = lsd(img, scale=0.5)
			org_segs = filter_length(org_segs)
			with open(csv_path, 'w', newline='') as csvfile:
				csvwriter = csv.writer(csvfile)
				for line in org_segs:
					x1, y1, x2, y2 = line
					csvwriter.writerow([x1, y1, x2, y2])
			if i % 50 == 0:
				print(f"Done writing {csv_path}. Image {i}")
			i += 1

	# case = file.split('/')[-1].split('-')[0]
	# csv_path = osp.join(folder, 'split', f'{case}-split.csv')

	# with open(csv_path, 'w', newline='') as csvfile:
	# 	csvwriter = csv.writer(csvfile)
	# 	for file in test_file:
	# 		csvwriter.writerow([file])

	# 	i += 1



if __name__ == '__main__':
	main_holicity('data_csv/holicity-test-split.csv', 'data/holicity')
	# main_hlw('data_csv/hlw_test.csv', 'data/hlw/images')
