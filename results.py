import pandas as pd
import numpy as np
import argparse
import os

def AA(df, threshold):
	x = df['hl-err']
	x = np.sort(x)
	# Calculate the cumulative distribution
	y = (1 + np.arange(len(x))) / len(x)
	index = np.searchsorted(x, threshold)
	x = np.concatenate([x[:index], [threshold]])
	y = np.concatenate([y[:index], [threshold]])
	return ((x[1:] - x[:-1]) * y[:-1]).sum() / threshold * 100

threshold = {
'gsv': 			[0.1, 0.15, 0.25],
'holicity': 	[0.1, 0.15, 0.25],
'hlw': 			[0.1, 0.15, 0.25],
'yud': 			[0.1, 0.15, 0.25]
}

def main(folder_path, dataset):
	items = os.listdir(os.path.join(folder_path, dataset))
	for item in items:
		item_path = os.path.join(folder_path, dataset, item)
		# Check if the item is a directory (subfolder)
		if os.path.isdir(item_path):
		    print(f"Subfolder found: {item}")
		    # List files within the subfolder
		    subfolder_items = os.listdir(item_path)
		    if subfolder_items:
		        print(f"Files found in subfolder {item}:")
		        subfolder_items.sort()
		        for sub_item in subfolder_items:
		        	print(f"opening {sub_item}")
		        	file_path = os.path.join(item_path, sub_item)
		        	df = pd.read_csv(file_path)
		        	mean_values = df.iloc[:, 1:6].mean()
		        	std_values = df.iloc[:, 1:6].std()
		        	median_values = df.iloc[:, 1:6].median()
		        	max_values = df.iloc[:, 1:6].max()
		        	stats_df = pd.DataFrame({
					    'Mean': mean_values,
					    'Std': std_values,
					    '50%': median_values,
					    'Max': max_values})
		        	if dataset not in ['hlw']:
		        		print(stats_df.transpose())
		        	print([AA(df, t) for t in threshold[dataset]])
		        		

		    else:
		        print(f"No files found in subfolder {item}")
if __name__ == '__main__':
	parser = argparse.ArgumentParser("Dataset evaluation") 
	parser.add_argument('--dataset', default='gsv')
	args = parser.parse_args()
	main("outputs", args.dataset)