import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

plt.rcParams.update({'font.size': 17})

def AA(df, threshold, model, plot=True):
	x = df['hl-err']
	x = np.sort(x)
	# Calculate the cumulative distribution
	y = (1 + np.arange(len(x))) / len(x)
	if plot:
		plt.plot(x, y, label=model, linewidth=4)
		plt.xlabel('Error')
		plt.ylabel('Cumulative Probability')
		# plt.title('Cumulative Error Distribution')
		plt.axis([0, threshold, 0, 1])
		plt.xticks(np.linspace(0, threshold, 6))
		plt.xticks(np.linspace(0, threshold, 26), minor=True)
	index = np.searchsorted(x, threshold)
	x = np.concatenate([x[:index], [threshold]])
	y = np.concatenate([y[:index], [threshold]])
	print(f"\t{model} AA: ", ((x[1:] - x[:-1]) * y[:-1]).sum() / threshold * 100)

def main(args):
	dataset = args.dataset
	ctrlc = pd.read_csv(f'outputs/{dataset}/ctrlc/028.csv')
	mscc = pd.read_csv(f'outputs/{dataset}/mscc/028.csv')
	sofi = pd.read_csv(f'outputs/{dataset}/sofi/028.csv')

	plt.figure()
	thresholds = [0.1, 0.15, 0.25]
	for t in thresholds:
		if t ==0.25:
			plot=args.plot
		else:
			plot=False
		print(f"threshold {t}")
		AA(ctrlc, t, 'CTRL-C', plot)
		AA(mscc, t, 'MSCC', plot)
		AA(sofi, t, 'SOFI', plot)
	plt.legend(loc=4)
	plt.grid(axis='y')
	plt.savefig(f'{dataset}_auc.png', pad_inches=0, bbox_inches='tight')

	# plt.figure()
	# ctrlc = pd.read_csv('outputs/ctrlc/HLW/028.csv')
	# mscc = pd.read_csv('outputs/mscc_part2/HLW/028.csv')
	# ctrlcplus = pd.read_csv('outputs/ctrlcplus/HLW/027.csv')

	# thresholds = [1, 1.5, 2.5]
	# for t in thresholds:
	# 	if t ==1.5:
	# 		plot=args.plot
	# 	else:
	# 		plot=False
	# 	print(f"threshold {t}")
	# 	AA(ctrlc, t, 'ctrlc', plot)
	# 	AA(ctrlcplus, t, 'ctrlcplus', plot)
	# 	AA(mscc, t, 'mscc', plot)
	# plt.legend(loc=4)
	# plt.grid(axis='y')
	# plt.savefig('hlw_auc.png', pad_inches=0, bbox_inches='tight')


	# plt.figure()
	# ctrlc = pd.read_csv('outputs/ctrlc/holicity/028.csv')
	# mscc = pd.read_csv('outputs/mscc_part2/holicity/028.csv')
	# ctrlcplus = pd.read_csv('outputs/ctrlcplus/holicity/027.csv')

	# thresholds = [0.1, 0.15, 0.25]
	# for t in thresholds:
	# 	if t ==1:
	# 		plot=args.plot
	# 	else:
	# 		plot=False
	# 	print(f"threshold {t}")
	# 	AA(ctrlc, t, 'ctrlc', plot)
	# 	AA(ctrlcplus, t, 'ctrlcplus', plot)
	# 	AA(mscc, t, 'mscc', plot)
	# plt.legend(loc=4)
	# plt.grid(axis='y')
	# plt.savefig('holicity_auc.png', pad_inches=0, bbox_inches='tight')
	# #plt.show()


if __name__ == '__main__':
	parser = argparse.ArgumentParser('Plotting AUC')
	parser.add_argument('--plot', default=False, type=bool)
	parser.add_argument('--dataset', default=False, type=str)
	args = parser.parse_args()
	main(args)
