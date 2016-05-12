import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_loss(log_fname):
	loss = []
	with open(log_fname) as f:
		for line in f:
			words = line.strip()
			loss.append(float(words))

	sns.set(style='whitegrid')

	fig = plt.figure()

	ax = fig.add_subplot(1, 1, 1)
	ax.set_ylim([-0.5, 1])
	ax.plot(loss, linewidth=3)
	ax.set_title('Train loss')
	ax.set_xlabel('iterations')

	plt.show()


if __name__ == '__main__':
    log_file = '/home/boyarov/Projects/cpp/feedforward_network/log_loss.txt'
    # log_file = '/media/datac/andrew_workspace/darknet_1/log.txt'

    plot_loss(log_file)


