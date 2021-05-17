import os
import matplotlib.pyplot as plt
import numpy as np

# Stage 1 Plot
batch_size_choices = [64]
net = 'ResNet50'
dataset = 'cifar10'
normalization_choices = ['bn', 'gn', 'gn_plus_sequential_gn_first']
epochs = 164
schedule = [81, 122]
weight_decay = 1e-4

train_acc = {}
val_acc = {}
for bs in batch_size_choices:
	lr = (1e-1 * bs) / 128
	train_acc[bs] = {}
	val_acc[bs] = {}
	for norm in normalization_choices:
		dir_name = '{}_{}_{}_lr{}_bs{}_epochs{}_sched{}_{}_wd{}'.format(
			net,
			dataset,
			norm,
			lr,
			bs,
			epochs,
			schedule[0],
			schedule[1],
			weight_decay
		)
		train_acc[bs][norm] = []
		val_acc[bs][norm] = []
		file_path = os.path.join(dir_name, 'log.txt')
		if os.path.isfile(file_path):
			f = open(file_path)
			lines = f.readlines()
			del lines[0]
			for line in lines:
				line = line.split('\t')
				train_acc[bs][norm].append(float(line[3]))
				val_acc[bs][norm].append(float(line[4]))

x = np.arange(epochs)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(x, train_acc[64]['bn'], 'r-.')
ax1.plot(x, train_acc[64]['gn'], 'g-.')
ax1.plot(x, train_acc[64]['gn_plus_sequential_gn_first'], 'b-.')
ax1.set_xticks(np.arange(0, epochs, 20))
ax1.set_yticks(np.arange(0, 100, 5))
ax1.set_ylim(60, 100)
ax1.yaxis.grid()

ax1.set_title('train accuracy', fontsize=11)
ax1.legend(['Batch Norm (BN)', 'Group Norm (GN)', 'Group Norm Plus (GN+)'], loc='lower right')
ax1.set_xlabel('epochs', fontsize=10)
ax1.set_ylabel('accuracy (%)', fontsize=10)


ax2.plot(x, val_acc[64]['bn'], 'r')
ax2.plot(x, val_acc[64]['gn'], 'g')
ax2.plot(x, val_acc[64]['gn_plus_sequential_gn_first'], 'b')
ax2.set_xticks(np.arange(0, epochs, 20))
ax2.set_yticks(np.arange(0, 100, 5))
ax2.set_ylim(60, 100)
ax2.yaxis.grid()

ax2.set_title('val accuracy', fontsize=11)
ax2.legend(['Batch Norm (BN)', 'Group Norm (GN)', 'Group Norm Plus (GN+)'], loc='lower right')
ax2.set_xlabel('epochs', fontsize=10)
ax2.set_ylabel('accuracy (%)', fontsize=10)
plt.show()