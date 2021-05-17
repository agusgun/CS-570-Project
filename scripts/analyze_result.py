import os


# Stage 1 Analysis
batch_size_choices = [8, 16, 32, 64, 128]
net = 'ResNet50'
dataset = 'cifar10'
normalization_choices = ['bn', 'gn', 'gn_plus_sequential_bn_first', 'gn_plus_sequential_gn_first', 'gn_plus_parallel']
epochs = 164
schedule = [81, 122]
weight_decay = 1e-4

result = {}
for bs in batch_size_choices:
	lr = (1e-1 * bs) / 128
	result[bs] = {}
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
		file_path = os.path.join(dir_name, 'log.txt')
		if os.path.isfile(file_path):
			f = open(file_path)
			lines = f.readlines()
			del lines[0]
			best_val_acc = 0
			for line in lines:
				line = line.split('\t')
				if float(line[4]) > best_val_acc:
					best_val_acc = float(line[4])
			result[bs][norm] = best_val_acc
		else:
			result[bs][norm] = 0


f = open('result_stage1.txt', 'w')
f.write('\t')
for norm in normalization_choices:
	f.write('{}\t'.format(norm))
f.write('\n')
for bs, res in result.items():
	f.write('{}\t'.format(bs))
	for method, best_acc in res.items():
		f.write('{}\t'.format(best_acc))
	f.write('\n')
f.close()
print(result)


