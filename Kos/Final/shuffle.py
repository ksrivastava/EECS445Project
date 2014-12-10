import numpy as np

data = np.genfromtxt('data.tsv', skip_header=False, delimiter='\t')

n = data.shape[0];
m = data.shape[1] - 1;

t_num_pos = sum(data[:, m] == 1);
t_num_neg = sum(data[:, m] == -1);

# Limit num_neg
num_neg = min(t_num_pos, t_num_neg);
num_pos = min(t_num_pos, t_num_neg);

new_data = np.ndarray( shape=( num_neg + num_pos, data.shape[1]))

idx = 0
for i in range(0, n):
	status = data[i, m]
	write = False;
	if status > 0 and num_pos > 0:
		num_pos -= 1;
		write = True;
	elif status < 0 and num_neg > 0:
		num_neg -= 1;
		write = True;
	if write:
		new_data[idx, :] = data[i, :]
		idx += 1

assert(sum(new_data[:, m] == 1) == sum(new_data[:, m] == -1))

np.random.shuffle(new_data)
# new_data.tofile('data_shuffled.tsv', sep='\t')

np.save('data_shuffled', new_data)