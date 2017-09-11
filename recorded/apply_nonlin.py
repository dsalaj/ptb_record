import numpy as np
from scipy.special import expit

train_data = np.load('gates_train_raw.npy')
# valid_data = np.load('gates_valid_raw.npy')
# test_data = np.load('gates_test_raw.npy')

data = train_data

np_data = []
prev_keys = []
for idx, step in enumerate(data):
    if idx > 0:
        assert prev_keys == list(step.keys())
    else:
        prev_keys = list(step.keys())
    values = np.array(list(step.values()))
    np_data.append(values)

del data
del train_data

np_data = np.array(np_data)
print(prev_keys)
print(np_data.shape)
print("Swapping step axis with gate axis")
np_data = np.swapaxes(np_data, 0, 1)
print(np_data.shape)

print("Swap back and save the same")
tmp = np.swapaxes(np_data, 0, 1)
print(tmp.shape)
# np.save('gates_train_np_dict_same_check.npy', tmp)
# del tmp

np_data_dict = []
for idx, step_new in enumerate(tmp):
    gates_dict = {}
    for idy, gate in enumerate(prev_keys):
        gates_dict[gate] = step_new[idy]
    np_data_dict.append(gates_dict)

np.save('gates_train_np_dict_sameold.npy', np_data_dict)
# np.savez_compressed('gates_train_npz.npz', np_data)
