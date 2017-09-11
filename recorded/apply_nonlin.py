import numpy as np
from scipy.special import expit

train_data = np.load('gates_train_raw.npy')
valid_data = np.load('gates_valid_raw.npy')
test_data = np.load('gates_test_raw.npy')
data_sets = {'train': train_data, 'valid': valid_data, 'test': test_data}

for set_name, data in data_sets.items():
    print('-------' + set_name + '-------')
    np_data = []
    prev_keys = []
    for idx, step in enumerate(data):
        if idx > 0:
            assert prev_keys == list(step.keys())
        else:
            prev_keys = list(step.keys())
        values = np.array(list(step.values()))
        np_data.append(values)

    np_data = np.array(np_data)
    print(prev_keys)
    print(np_data.shape)
    print("Swapping step axis with gate axis")
    np_data = np.swapaxes(np_data, 0, 1)
    print(np_data.shape)

    np_data = expit(np_data)

    np.save('gates_'+set_name+'_nonlin.npy', np_data)
    # np.savez_compressed('gates_train_npz.npz', np_data)
