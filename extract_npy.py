import numpy as np
array_loaded = np.load('contact.npy')
print('shape:')
print(array_loaded.shape)
#print('Loaded: ')
#for l in array_loaded:
#    print(*[e[0] for e in l])
trainX = np.array([array_loaded])
print(trainX.shape)

