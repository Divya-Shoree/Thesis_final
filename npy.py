import numpy as np
array = np.random.rand(583,274,274)
print(array)

np.save('test.npy', array)


