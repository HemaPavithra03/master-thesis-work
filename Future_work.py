# Import packages 
import numpy as np
import matplotlib.pyplot as plt

# TRY TO SAVE BYTEARRAY AS IMAGE
# Starting point
arrb = bytearray(b'\x00\x00\x01\x90\x88\xc1\xc5\x81\xd9\xff\xb9\xcc\x8b\xed\x86\xe5\xa6\x07\xaa\xc2\xe4\xfc\xb5\xad\x97')

# Approach that works well
arr2 = np.array([[0, 1, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 1, 1, 1, 0]])
print(arr2.shape)
imgplot = plt.imshow(arr2, cmap='Greys')
plt.show()
plt.imsave('name2.png', arr2)

# Test with our data
arr = np.frombuffer(arrb, dtype=np.uint8)
arr = np.reshape(arr, [5, 5])
print(arr.shape)
plt.imsave('name.png', arr)
