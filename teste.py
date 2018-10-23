import utils as ut
import numpy as np
img = np.array([[0,0,0,0,0,0],[0,1,1,1,1,0],[0,1,1,1,1,0],[0,1,1,1,1,0],[0,0,0,0,0,0]])
print(img.shape)
print(img)
new_img = ut.filtroMinimo(img,3,3)
print(new_img)
