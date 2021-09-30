# %% imports
import numpy as np

from niftypet import nipet
from niftypet import nimpa

Cnt, txLUT, axLUT = nipet.mmraux.mmrinit()
Cnt['SPN'] = 1
ind = np.array([-1], dtype=np.int32)
# partial sino: array of any transaxial index between 0 and 68516
xshape = (320, 320, 127)
yshape = (68516, 4084)
x=np.ones(xshape, 'float32')
y=np.zeros(yshape, 'float32')

# %% try proj
nipet.prj.petprj.fprj(y, x, txLUT, axLUT, ind, Cnt, 0)

