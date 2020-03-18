"""Load PET data and prepare it for reconstruction."""

from mMR import load_data

fdata_amyloid = '/home/cd902/siemens-biograph_data/amyloidPET_FBP_TP0_Matthias'

# I don't have the fdg data
# fdata_fdg = ''

data, background, factors, image, image_mr, image_ct = load_data(
        fdata_amyloid, time=(3000, 3600))

# data, background, factors, image, image_mr, image_ct = load_data(fdata_fdg)