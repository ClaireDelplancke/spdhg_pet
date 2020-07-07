"""Load PET data and prepare it for reconstruction."""

from mMR import load_data

fdata_amyloid = '/home/cd902/siemens-biograph_data/amyloidPET_FBP_TP0'


data, background, factors, image,  image_ct = load_data(
        fdata_amyloid, time=(3000, 3600))

# %%
""" Load with MRI data as well """
# delete output of load_data before or load_data_with_mri will just load it
# but not make computations

from mMR import load_data_with_mri

fdata_amyloid = '/home/cd902/siemens-biograph_data/amyloidPET_FBP_TP0'


data, background, factors, image, image_mr, image_ct = load_data_with_mri(
        fdata_amyloid, time=(3000, 3600))

