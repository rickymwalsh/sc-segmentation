import numpy as np
import os
import json
import random
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

# Add the spinalcordtoolbox location to the system path.
import sys
from os.path import dirname, abspath, join as oj
path_to_sct = oj(dirname(abspath(__file__)), 'spinalcordtoolbox')
sys.path.append(path_to_sct)

from spinalcordtoolbox.image import Image


# For each dataset:
#	Loop through each subject.
# 	Apply preprocessing
# 		Preprocessing for SCT training:
# 			1. Detect centerline
# 			2. Convert to RPI (? is this necessary for us?)
# 			3. Resample Image to 0.5mm isotropic
# 			4. Cropping image around SC (48mm )  (crop_image_around_centerline)
# 			5. Intensity normalization (with percentiles)  (apply_intensity_normalization)
#			6. Mean/Std. Dev normalization
# 			7. Patch Extraction
#
#	Save processed patches to .pkl



