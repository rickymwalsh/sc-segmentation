# from ..spinalcordtoolbox import spinalcordtoolbox

from os.path import dirname, abspath, join as oj
import os
import sys

path_to_sct = oj(dirname(dirname(abspath(__file__))), 'spinalcordtoolbox')

sys.path.append(path_to_sct)

from spinalcordtoolbox.image import Image