import os
import sys

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)
from deim_wapper.deimv2_det import DEIMv2Det
