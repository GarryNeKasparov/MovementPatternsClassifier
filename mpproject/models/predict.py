# import os
import sys

# import torch
from data_utils import get_model

if __name__ == "__main__":
    name = sys.argv[1]
    model = get_model(name)
