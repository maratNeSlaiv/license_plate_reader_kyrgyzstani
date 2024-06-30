# Libraries 
import numpy as np
import matplotlib.pyplot as plt
import contextlib

# Dependencies
from ..tools.model_hub import get_device_torch
from ..tools.image_processing import normalize_img, convert_cv_zones_rgb_to_bgr
from ..tools.model_hub import get_device_torch

device_torch = get_device_torch()

CLASS_REGION_ALL = [
    "xx-unknown",
    "eu-ua-2015",
    "eu-ua-2004",
    "eu-ua-1995",
    "eu",
    "xx-transit",
    "ru",
    "kz",
    "eu-ua-ordlo-dpr",
    "eu-ua-ordlo-lpr",
    "ge",
    "by",
    "su",
    "kg",
    "am",
    "ua-military",
    "ru-military",
    "md",
    "eu-ua-custom",
]

CLASS_LINES_ALL = [
    "0",  # garbage
    "1",  # one line
    "2",  # two line
    "3",  # three line
]

CLASS_STATE_ALL = [
    "garbage",  # garbage
    "filled",  # manual filled number
    "not filled",  # two line
    "empty"  # deprecated
]

def imshow(img: np.ndarray) -> None:
    """
    # functions to show an image
    """
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

@contextlib.contextmanager
def dummy_context_mgr():
    yield None
