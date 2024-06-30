import torch
from torchvision.models import resnet18
# Libraries
import matplotlib.pyplot as plt
from numpy import mean
import imgaug as ia
import collections
try:
    collections_abc = collections.abc
except AttributeError:
    collections_abc = collections

# Dependencies
from ..tools.model_hub import get_device_torch

device_torch = get_device_torch()

def aug_seed(num: int = None) -> None:
    if num is None:
        ia.seed()
    else:
        ia.seed(num)

def plot_loss(epoch: int,
              train_losses: list,
              val_losses: list,
              n_steps: int = 100):
    """
    Plots train and validation losses
    """

    # making titles
    train_title = f'Epoch:{epoch} | Train Loss:{mean(train_losses[-n_steps:]):.6f}'
    val_title = f'Epoch:{epoch} | Val Loss:{mean(val_losses[-n_steps:]):.6f}'

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(train_losses)
    ax[1].plot(val_losses)

    ax[0].set_title(train_title)
    ax[1].set_title(val_title)

    plt.show()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

DEFAULT_PRESETS = {
    "eu_ua_2004_2015_efficientnet_b2": {
        "for_regions": ["eu_ua_2015", "eu_ua_2004"],
        "model_path": "latest"
    },
    "eu_ua_1995_efficientnet_b2": {
        "for_regions": ["eu_ua_1995"],
        "model_path": "latest"
    },
    "eu_ua_custom_efficientnet_b2": {
        "for_regions": ["eu_ua_custom"],
        "model_path": "latest"
    },
    "eu_efficientnet_b2": {
        "for_regions": ["eu", "xx_transit", "xx_unknown"],
        "model_path": "latest"
    },
    "ru": {
        "for_regions": ["ru", "eu_ua_ordlo_lpr", "eu_ua_ordlo_dpr"],
        "model_path": "latest"
    },
    "kz": {
        "for_regions": ["kz"],
        "model_path": "latest"
    },
    "kg": {  # "kg_shufflenet_v2_x2_0"
        "for_regions": ["kg"],
        "model_path": "latest"
    },
    "ge": {
        "for_regions": ["ge"],
        "model_path": "latest"
    },
    "su_efficientnet_b2": {
        "for_regions": ["su"],
        "model_path": "latest"
    },
    "am": {
        "for_regions": ["am"],
        "model_path": "latest"
    },
    "by": {
        "for_regions": ["by"],
        "model_path": "latest"
    },
}

if __name__ == "__main__":
    h, w, c, b = 50, 200, 3, 1
    net = NPOcrNet(letters=["A", "B"],
                   letters_max=2,
                   max_text_len=8,
                   learning_rate=0.02,
                   bidirectional=True,
                   label_converter=None,
                   val_dataset=None,
                   height=h,
                   width=w,
                   color_channels=c,
                   weight_decay=1e-5,
                   momentum=0.9,
                   clip_norm=5,
                   hidden_size=32,
                   backbone=resnet18)
    device = get_device_torch()
    net = net.to(device)
    xs = torch.rand((b, c, h, w)).to(device)

    print("MODEL:")
    print("xs", xs.shape)
    y = net(xs)
    print("y", y.shape)
