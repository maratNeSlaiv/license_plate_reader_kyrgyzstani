# Libraries
import os
import cv2
import json
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from collections import Counter
from typing import Optional
from torch.nn import functional
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from typing import List, Tuple, Any, Dict
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from torchvision import models
from torch.utils.data import DataLoader
import itertools
import warnings
import copy
from torch import no_grad
import collections
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
try:
    collections_abc = collections.abc
except AttributeError:
    collections_abc = collections

# Dependencies
from ..tools.image_processing import convert_cv_zones_rgb_to_bgr, normalize_img
from ..tools.model_hub import modelhub
from ..tools.data_loaders import TextImageGenerator
from .ocr_tools import (weights_init, device_torch, aug_seed)

class StrLabelConverter(object):
    """Convert between str and label.
        Insert `blank` to the alphabet for CTC.
    Args:
        letters (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, letters: str,
                 max_text_len: int,
                 ignore_case: bool = True):
        self._ignore_case = ignore_case
        if self._ignore_case:
            letters = letters.lower()
        self.letters = letters
        self.letters_max = len(self.letters) + 1
        self.max_text_len = max_text_len

    def labels_to_text(self, labels: List) -> str:
        out_best = [k for k, g in itertools.groupby(labels)]
        outstr = ''
        for c in out_best:
            if c != 0:
                outstr += self.letters[c - 1]
        return outstr

    def encode(self, text):
        """Support batch or single str.
        Args:
            text (str or list of str): texts to convert.
        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        length = []
        if isinstance(text, str):
            text = list(map(lambda x: self.letters.index(x) + 1, text))
            while len(text) < self.max_text_len:
                text.append(0)
            length = [len(text)]
        elif isinstance(text, collections_abc.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return torch.IntTensor(text), torch.IntTensor(length)

    def decode(self, t, length):
        """Decode encoded texts back into strs.
        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        Raises:
            AssertionError: when the texts and its length does not match.
        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(),
                                                                                                         length)
            out_best = list(np.argmax(t[0, :], 1))
            out_best = [k for k, g in itertools.groupby(out_best)]
            outstr = ''
            for c in out_best:
                if c != 0:
                    outstr += self.letters[c - 1]
            return outstr
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(
                t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                texts.append(
                    self.decode(
                        t[index:index + length[i]], torch.IntTensor([length[i]])))
                index += length[i]
        return texts


def decode_prediction(logits: torch.Tensor,
                      label_converter: StrLabelConverter) -> str:
    tokens = logits.softmax(2).argmax(2)
    tokens = tokens.squeeze(1).numpy()

    text = label_converter.labels_to_text(tokens)
    return text

def decode_batch(net_out_value: torch.Tensor,
                 label_converter: StrLabelConverter) -> str or List:
    texts = []
    for i in range(net_out_value.shape[1]):
        logits = net_out_value[:, i:i+1, :]
        pred_texts = decode_prediction(logits, label_converter)
        texts.append(pred_texts)
    return texts

def print_prediction(model,
                     dataset,
                     device,
                     label_converter,
                     w=200,
                     h=50,
                     count_zones=16):

    idx = np.random.randint(len(dataset))
    path = dataset.pathes[idx]

    with torch.no_grad():
        model.eval()
        img, target_text = dataset[idx]
        img = img.unsqueeze(0)
        logits = model(img.to(device))

    pred_text = decode_prediction(logits.cpu(), label_converter)
    img = Image.open(path).convert('L')
    img = img.resize((w, h))
    draw = ImageDraw.Draw(img)
    for i in np.arange(0, w, w / count_zones):
        if 1 > i or i > w:
            continue
        draw.line((i, 0, i, img.size[0]), fill=256)
    img = np.asarray(img)
    title = f'Truth: {target_text} | Pred: {pred_text}'
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
    
class NPOcrNet(pl.LightningModule):
    def __init__(self,
                 letters: List = None,
                 letters_max: int = 0,
                 max_text_len: int = 8,
                 learning_rate: float = 0.02,
                 height: int = 50,
                 width: int = 200,
                 color_channels: int = 3,
                 bidirectional: bool = True,
                 label_converter: Any = None,
                 val_dataset: Any = None,
                 weight_decay: float = 1e-5,
                 momentum: float = 0.9,
                 clip_norm: int = 5,
                 hidden_size=32,
                 linear_size=512,
                 backbone=None):
        super().__init__()
        self.save_hyperparameters()

        self.width = width
        self.height = height
        self.linear_size = linear_size
        self.color_channels = color_channels

        self.letters = letters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.clip_norm = clip_norm
        self.momentum = momentum
        self.max_text_len = max_text_len

        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.label_converter = label_converter

        # convolutions
        if backbone is None:
            backbone = resnet18
        conv_nn = backbone(pretrained=True)
        if 'resnet' in str(backbone):
            conv_modules = list(conv_nn.children())[:-3]
        elif 'efficientnet' in str(backbone):
            conv_modules = list(conv_nn.children())[:-2]
        elif 'shufflenet' in str(backbone):
            conv_modules = list(conv_nn.children())[:-3]
        else:
            raise NotImplementedError(backbone)
        self.conv_nn = nn.Sequential(*conv_modules)
        _, backbone_c, backbone_h, backbone_w = self.conv_nn(torch.rand((1, color_channels, height, width))).shape

        assert backbone_w > max_text_len

        # RNN + Linear
        self.linear1 = nn.Linear(backbone_c*backbone_h, self.linear_size)
        self.recurrent_layer1 = BlockRNN(self.linear_size, hidden_size, hidden_size,
                                         bidirectional=bidirectional)
        self.recurrent_layer2 = BlockRNN(hidden_size, hidden_size, letters_max,
                                         bidirectional=bidirectional)

        self.linear2 = nn.Linear(hidden_size * 2, letters_max)

        self.automatic_optimization = True
        self.criterion = None
        self.val_dataset = val_dataset
        self.train_losses = []
        self.val_losses = []

    def forward(self, batch: torch.float64):
        """
        forward
        """
        batch_size = batch.size(0)
        
        # convolutions
        batch = self.conv_nn(batch)

        # make sequences of image features
        batch = batch.permute(0, 3, 1, 2)
        n_channels = batch.size(1)
        batch = batch.reshape(batch_size, n_channels, -1)
        batch = self.linear1(batch)

        # rnn layers
        batch = self.recurrent_layer1(batch, add_output=True)
        batch = self.recurrent_layer2(batch)

        # output
        batch = self.linear2(batch)
        batch = batch.permute(1, 0, 2)
        return batch

    def init_loss(self):
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction='mean')

    def calculate_loss(self, logits, texts):
        if self.criterion is None:
            self.init_loss()

        # get infomation from prediction
        device = logits.device
        input_len, batch_size, vocab_size = logits.size()
        # encode inputs
        logits = logits.log_softmax(2)
        encoded_texts, text_lens = self.label_converter.encode(texts)
        logits_lens = torch.full(size=(batch_size,), fill_value=input_len, dtype=torch.int32)
        # calculate ctc
        loss = self.criterion(
            logits,
            encoded_texts,
            logits_lens.to(device),
            text_lens)
        return loss

    def step(self, batch):
        x, texts = batch
        output = self.forward(x)
        loss = self.calculate_loss(output, texts)
        return loss

    def on_save_checkpoint(self, _):
        if self.current_epoch and self.val_dataset:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print_prediction(self, self.val_dataset, device, self.label_converter)
            plot_loss(self.current_epoch, self.train_losses, self.val_losses)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            nesterov=True,
            weight_decay=self.weight_decay,
            momentum=self.momentum)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        tqdm_dict = {
            'train_loss': loss,
        }
        return {
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        }

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        tqdm_dict = {
            'val_loss': loss,
        }
        return {
            'val_loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        }

    def test_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        tqdm_dict = {
            'test_loss': loss,
        }
        return {
            'test_loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        }

#
#
#
#
#
#
#
#
#
#
#
#
#

class OCRError(Exception):
    ...

class OcrNetDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_dir,
                 val_dir,
                 test_dir,
                 letters,
                 max_text_len,
                 width=128,
                 height=64,
                 batch_size=32,
                 num_workers=0,
                 seed=42,
                 with_aug=False):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        # init train generator
        self.train = None
        self.train_image_generator = TextImageGenerator(
            train_dir,
            letters,
            max_text_len,
            img_w=width,
            img_h=height,
            batch_size=batch_size,
            seed=seed,
            with_aug=with_aug)

        # init validation generator
        self.val = None
        self.val_image_generator = TextImageGenerator(
            val_dir,
            letters,
            max_text_len,
            img_w=width,
            img_h=height,
            batch_size=batch_size)

        # init test generator
        self.test = None
        self.test_image_generator = TextImageGenerator(
            test_dir,
            letters,
            max_text_len,
            img_w=width,
            img_h=height,
            batch_size=batch_size)

    def prepare_data(self):
        return

    def setup(self, stage=None):
        self.train = self.train_image_generator
        self.val = self.val_image_generator
        self.test = self.test_image_generator

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self.test_dataloader()

    def teardown(self, stage: Optional[str] = None):
        # clean up after fit or test
        # called on every process in DDP
        ...

#
#
#
#
#
#
#
#
#
#
#
#
        
class OCR(object):

    def __init__(self, model_name: str = None, letters: List = None, linear_size: int = 512,
                 max_text_len: int = 0, height: int = 50, width: int = 200, color_channels: int = 3,
                 hidden_size: int = 32, backbone: str = "resnet18",
                 off_number_plate_classification=True, **_) -> None:
        self.model_name = model_name
        self.letters = []
        if letters is not None:
            self.letters = letters

        # model
        self.dm = None
        self.model = None
        self.trainer = None

        # Input parameters
        self.linear_size = linear_size
        self.max_text_len = max_text_len
        self.height = height
        self.width = width
        self.color_channels = color_channels

        # Train hyperparameters
        self.hidden_size = hidden_size
        self.backbone = getattr(models, backbone)
        self.batch_size = 32
        self.epochs = 1
        self.gpus = 1

        self.label_converter = None
        self.path_to_model = None

    def init_label_converter(self):
        self.label_converter = StrLabelConverter("".join(self.letters), self.max_text_len)

    @staticmethod
    def get_counter(dirpath: str, verbose: bool = True) -> Tuple[Counter, int]:
        dir_name = os.path.basename(dirpath)
        ann_dirpath = os.path.join(dirpath, 'ann')
        letters = ''
        lens = []
        for file_name in os.listdir(ann_dirpath):
            json_filepath = os.path.join(ann_dirpath, file_name)
            description = json.load(open(json_filepath, 'r'))['description']
            lens.append(len(description))
            letters += description
        max_text_len = max(Counter(lens).keys())
        if verbose:
            print('Max plate length in "%s":' % dir_name, max_text_len)
        return Counter(letters), max_text_len

    def get_alphabet(self, train_path: str, test_path: str, val_path: str, verbose: bool = True) -> Tuple[List, int]:
        c_val, max_text_len_val = self.get_counter(val_path)
        c_train, max_text_len_train = self.get_counter(train_path)
        c_test, max_text_len_test = self.get_counter(test_path)

        letters_train = set(c_train.keys())
        letters_val = set(c_val.keys())
        letters_test = set(c_test.keys())
        if verbose:
            print("Letters train ", letters_train)
            print("Letters val ", letters_val)
            print("Letters test ", letters_test)

        if max_text_len_val == max_text_len_train:
            if verbose:
                print('Max plate length in train, test and val do match')
        else:
            raise OCRError('Max plate length in train, test and val do not match')

        if letters_train == letters_val:
            if verbose:
                print('Letters in train, val and test do match')
        else:
            raise OCRError('Letters in train, val and test do not match')

        self.letters = sorted(list(letters_train))
        self.max_text_len = max_text_len_train
        if verbose:
            print('Letters:', ' '.join(self.letters))
        return self.letters, self.max_text_len

    def prepare(self,
                path_to_dataset: str,
                use_aug: bool = False,
                seed: int = 42,
                verbose: bool = True,
                num_workers: int = 0) -> None:
        train_dir = os.path.join(path_to_dataset, "train")
        test_dir = os.path.join(path_to_dataset, "test")
        val_dir = os.path.join(path_to_dataset, "val")

        if verbose:
            print("GET ALPHABET")
        self.letters, self.max_text_len = self.get_alphabet(
            train_dir,
            test_dir,
            val_dir,
            verbose=verbose)
        self.init_label_converter()

        if verbose:
            print("START BUILD DATA")
        # compile generators
        self.dm = OcrNetDataModule(
            train_dir,
            val_dir,
            test_dir,
            self.letters,
            self.max_text_len,
            width=self.width,
            height=self.height,
            batch_size=self.batch_size,
            num_workers=num_workers,
            seed=seed,
            with_aug=use_aug)
        if verbose:
            print("DATA PREPARED")

    def create_model(self):
        """
        TODO: describe method
        """
        self.model = NPOcrNet(self.letters,
                              linear_size=self.linear_size,
                              hidden_size=self.hidden_size,
                              backbone=self.backbone,
                              letters_max=len(self.letters) + 1,
                              label_converter=self.label_converter,
                              height=self.height,
                              width=self.width,
                              color_channels=self.color_channels,
                              max_text_len=self.max_text_len)
        if 'resnet' in str(self.backbone):
            self.model.apply(weights_init)
        self.model = self.model.to(device_torch)
        return self.model

    def train(self,
              log_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../data/logs/ocr')),
              seed: int = None,
              ckpt_path: str = None
              ) -> NPOcrNet:
        """
        TODO: describe method
        """
        if seed is not None:
            aug_seed(seed)
            pl.seed_everything(seed)
        if self.model is None:
            self.create_model()
        checkpoint_callback = ModelCheckpoint(dirpath=log_dir, monitor='val_loss')
        lr_monitor = LearningRateMonitor(logging_interval='step')
        if self.gpus:
            self.trainer = pl.Trainer(max_epochs=self.epochs,
                                      accelerator='gpu', devices=self.gpus,
                                      callbacks=[checkpoint_callback, lr_monitor])
        else:
            self.trainer = pl.Trainer(max_epochs=self.epochs,
                                      accelerator='cpu',
                                      callbacks=[checkpoint_callback, lr_monitor])
        self.trainer.fit(self.model, self.dm, ckpt_path=ckpt_path)
        print("[INFO] best model path", checkpoint_callback.best_model_path)
        return self.model

    def validation(self, val_losses, device):
        with torch.no_grad():
            self.model.eval()
            for batch_img, batch_text in self.dm.val_dataloader():
                logits = self.model(batch_img.to(device))
                val_loss = self.model.calculate_loss(logits, batch_text)
                val_losses.append(val_loss.item())
        return val_losses

    def tune(self, percentage=0.05) -> Dict:
        """
        TODO: describe method
        """
        if self.model is None:
            self.create_model()

        if self.gpus:
            trainer = pl.Trainer(max_epochs=self.epochs,
                                 accelerator='gpu', devices=self.gpus,
                                 )
        else:
            trainer = pl.Trainer(max_epochs=self.epochs,
                                 accelerator='cpu'
                                 )


        num_training = int(len(self.dm.train_image_generator)*percentage) or 1
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(self.model,
                                  self.dm,
                                  num_training=num_training,
                                  early_stop_threshold=None)
        lr = lr_finder.suggestion()
        print(f"Found lr: {lr}")
        self.model.hparams["learning_rate"] = lr

        return lr_finder

    def preprocess(self, imgs, need_preprocess=True):
        xs = []
        if need_preprocess:
            for img in imgs:
                x = normalize_img(img,
                                  width=self.width,
                                  height=self.height)
                xs.append(x)
            xs = np.moveaxis(np.array(xs), 3, 1)
        else:
            xs = np.array(imgs)
        xs = torch.tensor(xs)
        xs = xs.to(device_torch)
        return xs

    def forward(self, xs):
        return self.model(xs)

    def postprocess(self, net_out_value):
        net_out_value = [p.cpu().numpy() for p in net_out_value]
        pred_texts = decode_batch(torch.Tensor(net_out_value), self.label_converter)
        pred_texts = [pred_text.upper() for pred_text in pred_texts]
        return pred_texts

    @torch.no_grad()
    def predict(self, xs: List or torch.Tensor, return_acc: bool = False) -> Any:
        net_out_value = self.model(xs)
        net_out_value = [p.cpu().numpy() for p in net_out_value]
        pred_texts = decode_batch(torch.Tensor(net_out_value), self.label_converter)
        pred_texts = [pred_text.upper() for pred_text in pred_texts]
        if return_acc:
            if len(net_out_value):
                net_out_value = np.array(net_out_value)
                net_out_value = net_out_value.reshape((net_out_value.shape[1],
                                                       net_out_value.shape[0],
                                                       net_out_value.shape[2]))
            return pred_texts, net_out_value
        return pred_texts

    def save(self, path: str, verbose: bool = True, weights_only=True) -> None:
        """
        TODO: describe method
        """
        if bool(verbose):
            print("model save to {}".format(path))
        if self.model is None:
            raise ValueError("self.model is not defined")
        if self.trainer is None:
            torch.save({"state_dict": self.model.state_dict()}, path)
        else:
            self.trainer.save_checkpoint(path, weights_only=weights_only)

    def is_loaded(self) -> bool:
        """
        TODO: describe method
        """
        if self.model is None:
            return False
        return True

    def load_model(self, path_to_model, nn_class=NPOcrNet):
        self.path_to_model = path_to_model
        print(f'Here I am {self.path_to_model}')
        self.model = nn_class.load_from_checkpoint(path_to_model,
                                                   map_location=torch.device('cpu'),
                                                   letters=self.letters,
                                                   linear_size=self.linear_size,
                                                   hidden_size=self.hidden_size,
                                                   backbone=self.backbone,
                                                   letters_max=len(self.letters) + 1,
                                                   label_converter=self.label_converter,
                                                   height=self.height,
                                                   width=self.width,
                                                   color_channels=self.color_channels,
                                                   max_text_len=self.max_text_len,
                                                   **{'pytorch_lightning_version': '0.0.0'})
        
        self.model = self.model.to(device_torch)
        self.model.eval()
        return self.model

    def load_meta(self, path_to_model: str = "latest") -> str:
        model_info = {}
        if path_to_model == "latest":
            model_info = modelhub.download_model_by_name(self.model_name)
            # print(f'model_info:{model_info}')
            path_to_model = model_info["path"]
            # print(f"path_to_model: {path_to_model}")
        elif path_to_model.startswith("http"):
            model_info = modelhub.download_model_by_url(path_to_model,
                                                        self.model_name,
                                                        self.model_name)
            path_to_model = model_info["path"]
        elif path_to_model.startswith("modelhub://"):
            path_to_model = path_to_model.split("modelhub://")[1]
            model_info = modelhub.download_model_by_name(path_to_model)
            path_to_model = model_info["path"]
        self.hidden_size = model_info.get("hidden_size", self.hidden_size)
        self.backbone = model_info.get("backbone", self.backbone)
        if type(self.backbone) == str:
            self.backbone = getattr(models, self.backbone)
        self.letters = model_info.get("letters", self.letters)
        self.max_text_len = model_info.get("max_text_len", self.max_text_len)
        self.height = model_info.get("height", self.height)
        self.width = model_info.get("width", self.width)
        self.color_channels = model_info.get("color_channels", self.color_channels)
        self.linear_size = model_info.get("linear_size", self.linear_size)
        return path_to_model

    def load(self, path_to_model: str = "latest", nn_class=NPOcrNet) -> NPOcrNet:
        """
        TODO: describe method
        """
        path_to_model = self.load_meta(path_to_model)
        self.create_model()
        return self.load_model(path_to_model, nn_class=nn_class)

    @torch.no_grad()
    def get_acc(self, predicted: List, decode: List) -> torch.Tensor:
        decode = [pred_text.lower() for pred_text in decode]
        self.init_label_converter()

        logits = torch.tensor(predicted)
        logits = logits.reshape(logits.shape[1],
                                logits.shape[0],
                                logits.shape[2])
        input_len, batch_size, vocab_size = logits.size()
        device = logits.device

        logits = logits.log_softmax(2)

        encoded_texts, text_lens = self.label_converter.encode(decode)
        text_lens = torch.tensor([self.max_text_len for _ in range(batch_size)])
        logits_lens = torch.full(size=(batch_size,), fill_value=input_len, dtype=torch.int32)

        acc = functional.ctc_loss(
            logits,
            encoded_texts,
            logits_lens.to(device),
            text_lens
        )
        return 1 - acc / len(self.letters)
    
    @torch.no_grad()
    def acc_calc(self, dataset, verbose: bool = False, save_test_result = False) -> float:
        acc = 0
        self.model = self.model.to(device_torch)
        self.model.eval()
        for idx in range(len(dataset)):
            img, text = dataset[idx]
            img = img.unsqueeze(0).to(device_torch)
            logits = self.model(img)
            pred_text = decode_prediction(logits.cpu(), self.label_converter)

            if save_test_result:
                img_path = dataset.paths[idx]
                ann_path = img_path.replace("/img/", "/ann/").replace(".png", ".json")
                ann_data = json.load(open(ann_path, 'r'))
                if "moderation" not in ann_data:
                    ann_data["moderation"] = {}

            if pred_text == text:
                acc += 1
                if save_test_result:
                    ann_data["moderation"]["isModerated"]=1
            elif verbose:
                print(f'\n[INFO] {dataset.paths[idx]}\nPredicted: {pred_text.upper()} \t\t\t True: {text.upper()}')
                if save_test_result:
                    ann_data["moderation"]["isModerated"]=0
                    ann_data["moderation"]["predicted"]=pred_text.upper()
            if save_test_result:
                with open(ann_path, "w") as outfile:
                    outfile.write(json.dumps(ann_data, indent=4))
        return acc / len(dataset)

    def val_acc(self, verbose=False) -> float:
        acc = self.acc_calc(self.dm.val_image_generator, verbose=verbose)
        print('Validaton Accuracy: ', acc, "in", len(self.dm.val_image_generator))
        return acc

    def test_acc(self, verbose=True, save_test_result=False) -> float:
        acc = self.acc_calc(self.dm.test_image_generator, verbose=verbose, save_test_result=save_test_result)
        print('Testing Accuracy: ', acc, "in", len(self.dm.test_image_generator))
        return acc

    def train_acc(self, verbose=False) -> float:
        acc = self.acc_calc(self.dm.train_image_generator, verbose=verbose)
        print('Training Accuracy: ', acc, "in", len(self.dm.train_image_generator))
        return acc
#
#
#
#
#
#
#
#
#
#    
#
#
#

class TextDetectorError(Exception):
    ...

class TextDetector(object):
    @classmethod
    def get_classname(cls: object) -> str:
        return cls.__name__

    def __init__(self,
                 presets: Dict = None,
                 default_label: str = "eu_ua_2015",
                 default_lines_count: int = 1,
                 load_models=True,
                 option_detector_width=0,
                 option_detector_height=0,
                 off_number_plate_classification=True) -> None:
        if presets is None:
            presets = {}
        self.presets = presets

        self.detectors_map = {}
        self.detectors = []
        self.detectors_names = []

        self.option_detector_width = option_detector_width
        self.option_detector_height = option_detector_height

        self.default_label = default_label
        self.default_lines_count = default_lines_count
        self.off_number_plate_classification = off_number_plate_classification

        i = 0
        for preset_name in self.presets:
            preset = self.presets[preset_name]
            for region in preset["for_regions"]:
                self.detectors_map[region.replace("-", '_')] = i
            _label = preset_name
            if modelhub.models.get(_label, None) is None:
                raise TextDetectorError("Text detector {} not exists.".format(_label))
            self.detectors_names.append(_label)
            i += 1

        if load_models:
            self.load()

    def load(self):
        """
        TODO: support reloading
        """
        self.detectors = []
        for i, detector_name in enumerate(self.detectors_names):
            model_conf = copy.deepcopy(modelhub.models[detector_name])
            model_conf.update(self.presets[detector_name])
            detector = OCR(model_name=detector_name, letters=model_conf["letters"],
                           linear_size=model_conf["linear_size"], max_text_len=model_conf["max_text_len"],
                           height=model_conf["height"], width=model_conf["width"],
                           color_channels=model_conf["color_channels"],
                           hidden_size=model_conf["hidden_size"], backbone=model_conf["backbone"])
            detector.load(self.presets[detector_name]['model_path'])
            detector.init_label_converter()
            self.detectors.append(detector)

    def define_predict_classes(self,
                               zones: List[np.ndarray],
                               labels: List[int] = None,
                               lines: List[int] = None) -> Tuple:
        if labels is None:
            labels = []
        if lines is None:
            lines = []

        while len(labels) < len(zones):
            labels.append(self.default_label)
        while len(lines) < len(zones):
            lines.append(self.default_lines_count)
        return labels, lines

    def define_order_detector(
            self,
            zones: List[np.ndarray],
            labels: List[int] = None) -> Dict:
        predicted = {}
        i = 0
        # print('======: ', type(zones), zones)
        for zone, label in zip(zones, labels):
            # cv2.imwrite(f'/Users/adilet/nomeroff-net/media/test_zone_====={label}.jpg', zone)
            # print('======: ', type(zone), zone)
            if label not in self.detectors_map.keys():
                warnings.warn(f"Label '{label}' not in {self.detectors_map.keys()}! "
                              f"Label changed on default '{self.default_label}'.")
                label = self.default_label
            detector = self.detectors_map[label]
            if detector not in predicted.keys():
                predicted[detector] = {"zones": [], "order": []}
            predicted[detector]["zones"].append(zone)
            predicted[detector]["order"].append(i)
            i += 1
        return predicted

    def get_avalible_module(self) -> List[str]:
        return self.detectors_names

    def preprocess(self,
                   zones: List[np.ndarray],
                   labels: List[str] = None,
                   lines: List[int] = None):
        labels, lines = self.define_predict_classes(zones, labels, lines)
        predicted = self.define_order_detector(zones, labels)
        # print('zones: ', zones, type(zones))
        for key in predicted.keys():
            if self.off_number_plate_classification:
                zones = convert_cv_zones_rgb_to_bgr(predicted[key]["zones"])
                predicted[key]["xs"] = self.detectors[int(key)].preprocess(zones)
            elif (self.option_detector_width != self.detectors[key].width or
                    self.option_detector_height != self.detectors[key].height):
                zones = [np.moveaxis(cv2.resize(np.moveaxis(zone, 0, 2),
                                    (self.detectors[int(key)].width, self.detectors[int(key)].height)), 2, 0)
                         for zone in zones]
                predicted[key]["xs"] = self.detectors[int(key)].preprocess(zones, need_preprocess=False)
            else:
                predicted[key]["xs"] = self.detectors[int(key)].preprocess(predicted[key]["zones"],
                                                                           need_preprocess=False)
        # print('predicted', predicted)
        # print('images: ', predicted[6]['zones'][0], type(zones[0]), '-----')
        # cv2.imwrite(f'/Users/adilet/nomeroff-net/media/test_zone_=====.jpg', predicted[6]['zones'][0])
        return predicted

    @no_grad()
    def forward(self, predicted):
        for key in predicted.keys():
            xs = predicted[key]["xs"]
            # print('=== XS', xs, type(xs))
            predicted[key]["ys"] = self.detectors[int(key)].forward(xs)
        return predicted

    def postprocess(self, predicted):
        res_all, order_all = [], []
        for key in predicted.keys():
            predicted[key]["ys"] = self.detectors[int(key)].postprocess(predicted[key]["ys"])
            res_all = res_all + predicted[key]["ys"]
            order_all = order_all + predicted[key]["order"]
        return [x for _, x in sorted(zip(order_all, res_all), key=lambda pair: pair[0])]

    def predict(self,
                zones: List[np.ndarray],
                labels: List[str] = None,
                lines: List[int] = None,
                return_acc: bool = False) -> List:

        labels, lines = self.define_predict_classes(zones, labels, lines)
        predicted = self.define_order_detector(zones, labels)

        res_all, scores, order_all = [], [], []
        for key in predicted.keys():
            if return_acc:
                buff_res, acc = self.detectors[int(key)].predict(predicted[key]["zones"], return_acc=return_acc)
                res_all = res_all + buff_res
                scores = scores + list(acc)
            else:
                res_all = res_all + self.detectors[int(key)].predict(predicted[key]["zones"], return_acc=return_acc)
            order_all = order_all + predicted[key]["order"]

        if return_acc:
            return [
                [x for _, x in sorted(zip(order_all, res_all), key=lambda pair: pair[0])],
                [x for _, x in sorted(zip(order_all, scores), key=lambda pair: pair[0])]
            ]
        return [x for _, x in sorted(zip(order_all, res_all), key=lambda pair: pair[0])]

    @staticmethod
    def get_static_module(name: str, **kwargs) -> object:
        model_conf = copy.deepcopy(modelhub.models[name])
        model_conf.update(**kwargs)
        detector = OCR(model_name=name, letters=model_conf["letters"],
                       linear_size=model_conf["linear_size"], max_text_len=model_conf["max_text_len"],
                       height=model_conf["height"], width=model_conf["width"],
                       color_channels=model_conf["color_channels"],
                       hidden_size=model_conf["hidden_size"], backbone=model_conf["backbone"])
        detector.init_label_converter()
        return detector

    def get_acc(self, predicted: List, decode: List, regions: List[str]) -> List[List[float]]:
        acc = []
        for i, region in enumerate(regions):
            if self.detectors_map.get(region, None) is None or len(decode[i]) == 0:
                acc.append([0.])
            else:
                detector = self.detectors[int(self.detectors_map[region])]
                _acc = detector.get_acc([predicted[i]], [decode[i]])
                acc.append([float(_acc)])
        return acc

    def get_module(self, name: str) -> object:
        ind = self.detectors_names.index(name)
        return self.detectors[ind]

#
#
#
#
#
#
#
#
#
#
#
#

class BlockRNN(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, bidirectional, recurrent_nn=nn.LSTM):
        super().__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.bidirectional = bidirectional

        # layers
        self.rnn = recurrent_nn(in_size, hidden_size, bidirectional=bidirectional, batch_first=True)

    def forward(self, batch, add_output=False):
        """
        in array:
            batch - [seq_len , batch_size, in_size]
        out array:
            out - [seq_len , batch_size, out_size]
        """
        outputs, hidden = self.rnn(batch)
        out_size = int(outputs.size(2) / 2)
        if add_output:
            outputs = outputs[:, :, :out_size] + outputs[:, :, out_size:]
        return outputs


if __name__ == "__main__":
    det = OCR()
    det.get_classname = lambda: "Eu"
    det.letters = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I",
                   "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    det.max_text_len = 9
    det.letters_max = len(det.letters)+1
    det.init_label_converter()
    det.load()

    image_path = os.path.join(os.getcwd(), "./data/examples/numberplate_zone_images/JJF509.png")
    img = cv2.imread(image_path)
    xs = det.preprocess([img])
    y = det.predict(xs)
    print("y", y)

    image_path = os.path.join(os.getcwd(), "./data/examples/numberplate_zone_images/RP70012.png")
    img = cv2.imread(image_path)
    xs = det.preprocess([img])
    y = det.predict(xs)
    print("y", y)

