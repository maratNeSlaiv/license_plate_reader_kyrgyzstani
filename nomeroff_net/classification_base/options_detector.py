# Libraries
import copy
import os
import json
import cv2
import torch
import numpy as np
import random
from tqdm import tqdm
from typing import List, Tuple, Generator
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from torch.nn import functional
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from torchvision.models import efficientnet_v2_s
from typing import Any, Optional, Dict
import pytorch_lightning as pl
import sys
from pytorch_lightning import LightningModule

# Dependencies
from ..tools.model_hub import modelhub
from .classification_tools import CLASS_LINES_ALL, CLASS_REGION_ALL, CLASS_STATE_ALL, device_torch
from ..tools.dataset import Dataset
from ..tools.image_processing import normalize_img
from ..tools.augmentation import aug


class NPOptionsNetError(Exception):
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
#
#

class ClassificationNet(LightningModule):
    """
    Base Classification Model

    Examples:
        classification_net = ClassificationNet()
        device = get_device_torch()
        net = classification_net.to(device)
        xs = torch.rand((1, 64, 295)).to(device)
        y = classification_net(xs)
        print(y)
    """
    def __init__(self):
        """

        """
        super(ClassificationNet, self).__init__()

    def forward(self, *args, **kwargs) -> Any:
        """

        """
        pass

    def training_step(self, batch, batch_idx):
        """

        """
        loss, acc = self.step(batch)
        self.log(f'Batch {batch_idx} train_loss', loss)
        self.log(f'Batch {batch_idx} accuracy', acc)
        return {
            'loss': loss,
            'acc': acc,
        }

    def validation_step(self, batch, batch_idx):
        """

        """
        loss, acc = self.step(batch)
        self.log('val_loss', loss)
        self.log(f'val_accuracy', acc)
        return {
            'val_loss': loss,
            'val_acc': acc,
        }

    def test_step(self, batch, batch_idx):
        """

        """
        loss, acc = self.step(batch)
        self.log('test_loss', loss)
        self.log(f'test_accuracy', acc)
        return {
            'test_loss': loss,
            'test_acc': acc,
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
#
#
#


class DoubleLinear(torch.nn.Module):
    def __init__(self, linear1, linear2):
        super(DoubleLinear, self).__init__()
        self.linear1 = linear1
        self.linear2 = linear2

    def forward(self, input):
        return self.linear1(input), self.linear2(input)


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
#


class NPOptionsNet(ClassificationNet):
    def __init__(self,
                 region_output_size: int,
                 count_line_output_size: int,
                 batch_size: int = 1,
                 learning_rate: float = 0.001,
                 train_regions=True,
                 train_count_lines=True,
                 backbone=None):
        super(NPOptionsNet, self).__init__() 
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self.train_regions = train_regions
        self.train_count_lines = train_count_lines

        if backbone is None:
            backbone = efficientnet_v2_s
        self.model = backbone()

        if 'efficientnet' in str(backbone):
            in_features = self.model.classifier[1].in_features
        else:
            raise NotImplementedError(backbone)

        linear_region = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=in_features,
                      out_features=region_output_size,
                      bias=True)
        )
        if not self.train_regions:
            for name, param in linear_region.named_parameters():
                param.requires_grad = False
        linear_line = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=in_features,
                      out_features=count_line_output_size,
                      bias=True)
        )
        if not self.train_count_lines:
            for name, param in linear_line.named_parameters():
                param.requires_grad = False
        if 'efficientnet' in str(backbone):
            self.model.classifier = DoubleLinear(linear_region, linear_line)
        else:
            raise NotImplementedError(backbone)

    def training_step(self, batch, batch_idx):
        loss, acc, acc_reg, acc_line = self.step(batch)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'train_accuracy', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc_reg', acc_reg, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'train_acc_line', acc_line, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        tqdm_dict = {
            'train_loss': loss,
            'acc': acc,
            'acc_reg': acc_reg,
            'acc_line': acc_line,
        }
        return {
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        }

    def validation_step(self, batch, batch_idx):
        loss, acc, acc_reg, acc_line = self.step(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'val_accuracy', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc_reg', acc_reg, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'val_acc_line', acc_line, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        tqdm_dict = {
            'val_loss': loss,
            'acc': acc,
            'acc_reg': acc_reg,
            'acc_line': acc_line,
        }
        return {
            'val_loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        }

    def test_step(self, batch, batch_idx):
        loss, acc, acc_reg, acc_line = self.step(batch)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'test_accuracy', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc_reg', acc_reg, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'test_acc_line', acc_line, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        tqdm_dict = {
            'test_loss': loss,
            'acc': acc,
            'acc_reg': acc_reg,
            'acc_line': acc_line,
        }
        return {
            'test_loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        }

    def forward(self, x):

        x1, x2 = self.model(x)
        x1 = functional.softmax(x1)
        x2 = functional.softmax(x2)

        return x1, x2

    def step(self, batch):
        x, ys = batch

        outputs = self.forward(x)
        label_reg = ys[0]
        label_cnt = ys[1]
        
        loss_reg = functional.cross_entropy(outputs[0], torch.max(label_reg, 1)[1])
        loss_line = functional.cross_entropy(outputs[1], torch.max(label_cnt, 1)[1])
        if self.train_regions and self.train_count_lines:
            loss = (loss_reg + loss_line) / 2
        elif self.train_regions:
            loss = loss_reg
        elif self.train_count_lines:
            loss = loss_line
        else:
            raise NPOptionsNetError("train_regions and train_count_lines can not to be False both!")

        acc_reg = (torch.max(outputs[0], 1)[1] == torch.max(label_reg, 1)[1]).float().sum() / self.batch_size
        acc_line = (torch.max(outputs[1], 1)[1] == torch.max(label_cnt, 1)[1]).float().sum() / self.batch_size
        acc = (acc_reg + acc_line) / 2

        return loss, acc, acc_reg, acc_line

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)

        # # Try this
        # from lion_pytorch import Lion
        # optimizer = Lion(self.parameters(), lr=self.learning_rate)

        # # Old optimizer
        # optimizer = torch.optim.ASGD(self.parameters(),
        #                              lr=self.learning_rate,
        #                              lambd=0.0001,
        #                              alpha=0.75,
        #                              t0=1000000.0,
        #                              weight_decay=0)
        return optimizer

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
#

class ImgGenerator(Dataset):
    def __init__(self,
                 dirpath: str,
                 img_w: int = 295,
                 img_h: int = 64,
                 batch_size: int = 32,
                 labels_counts: List = (14, 4, 2),
                 with_aug: bool = False) -> None:
        self.with_aug = with_aug
        self.cur_index = 0
        self.paths = []
        self.discs = []

        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size

        self.labels_counts = labels_counts
        self.dirpath = dirpath
        self.samples = []
        self.images_path = []
        self.list_transforms = None

        self.prepare_transformers()
        self.load_dataset(with_aug, dirpath)

        self.n = len(self.samples)
        self.indexes = list(range(self.n))
        self.batch_count = int(self.n/batch_size)
        self.rezero()

    def load_dataset(self, with_aug: bool, dirpath: str, cache_postfix: str = "cache_options"):
        img_dirpath = os.path.join(self.dirpath, 'img')
        ann_dirpath = os.path.join(self.dirpath, 'ann')

        if with_aug:
            cache_postfix = f"{cache_postfix}_aug"
        cache_dirpath = os.path.join(dirpath, cache_postfix)
        os.makedirs(cache_dirpath, exist_ok=True)
        self.samples = []
        self.images_path = []
        for file_name in tqdm(os.listdir(img_dirpath)):
            name, ext = os.path.splitext(file_name)
            if ext == '.png':
                img_filepath = os.path.join(img_dirpath, file_name)
                self.images_path.append(img_filepath)
                json_filepath = os.path.join(ann_dirpath, name + '.json')
                x_filepath = self.generate_cache_x_in_path(img_filepath, cache_dirpath)
                if os.path.exists(json_filepath):
                    description = json.load(open(json_filepath, 'r'))
                    self.samples.append([x_filepath, [
                        int(description.get("region_id", -1)),
                        int(description.get("count_lines", -1)),
                        int(description.get("orientation", -1))]])

        self.added_samples_to_round_batch()
        self.n = len(self.samples)
        self.indexes = list(range(self.n))
        self.batch_count = int(self.n / self.batch_size)

    def added_samples_to_round_batch(self):
        while len(self.samples) % self.batch_size != 0 and len(self.samples):
            for sample, images_path in zip(self.samples, self.images_path):
                self.samples.append(sample)
                self.images_path.append(images_path)
                if len(self.samples) % self.batch_size == 0:
                    break

    def __len__(self):
        """
        Denotes the total number of samples
        """
        return self.n

    @staticmethod
    def get_x_from_path(x_path: str) -> torch.Tensor:
        return torch.load(x_path)

    def generate_cache_x_in_path(self, img_path: str, cache_dirpath: str, newsize: Tuple = None) -> str:
        x_path = self.generate_x_path(img_path, cache_dirpath)

        if os.path.exists(x_path):
            return x_path

        if newsize is None:
            newsize = (self.img_w, self.img_h)
        img = Image.open(img_path).convert('RGB')
        img = img.resize(newsize)
        if self.with_aug:
            img = np.array(img)
            imgs = aug([img])
            img = Image.fromarray(imgs[0])
        x = self.transform(img)
        torch.save(x, x_path)
        return x_path

    @staticmethod
    def generate_x_path(img_path: str, cache_dirpath: str):
        filename, file_extension = os.path.splitext(img_path)
        filename = os.path.basename(filename)
        x_path = os.path.join(cache_dirpath, f'{filename}.pt')
        return x_path

    def __getitem__(self, index):
        """
        Generates one sample of data
        """

        x = copy.deepcopy(self.paths[self.indexes[index]])
        y = copy.deepcopy(self.discs[self.indexes[index]])
        x = self.get_x_from_path(x)
        y[0] = torch.from_numpy(y[0])
        y[1] = torch.from_numpy(y[1])
        return x, y

    def prepare_transformers(self):
        self.list_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

    def transform(self, img) -> torch.Tensor:
        x = self.list_transforms(img)
        return x

    def rezero(self) -> None:
        self.cur_index = 0
        random.shuffle(self.indexes)

    def build_data(self) -> None:
        self.paths = []
        self.discs = []
        for i, (img_filepath, disc) in enumerate(self.samples):
            self.paths.append(img_filepath)
            self.discs.append(
                [
                    np.eye(self.labels_counts[0])[disc[0]],
                    np.eye(self.labels_counts[1])[disc[1]]
                ]
            )

    def next_sample(self) -> Tuple:
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
        return self.images_path[self.indexes[self.cur_index]], self.discs[self.indexes[self.cur_index]]

    def run_iteration(self, with_aug=False):
        ys = [[], []]
        xs = []
        paths = []
        for _ in np.arange(self.batch_size):
            x, y = self.next_sample()
            paths.append(x)
            img = cv2.imread(x)
            img = img[:, :, ::-1]
            x = normalize_img(img, with_aug=with_aug, width=self.img_w, height=self.img_h)
            xs.append(x)
            ys[0].append(y[0])
            ys[1].append(y[1])
        ys[0] = np.array(ys[0], dtype=np.float32)
        ys[1] = np.array(ys[1], dtype=np.float32)
        xs = np.moveaxis(np.array(xs), 3, 1)
        return paths, xs, ys

    def generator(self, with_aug: bool = False) -> Generator:
        for _ in np.arange(self.batch_count):
            _, xs, ys = self.run_iteration(with_aug)
            yield xs, ys

    def torch_generator(self, with_aug: bool = False) -> Generator:
        for _ in np.arange(self.batch_count):
            _, xs, ys = self.run_iteration(with_aug)
            xs = torch.from_numpy(ys)
            ys = torch.from_numpy(ys)
            yield xs, ys

    def path_generator(self, with_aug: bool = False) -> Generator:
        for _ in np.arange(self.batch_count):
            paths, xs, ys = self.run_iteration(with_aug)
            yield paths, xs, ys




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

class OptionsNetDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_dir=None,
                 val_dir=None,
                 test_dir=None,
                 class_region=None,
                 class_count_line=None,
                 orientations=None,
                 data_loader=ImgGenerator,
                 width=295,
                 height=64,
                 batch_size=32,
                 num_workers=0):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        if orientations is None:
            orientations = [
                "0°", 
                "90°", 
                "180°", 
                "270°"
            ]
        if class_region is None:
            class_region = []
        if class_count_line is None:
            class_count_line = []

        # init train generator
        self.train = None
        self.train_image_generator = None
        if train_dir is not None:
            self.train_image_generator = data_loader(
                train_dir,
                width,
                height,
                batch_size,
                [len(class_region), len(class_count_line), len(orientations)])

        # init validation generator
        self.val = None
        self.val_image_generator = None
        if val_dir is not None:
            self.val_image_generator = data_loader(
                val_dir,
                width,
                height,
                batch_size,
                [len(class_region), len(class_count_line), len(orientations)])

        # init test generator
        self.test = None
        self.test_image_generator = None
        if test_dir is not None:
            self.test_image_generator = data_loader(
                test_dir,
                width,
                height,
                batch_size,
                [len(class_region), len(class_count_line), len(orientations)])

    def prepare_data(self):
        self.train_image_generator.build_data()
        self.val_image_generator.build_data()
        self.test_image_generator.build_data()

    def setup(self, stage=None):
        self.train_image_generator.rezero()
        self.train = self.train_image_generator

        self.val_image_generator.rezero()
        self.val = self.val_image_generator

        self.test_image_generator.rezero()
        self.test = self.test_image_generator

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

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
#
#
class OptionsDetector(object):
    """
    TODO: describe class
    """

    def __init__(self, options: Dict = None) -> None:
        """
        TODO: describe __init__
        """
        if options is None:
            options = dict()

        # input
        self.height = 50
        self.width = 200
        self.color_channels = 3

        # outputs 1
        self.class_region = options.get("class_region", CLASS_REGION_ALL)

        # outputs 2
        self.count_lines = options.get("count_lines", CLASS_LINES_ALL)

        # model
        self.model = None
        self.trainer = None

        # data module
        self.dm = None

        # train hyperparameters
        self.batch_size = 64
        self.epochs = 100
        self.gpus = 0
        self.train_regions = True
        self.train_count_lines = True

        self.class_region_indexes = None
        self.class_region_indexes_global = None

        self.class_lines_indexes = None
        self.class_lines_indexes_global = None

    @classmethod
    def get_classname(cls: object) -> str:
        return cls.__name__

    @staticmethod
    def get_class_region_all() -> List:
        return CLASS_REGION_ALL

    @staticmethod
    def get_class_count_lines_all() -> List:
        return CLASS_LINES_ALL

    @staticmethod
    def get_class_state_all() -> List:
        return CLASS_STATE_ALL

    def get_class_region_for_report(self) -> List:
        """
        TODO: Get class_region list for classification_report routine
        """
        class_regions = []
        for region in self.class_region:
            region_item = region
            if type(region) == list:
                region_item = ','.join(region_item)
            class_regions.append(region_item)
        return class_regions

    def create_model(self) -> NPOptionsNet:
        """
        TODO: describe method
        """
        if self.model is None:
            self.model = NPOptionsNet(len(self.class_region),
                                      len(self.count_lines),
                                      batch_size=self.batch_size,
                                      train_regions=self.train_regions,
                                      train_count_lines=self.train_count_lines, )
            self.model = self.model.to(device_torch)
        return self.model

    def prepare(self,
                base_dir: str,
                num_workers: int = 0,
                verbose: bool = True) -> None:
        """
        TODO: describe method
        """
        if verbose:
            print("START PREPARING")
        # you mast split your data on 3 directory
        train_dir = os.path.join(base_dir, 'train')
        validation_dir = os.path.join(base_dir, 'val')
        test_dir = os.path.join(base_dir, 'test')

        # compile generators
        self.dm = OptionsNetDataModule(
            train_dir,
            validation_dir,
            test_dir,
            self.class_region,
            self.count_lines,
            width=self.width,
            height=self.height,
            batch_size=self.batch_size,
            num_workers=num_workers)

        if verbose:
            print("DATA PREPARED")

    @staticmethod
    def define_callbacks(log_dir):
        checkpoint_callback = ModelCheckpoint(dirpath=log_dir, monitor='val_loss')
        lr_monitor = LearningRateMonitor(logging_interval='step')
        return [checkpoint_callback, lr_monitor]

    def train(self,
              log_dir=sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/logs/options')))
              ) -> NPOptionsNet:
        """
        TODO: describe method
        """
        self.create_model()
        if self.gpus:
            self.trainer = pl.Trainer(max_epochs=self.epochs,
                                      accelerator='gpu', devices=self.gpus,
                                      callbacks=self.define_callbacks(log_dir))
        else:
            self.trainer = pl.Trainer(max_epochs=self.epochs,
                                      accelerator='cpu',
                                      callbacks=self.define_callbacks(log_dir))
        self.trainer.fit(self.model, self.dm)
        return self.model

    def tune(self, percentage=0.1) -> Dict:
        """
        TODO: describe method
        TODO: add ReduceLROnPlateau callback
        """
        model = self.create_model()
        if self.gpus:
            trainer = pl.Trainer(
                max_epochs=self.epochs,
                accelerator='gpu', devices=self.gpus
            )
        else:
            trainer = pl.Trainer(
                max_epochs=self.epochs,
                accelerator='cpu'
            )
        num_training = int(len(self.dm.train_image_generator) * percentage) or 1
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(model,
                                  self.dm,
                                  num_training=num_training,
                                  early_stop_threshold=None)
        lr = lr_finder.suggestion()
        print(f"Found lr: {lr}")
        model.hparams["learning_rate"] = lr

        return lr_finder

    def test(self) -> List:
        """
        TODO: describe method
        """
        return self.trainer.test()

    def save(self, path: str, verbose: bool = True) -> None:
        """
        TODO: describe method
        """
        if self.model is not None:
            if bool(verbose):
                print("model save to {}".format(path))
            if self.trainer is None:
                torch.save({"state_dict": self.model.state_dict()}, path)
            else:
                self.trainer.save_checkpoint(path, weights_only=True)

    def is_loaded(self) -> bool:
        """
        TODO: describe method
        """
        if self.model is None:
            return False
        return True

    def load_model(self, path_to_model):
        self.model = NPOptionsNet.load_from_checkpoint(path_to_model,
                                                       map_location=torch.device('cpu'),
                                                       region_output_size=len(self.class_region),
                                                       count_line_output_size=len(self.count_lines),
                                                       img_h=self.height,
                                                       img_w=self.width,
                                                       batch_size=self.batch_size,
                                                       train_regions=self.train_regions,
                                                       train_count_lines=self.train_count_lines, )
        self.model = self.model.to(device_torch)
        self.model.eval()
        return self.model

    def get_region_label(self, index: int) -> str:
        """
        TODO: describe method
        """
        return self.class_region[index].replace("-", "_")

    def get_region_labels(self, indexes: List[int]) -> List[str]:
        """
        TODO: describe method
        """
        return [self.class_region[index].replace("-", "_") for index in indexes]

    def custom_regions_id_to_all_regions(self, indexes: List[int]) -> List[int]:
        """
        TODO: describe method
        """
        return [CLASS_REGION_ALL.index(str(self.class_region[index].replace("_", "-"))) for index in indexes]

    @staticmethod
    def get_regions_label_global(indexes: List[int]) -> List[str]:
        """
        TODO: describe method
        """
        return [CLASS_REGION_ALL[index].replace("-", "_") for index in indexes]

    def get_count_lines_label(self, index: int) -> int:
        """
        TODO: describe method
        """
        return int(self.count_lines[index])

    def custom_regions_id_to_all_regions_with_confidences(self,
                                                          indexes: List[int],
                                                          confidences: List) -> Tuple[List[int],
                                                                                      List]:
        """
        TODO: describe method
        """
        global_indexes = self.custom_regions_id_to_all_regions(indexes)
        self.class_region_indexes = [i for i, _ in enumerate(self.class_region)]
        self.class_region_indexes_global = self.custom_regions_id_to_all_regions(
            self.class_region_indexes)
        global_confidences = [[confidence[self.class_region_indexes.index(self.class_region_indexes_global.index(i))]
                               if i in self.class_region_indexes_global
                               else 0
                               for i, _
                               in enumerate(CLASS_REGION_ALL)]
                              for confidence in confidences]
        return global_indexes, global_confidences

    def custom_count_lines_id_to_all_count_lines(self, indexes: List[int]) -> List[int]:
        """
        TODO: describe method
        """
        return [CLASS_LINES_ALL.index(str(self.count_lines[index])) for index in indexes]

    def custom_count_lines_id_to_all_count_lines_with_confidences(self,
                                                                  global_indexes: List[int],
                                                                  confidences: List) -> Tuple[List[int],
                                                                                              List]:
        """
        TODO: describe method
        """
        self.class_lines_indexes = [i for i, _ in enumerate(self.count_lines)]
        self.class_lines_indexes_global = self.custom_count_lines_id_to_all_count_lines(
            self.class_lines_indexes)
        global_confidences = [[confidence[self.class_lines_indexes.index(self.class_lines_indexes_global.index(i))]
                               if i in self.class_lines_indexes_global
                               else 0
                               for i, _
                               in enumerate(CLASS_LINES_ALL)]
                              for confidence in confidences]
        return global_indexes, global_confidences

    @staticmethod
    def get_count_lines_labels_global(indexes: List[int]) -> List[int]:
        """
        TODO: describe method
        """
        return [int(CLASS_LINES_ALL[index]) for index in indexes]

    def get_count_lines_labels(self, indexes: List[int]) -> List[int]:
        """
        TODO: describe method
        """
        return [int(self.count_lines[index]) for index in indexes]

    def load_meta(self, path_to_model: str = "latest", options: Dict = None) -> NPOptionsNet:
        if options is None:
            options = dict()
        self.__dict__.update(options)

        if path_to_model == "latest":
            model_info = modelhub.download_model_by_name("numberplate_options")
            path_to_model = model_info["path"]
            self.class_region = model_info["class_region"]
            self.count_lines = model_info["count_lines"]
            self.height = model_info.get("height", self.height)
            self.width = model_info.get("width", self.width)
        elif path_to_model.startswith("http"):
            model_info = modelhub.download_model_by_url(path_to_model, self.get_classname(), "numberplate_options")
            path_to_model = model_info["path"]
        elif path_to_model.startswith("modelhub://"):
            path_to_model = path_to_model.split("modelhub://")[1]
            model_info = modelhub.download_model_by_name(path_to_model)
            path_to_model = model_info["path"]
            self.class_region = model_info["class_region"]
            self.count_lines = model_info["count_lines"]
            self.height = model_info.get("height", self.height)
            self.width = model_info.get("width", self.width)
        return path_to_model

    def load(self, path_to_model: str = "latest", options: Dict = None) -> NPOptionsNet:
        """
        TODO: describe method
        """
        path_to_model = self.load_meta(path_to_model, options)
        self.create_model()
        return self.load_model(path_to_model)

    def predict(self, imgs: List[np.ndarray], return_acc: bool = False) -> Tuple:
        """
        Predict options(region, count lines) by numberplate images
        """
        region_ids, count_lines, confidences, predicted = self.predict_with_confidence(imgs)
        if return_acc:
            return region_ids, count_lines, predicted
        return region_ids, count_lines

    def _predict(self, xs):
        x = torch.tensor(np.moveaxis(np.array(xs), 3, 1))
        x = x.to(device_torch)
        predicted = [p.cpu().numpy() for p in self.model(x)]
        return predicted

    @staticmethod
    def unzip_predicted(predicted):
        confidences, region_ids, count_lines = [], [], []
        for region, count_line in zip(predicted[0], predicted[1]):
            region_ids.append(int(np.argmax(region)))
            count_lines.append(int(np.argmax(count_line)))
            region = region.tolist()
            count_line = count_line.tolist()
            region_confidence = region[int(np.argmax(region))]
            count_lines_confidence = count_line[int(np.argmax(count_line))]
            confidences.append([region_confidence, count_lines_confidence])
        return confidences, region_ids, count_lines

    def preprocess(self, images):
        x = [normalize_img(img, height=self.height, width=self.width) for img in images]
        x = np.moveaxis(np.array(x), 3, 1)
        return x

    def forward(self, inputs):
        x = torch.tensor(inputs)
        x = x.to(device_torch)
        model_output = self.model(x)
        return model_output

    @torch.no_grad()
    def predict_with_confidence(self, imgs: List[np.ndarray or List]) -> Tuple:
        """
        Predict options(region, count lines) with confidence by numberplate images
        """
        xs = [normalize_img(img, height=self.height, width=self.width) for img in imgs]
        if not bool(xs):
            return [], [], [], []
        predicted = self._predict(xs)

        confidences, region_ids, count_lines = self.unzip_predicted(predicted)
        count_lines = self.custom_count_lines_id_to_all_count_lines(count_lines)
        return region_ids, count_lines, confidences, predicted

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
#
#
