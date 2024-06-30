import os
import time
import ujson
import cv2
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored
from abc import abstractmethod
from typing import Any, Dict, Optional, Union
from collections import Counter
import gevent
import itertools
from gevent import Greenlet
import numpy
from abc import abstractmethod
from PIL import Image
from turbojpeg import TurboJPEG
from turbojpeg import TJPF_RGB

def empty_method(func):
    """
    if in your pipeline you want to off some thing
    """
    func.is_empty = True
    return func

class BaseImageLoader(object):

    @abstractmethod
    def load(self, **kwargs):
        raise NotImplementedError("load not implemented")
    
class DumpyImageLoader(BaseImageLoader):

    def load(self, img):
        return img
    
class OpencvImageLoader(BaseImageLoader):
    def load(self, img_path):
        img = cv2.imread(img_path)
        img = img[..., ::-1]
        return img

class PillowImageLoader(BaseImageLoader):
    def load(self, img_path):
        im = Image.open(img_path)
        img = np.asarray(im)
        return img

class TurboImageLoader(BaseImageLoader):
    def __init__(self, **kwargs):
        self.jpeg = TurboJPEG(**kwargs)

    def load(self, img_path):
        with open(img_path, 'rb') as in_file:
            img = self.jpeg.decode(in_file.read(), TJPF_RGB)
        return img

image_loaders_map = {
    "opencv": OpencvImageLoader,
    "cv2": OpencvImageLoader,
    "pillow": PillowImageLoader,
    "turbo": TurboImageLoader,
}

def chunked_iterable(iterable, size):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk

def unzip(zipped):
    return list(zip(*zipped))

def process_job(func):
    res = func["function"](*func.get("args", []), **func.get("kwargs", {}))
    return res

def promise_all(function_list):
    """
    [
        {
            "function": func,
            "args": args,
            "kwargs": kwargs
        }
    ]
    :return: List response
    """
    jobs = [Greenlet.spawn(process_job, item) for item in function_list]
    gevent.joinall(jobs)
    res = [job.value for job in jobs]
    return res


class Pipeline(
    # AccuracyTestPipeline
               ):
    """
    The Pipeline class is the class from which all pipelines inherit. Refer to this class for methods shared across
    different pipelines.
    Base class implementing pipelined operations. Pipeline workflow is defined as a sequence of the following
    operations:
        Input -> Pre-Processing -> Model Inference -> Post-Processing (task dependent) -> Output
    Pipeline supports running on CPU or GPU through the device argument (see below).
    """

    default_input_names = None

    def __init__(
        self,
        task: str = "",
        image_loader: Optional[Union[str, BaseImageLoader]] = None,
        **kwargs,
    ):
        """
        TODO: write description
        """
        self.task = task
        self.image_loader = self._init_image_loader(image_loader)

        self._preprocess_params, self._forward_params, self._postprocess_params = self.sanitize_parameters(**kwargs)

    @staticmethod
    def _init_image_loader(image_loader):
        """
        TODO: write description
        """
        if image_loader is None:
            image_loader_class = DumpyImageLoader
        elif type(image_loader) == str:
            image_loader_class = image_loaders_map.get(image_loader, None)
            if image_loader is None:
                raise ValueError(f"{image_loader} not in {image_loaders_map.keys()}.")
        elif issubclass(image_loader, BaseImageLoader):
            image_loader_class = image_loader
        else:
            raise TypeError(f"The image_loader type must by in None, BaseImageLoader, str")
        return image_loader_class()

    def sanitize_parameters(self, **pipeline_parameters):
        """
        sanitize_parameters will be called with any excessive named arguments from either `__init__` or `__call__`
        methods. It should return 3 dictionnaries of the resolved parameters used by the various `preprocess`,
        `forward` and `postprocess` methods. Do not fill dictionnaries if the caller didn't specify a kwargs. This
        let's you keep defaults in function signatures, which is more "natural".
        It is not meant to be called directly, it will be automatically called and the final parameters resolved by
        `__init__` and `__call__`
        """
        return pipeline_parameters, pipeline_parameters, pipeline_parameters

    # @abstractmethod
    # @may_by_empty_method
    # def preprocess(self, inputs: Any, **preprocess_parameters: Dict) -> Dict[str, Any]:
    #     """
    #     Preprocess will take the `input_` of a specific pipeline and return a dictionnary of everything necessary for
    #     `_forward` to run properly.
    #     """
    #     raise NotImplementedError("preprocess not implemented")

    # @abstractmethod
    # @may_by_empty_method
    # def forward(self, inputs: Any, **forward_parameters: Dict) -> Dict[str, Any]:
    #     """
    #     _forward will receive the prepared dictionnary from `preprocess` and run it on the model. This method might
    #     involve the GPU or the CPU and should be agnostic to it. Isolating this function is the reason for `preprocess`
    #     and `postprocess` to exist, so that the hot path, this method generally can run as fast as possible.
    #     It is not meant to be called directly, `forward` is preferred. It is basically the same but contains additional
    #     code surrounding `_forward` making sure tensors and models are on the same device, disabling the training part
    #     of the code (leading to faster inference).
    #     """
    #     raise NotImplementedError("forward not implemented")

    # @abstractmethod
    # @may_by_empty_method
    # def postprocess(self, inputs: Any, **postprocess_parameters: Dict) -> Any:
    #     """
    #     Postprocess will receive the raw outputs of the `forward` method, generally tensors, and reformat them into
    #     something more friendly. Generally it will output a list or a dict or results (containing just strings and
    #     numbers).
    #     """
    #     raise NotImplementedError("postprocess not implemented")

    def __call__(self, inputs, batch_size=1, num_workers=1, **kwargs):
        """
        TODO: write description
        """
        kwargs["batch_size"] = batch_size
        kwargs["num_workers"] = num_workers
        preprocess_params, forward_params, postprocess_params = self.sanitize_parameters(**kwargs)

        # Fuse __init__ params and __call__ params without modifying the __init__ ones.
        preprocess_params = {**self._preprocess_params, **preprocess_params}
        forward_params = {**self._forward_params, **forward_params}
        postprocess_params = {**self._postprocess_params, **postprocess_params}

        if num_workers < 0 or num_workers > batch_size:
            raise ValueError("num_workers must by grater 0 and less or equal batch_size")
        outputs = self.run_multi(inputs, batch_size, num_workers,
                                 preprocess_params, forward_params, postprocess_params)
        return outputs

    @staticmethod
    def process_worker(func, inputs, params, num_workers=1):
        """
        TODO: write description
        """
        if num_workers == 1:
            return func(inputs, **params)
        promises_outputs = []
        promise_all_args = []
        for chunk_inputs in chunked_iterable(inputs, num_workers):
            for inp in chunked_iterable(chunk_inputs, 1):
                promise_all_args.append(
                    {
                        "function": func,
                        "args": [inp],
                        "kwargs": params
                    }
                )
            # print(f"RUN promise_all {func} functions {len(promise_all_args)}")
            promise_outputs = promise_all(promise_all_args)
            promises_outputs.append(promise_outputs)

        outputs = []
        for promise_output in promises_outputs:
            for chunk in promise_output:
                for item in chunk:
                    outputs.append(item)
        return outputs

    def run_multi(self, inputs, batch_size, num_workers, preprocess_params, forward_params, postprocess_params):
        """
        TODO: write description
        """
        outputs = []
        # for n, input1 in enumerate(inputs):
            # print('chunk input: ', n, type(input1))
            # if isinstance(input1, numpy.ndarray):
                # cv2.imwrite('./media/inputs_in_base.jpg', input1)
            # print('input{}:'.format(n), input1, type(input1), '--------\n')
            # for i in input1:
                # cv2.imwrite('./media/inputs_in_base.jpg', input1[0])
            #     print('input one: ', type(i), len(input1), '++++++\n')
        # print(batch_size, type(batch_size))
        # print(len(inputs[2]), type(inputs[2]), inputs[2])
        # print('========= ', inputs, '===========')
        for chunk_inputs in chunked_iterable(inputs, batch_size):
            chunk_outputs = self.run_single(chunk_inputs, num_workers,
                                            preprocess_params, forward_params, postprocess_params)
            for output in chunk_outputs:
                outputs.append(output)
        # print('output: ', outputs)

        return outputs

    def run_single(self, inputs, num_workers, preprocess_params, forward_params, postprocess_params):
        """
        TODO: write description
        """
        _inputs = inputs
        if not hasattr(self.preprocess, "is_empty") or not self.preprocess.is_empty:
            _inputs = self.process_worker(self.preprocess, _inputs, preprocess_params, num_workers)
        if not hasattr(self.forward, "is_empty") or not self.forward.is_empty:
            _inputs = self.forward(_inputs, **forward_params)
        if not hasattr(self.postprocess, "is_empty") or not self.postprocess.is_empty:
            _inputs = self.process_worker(self.postprocess, _inputs, postprocess_params, num_workers)
        return _inputs
    