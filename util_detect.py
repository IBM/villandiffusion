import os
from typing import Union, Tuple, List, Dict, Set
import pickle
import pathlib
import glob

from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from matplotlib import pyplot as plt
from safetensors.torch import save_file, safe_open
from diffusers import StableDiffusionPipeline, DiffusionPipeline, DDPMPipeline, DDIMScheduler, DDIMInverseScheduler
import bz2file as bz2

def list_all_images(root: Union[str, os.PathLike, pathlib.PurePath], image_exts: List[str]=['png', 'jpg', 'jpeg', 'webp'], ext_case_sensitive: bool=False):
    img_ls_all: List[str] = glob.glob(f"{str(root)}/**", recursive=True)
    img_ls_filtered: List[str] = []
    # print(f"img_ls_all: {len(img_ls_all)}")
    for img in img_ls_all:
        img_path: str = str(img)
        img_path_ext: str = img_path.split('.')[-1]
        # print(f"img_path: {img_path}, img_path_ext: {img_path_ext}")
        for image_ext in image_exts:
            # print(f"img_path_ext: {img_path_ext}, image_ext: {image_ext}")
            if ext_case_sensitive:
                if img_path_ext == str(image_ext):
                    img_ls_filtered.append(img_path)
            else:
                # print(f"Case In-sensitive")
                if img_path_ext.lower() == str(image_ext).lower():
                    img_ls_filtered.append(img_path)
    return img_ls_filtered

def normalize(x: Union[np.ndarray, torch.Tensor], vmin_in: float=None, vmax_in: float=None, vmin_out: float=0, vmax_out: float=1, eps: float=1e-5) -> Union[np.ndarray, torch.Tensor]:
    if vmax_out == None and vmin_out == None:
        return x

    if isinstance(x, np.ndarray):
        if vmin_in == None:
            min_x = np.min(x)
        else:
            min_x = vmin_in
        if vmax_in == None:
            max_x = np.max(x)
        else:
            max_x = vmax_in
    elif isinstance(x, torch.Tensor):
        if vmin_in == None:
            min_x = torch.min(x)
        else:
            min_x = vmin_in
        if vmax_in == None:
            max_x = torch.max(x)
        else:
            max_x = vmax_in
    else:
        raise TypeError("x must be a torch.Tensor or a np.ndarray")
    if vmax_out == None:
        vmax_out = max_x
    if vmin_out == None:
        vmin_out = min_x
    return ((x - min_x) / (max_x - min_x + eps)) * (vmax_out - vmin_out) + vmin_out

def set_generator(generator: Union[int, torch.Generator]) -> torch.Generator:
    if isinstance(generator, int):
        rng: torch.Generator = torch.Generator()
        rng.manual_seed(generator)
    elif isinstance(generator, torch.Generator):
        rng = generator
    return rng

def mse(x0: torch.Tensor, pred_x0: torch.Tensor) -> torch.Tensor:
    return ((x0 - pred_x0) ** 2).sqrt().mean(list(range(x0.dim()))[1:])

def mse_series(x0: torch.Tensor, pred_orig_images: List[torch.Tensor]):
    # pred_orig_images_stck: torch.Tensor = torch.stack(pred_orig_images, dim=0)
    t: int = len(pred_orig_images)
    n: int = pred_orig_images.shape[1]
    print(f"x0: {x0.shape}, pred_orig_images: {pred_orig_images.shape}")
    x0_stck = x0.repeat(t, *([1] * (len(pred_orig_images.shape) - 1)))
    print(f"t: {t}, n: {n}, x0_stck: {x0_stck.shape}, pred_orig_images: {pred_orig_images.shape}")
    residual: torch.Tensor = ((x0_stck - pred_orig_images) ** 2).sqrt().mean(list(range(x0_stck.dim()))[2:]).transpose(0, 1)
    # print(f"residual: {residual.shape}")
    
    return residual

def mse_traj(pred_orig_images: List[torch.Tensor]):
    # pred_orig_images_stck: torch.Tensor = torch.stack(pred_orig_images, dim=0)
    t: int = len(pred_orig_images)
    n: int = pred_orig_images.shape[1]
    # print(f"x0: {x0.shape}, pred_orig_images: {pred_orig_images.shape}")
    # x0_stck = x0.repeat(t, *([1] * (len(pred_orig_images.shape) - 1)))
    # print(f"t: {t}, n: {n}, x0_stck: {x0_stck.shape}, pred_orig_images: {pred_orig_images.shape}")
    residual: torch.Tensor = ((pred_orig_images - pred_orig_images.roll(shifts=1))[:-1] ** 2).sqrt().mean(list(range(pred_orig_images.dim()))[2:]).transpose(0, 1)
    # print(f"residual: {residual.shape}")
    
    return residual

def center_mse_series(pred_orig_images: List[torch.Tensor]):
    # pred_orig_images_stck: torch.Tensor = torch.stack(pred_orig_images, dim=0)
    t: int = len(pred_orig_images)
    n: int = pred_orig_images.shape[1]
    pred_orig_images_mean: torch.Tensor = pred_orig_images.mean(dim=1, keepdim=True)
    pred_orig_images_mean_stck = pred_orig_images_mean.repeat(1, n, *([1] * (len(pred_orig_images.shape) - 2)))
    # print(f"t: {t}, n: {n}, pred_orig_images_mean: {pred_orig_images_mean.shape}, pred_orig_images_mean_stck: {pred_orig_images_mean_stck.shape}, pred_orig_images: {pred_orig_images.shape}")
    residual: torch.Tensor = ((pred_orig_images_mean_stck - pred_orig_images) ** 2).sqrt().mean(list(range(pred_orig_images_mean_stck.dim()))[2:]).transpose(0, 1)
    # print(f"residual: {residual.shape}")
    
    return residual

class Recorder:
    EMPTY_HANDLER_SKIP: str = "SKIP"
    EMPTY_HANDLER_ERR: str = "ERR"
    EMPTY_HANDLER_DEFAULT: str = "DEFAULT"
    def __init__(self) -> None:
        self.__data__: dict = {}
        
    def __init_data_by_key__(self, key: Union[str, int]) -> None:
        if not key in self.__data__:
            self.__data__[key] = {}
    
    def __handle_values__(self, values, indices: Union[int, float, List[Union[int, float]], torch.Tensor, slice]):
        if isinstance(indices, int) or isinstance(indices, float):
            return [values]
        return values
    
    def __handle_indices__(self, indices: Union[int, float, List[Union[int, float]], torch.Tensor, slice], indice_max_length: int=None):
        if indice_max_length is None:
            indice_max_length = len(self.__data__)
        if isinstance(indices, slice):
            indices = [x for x in range(*indices.indices(indice_max_length))]
        elif not hasattr(indices, '__len__'):
            indices = [indices]
        return indices
    
    def __handle_values_indices__(self, values, indices: Union[int, float, List[Union[int, float]], torch.Tensor, slice], indice_max_length: int=None):
        return self.__handle_values__(values=values, indices=indices), self.__handle_indices__(indices=indices, indice_max_length=indice_max_length)
    
    def __update_by_indices__(self, key: Union[str, int], values, indices: Union[int, float, List[Union[int, float]], torch.Tensor, slice], err_if_replace: bool=False, replace: bool=False):
        if err_if_replace and replace:
            raise ValueError(f"Arguement err_if_replace and replace shouldn't be true at the same time.")
        
        for i, idx in enumerate(indices):
            if idx in self.__data__[key]:
                if replace and not err_if_replace:
                    self.__data__[key][idx] = values[i]
                elif not replace and err_if_replace:
                    raise ValueError(f"Cannot update existing value with key: {key} and indice: {idx}")
            else:
                self.__data__[key][idx] = values[i]
    
    def update_by_key(self, key: Union[str, int], values, indices: Union[int, float, List[Union[int, float]], torch.Tensor, slice], indice_max_length: int=None, err_if_replace: bool=False, replace: bool=False) -> None:
        if err_if_replace and replace:
            raise ValueError(f"Arguement err_if_replace and replace shouldn't be true at the same time.")
        self.__init_data_by_key__(key=key)
        
        values, indices = self.__handle_values_indices__(values=values, indices=indices, indice_max_length=indice_max_length)
        # print(f"Handled values: {values}, Handled indices: {indices}")
            
        if len(indices) != len(values):
            raise ValueError(f"values and indices should have the same length.")
        self.__update_by_indices__(key=key, values=values, indices=indices, err_if_replace=err_if_replace, replace=replace)
        
    def get_by_key(self, key: Union[str, int], indices: Union[int, float, List[Union[int, float]], torch.Tensor, slice], indice_max_length: int=None, empty_handler: str=EMPTY_HANDLER_DEFAULT, default_val=None):
        self.__init_data_by_key__(key=key)
        
        indices = self.__handle_indices__(indices=indices, indice_max_length=indice_max_length)
        ret_ls = []
        for idx in indices:
            if idx in self.__data__[key]:
                # print(f"Get [{key}][{idx}]: {self.__data__[key][idx]}")
                ret_ls.append(self.__data__[key][idx])
            elif empty_handler == Recorder.EMPTY_HANDLER_DEFAULT:
                # print(f"Get [{key}][{idx}]: None")
                ret_ls.append(default_val)
            elif empty_handler == Recorder.EMPTY_HANDLER_ERR:
                raise ValueError(f"Value at [{key}][{idx}] is empty.")
        return ret_ls
    
class TensorDict:
    __EMBED_KEY__ = "EMBED"
    __DATA_KEY__ = "DATA"
    def __init__(self, max_size: int=10000) -> None:
        self.__max_size__: int = max_size
        
        # self.__embed__: torch.nn.Embedding = torch.nn.Embedding(num_embeddings=self.__max_size__, embedding_dim=1)
        self.__data__: Dict[list, any] = {}
        
    def __get_key__(self, key: torch.Tensor) -> int:
        # print(f"key: {hash(tuple(key.reshape(-1).tolist()))}")
        return hash(tuple(key.reshape(-1).tolist()))
    
    def __getitem__(self, key: torch.Tensor):
        return self.__data__[self.__get_key__(key)]
    
    def __setitem__(self, key: torch.Tensor, value: any):
        self.__data__[self.__get_key__(key)] = value
    
    def is_key_exist(self, key: torch.Tensor) -> bool:
        embed_key = self.__get_key__(key)
        return embed_key in self.__data__
    
    def __pack_internal__(self) -> dict:
        # return {TensorDict.__EMBED_KEY__: self.__embed__, TensorDict.__DATA_KEY__: self.__data__}
        return {TensorDict.__DATA_KEY__: self.__data__}
    
    def __unpack_internal__(self, input: dict) -> None:
        # self.__embed__ = input[TensorDict.__EMBED_KEY__]
        self.__data__ = input[TensorDict.__DATA_KEY__]
    
    def save(self, path: Union[str, os.PathLike], file_ext: str='pkl') -> None:
        file_path: str = f"{path}.{file_ext}"
        if file_ext is None or file_ext == "":
            file_path: str = path
        pickle.dump(self.__pack_internal__(), file_path, pickle.HIGHEST_PROTOCOL)
        
    def load(self, path: Union[str, os.PathLike]) -> None:
        with open(path, 'rb') as f:
            self.__unpack_internal__(input=pickle.load(f))
            
class DirectRecorder:
    TOP_DICT_KEY: str = "TOP_DICT"
    TOP_DICT_MAX_SIZE_KEY: str = "TOP_DICT_MAX_SIZE"
    SUB_DICT_MAX_SIZE_KEY: str = "SUB_DICT_MAX_SIZE"
    
    SEQ_KEY: str = 'SEQ'
    RECONST_KEY: str = 'RECONST'
    IMAGE_KEY: str = 'IMAGE'
    NOISE_KEY: str = 'NOISE'
    NOISY_IMAGE_KEY: str = 'NOISY_IMAGE'
    TS_KEY: str = "ts"
    LABEL_KEY: str = "LABEL"
    RESIDUAL_KEY: str = 'RESIDUAL'
    TRAJ_RESIDUAL_KEY: str = 'TRAJ_RESIDUAL'
    def __init__(self, top_dict_max_size: int=10000, sub_dict_max_size: int=10000) -> None:
        self.__top_dict__: TensorDict[torch.Tensor, TensorDict[torch.Tensor, dict]] = TensorDict(max_size=top_dict_max_size)
        self.__top_dict_max_size__: int = top_dict_max_size
        self.__sub_dict_max_size__: int = sub_dict_max_size
        
    def __getitem__(self, key: torch.Tensor) -> TensorDict:
        if self.__top_dict__.is_key_exist(key=key):
            return self.__top_dict__[key]
        else:
            self.__top_dict__[key] = TensorDict(max_size=self.__sub_dict_max_size__)
            return self.__top_dict__[key]
    
    def __init_key__(self, top_key: torch.Tensor, sub_key: torch.Tensor):
        if top_key is None or sub_key is None:
            raise TypeError("")
        if not self.__top_dict__.is_key_exist(key=top_key):
            self.__top_dict__[top_key] = TensorDict(max_size=self.__sub_dict_max_size__)
            
        if not self.__top_dict__[top_key].is_key_exist(key=sub_key):
            self.__top_dict__[top_key][sub_key] = {DirectRecorder.SEQ_KEY: Recorder(), DirectRecorder.RECONST_KEY: None}
    
    def update_seq(self, top_key: torch.Tensor, sub_key: torch.Tensor, values, indices: Union[int, float, List[Union[int, float]], torch.Tensor, slice], indice_max_length: int=None, err_if_replace: bool=False, replace: bool=False):
        self.__init_key__(top_key, sub_key)
        self.__top_dict__[top_key][sub_key][DirectRecorder.SEQ_KEY].update_by_key(key='seq', values=values, indices=indices, indice_max_length=indice_max_length, err_if_replace=err_if_replace, replace=replace)        
    
    def update_seq(self, top_key: torch.Tensor, sub_key: torch.Tensor, values):
        self.__init_key__(top_key, sub_key)
        self.__top_dict__[top_key][sub_key][DirectRecorder.SEQ_KEY].update_by_key(key='seq', values=values, indices=[i for i in range(len(values))], indice_max_length=None, err_if_replace=False, replace=False)
        
    def update_reconst(self, top_key: torch.Tensor, sub_key: torch.Tensor, values):
        self.__init_key__(top_key, sub_key)
        self.__top_dict__[top_key][sub_key][DirectRecorder.RECONST_KEY] = values
        
    def update_noise(self, top_key: torch.Tensor, sub_key: torch.Tensor, values):
        self.__init_key__(top_key, sub_key)
        self.__top_dict__[top_key][sub_key][DirectRecorder.NOISE_KEY] = values
        
    def update_ts(self, top_key: torch.Tensor, sub_key: torch.Tensor, values):
        self.__init_key__(top_key, sub_key)
        self.__top_dict__[top_key][sub_key][DirectRecorder.TS_KEY] = values
        
    def update_label(self, top_key: torch.Tensor, sub_key: torch.Tensor, values):
        self.__init_key__(top_key, sub_key)
        self.__top_dict__[top_key][sub_key][DirectRecorder.LABEL_KEY] = values
        
    def update_image(self, top_key: torch.Tensor, sub_key: torch.Tensor, values):
        self.__init_key__(top_key, sub_key)
        self.__top_dict__[top_key][sub_key][DirectRecorder.IMAGE_KEY] = values
        
    def update_noisy_image(self, top_key: torch.Tensor, sub_key: torch.Tensor, values):
        self.__init_key__(top_key, sub_key)
        self.__top_dict__[top_key][sub_key][DirectRecorder.NOISY_IMAGE_KEY] = values
        
    def update_residual(self, top_key: torch.Tensor, sub_key: torch.Tensor, values):
        self.__init_key__(top_key, sub_key)
        self.__top_dict__[top_key][sub_key][DirectRecorder.RESIDUAL_KEY] = values
        
    def batch_update(self, top_keys: torch.Tensor, sub_keys: torch.Tensor, seq: torch.Tensor, reconst: torch.Tensor, noise: torch.Tensor, residual: torch.Tensor, ts: int, label: str):
        Ts: int = seq.shape[1]
        for i, (top_key, sub_key) in enumerate(zip(top_keys, sub_keys)):
            # print(f"top_key: {top_key.shape}, sub_key: {sub_key.shape}")
            # self.set_seq(top_key=top_key, sub_key=sub_key, values=torch.squeeze(seq[:, i, :, :, :]), indices=[i for i in range(Ts)])
            self.update_seq(top_key=top_key, sub_key=sub_key, values=torch.squeeze(seq[i]))
            self.update_reconst(top_key=top_key, sub_key=sub_key, values=reconst)
            self.update_noise(top_key=top_key, sub_key=sub_key, values=noise)
            self.update_image(top_key=top_key, sub_key=sub_key, values=top_key)
            self.update_noisy_image(top_key=top_key, sub_key=sub_key, values=sub_key)
            self.update_ts(top_key=top_key, sub_key=sub_key, values=ts)
            self.update_label(top_key=top_key, sub_key=sub_key, values=label)
            self.update_residual(top_key=top_key, sub_key=sub_key, values=residual)
            
    def __pack_internal__(self) -> dict:
        return {DirectRecorder.TOP_DICT_KEY: self.__top_dict__, DirectRecorder.TOP_DICT_MAX_SIZE_KEY: self.__top_dict_max_size__, DirectRecorder.SUB_DICT_MAX_SIZE_KEY: self.__sub_dict_max_size__}
    
    def __unpack_internal__(self, input: dict) -> None:
        self.__top_dict__ = input[DirectRecorder.TOP_DICT_KEY] 
        self.__top_dict_max_size__ = input[DirectRecorder.TOP_DICT_MAX_SIZE_KEY]
        self.__sub_dict_max_size__ = input[DirectRecorder.SUB_DICT_MAX_SIZE_KEY] 
    
    def save(self, path: Union[str, os.PathLike], file_ext: str='pkl') -> None:
        file_path: str = f"{path}.{file_ext}"
        if file_ext is None or file_ext == "":
            file_path: str = path
        # with open(file_path, "wb") as f:
        with bz2.BZ2File(file_path, 'w') as f:
            pickle.dump(self.__pack_internal__(), f, pickle.HIGHEST_PROTOCOL)
        # save_file(self.__pack_internal__(), file_path)
        
    def load(self, path: Union[str, os.PathLike]) -> None:
        # with open(path, 'rb') as f:
        with bz2.BZ2File(path, 'rb') as f:
            self.__unpack_internal__(input=pickle.load(f))
        # loaded_data: dict = {}
        # with safe_open(path, framework="pt", device='cpu') as f:
        #     for k in f.keys():
        #         loaded_data[k] = f.get_tensor(k)
        #     self.__unpack_internal__(input=loaded_data)
        
class SafetensorRecorder(DirectRecorder):
    PROC_BEF_SAVE_MODE_STACK: str = "stack"
    PROC_BEF_SAVE_MODE_CAT: str = "cat"
    def __init__(self) -> None:
        self.__data__ = {}
        
    def __pack_internal__(self) -> dict:
        # self.process_before_saving(mode=proc_mode)
        return self.__data__
    
    def __unpack_internal__(self, input: dict) -> None:
        self.__data__: Dict[str, torch.Tensor] = input
        
    def update_seq(self, values: torch.Tensor):
        if SafetensorRecorder.SEQ_KEY in self.__data__:
            self.__data__[SafetensorRecorder.SEQ_KEY].append(values)
        else:
            self.__data__[SafetensorRecorder.SEQ_KEY] = [values]
        
    def update_reconst(self, values: torch.Tensor):
        if SafetensorRecorder.RECONST_KEY in self.__data__:
            self.__data__[SafetensorRecorder.RECONST_KEY].append(values)
        else:
            self.__data__[SafetensorRecorder.RECONST_KEY] = [values]
        
    def update_noise(self, values: torch.Tensor):
        if SafetensorRecorder.NOISE_KEY in self.__data__:
            self.__data__[SafetensorRecorder.NOISE_KEY].append(values)
        else:
            self.__data__[SafetensorRecorder.NOISE_KEY] = [values]
        
    def update_ts(self, values: torch.Tensor):
        if SafetensorRecorder.TS_KEY in self.__data__:
            self.__data__[SafetensorRecorder.TS_KEY].append(values)
        else:
            self.__data__[SafetensorRecorder.TS_KEY] = [values]
        
    def update_label(self, values: torch.Tensor):
        if SafetensorRecorder.LABEL_KEY in self.__data__:
            self.__data__[SafetensorRecorder.LABEL_KEY].append(values)
        else:
            self.__data__[SafetensorRecorder.LABEL_KEY] = [values]
        
    def update_image(self, values: torch.Tensor):
        if SafetensorRecorder.IMAGE_KEY in self.__data__:
            self.__data__[SafetensorRecorder.IMAGE_KEY].append(values)
        else:
            self.__data__[SafetensorRecorder.IMAGE_KEY] = [values]
        
    def update_noisy_image(self, values: torch.Tensor):
        if SafetensorRecorder.NOISY_IMAGE_KEY in self.__data__:
            self.__data__[SafetensorRecorder.NOISY_IMAGE_KEY].append(values)
        else:
            self.__data__[SafetensorRecorder.NOISY_IMAGE_KEY] = [values]
            
    def update_residual(self, values: torch.Tensor):
        if SafetensorRecorder.RESIDUAL_KEY in self.__data__:
            self.__data__[DirectRecorder.RESIDUAL_KEY].append(values)
        else:
            self.__data__[SafetensorRecorder.RESIDUAL_KEY] = [values]
            
    def update_traj_residual(self, values: torch.Tensor):
        if SafetensorRecorder.TRAJ_RESIDUAL_KEY in self.__data__:
            self.__data__[DirectRecorder.TRAJ_RESIDUAL_KEY].append(values)
        else:
            self.__data__[SafetensorRecorder.TRAJ_RESIDUAL_KEY] = [values]
        
    def batch_update(self, images: torch.Tensor, noisy_images: torch.Tensor, reconsts: torch.Tensor, noises: torch.Tensor, timestep: int, label: str, seqs: torch.Tensor=None, residuals: torch.Tensor=None, traj_residuals: torch.Tensor=None):
        n: int = len(images)
        labels: List[str] = [label] * n
        tss: List[int] = [timestep] * n
        if seqs is None:
            seqs = [None] * n
        if residuals is None:
            residuals = [None] * n
        if traj_residuals is None:
            traj_residuals = [None] * n
        for i, (image, noisy_image, noise, reconst, seq, residual, traj_residual, lab, ts) in enumerate(zip(images, noisy_images, noises, reconsts, seqs, residuals, traj_residuals, labels, tss)):
            # print(f"top_key: {top_key.shape}, sub_key: {sub_key.shape}")
            # self.set_seq(top_key=top_key, sub_key=sub_key, values=torch.squeeze(seq[:, i, :, :, :]), indices=[i for i in range(Ts)])
            # print(f"Update: ts: {torch.LongTensor([ts])}, label: {torch.LongTensor([lab])}")
            if seq is not None:
                self.update_seq(values=seq)
            self.update_reconst(values=reconst)
            self.update_noise(values=noise)
            self.update_image(values=image)
            self.update_noisy_image(values=noisy_image)
            if residual is not None:
                self.update_residual(values=residual)
            if traj_residual is not None:
                self.update_traj_residual(values=traj_residual)
            self.update_ts(values=torch.LongTensor([ts]))
            self.update_label(values=torch.LongTensor([lab]))
            
    def process_before_saving(self, mode: str):
        if mode == SafetensorRecorder.PROC_BEF_SAVE_MODE_STACK:
            for key, val in self.__data__.items():
                self.__data__[key] = torch.stack(val, dim=0)
        elif mode == SafetensorRecorder.PROC_BEF_SAVE_MODE_CAT:
            for key, val in self.__data__.items():
                self.__data__[key] = torch.cat(val, dim=0)
        else:
            raise ValueError(f"Arguement mode should be {SafetensorRecorder.PROC_BEF_SAVE_MODE_STACK} or {SafetensorRecorder.PROC_BEF_SAVE_MODE_CAT}")
            
    def save(self, path: Union[str, os.PathLike], file_ext: str='safetensors', proc_mode: str=PROC_BEF_SAVE_MODE_CAT) -> None:
        if SafetensorRecorder.RESIDUAL_KEY not in self.__data__:
            path = f"{path}_woRes"
        if SafetensorRecorder.SEQ_KEY not in self.__data__:
            path = f"{path}_woSeq"
            
        file_path: str = f"{path}.{file_ext}"
        if file_ext is None or file_ext == "":
            file_path: str = path
        # with open(file_path, "wb") as f:
        # with bz2.BZ2File(file_path, 'w') as f:
        #     pickle.dump(self.__pack_internal__(), f, pickle.HIGHEST_PROTOCOL)
        self.process_before_saving(mode=proc_mode)
        save_file(self.__pack_internal__(), file_path)
        
    def load(self, path: Union[str, os.PathLike], enable_update: bool=False) -> 'SafetensorRecorder':
        # with open(path, 'rb') as f:
        # with bz2.BZ2File(path, 'rb') as f:
        #     self.__unpack_internal__(input=pickle.load(f))
        loaded_data: dict = {}
        with safe_open(path, framework="pt", device='cpu') as f:
            for k in f.keys():
                if enable_update:
                    loaded_data[k] = [f.get_tensor(k)]
                else:
                    loaded_data[k] = f.get_tensor(k)
            self.__unpack_internal__(input=loaded_data)
        return self

def name_prefix(model: str):
    name: str = ""
    if model == "google/ddpm-cifar10-32":
        name = "DDPM_CIFAR10_32"
    elif model == "google/ddpm-celebahq-256":
        name = "DDPM_CelebA_HQ_256"
    return f"{name}_"

class ModelDataset():
    MD_DDPM: str = "DDPM"
    MD_DDPM_EMA: str = "DDPM_EMA"
    MD_NCSN: str = "NCSN"
    MD_NCSNPP: str = "NCSNPP"
    MD_LDM: str = "LDM"
    MD_SD: str = "SD"
    
    DS_CIFAR10_32: str = "CIFAR10_32"
    DS_CELEBA_HQ_256: str = "CELEBA_HQ_256"
    DS_BEDROOM_256: str = "BEDROOM_256"
    DS_CHURCH_256: str = "CHURCH_256"
    DS_CAT_256: str = "CAT_256"
    DS_FFHQ_256: str = "FFHQ_256"
    DS_FFHQ_1024: str = "FFHQ_1024"
    DS_IMAGENET_64: str = "IMAGENET_64"
    
    real_images_dir: str = "real_images"
    fake_images_dir: str = "fake_images"
    
    MDID_GOOGLE_DDPM_CIFAR10_32: str = "google/ddpm-cifar10-32"
    MDID_GOOGLE_DDPM_CELEBA_HQ_256: str = "google/ddpm-celebahq-256"
    MDID_GOOGLE_DDPM_BEDROOM_256: str = "google/ddpm-bedroom-256"
    MDID_GOOGLE_DDPM_CHURCH_256: str = "google/ddpm-church-256"
    MDID_GOOGLE_DDPM_CAT_256: str = "google/ddpm-cat-256"
    MDID_GOOGLE_DDPM_EMA_CELEBA_HQ_256: str = "google/ddpm-ema-celebahq-256"
    MDID_GOOGLE_DDPM_EMA_BEDROOM_256: str = "google/ddpm-ema-bedroom-256"
    MDID_GOOGLE_DDPM_EMA_CHURCH_256: str = "google/ddpm-ema-church-256"
    MDID_GOOGLE_DDPM_EMA_CAT_256: str = "google/ddpm-ema-cat-256"
    
    MDID_MY_NCSN_CIFAR10_256: str = "newsyctw/NCSN_CIFAR10_my"
    MDID_GOOGLE_NCSNPP_CELEBA_HQ_256: str = "google/ncsnpp-celebahq-256"
    MDID_GOOGLE_NCSNPP_FFHQ_256: str = "google/ncsnpp-ffhq-256"
    MDID_GOOGLE_NCSNPP_FFHQ_1024: str = "google/ncsnpp-ffhq-1024"
    MDID_GOOGLE_NCSNPP_BEDROOM_256: str = "google/ncsnpp-bedroom-256"
    MDID_GOOGLE_NCSNPP_CHURCH_256: str = "google/ncsnpp-church-256"
    
    MDID_CV_LDM_CELEBA_HQ_256: str = "CompVis/ldm-celebahq-256"
    
    MDID_SD_1v1: str = "CompVis/stable-diffusion-v1-1"
    MDID_SD_1v4: str = "CompVis/stable-diffusion-v1-4"
    MDID_SD_2: str = "stabilityai/stable-diffusion-2"
    MDID_SD_2v1: str = "stabilityai/stable-diffusion-2-1"
    
    MDID_OPENAI_CM_IMAGENET_64: str = "openai/diffusers-cd_imagenet64_lpips"
    
    OUT_DIST_UNSPLASH_FACE: str = "UNSPLASH_FACE"
    OUT_DIST_UNSPLASH_HORSE: str = "UNSPLASH_HORSE"
    OUT_DIST_FFHQ: str = "FFHQ"
    
    def __init__(self):
        self.__model__: str = None
        self.__real_images__: str = None
        self.__out_dist_real_images__: str = None
        self.__fake_images__: str = None
    
    @staticmethod    
    def get_path(*args, **kwagrs):
        return str(os.path.join(*args, **kwagrs))
    
    def __prep_out_dist__(self, name: str) -> str:
        res: str = None
        if name == ModelDataset.OUT_DIST_UNSPLASH_HORSE:
            res = ModelDataset.get_path(ModelDataset.real_images_dir, "out_dist_horse")
        elif name == ModelDataset.OUT_DIST_UNSPLASH_FACE:
            res = ModelDataset.get_path(ModelDataset.real_images_dir, "out_dist_human_face")
        elif name == ModelDataset.OUT_DIST_FFHQ:
            res = ModelDataset.get_path(ModelDataset.real_images_dir, "out_dist_ffhq")
        return res
    
    def __prep_real_fake__(self, real_fodler: str, fake_folder: str):
        return [ModelDataset.get_path(ModelDataset.real_images_dir, real_fodler), ModelDataset.get_path(ModelDataset.fake_images_dir, fake_folder)]
    
    def prep_model_dataset(self, model: str, dataset: str, out_dist: str):
        res: List[str] = None
        if model == ModelDataset.MD_DDPM:
            if dataset == ModelDataset.DS_CIFAR10:
                res = [ModelDataset.MDID_GOOGLE_DDPM_CIFAR10_32] + self.__prep_real_fake__(real_fodler="cifar10", fake_folder="cifar10_ddpm")
            elif dataset == ModelDataset.DS_CELEBA_HQ_256:
                res = [ModelDataset.MDID_GOOGLE_DDPM_CELEBA_HQ_256] + self.__prep_real_fake__(real_fodler="celeba_hq_256", fake_folder="celeba_hq_256_ddpm")
            elif dataset == ModelDataset.DS_BEDROOM_256:
                res = [ModelDataset.MDID_GOOGLE_DDPM_BEDROOM_256] + self.__prep_real_fake__(real_fodler="bedroom_256", fake_folder="bedroom_256_ddpm")
            elif dataset == ModelDataset.DS_CHURCH_256:
                res = [ModelDataset.MDID_GOOGLE_DDPM_CHURCH_256] + self.__prep_real_fake__(real_fodler="church_256", fake_folder="church_256_ddpm")
            else:
                raise ValueError(f"Model, {model}, does not support  Dataset, {dataset}.")
        if model == ModelDataset.MD_DDPM_EMA:
            if dataset == ModelDataset.DS_CELEBA_HQ_256:
                res = [ModelDataset.MDID_GOOGLE_DDPM_EMA_CELEBA_HQ_256, ModelDataset.get_path(ModelDataset.real_images_dir, "celeba_hq_256"), ModelDataset.get_path(ModelDataset.fake_images_dir, "celeba_hq_256_ddpm")]
            else:
                raise ValueError(f"Model, {model}, does not support Dataset, {dataset}.")
        else:
            raise ValueError(f"Model, {model}, is not supported.")
        
        out_dist_path: str = self.__prep_out_dist__(name=out_dist)
        return res + [out_dist_path]
    
    @staticmethod
    def __model_dataset_name_fn__(model: str, dataset: str):
        return f"{dataset}_{model}"
    
    def unroll_gen_dataset_combination(self, combinations: Union[str, List[str], Dict[str, str]]):
        if combinations is None:
            return []
        elif isinstance(combinations, str):
            return [combinations]
        elif isinstance(combinations, list):
            unrolled_combs: List[str] = []
            for comb in combinations:
                unrolled_combs = unrolled_combs + ModelDataset.unroll_dataset_combination(combinations=comb)
            return unrolled_combs
        elif isinstance(combinations, dict):
            unrolled_combs: List[str] = []
            all_models = self.filter_md_by(archs=None)
            for key, val in combinations.items():
                if key in all_models:
                    unrolled_combs.append(ModelDataset.__model_dataset_name_fn__(model=key, dataset=val))
                else:
                    unrolled_combs.append(ModelDataset.__model_dataset_name_fn__(model=val, dataset=key))
            return combinations
        
    @staticmethod
    def make_grid_comb(models: Union[str, List[str]], datasets: Union[str, List[str]]) -> Dict[str, List[str]]:
        if isinstance(models, str):
            models = [models]
        if isinstance(datasets, str):
            datasets = [datasets]
        comb: Dict[str, List[str]] = {}
        for model in models:
            comb[model] = datasets
        return comb
        
    def set_model_dataset(self, model: str, dataset: str, out_dist: str, real: Union[str, List[str]]=None, fake: Union[str, List[str]]=None):
        self.__model__, self.__real_images__, self.__fake_images__, self.__out_dist_real_images__ = self.prep_model_dataset(model=model, dataset=dataset, out_dist=out_dist)
        print(f"Model: {self.__model__}, Real: {self.__real_images__}, Fake: {self.__fake_images__}, Out Dist: {self.__out_dist_real_images__}")
        return self
    
    @property
    def pipeline(self):
        if self.__model__ in self.filter_mdid_by(archs=[ModelDataset.MD_DDPM, ModelDataset.MD_DDPM_EMA]):
            return DDPMPipeline.from_pretrained(self.__model__)
        elif self.__model__ in self.filter_mdid_by(archs=[ModelDataset.MD_SD]):
            return StableDiffusionPipeline.from_pretrained(self.__model__, torch_dtype=torch.float16)
        elif self.__model__ in self.filter_mdid_by(archs=[ModelDataset.MD_LDM]):
            pass
        elif self.__model__ in self.filter_mdid_by(archs=[ModelDataset.MD_NCSN, ModelDataset.MD_NCSNPP]):
            pass
        else:
            raise ValueError(f"Model, {self.__model__}, is not supported.")
    
    @property
    def name_prefix(self):
        key: str = 'MDID_'
        for var in self.STATIC_VARS:
            if var[:len(key)] == key:
                if getattr(self, var) == self.__model__:
                    return f"{var[len(key):]}_"
    
    @property
    def model(self):
        return self.__model__
    
    @property
    def out_dist_real_images(self):
        return self.__out_dist_real_images__
    
    @property
    def fake_images(self):
        return self.__fake_images__
    
    @property
    def real_images(self):
        return self.__real_images__
    
    @property
    def STATIC_VARS(self):
        return [attr for attr in dir(ModelDataset) if not callable(getattr(ModelDataset, attr)) and not attr.startswith("__")]
    
    def __scan_by__(self, var: str, conds: Union[int, List[int], Set[int], str, List[str], Set[str], None]=None):
        if conds is None or (isinstance(conds, list) and len(conds) == 0) or (isinstance(conds, set) and len(conds) == 0):
            return True
        elif isinstance(conds, int) or isinstance(conds, str):
            conds = [conds]
        # else:
        #     raise TypeError(f"Arguement conds is not supported, should be Union[int, List[int], Set[int], str, List[str], Set[str]], not {type(conds)}")
            
        for cond in conds:
            if f'_{cond}' in var:
                return True
        return False
    
    def __scan_ds_by__(self, datasets: Union[int, List[int], Set[int], None]=None, sizes: Union[int, List[int], Set[int], None]=None):
        if isinstance(sizes, int):
            sizes = [sizes]
        res: List[str] = []
        key: str = 'DS_'
        for var in self.STATIC_VARS:
            if var[:len(key)] == key:
                if self.__scan_by__(var=var, conds=datasets) and self.__scan_by__(var=var, conds=sizes):
                    res.append(getattr(self, var))
        return list(set(res))
    
    def __scan_md_by__(self, archs: Union[str, List[str], Set[str], None]=None):
        if isinstance(sizes, int):
            sizes = [sizes]
        res: List[str] = []
        key: str = 'MD_'
        for var in self.STATIC_VARS:
            if var[:len(key)] == key:
                if self.__scan_by__(var=var, conds=archs):
                    res.append(getattr(self, var))
        return list(set(res))
    
    def __scan_mdid_by__(self, archs: Union[str, List[str], Set[str], None]=None, datasets: Union[str, List[str], Set[str], None]=None, sizes: Union[int, List[int], Set[int], None]=None):
        if isinstance(sizes, int):
            sizes = [sizes]
        res: List[str] = []
        key: str = 'MDID_'
        for var in self.STATIC_VARS:
            if var[:len(key)] == key:
                # print(f"MDID: {var}, {self.__scan_by__(var=var, conds=archs)}, {self.__scan_by__(var=var, conds=datasets)}, {self.__scan_by__(var=var, conds=sizes)}")
                if self.__scan_by__(var=var, conds=archs) and self.__scan_by__(var=var, conds=datasets) and self.__scan_by__(var=var, conds=sizes):
                    res.append(getattr(self, var))
        return list(set(res))
    
    @property
    def ALL_DS32(self):
        return self.__scan_ds_by__(sizes=32)
    
    @property
    def ALL_DS64(self):
        return self.__scan_ds_by__(sizes=64)
    
    @property
    def ALL_DS256(self):
        return self.__scan_ds_by__(sizes=256)
    
    @property
    def ALL_DS21024(self):
        return self.__scan_ds_by__(sizes=1024)
    
    @property
    def ALL_DS(self):
        return self.__scan_ds_by__(sizes=None)
    
    @property
    def ALL_MD(self):
        return self.__scan_md_by__(sizes=None)
    
    def filter_ds_by(self, datasets: Union[str, List[str], Set[str], None]=None, sizes: Union[int, List[int], Set[int], None]=None):
        return self.__scan_ds_by__(datasets=datasets, sizes=sizes)
    
    def filter_md_by(self, archs: Union[str, List[str], Set[str], None]=None):
        return self.__scan_md_by__(archs=archs)
    
    def filter_mdid_by(self, archs: Union[str, List[str], Set[str], None]=None, datasets: Union[str, List[str], Set[str], None]=None, sizes: Union[int, List[int], Set[int], None]=None):
        return self.__scan_mdid_by__(archs=archs, datasets=datasets, sizes=sizes)

def nd_cicle(shape: Union[Tuple[int], int], diamiter: int):
    '''
    Input:
    shape    : tuple (height, width)
    diameter : scalar
    
    Output:
    np.array of shape  that says True within a circle with diamiter =  around center 
    '''
    # assert len(shape) == 2
    if isinstance(shape, int):
        shape = (shape, shape)
    center: torch.Tensor = torch.tensor(shape) / 2.0
    
    # Generate Grids
    idx_list: List[torch.Tensor] = []
    for d in shape:
        idx_list.append(torch.arange(d))
    grids: List[torch.Tensor] = torch.meshgrid(idx_list, indexing='ij')
    
    # Compute distance to center per dimension
    grid_residuals: List[torch.Tensor] = []
    for grid, c in zip(grids, center):
        grid_residuals.append((grid - c) ** 2)
        
    mask: torch.Tensor = torch.stack(grid_residuals, dim=0).sum(dim=0) < diamiter ** 2

    return mask.int()

def fft_nd(x: torch.Tensor, dim: Union[Tuple[int], int]=None, nd: int=None):
    if (dim is not None) and (nd is not None):
        raise ValueError(f"Arguements dim and nd can not be used at the same time.")
    if dim is not None:
        x_fft = torch.fft.fftn(x, dim=dim)
    elif nd is not None:
        x_fft = torch.fft.fftn(x, dim=x.shape[-nd:])
    else:
        x_fft = torch.fft.fftn(x)
    
    # print(f"x_fft: {torch.fft.fftshift(x_fft)}, .abs().log(): {torch.fft.fftshift(x_fft).abs().log()}")
    print(f"x_fft: {torch.fft.fftshift(x_fft).isfinite().all()}, .abs().log(): {torch.fft.fftshift(x_fft).abs().log().isfinite().all()}")
    return torch.fft.fftshift(x_fft).abs().log()

def get_freq_circle(shape: Union[Tuple[int], int], in_diamiter: int, out_diamiter: int):
    return nd_cicle(shape=shape, diamiter=out_diamiter) - nd_cicle(shape=shape, diamiter=in_diamiter)

def get_freq_circle_all(shape: Union[Tuple[int], int]):
    n: int = max(shape)
    freq_circles: List[torch.Tensor] = []
    for i in range(n):
        freq_circles.append(get_freq_circle(shape=shape, in_diamiter=i, out_diamiter=i + 1))
        print(f"Any < 0: {(get_freq_circle(shape=shape, in_diamiter=i, out_diamiter=i + 1) < 0).any()}")
    return freq_circles

def filtered_by_freq_all(x: torch, dim: Union[Tuple[int], int]):
    if isinstance(dim, int):
        dim = [dim]
    filter_shape = tuple(x.shape[i] for i in dim)
    all_freq_circles: List[torch.Tensor] = get_freq_circle_all(shape=filter_shape)
    
    print(f"x: {x.shape}")
    def reshape_repeat(circle: torch.Tensor, target_shape: Union[Tuple[int], List[int]], dim: Union[Tuple[int], int]):
        circle_reshape = [1] * len(target_shape)
        circle_repeat = list(target_shape)
        for i, elem in enumerate(dim):
            circle_reshape[elem] = target_shape[elem]
            circle_repeat[elem] = 1
        print(f"dim: {dim}, circle: {circle.shape}, {circle.isfinite().all()}, circle_reshape: {circle_reshape}, circle_repeat: {circle_repeat}")
        return circle.reshape(circle_reshape).repeat(circle_repeat)
    
    return [fft_nd(x=x, dim=dim) * reshape_repeat(circle=circle, target_shape=x.shape, dim=dim) for circle in all_freq_circles]

def filtered_by_freq(x: torch, dim: Union[Tuple[int], int], in_diamiter: int, out_diamiter: int):
    if isinstance(dim, int):
        dim = [dim]
    filter_shape = tuple(x.shape[i] for i in dim)
    freq_circle: torch.Tensor = get_freq_circle(shape=filter_shape, in_diamiter=in_diamiter, out_diamiter=out_diamiter)
    
    print(f"x: {x.shape}")
    def reshape_repeat(circle: torch.Tensor, target_shape: Union[Tuple[int], List[int]], dim: Union[Tuple[int], int]):
        circle_reshape = [1] * len(target_shape)
        circle_repeat = list(target_shape)
        for i, elem in enumerate(dim):
            circle_reshape[elem] = target_shape[elem]
            circle_repeat[elem] = 1
        print(f"dim: {dim}, circle: {circle.shape}, {circle.isfinite().all()}, circle_reshape: {circle_reshape}, circle_repeat: {circle_repeat}")
        return circle.reshape(circle_reshape).repeat(circle_repeat)
    
    return fft_nd(x=x, dim=dim) * reshape_repeat(circle=freq_circle, target_shape=x.shape, dim=dim)

def freq_magnitude(filtered_freqs: Union[List[torch.Tensor], torch.Tensor], dim: Union[Tuple[int], int], method: str='mean'):
    METHOD_SUM: str = 'sum'
    METHOD_MEAN: str = 'mean'
    
    if isinstance(dim, int):
        dim = [dim]
    if not isinstance(filtered_freqs, list):
        filtered_freqs = [filtered_freqs]
    if method == METHOD_SUM:
        return torch.stack([filtered_freq.sum(dim) for filtered_freq in filtered_freqs], dim=dim[0])
    elif method == METHOD_MEAN:
        return torch.stack([filtered_freq.mean(dim) for filtered_freq in filtered_freqs], dim=dim[0])
    else:
        raise ValueError(f"Arguement method cannot be {method}, should be {METHOD_SUM} or {METHOD_MEAN}")

def fft_nd_to_1d(x: torch.Tensor, dim: Union[Tuple[int], int]=None, nd: int=None):
    if (dim is not None) and (nd is not None):
        raise ValueError(f"Arguements dim and nd can not be used at the same time.")
    if dim is not None:
        x_fft = freq_magnitude(filtered_freqs=filtered_by_freq_all(x=x, dim=dim), dim=dim, method='mean')
    elif nd is not None:
        dim = list(range(len(x.shape)))[-nd:]
        print(f"dim: {dim}")
        x_fft = freq_magnitude(filtered_freqs=filtered_by_freq_all(x=x, dim=dim), dim=dim, method='mean')
    else:
        x_fft = torch.fft.fftn(x)
    
    # return torch.fft.fftshift(x_fft).log().abs()
    return x_fft

if __name__ == "__main__":
    rec: Recorder = Recorder()
    rec.update_by_key(key='seq', values=[1], indices=[1])
    assert rec.get_by_key(key='seq', indices=[0, 1]) == [None, 1]
    rec.update_by_key(key='seq', values=[3], indices=[3])
    assert rec.get_by_key(key='seq', indices=[0, 1, 2, 3]) == [None, 1, None, 3]
    assert rec.get_by_key(key='seq', indices=[0, 1, 3]) == [None, 1, 3]
    assert rec.get_by_key(key='seq', indices=[0, 1, 2]) == [None, 1, None]
    assert rec.get_by_key(key='seq', indices=[0, 1, 1]) == [None, 1, 1]
    assert rec.get_by_key(key='seq', indices=[0, 1, 0]) == [None, 1, None]
    
    md: ModelDataset = ModelDataset().set_model_dataset(model=ModelDataset.MD_DDPM_EMA, dataset=ModelDataset.DS_CELEBA_HQ_256, out_dist=ModelDataset.OUT_DIST_FFHQ)
    print(md.STATIC_VARS)
    
    print(md.ALL_DS32)
    assert sorted(md.ALL_DS32) == sorted(['CIFAR10_32'])
    
    print(md.ALL_DS256)
    assert sorted(md.ALL_DS256) == sorted(['CHURCH_256', 'CAT_256', 'BEDROOM_256', 'CELEBA_HQ_256', 'FFHQ_256'])
    
    print(md.ALL_DS)
    assert sorted(md.ALL_DS) == sorted(['BEDROOM_256', 'CAT_256', 'FFHQ_1024', 'FFHQ_256', 'CELEBA_HQ_256', 'IMAGENET_64', 'CIFAR10_32', 'CHURCH_256'])
    
    print(md.filter_mdid_by(archs=ModelDataset.MD_DDPM, datasets=ModelDataset.DS_CELEBA_HQ_256, sizes=None))
    assert sorted(md.filter_mdid_by(archs=ModelDataset.MD_DDPM, datasets=ModelDataset.DS_CELEBA_HQ_256, sizes=None)) == sorted(['google/ddpm-celebahq-256', 'google/ddpm-ema-celebahq-256'])
                  
    print(md.filter_mdid_by(archs=ModelDataset.MD_DDPM, datasets=[ModelDataset.DS_CELEBA_HQ_256, ModelDataset.DS_FFHQ_256], sizes=None))
    assert sorted(md.filter_mdid_by(archs=ModelDataset.MD_DDPM, datasets=[ModelDataset.DS_CELEBA_HQ_256, ModelDataset.DS_FFHQ_256], sizes=None)) == sorted(['google/ddpm-celebahq-256', 'google/ddpm-ema-celebahq-256'])
                  
    print(md.filter_mdid_by(archs=ModelDataset.MD_SD, datasets=[ModelDataset.DS_CELEBA_HQ_256, ModelDataset.DS_FFHQ_256], sizes=None))
    assert sorted(md.filter_mdid_by(archs=ModelDataset.MD_SD, datasets=[ModelDataset.DS_CELEBA_HQ_256, ModelDataset.DS_FFHQ_256], sizes=None)) == sorted([])
    
    print(md.filter_mdid_by(archs=ModelDataset.MD_NCSN, datasets=[ModelDataset.DS_CELEBA_HQ_256, ModelDataset.DS_FFHQ_256], sizes=None))
    assert sorted(md.filter_mdid_by(archs=ModelDataset.MD_NCSN, datasets=[ModelDataset.DS_CELEBA_HQ_256, ModelDataset.DS_FFHQ_256], sizes=None)) == sorted(['google/ncsnpp-ffhq-256', 'google/ncsnpp-celebahq-256'])
    
    print(md.filter_mdid_by(archs=ModelDataset.MD_SD, datasets=None, sizes=None))
    assert sorted(md.filter_mdid_by(archs=ModelDataset.MD_SD, datasets=None, sizes=None)) == sorted(['CompVis/stable-diffusion-v1-4', 'stabilityai/stable-diffusion-2', 'CompVis/stable-diffusion-v1-1', 'stabilityai/stable-diffusion-2-1'])
    
    print(md.name_prefix)
    assert md.name_prefix == 'GOOGLE_DDPM_EMA_CELEBA_HQ_256_'
    
    n: int = 10
    size: int = 256
    vmin, vmax = 0, 1
    trans = transforms.Compose([
                        transforms.ToTensor(),
                        # transforms.Resize(size=size),
                        transforms.ConvertImageDtype(torch.float),
                        # transforms.Lambda(lambda x: normalize(vmin_in=0, vmax_in=1, vmin_out=, vmax_out=, x=x)),
                        ])
    imgs: List[torch.Tensor] = []
    for i in range(n):
        # image_file: str = f'real_images/celeba_hq_256_jpg/{i}.jpg'
        image_file: str = f'fake_images/celeba_hq_256_ddpm/image{i}.jpg'
        img = trans(Image.open(image_file).convert('RGB'))
        imgs.append(img)
        
    print(f"imgs: {imgs[0].shape}")
    imgs = torch.stack(imgs)
    print(f"imgs: {imgs.shape}")
    fft_1d = fft_nd_to_1d(x=imgs, dim=None, nd=2)
    print(f"fft_1d: {fft_1d.shape}")
    
    # fig, ax = plt.subplots(1,3,figsize=(15,15))
    # ax[0].imshow(fft_1d[0][0])
    # ax[1].imshow(fft_1d[0][1])
    # ax[2].imshow(fft_1d[0][2])
    # fig.save('fourier.jpg')
    
    print(f"fft_1d[0][0]: {fft_1d[0][0].shape}, fft_1d[0][1]: {fft_1d[0][1].shape}, fft_1d[0][2]: {fft_1d[0][2].shape}")
    print(fft_1d[0][2])
    
    for i, img in enumerate(fft_1d):
        plt.plot(img[0], label='R', color='red')
        plt.plot(img[1], label='G', color='green')
        plt.plot(img[2], label='B', color='blue')
        plt.legend()
        plt.grid()
        plt.title(f"CelebA-HQ 256 {i}")
        # plt.savefig(f"fourier_spectral/fourier_spectral_{i}.jpg")
        plt.savefig(f"fourier_spectral_fake/fourier_spectral_fake_{i}.jpg")
        plt.show()
        plt.clf()