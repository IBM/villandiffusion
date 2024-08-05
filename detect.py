# %%
import glob
import os
import pathlib
import argparse
from typing import Union, Tuple, List, Set, Dict

from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from matplotlib import pyplot as plt

from diffusers import StableDiffusionPipeline, DiffusionPipeline, DDPMPipeline, DDIMScheduler, DDIMInverseScheduler

from util_detect import DirectRecorder, SafetensorRecorder, normalize, set_generator, mse, mse_series, mse_traj, ModelDataset, name_prefix, list_all_images
            
class NoisyDataset(Dataset):
    def __init__(self, image: Union[torch.Tensor, Image.Image, str, os.PathLike, pathlib.PurePath], size: Union[int, Tuple[int, int], List[int]], epsilon_scale: float=1.0, num: int=1000, vmin: float=-1.0, vmax: float=1.0, generator: Union[int, torch.Generator]=0):
        if isinstance(image, torch.Tensor):
            self.__image__ = image
        elif isinstance(image, Image.Image):
            self.__image__ = transforms.ToTensor()(image)
        elif isinstance(image, str) or isinstance(image, os.PathLike) or isinstance(image, pathlib.PurePath):
            self.__image__ = transforms.ToTensor()(Image.open(image).convert('RGB'))
        else:
            raise TypeError(f"The arguement image should be torch.Tensor, Image.Image, str, os.PathLike, or pathlib.PurePath, not {type(image)}")
        
        self.__vmin: float = vmin
        self.__vmax: float = vmax
        self.__trans__ = transforms.Compose([
                            transforms.Resize(size=size),
                            transforms.ConvertImageDtype(torch.float),
                            transforms.Lambda(lambda x: normalize(vmin_in=0, vmax_in=1, vmin_out=self.__vmin, vmax_out=self.__vmax, x=x)),
                         ])
        self.__image__: torch.Tensor = self.__trans__(self.__image__)
        
        # if isinstance(size, int):
        #     h, w = size, size
        # elif isinstance(size, list) or isinstance(size, tuple):
        #     h, w = size[1], size[0]
        # else:
        #     raise TypeError(f"Arguement size should be int, tuple, or list, not {type(size)}")
        
        self.__image__ = transforms.Resize(size=size)(self.__image__)
        self.__generator__ = set_generator(generator=generator)
        
        # print(f"self.__image__: {self.__image__.shape}")
        self.__labels__: torch.Tensor = torch.randn(size=(num, *self.__image__.shape), generator=self.__generator__) * epsilon_scale
        self.__images__: torch.Tensor = self.__image__.repeat((num, *([1] * self.__image__.dim())))

    def __len__(self):
        return len(self.__images__)
    
    def to_tensor(self):
        return self.__images__, self.__labels__

    def __getitem__(self, idx):
        return self.__images__[idx], self.__labels__[idx]
    
class NoisyDatasets(Dataset):
    IMAGE_KEY: str = "Image"
    NOISE_KEY: str = "Noise"
    LATENT_KEY: str = "Latent"
    RECONST_KEY: str = "Reconst"
    RESIDUAL_KEY: str = "Residual"
    def __init__(self, images: Union[torch.Tensor, Image.Image, str, os.PathLike, pathlib.PurePath], size: Union[int, Tuple[int, int], List[int]], epsilon_scale: float, num: int=1000, vmin: float=-1.0, vmax: float=1.0, generator: Union[int, torch.Generator]=0):
        self.__vmin__: float = vmin
        self.__vmax__: float = vmax
        self.__trans__ = transforms.Compose([
                            transforms.Resize(size=size),
                            transforms.ConvertImageDtype(torch.float),
                            transforms.Lambda(lambda x: normalize(vmin_in=0, vmax_in=1, vmin_out=self.__vmin__, vmax_out=self.__vmax__, x=x)),
                         ])
        # self.__data__: Dict[int, Dict[str, Dict[int, Union[torch.Tensor, np.ndarray]]]] = {}
        self.__generator__ = set_generator(generator=generator)
        self.__epsilon_scale__ = epsilon_scale
        
        if isinstance(images, torch.Tensor) or isinstance(images, Image.Image) or isinstance(images, str) or isinstance(images, os.PathLike) or isinstance(images, pathlib.PurePath) or isinstance(images, list):
            self.__images__ = self.__load_images__(images=images, num=num, epsilon_scale=epsilon_scale)
        else:
            raise TypeError(f"The arguement image should be torch.Tensor, Image.Image, str, os.PathLike, or pathlib.PurePath, not {type(images)}")
        
        self.__labels__: torch.Tensor = self.__build_noises__(images=self.__images__)
        
        # self.__image__: torch.Tensor = self.__trans__(self.__image__)
        
        # if isinstance(size, int):
        #     h, w = size, size
        # elif isinstance(size, list) or isinstance(size, tuple):
        #     h, w = size[1], size[0]
        # else:
        #     raise TypeError(f"Arguement size should be int, tuple, or list, not {type(size)}")
        
        # self.__image__ = transforms.Resize(size=size)(self.__image__)
        
        
        # print(f"self.__image__: {self.__image__.shape}")
        # self.__labels__: torch.Tensor = torch.randn(size=(num, *self.__image__.shape), generator=self.__generator__) * epsilon_scale
        # self.__images__: torch.Tensor = self.__image__.repeat((num, *([1] * self.__image__.dim())))

    def __process_img__(self, image: Union[torch.Tensor, Image.Image, str, os.PathLike, pathlib.PurePath]):
        if isinstance(image, torch.Tensor):
            return self.__trans__(image)
        elif isinstance(image, Image.Image):
            return self.__trans__(transforms.ToTensor()(image))
        elif isinstance(image, str) or isinstance(image, os.PathLike) or isinstance(image, pathlib.PurePath):
            return self.__trans__(transforms.ToTensor()(Image.open(image)))
        else:
            raise TypeError(f"The arguement image should be torch.Tensor, Image.Image, str, os.PathLike, or pathlib.PurePath, not {type(image)}")
        
    def __init_entry__(self, image: Union[torch.Tensor, Image.Image, str, os.PathLike, pathlib.PurePath], num: int, epsilon_scale: float):
        proc_img = self.__process_img__(image=image)
        noise = torch.randn(size=(num, *proc_img.shape), generator=self.__generator__) * epsilon_scale
        return {NoisyDataset.IMAGE_KEY: proc_img, NoisyDataset.NOISE_KEY: noise, NoisyDataset.LATENT_KEY: {}, NoisyDataset.RECONST_KEY: {}, NoisyDataset.RESIDUAL_KEY: {}}        

    def __load_images__(self, images: Union[List[Union[torch.Tensor, Image.Image, str, os.PathLike, pathlib.PurePath]], torch.Tensor, Image.Image, str, os.PathLike, pathlib.PurePath], num: int, epsilon_scale: float) -> torch.Tensor:
        if isinstance(images, torch.Tensor):
            if len(images.shape) == 3:
                images = images.unsqueeze(0)
        elif isinstance(images, np.ndarray):
            if len(images.shape) == 3:
                images = torch.from_numpy(images).unsqueeze(0)
        elif not isinstance(images, list):
            images = [images]
        
        img_ls = []
        for img in images:
            processed_img: torch.Tensor = self.__process_img__(image=img)
            # processed_img_dim: int = processed_img.dim()
            # processed_imgs: torch.Tensor = processed_img.repeat(num, *([1] * processed_img_dim))
            img_ls.append(processed_img)
        return torch.stack(img_ls)
    
    def __build_noises__(self, images: torch.Tensor) -> torch.Tensor:
        return torch.randn(size=images.shape, generator=self.__generator__) * self.__epsilon_scale__
        
    def __len__(self):
        return len(self.__images__)
    
    def to_tensor(self):
        return self.__images__, self.__labels__

    def __getitem__(self, idx):
        return self.__images__[idx], self.__labels__[idx]

def prep_noisy_dataloader(batch_size: int, image: Union[torch.Tensor, Image.Image, str, os.PathLike, pathlib.PurePath], epsilon_scale: float=1.0, size: int=32, num: int=1000, generator: Union[int, torch.Generator]=0, device: Union[str, torch.device]='cuda') -> DataLoader:
    rng: torch.Generator = set_generator(generator=generator)
            
    if isinstance(size, int):
        size_hw = (size, size)
    ds: NoisyDataset = NoisyDataset(image=image, epsilon_scale=epsilon_scale, size=size_hw, num=num, generator=generator)
    dl: DataLoader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=rng, collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))
    return dl

class RealFakeDataset(Dataset):
    def __init__(self, root: Union[str, os.PathLike, pathlib.PurePath], size: Union[int, Tuple[int, int], List[int]], dir_label_map: Dict[str, int]={'real': 0, 'fake': 1}, vmin: float=-1.0, vmax: float=1.0, generator: Union[int, torch.Generator]=0, image_exts: List[str]=['png, jpg, jpeg, webp'], ext_case_sensitive: bool=False):
        self.__images__: List[str] = []
        self.__labels__: List[int] = []
        if isinstance(root, str) or isinstance(root, os.PathLike) or isinstance(root, pathlib.PurePath):
            for dir, label in dir_label_map.items():
                appended_list: List[str] = list_all_images(os.path.join(root, dir), image_exts=image_exts, ext_case_sensitive=ext_case_sensitive)
                self.__images__ += appended_list
                self.__labels__ += [label] * len(appended_list)
        else:
            raise TypeError(f"The arguement image should be torch.Tensor, Image.Image, str, os.PathLike, or pathlib.PurePath, not {type(image)}")

        self.__vmin: float = vmin
        self.__vmax: float = vmax
        self.__generator__ = set_generator(generator=generator)
        self.__trans__ = transforms.Compose([
                            transforms.Resize(size=size),
                            transforms.ToTensor(),
                            transforms.ConvertImageDtype(torch.float),
                            transforms.Lambda(lambda x: normalize(vmin_in=0, vmax_in=1, vmin_out=self.__vmin, vmax_out=self.__vmax, x=x)),
                         ])

    def __len__(self):
        return len(self.__images__)
    
    def to_tensor(self):
        return self.__images__, self.__labels__

    def __getitem__(self, idx):
        return self.__trans__(self.__images__[idx]), self.__labels__[idx]

def prep_real_fake_dataloader(batch_size: int, root: Union[str, os.PathLike, pathlib.PurePath], size: Union[int, Tuple[int, int], List[int]], dir_label_map: Dict[str, int]={'real': 0, 'fake': 1}, generator: Union[int, torch.Generator]=0, image_exts: List[str]=['png, jpg, jpeg, webp'], ext_case_sensitive: bool=False) -> DataLoader:
    rng: torch.Generator = set_generator(generator=generator)
    
    ds: RealFakeDataset = RealFakeDataset(root=root, size=size, dir_label_map=dir_label_map, image_exts=image_exts, generator=generator, ext_case_sensitive=ext_case_sensitive)
    dl: DataLoader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=rng)
    return dl

def ddpm_pred(pipeline: DiffusionPipeline, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    model_output = pipeline.unet(xt, t).sample
    return model_output

def reconstruct_xt(pipeline: DiffusionPipeline, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    alpha_prod_t = pipeline.scheduler.alphas_cumprod[t]
    beta_prod_t = 1 - alpha_prod_t
    
    pred_original_sample = (xt - beta_prod_t ** (0.5) * ddpm_pred(pipeline, xt, t)) / alpha_prod_t ** (0.5)
    return pred_original_sample

def reconstruct_x0(pipeline: DiffusionPipeline, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor=None, generator: torch.Generator=0) -> torch.Tensor:
    if x0.dim() != 4:
        x0 = x0.unsqueeze(0)
    
    num: int = x0.shape[0]
    if isinstance(t, int):
        t = torch.LongTensor([t] * num)
    alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(pipeline.device)
    alpha_prod_t = (alphas_cumprod[t]).reshape((-1, *([1] * len(x0.shape[1:]))))
    beta_prod_t = 1 - alpha_prod_t
    
    generator = set_generator(generator=generator)
    if noise is None:
        noise = torch.randn(size=x0.shape, generator=generator)
    # print(f"noise: {noise.shape}, beta_prod_t: {beta_prod_t.shape}")
    scaled_noise = noise * (beta_prod_t ** 0.5)
        
    pred_original_sample = reconstruct_xt(pipeline=pipeline, xt=(alpha_prod_t ** 0.5) * x0 + scaled_noise, t=t)
    return pred_original_sample

def reconstruct_epsilon(pipeline: DiffusionPipeline, x0: torch.Tensor, t: Union[int, torch.IntTensor, torch.LongTensor], noise: torch.Tensor=None, generator: torch.Generator=0) -> torch.Tensor:
    if x0.dim() != 4:
        x0 = x0.unsqueeze(0)
    
    num: int = x0.shape[0]
    if isinstance(t, int):
        t = torch.LongTensor([t] * num)
    alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(pipeline.device)
    alpha_prod_t = (alphas_cumprod[t]).reshape((-1, *([1] * len(x0.shape[1:]))))
    beta_prod_t = 1 - alpha_prod_t
    
    generator = set_generator(generator=generator)
    if noise is None:
        noise = torch.randn(size=x0.shape, generator=generator)
    # print(f"noise: {noise.shape}, beta_prod_t: {beta_prod_t.shape}")
    scaled_noise = noise * (beta_prod_t ** 0.5)
        
    pred_epsilon = ddpm_pred(pipeline=pipeline, xt=(alpha_prod_t ** 0.5) * x0 + scaled_noise, t=t)
    return pred_epsilon

def epsilon_distance(epsilon: torch.Tensor, pred_epsilon: torch.Tensor) -> torch.Tensor:
    return ((epsilon ** 2) - (epsilon - pred_epsilon) ** 2).mean(list(range(epsilon.dim()))[1:])

# def reconstruct_x0_n_steps(pipeline: DDPMPipeline, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor=None, generator: torch.Generator=0) -> torch.Tensor:
#     if x0.dim() != 4:
#         x0 = x0.unsqueeze(0)
    
#     num: int = x0.shape[0]
#     if isinstance(t, int):
#         t = torch.LongTensor([t] * num)
#     alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(pipeline.device)
#     alpha_prod_t = (alphas_cumprod[t]).reshape((-1, *([1] * len(x0.shape[1:]))))
#     beta_prod_t = 1 - alpha_prod_t
    
#     generator = set_generator(generator=generator)
#     if noise is None:
#         noise = torch.randn(size=x0.shape, generator=generator)
#     # print(f"noise: {noise.shape}, beta_prod_t: {beta_prod_t.shape}")
#     scaled_noise = noise * (beta_prod_t ** 0.5)
#     xt=(alpha_prod_t ** 0.5) * x0 + scaled_noise
        
#     latents = pipeline.invert(init=xt).latents
#     images = pipeline(init=latents).latents
#     return images

@torch.no_grad()
def compute_flow_timestep(pipeline: DiffusionPipeline, timestep: Union[int, torch.IntTensor, torch.LongTensor], batch_size: int, image: Union[torch.Tensor, Image.Image, str, os.PathLike, pathlib.PurePath], num: int=1000, generator: Union[int, torch.Generator]=0, device: Union[str, torch.device]='cuda'):
    pipeline = pipeline.to(device)
    dl = prep_noisy_dataloader(batch_size=batch_size, image=image, size=pipeline.unet.config.sample_size, num=num, generator=generator, device=device)
    if isinstance(timestep, int):
        timestep: torch.LongTensor = torch.LongTensor([timestep] * batch_size).to(device)
    timestep: torch.LongTensor = timestep.long().to(device)
    timestep_dist: torch.Tensor = torch.tensor([1.0] * len(timestep)).to(device)
        
    alpha_prod_t = pipeline.scheduler.alphas_cumprod.to(device)[timestep.unique()]
    beta_prod_t = 1 - alpha_prod_t
    print(f"Noise Scale(Beta Bar T): {beta_prod_t ** 0.5}, Content Scale(Alpha Bar T): {alpha_prod_t ** 0.5}")
    
    eps_dis_ls: List[torch.Tensor] = []
    for img, noise in dl:
        # print(f"Img: {img.shape}, noise: {noise.shape}")
        t: torch.IntTensor = timestep[torch.multinomial(timestep_dist, len(img), replacement=True)]
        pred_epsilon = reconstruct_epsilon(pipeline=pipeline, x0=img, t=t, noise=noise, generator=generator)
        eps_dis = epsilon_distance(epsilon=noise, pred_epsilon=pred_epsilon)
        # print(f"eps_dis: {eps_dis.shape}")
        eps_dis_ls.append(eps_dis)
        
    eps_dis_ls: torch.Tensor = torch.cat(eps_dis_ls)
    return eps_dis_ls, eps_dis_ls.mean(), eps_dis_ls.var()

def get_xt(x0: torch.Tensor, noise_scale: Union[int, torch.FloatTensor], noise: torch.Tensor):
    if x0.dim() != 4:
        x0 = x0.unsqueeze(0)
    noise_scale = noise_scale.reshape((-1, *([1] * len(x0.shape[1:]))))
    # print(f"x0: {x0.shape}, noise_scale: {noise_scale.shape}, noise: {noise.shape}")
    return x0 + noise_scale * noise

def get_xt_by_t(pipeline, x0: torch.Tensor, t: int, noise: torch.Tensor):
    alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(x0.device)
    alpha_prod_t = (alphas_cumprod[t]).reshape((-1, *([1] * len(x0.shape[1:]))))
    beta_prod_t = 1 - alpha_prod_t
    
    return alpha_prod_t * x0 + beta_prod_t * noise

def reconstruct_x0_n_steps(pipeline: DiffusionPipeline, x0: torch.Tensor, noise_scale: Union[int, torch.FloatTensor], noise: torch.Tensor=None, generator: torch.Generator=0) -> torch.Tensor:
    if x0.dim() != 4:
        x0 = x0.unsqueeze(0)
    
    num: int = x0.shape[0]
    # if isinstance(t, int):
    #     t = torch.LongTensor([t] * num)
    # alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(pipeline.device)
    # alpha_prod_t = (alphas_cumprod[t]).reshape((-1, *([1] * len(x0.shape[1:]))))
    # beta_prod_t = 1 - alpha_prod_t
    
    generator = set_generator(generator=generator)
    if noise is None:
        noise = torch.randn(size=x0.shape, generator=generator)
    # print(f"noise: {noise.shape}, beta_prod_t: {beta_prod_t.shape}")
    # scaled_noise = noise * (beta_prod_t ** 0.5)
    # noise_scale = noise_scale.reshape((-1, *([1] * len(x0.shape[1:]))))
    # xt = x0 + noise_scale * noise
    xt = get_xt(x0=x0, noise_scale=noise_scale, noise=noise)
        
    latents = pipeline.invert(init=xt).latents
    images = pipeline(init=latents).latents
    return images

def reconstruct_x0_direct_n_steps(pipeline: DiffusionPipeline, x0: torch.Tensor, timestep: int, noise: torch.Tensor=None, generator: torch.Generator=0) -> torch.Tensor:
    if x0.dim() != 4:
        x0 = x0.unsqueeze(0)
    
    num: int = x0.shape[0]
    
    generator = set_generator(generator=generator)
    if noise is None:
        noise = torch.randn(size=x0.shape, generator=generator)
    xt = get_xt_by_t(pipeline=pipeline, x0=x0, t=timestep, noise=noise)
    
    print(f"pipeline xt: {xt.shape}")
    pipeline_output = pipeline(init=xt, start_ratio_inference_steps=timestep/1000)
    print(f"pipeline_output.pred_orig_samples: {pipeline_output.pred_orig_samples.shape}")
    return pipeline_output.latents, pipeline_output.pred_orig_samples

@torch.no_grad()
def compute_gaussian_reconstruct(pipeline: DiffusionPipeline, noise_scale: Union[float, torch.FloatTensor, torch.Tensor], batch_size: int, image: Union[torch.Tensor, Image.Image, str, os.PathLike, pathlib.PurePath], num: int=1000, generator: Union[int, torch.Generator]=0, device: Union[str, torch.device]='cuda'):
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)
    
    dl = prep_noisy_dataloader(batch_size=batch_size, image=image, size=pipeline.unet.config.sample_size, num=num, generator=generator, device=device)
    if isinstance(noise_scale, float):
        noise_scale: torch.FloatTensor = torch.FloatTensor([noise_scale] * batch_size).to(device)
    noise_scales_dist: torch.Tensor = torch.tensor([1.0] * len(noise_scale)).to(device)
        
    # alpha_prod_t = pipeline.scheduler.alphas_cumprod.to(device)[timestep.unique()]
    # beta_prod_t = 1 - alpha_prod_t
    # print(f"Noise Scale(Beta Bar T): {beta_prod_t ** 0.5}, Content Scale(Alpha Bar T): {alpha_prod_t ** 0.5}")
    
    x0_dis_ls: List[torch.Tensor] = []
    for img, noise in dl:
        # print(f"Img: {img.shape}, noise: {noise.shape}")
        noise_scale_sample: torch.FloatTensor = noise_scale[torch.multinomial(noise_scales_dist, len(img), replacement=True)]
        pred_x0 = reconstruct_x0_n_steps(pipeline=pipeline, x0=img, noise_scale=noise_scale_sample, noise=noise, generator=generator).to(device)
        x0_dis = mse(x0=img, pred_x0=pred_x0)
        # print(f"x0_dis: {x0_dis.shape}")
        x0_dis_ls.append(x0_dis)
        
    x0_dis_ls: torch.Tensor = torch.cat(x0_dis_ls)
    return x0_dis_ls, x0_dis_ls.mean(), x0_dis_ls.var()

@torch.no_grad()
def compute_direct_reconst(pipeline: DiffusionPipeline, timestep: int, batch_size: int, image: Union[torch.Tensor, Image.Image, str, os.PathLike, pathlib.PurePath], num: int=1000, generator: Union[int, torch.Generator]=0, device: Union[str, torch.device]='cuda', recorder: SafetensorRecorder=None, label: str=None):
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    # pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)
    
    dl = prep_noisy_dataloader(batch_size=batch_size, image=image, size=pipeline.unet.config.sample_size, num=num, generator=generator, device=device)
    
    x0_dis_ls: List[torch.Tensor] = []
    x0_dis_trend_ls: List[torch.Tensor] = []
    # pred_dis_traj_ls: List[torch.Tensor] = []
    for img, noise in dl:
        # print(f"Img: {img.shape}, noise: {noise.shape}")
        # noise_scale_sample: torch.FloatTensor = timestep[torch.multinomial(noise_scales_dist, len(img), replacement=True)]
        pred_x0, pred_orig_images = reconstruct_x0_direct_n_steps(pipeline=pipeline, x0=img, timestep=timestep, noise=noise, generator=generator)
        # pred_x0, pred_orig_images = pred_x0.detach().to(device), pred_orig_images.detach().to(device)
        img, noise, pred_x0, pred_orig_images = img.detach().cpu(), noise.detach().cpu(), pred_x0.detach().cpu(), pred_orig_images.detach().cpu()
        x0_dis = mse(x0=img, pred_x0=pred_x0)
        x0_dis_trend = mse_series(x0=img.detach().cpu(), pred_orig_images=pred_orig_images.detach().cpu())
        pred_dis_traj = mse_traj(pred_orig_images=pred_orig_images.detach().cpu())
        x0_dis_ls.append(x0_dis)
        x0_dis_trend_ls.append(x0_dis_trend)
        # pred_dis_traj_ls.append(pred_dis_traj)
        
        pred_orig_images_trans: torch.Tensor = pred_orig_images.transpose(0, 1)
        noisy_image: torch.Tensor = get_xt_by_t(pipeline=pipeline, x0=img, t=timestep, noise=noise)
        print(f"x0_dis_trend: {x0_dis_trend.shape}, pred_dis_traj: {pred_dis_traj.shape}")
        print(f"img: {img.shape}, x0_dis: {x0_dis.shape}, noise: {noise.shape}, noisy_image: {noisy_image.shape}, pred_orig_images: {pred_orig_images_trans.shape}, pred_x0: {pred_x0.shape}, timestep: {timestep}, label: {label}")
        # recorder.batch_update(images=img.cpu(), noisy_images=get_xt_by_t(pipeline=pipeline, x0=img, t=timestep, noise=noise).cpu(), seqs=x0_dis_trend.cpu(), reconsts=pred_x0.cpu(), noises=noise.cpu(), timestep=timestep, label=label)
        # recorder.batch_update(images=img.cpu(), noisy_images=noisy_image.cpu(), seqs=pred_orig_images_trans.cpu(), reconsts=pred_x0.cpu(), residuals=x0_dis_trend.cpu(), noises=noise.cpu(), timestep=timestep, label=label)
        recorder.batch_update(images=img.cpu(), noisy_images=noisy_image.cpu(), reconsts=pred_x0.cpu(), residuals=x0_dis_trend.cpu(), traj_residuals=pred_dis_traj.cpu(), noises=noise.cpu(), timestep=timestep, label=label)
        
    x0_dis_ls: torch.Tensor = torch.cat(x0_dis_ls)
    x0_dis_trend_ls: torch.Tensor = torch.cat(x0_dis_trend_ls)
    print(f"x0_dis_trend_ls: {x0_dis_trend_ls.shape}")
    return x0_dis_ls, x0_dis_ls.mean(), x0_dis_ls.var(), x0_dis_trend_ls, recorder

def plot_scatter(x_set_list: List[List[torch.Tensor]], y_set_list: List[List[torch.Tensor]], color_set_list: List[str]=['tab:blue', 'tab:orange', 'tab:red'], zorder_set_list: List[str]=[0, 2, 1], label_set_list: List[str]=['Training Real', 'Out Dist Real', 'Fake'], title: str=f"Flow-In Rate at Timestep", fig_name: str='scatter.jpg', xlabel: str='Mean', ylabel: str='Variance', xscale: str='linear', yscale: str='linear'):
    min_len: int = min(len(x_set_list), len(y_set_list), len(color_set_list), len(label_set_list), len(zorder_set_list))
    fig, ax = plt.subplots(figsize=(8, 5))
    for x, y, color, label, zorder in zip(x_set_list[:min_len], y_set_list[:min_len], color_set_list[:min_len], label_set_list[:min_len], zorder_set_list[:min_len]):
        # n = 750
        # x, y = np.random.rand(2, n)
        # scale = 200.0 * np.random.rand(n)
        # if isinstance(x, torch.Tensor):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()
        # if isinstance(y, torch.Tensor):
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu()
        ax.scatter(x, y, c=color, label=label, alpha=0.8, edgecolors='none', zorder=zorder)

    ax.legend()
    ax.grid(True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    plt.savefig(fig_name)
    plt.show()
    
def plot_run(x_set_list: List[List[torch.Tensor]], color_set_list: List[str]=['tab:blue', 'tab:orange', 'tab:red'], label_set_list: List[str]=['Training Real', 'Out Dist Real', 'Fake'], title: str=f"Reconst Loss at Timestep", fig_name: str='line.jpg', is_plot_var: bool=True):
    min_len: int = min(len(x_set_list), len(color_set_list), len(label_set_list))
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for x, color, label in zip(x_set_list[:min_len], color_set_list[:min_len], label_set_list[:min_len]):
        # n = 750
        # x, y = np.random.rand(2, n)
        # scale = 200.0 * np.random.rand(n)
        print(f"x: {type(x)}")
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()
        else:
            x = torch.tensor(x)
        print(f"x: {len(x)}")
        for i, line in enumerate(x):
            if len(line.shape) == 2:
                if is_plot_var:
                    line_mean: torch.Tensor = line.mean(0)
                    line_var: torch.Tensor = line.var(0)
                    if i == 0:
                        print(f"line: {line.shape}, line_mean: {line_mean.shape}, line_var: {line_var.shape}")
                        print(f"Max line_var: {line_var.max()}")
                        ax.plot(line_mean, c=color, label=label, alpha=0.8)
                        # ax.fill_between(line_mean, line_mean + line_var, line_mean - line_var, color=color, alpha=0.2)
                    else:
                        ax.plot(line_mean, c=color, alpha=0.8)
                        # ax.fill_between(line_mean, line_mean + line_var, line_mean - line_var, color=color, alpha=0.2)   
                else:
                    line_mean: torch.Tensor = line.mean(0)                    
                    if i == 0:
                        print(f"line: {line.shape}, line_mean: {line_mean.shape}")
                        ax.plot(line_mean, c=color, label=label, alpha=0.8)
                    else:
                        ax.plot(line_mean, c=color, alpha=0.8)
            elif len(line.shape) == 1:
                if is_plot_var:
                    raise ValueError(f"Should be 3 dimensions")
                else:
                    if i == 0:
                        ax.plot(line, c=color, label=label, alpha=0.8)
                    else:
                        ax.plot(line, c=color, alpha=0.8)
            else:
                raise ValueError(f"Should be 3 or 2 dimensions")
                

    ax.legend()
    ax.grid(True)
    ax.set_title(title)
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Reconstruction Loss')
    plt.savefig(fig_name)
    plt.show()
    
def plot_flow_timesteps(pipeline: DDPMPipeline, real_images, out_dist_real_images, fake_images, sample_num: int = 100, batch_size: int = 1024, timesteps: List[int] = [1, 2, 4, 8, 16, 32, 50, 100, 200, 400, 800]):
    real_images_list: List[str] = list_all_images(root=real_images)
    out_dist_real_images_list: List[str] = list_all_images(root=out_dist_real_images)
    fake_images_list: List[str] = list_all_images(root=fake_images)
    
    image_lists = [real_images_list, out_dist_real_images_list, fake_images_list]
    
    for timestep in timesteps:
        eps_dis1_set_list = []
        eps_dis2_set_list = []
        for image_list in image_lists:
            eps_dis1_set = []
            eps_dis2_set = []
            print(f"image_list: {len(image_list)}")
            for image in tqdm(image_list[:100]):
                eps_dis_ls, eps_dis1, eps_dis2 = compute_flow_timestep(pipeline=pipeline, timestep=timestep, batch_size=batch_size, image=image, num=sample_num)
                print(f"Moment: {eps_dis1}, {eps_dis2}")
                print(f"eps_dis_ls: {eps_dis_ls.shape}")
                
                eps_dis1_set.append(eps_dis1)
                eps_dis2_set.append(eps_dis2)
            eps_dis1_set_list.append(torch.stack(eps_dis1_set).squeeze())
            eps_dis2_set_list.append(torch.stack(eps_dis2_set).squeeze())
        
        plot_scatter(x_set_list=eps_dis1_set_list, y_set_list=eps_dis2_set_list, fig_name=f'scatter_{timestep}.jpg', title=f"Flow-In Rate at Timestep {timestep}")
        
def plot_gaussian_reconstruction(pipeline: DDPMPipeline, real_images, out_dist_real_images, fake_images, sample_num: int=100, batch_size: int=1024, noise_scale: Union[List[float], float]=0.01, n: int=10):
    real_images_list: List[str] = list_all_images(root=real_images)
    out_dist_real_images_list: List[str] = list_all_images(root=out_dist_real_images)
    fake_images_list: List[str] = list_all_images(root=fake_images)
    
    image_lists = [real_images_list, out_dist_real_images_list, fake_images_list]
    
    
    x0_dis1_set_list = []
    x0_dis2_set_list = []
    for image_list in image_lists:
        x0_dis1_set = []
        x0_dis2_set = []
        print(f"image_list: {len(image_list)}")
        for image in tqdm(image_list[:n]):
            x0_dis_ls, x0_dis1, x0_dis2 = compute_gaussian_reconstruct(pipeline=pipeline, noise_scale=noise_scale, batch_size=batch_size, image=image, num=sample_num)
            print(f"Moment: {x0_dis1}, {x0_dis2}")
            print(f"eps_dis_ls: {x0_dis_ls.shape}")
            
            x0_dis1_set.append(x0_dis1)
            x0_dis2_set.append(x0_dis2)
        x0_dis1_set_list.append(torch.stack(x0_dis1_set).squeeze())
        x0_dis2_set_list.append(torch.stack(x0_dis2_set).squeeze())
    
    ns_str: str = str(noise_scale).replace(",", "").replace("[", "").replace("]", "")
    plot_scatter(x_set_list=x0_dis1_set_list, y_set_list=x0_dis2_set_list, fig_name=f'scatter_shift_ns{ns_str}_n{n}.jpg', title=f"Flow-In Rate at Timestep {noise_scale}")
    
def plot_direct_reconst(pipeline: DDPMPipeline, real_images, out_dist_real_images, fake_images, sample_num: int=100, batch_size: int=1024, timestep: int=0, n: int=10, name_prefix: str=""):
    recorder: SafetensorRecorder = SafetensorRecorder()
    real_images_list: List[str] = list_all_images(root=real_images)
    out_dist_real_images_list: List[str] = list_all_images(root=out_dist_real_images)
    fake_images_list: List[str] = list_all_images(root=fake_images)
    
    image_lists = [real_images_list, out_dist_real_images_list, fake_images_list]
    # image_lists_labels = ['real', 'out_dist', 'fake']
    image_lists_labels = [0, 1, 2]
    
    x0_dis1_set_list = []
    x0_dis2_set_list = []
    x0_dis_trend_set_list = []
    for image_list, image_lists_label in zip(image_lists, image_lists_labels):
        x0_dis1_set = []
        x0_dis2_set = []
        x0_dis_trend_set = []
        print(f"image_list: {len(image_list)}")
        for image in tqdm(image_list[:n]):
            x0_dis_ls, x0_dis1, x0_dis2, x0_dis_trend, recorder = compute_direct_reconst(pipeline=pipeline, timestep=timestep, batch_size=batch_size, image=image, num=sample_num, recorder=recorder, label=image_lists_label)
            print(f"Moment: {x0_dis1}, {x0_dis2}")
            print(f"eps_dis_ls: {x0_dis_ls.shape}")
            print(f"x0_dis_trend: {x0_dis_trend.shape}, x0_dis_trend_set: {len(x0_dis_trend_set)}")
            
            x0_dis1_set.append(x0_dis1)
            x0_dis2_set.append(x0_dis2)
            x0_dis_trend_set.append(x0_dis_trend)
            # for i, a in enumerate(x0_dis_trend_set):
            #     print(f"x0_dis_trend_set[{i}] {a.shape}")
            
        x0_dis1_set_list.append(torch.stack(x0_dis1_set).squeeze())
        x0_dis2_set_list.append(torch.stack(x0_dis2_set).squeeze())
        x0_dis_trend_set_list.append(torch.stack(x0_dis_trend_set).squeeze())
    
    ts_str: str = str(timestep).replace(",", "").replace("[", "").replace("]", "")
    recorder.save(f'{name_prefix}direct_record_ts{ts_str}_n{n}', proc_mode=SafetensorRecorder.PROC_BEF_SAVE_MODE_STACK)
    plot_scatter(x_set_list=x0_dis1_set_list, y_set_list=x0_dis2_set_list, fig_name=f'{name_prefix}scatter_direct_ts{ts_str}_n{n}.jpg', title=f"Direct Reconstruction at Timestep {timestep}")
    plot_run(x_set_list=x0_dis_trend_set_list, fig_name=f'{name_prefix}line_direct_ts{ts_str}_n{n}.jpg', title=f"Direct Reconstruction", is_plot_var=False)

def detect_fourier_lora():
    ts = 1
    n = 10
    eps_dis_num: int = 100
    batch_size: int = 1024
    # timesteps: List[int] = [1, 2, 4, 8, 16]
    model_id = "VillanDiffusion/fres_DDPM-CELEBA-HQ-256_CELEBA-HQ_ep50_sde_c1.0_p0.0_epr0.0_BOX_14-HAT_psi1_lr6e-05_vp1.0_ve1.0"
    lora_id = "VillanDiffusion/fres_DDPM-CELEBA-HQ-256_CELEBA-HQ_ep50_sde_c1.0_p0.0_epr0.0_BOX_14-HAT_psi1_lr6e-05_vp1.0_ve1.0/lora"
    name_prefix = "fourier_lora"
    real_images = "../real_images/celeba_hq_256_jpg_n2048"
    out_dist_real_images = "../fake_images/celeba_hq_256_ddpm"
    fake_images = "../fake_images/celeba_hq_256_ddpm_1"
    pipeline = DDPMPipeline.from_pretrained(model_id, low_cpu_mem_usage=False, device_map=None)
    pipeline.load_lora_weights(lora_id)
    
    plot_direct_reconst(name_prefix=name_prefix, pipeline=pipeline, real_images=real_images, out_dist_real_images=out_dist_real_images, fake_images=fake_images, sample_num=100, batch_size=1024, timestep=ts, n=n)

# %%
if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--n', type=int)
    # parser.add_argument('--ns', type=float, required=False)
    # parser.add_argument('--ts', type=int, required=False)
    
    # args = parser.parse_args()
    # print(f"n: {args.n}, ns: {args.ns}, ts: {args.ts}")
    
    # md: ModelDataset = ModelDataset().set_model_dataset(model=ModelDataset.DDPM_EMA, dataset=ModelDataset.CELEBA_HQ_256, out_dist=ModelDataset.OUT_DIST_FFHQ)
    
    # eps_dis_num: int = 100
    # batch_size: int = 1024
    # # timestep: int = 100
    # timesteps: List[int] = [1, 2, 4, 8, 16, 32, 50, 100, 200, 400, 800]
    # # model_id = "google/ddpm-cifar10-32"
    # # model_id = "google/ddpm-celebahq-256"
    # pipeline = DDPMPipeline.from_pretrained(md.model)
    
    
    # # eps_dis_ls, eps_dis1, eps_dis2 = compute_flow_timestep(pipeline=pipeline, timestep=1, batch_size=512, image=image)
    # # print(f"Moment: {eps_dis1}, {eps_dis2}")
    # # print(f"eps_dis_ls: {eps_dis_ls.shape}")
    # # plt.hist(eps_dis_ls, bins=100)
    # # plt.show()
    
    # # cifar10_real_images: str = "real_images/cifar10"
    # # cifar10_real_images: str = "real_images/celeba_hq_256"
    # # out_dist_real_images: str = "real_images/out_dist"
    # # fake_images: str = "fake_images/cifar10_ddpm"
    # # fake_images: str = "fake_images/celeba_hq_256_ddpm"
    
    # # plot_flow_timesteps(pipeline=pipeline, real_images=cifar10_real_images, out_dist_real_images=out_dist_real_images, fake_images=fake_images, sample_num=100, batch_size=1024, timesteps=timesteps)
    # # plot_gaussian_reconstruction(pipeline=pipeline, real_images=cifar10_real_images, out_dist_real_images=out_dist_real_images, fake_images=fake_images, sample_num=100, batch_size=1024, noise_scale=args.ns, n=args.n)
    # plot_direct_reconst(name_prefix=md.name_prefix, pipeline=pipeline, real_images=md.real_images, out_dist_real_images=md.out_dist_real_images, fake_images=md.fake_images, sample_num=100, batch_size=1024, timestep=args.ts, n=args.n)
    
    
    detect_fourier_lora()

# %%
