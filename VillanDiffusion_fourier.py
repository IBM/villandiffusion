import copy
from dataclasses import dataclass
import argparse
from math import ceil, sqrt
import os
import json
import traceback
from typing import Callable, Dict, List, Tuple, Union
import warnings

import torch
import wandb

from dataset import DatasetLoader, Backdoor, ImagePathDataset
# from model import DiffuserModelSched

MODE_TRAIN: str = 'train'
MODE_RESUME: str = 'resume'
MODE_SAMPLING: str = 'sampling'
MODE_MEASURE: str = 'measure'
MODE_TRAIN_MEASURE: str = 'train+measure'

TASK_GENERATE: str = 'generate'
TASK_UNPOISONED_DENOISE: str = 'unpoisoned_denoise'
TASK_POISONED_DENOISE: str = 'poisoned_denoise'
TASK_UNPOISONED_INPAINT_BOX: str = 'unpoisoned_inpaint_box'
TASK_POISONED_INPAINT_BOX: str = 'poisoned_inpaint_box'
TASK_UNPOISONED_INPAINT_LINE: str = 'unpoisoned_inpaint_line'
TASK_POISONED_INPAINT_LINE: str = 'poisoned_inpaint_line'

DEFAULT_TASK: str = TASK_GENERATE
DEFAULT_PROJECT: str = "Default"
DEFAULT_BATCH: int = 512
DEFAULT_SCHED: str = None
DEFAULT_EVAL_MAX_BATCH: int = 1500
DEFAULT_EPOCH: int = 50
DEFAULT_DDIM_ETA: float = None
DEFAULT_INFER_STEPS: int = 1000
DEFAULT_INFER_START: int = 0
DEFAULT_INPAINT_MUL: int = 1.0
DEFAULT_LEARNING_RATE: float = None
DEFAULT_LEARNING_RATE_32: float = 2e-4
DEFAULT_LEARNING_RATE_256: float = 6e-5
DEFAULT_CLEAN_RATE: float = 1.0
DEFAULT_POISON_RATE: float = 0.007
DEFAULT_EXTEND_POISON_RATE: float = 0.0
DEFAULT_TRIGGER: str = Backdoor.TRIGGER_SM_BOX
DEFAULT_TARGET: str = Backdoor.TARGET_BOX
DEFAULT_DATASET_LOAD_MODE: str = DatasetLoader.MODE_FIXED
DEFAULT_SOLVER_TYPE: str = 'sde'
DEFAULT_PSI: float = 1
DEFAULT_SDE_TYPE: str = "SDE-VP"
DEFAULT_VE_SCALE: float = 1.0
DEFAULT_VP_SCALE: float = 1.0
DEFAULT_GPU = '0'
DEFAULT_CKPT: str = None
DEFAULT_OVERWRITE: bool = False
DEFAULT_POSTFIX: str = ""
DEFAULT_FCLIP: str = 'w'
DEFAULT_SAVE_IMAGE_EPOCHS: int = 5
DEFAULT_SAVE_MODEL_EPOCHS: int = 5
DEFAULT_IS_SAVE_ALL_MODEL_EPOCHS: bool = False
DEFAULT_SAMPLE_EPOCH: int = None
DEFAULT_RESULT: int = '.'
DEFAULT_LORA_RANK: int = None
DEFAULT_LORA_ALPHA: int = None

NOT_MODE_TRAIN_OPTS = ['sample_ep']
NOT_MODE_TRAIN_MEASURE_OPTS = ['sample_ep']
MODE_RESUME_OPTS = ['project', 'task', 'sched', 'ddim_eta', 'infer_steps', 'mode', 'gpu', 'ckpt']
MODE_SAMPLING_OPTS = ['project', 'task', 'sched', 'ddim_eta', 'infer_steps', 'mode', 'eval_max_batch', 'gpu', 'fclip', 'ckpt', 'sample_ep', 'infer_start', 'inpaint_mul']
MODE_MEASURE_OPTS = ['project', 'task', 'sched', 'ddim_eta', 'infer_steps', 'mode', 'eval_max_batch', 'gpu', 'fclip', 'ckpt', 'sample_ep', 'infer_start', 'inpaint_mul']
# IGNORE_ARGS = ['overwrite']
IGNORE_ARGS = ['overwrite', 'is_save_all_model_epochs', 'R_trigger_only']

def parse_args():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--project', '-pj', required=False, type=str, help='Project name')
    parser.add_argument('--mode', '-m', required=True, type=str, help='Train or test the model', choices=[MODE_TRAIN, MODE_RESUME, MODE_SAMPLING, MODE_MEASURE, MODE_TRAIN_MEASURE])
    parser.add_argument('--task', '-t', required=False, type=str, help='Type of task for performance measurement', choices=[TASK_GENERATE, TASK_UNPOISONED_DENOISE, TASK_POISONED_DENOISE, TASK_UNPOISONED_INPAINT_BOX, TASK_POISONED_INPAINT_BOX, TASK_UNPOISONED_INPAINT_LINE, TASK_POISONED_INPAINT_LINE])
    parser.add_argument('--dataset', '-ds', type=str, help='Training dataset', choices=[DatasetLoader.MNIST, DatasetLoader.CIFAR10, DatasetLoader.CELEBA, DatasetLoader.CELEBA_HQ, DatasetLoader.CELEBA_HQ_LATENT_PR05, DatasetLoader.CELEBA_HQ_LATENT])
    parser.add_argument('--sched', '-sc', type=str, help='Noise scheduler', choices=["DDPM-SCHED", "DDIM-SCHED", "DPM_SOLVER_PP_O1-SCHED", "DPM_SOLVER_O1-SCHED", "DPM_SOLVER_PP_O2-SCHED", "DPM_SOLVER_O2-SCHED", "DPM_SOLVER_PP_O3-SCHED", "DPM_SOLVER_O3-SCHED", "UNIPC-SCHED", "PNDM-SCHED", "DEIS-SCHED", "HEUN-SCHED", "LMSD-SCHED", "SCORE-SDE-VE-SCHED", "EDM-VE-SDE-SCHED", "EDM-VE-ODE-SCHED"])
    parser.add_argument('--ddim_eta', '-det', type=float, help=f'Randomness hyperparameter \eta of DDIM, range: [0, 1], default: {DEFAULT_DDIM_ETA}')
    parser.add_argument('--infer_steps', '-is', type=int, help='Number of inference steps')
    parser.add_argument('--infer_start', '-ist', type=float, help='Inference start timestep')
    parser.add_argument('--inpaint_mul', '-im', type=float, help='Inpainting initial sampler multiplier')
    parser.add_argument('--batch', '-b', type=int, help=f"Batch size, default for train: {DEFAULT_BATCH}")
    parser.add_argument('--eval_max_batch', '-eb', type=int, help=f"Batch size of sampling, default for train: {DEFAULT_EVAL_MAX_BATCH}")
    parser.add_argument('--epoch', '-e', type=int, help=f"Epoch num, default for train: {DEFAULT_EPOCH}")
    parser.add_argument('--learning_rate', '-lr', type=float, help=f"Learning rate, default for 32 * 32 image: {DEFAULT_LEARNING_RATE_32}, default for larger images: {DEFAULT_LEARNING_RATE_256}")
    parser.add_argument('--clean_rate', '-cr', type=float, help=f"Clean rate, default for train: {DEFAULT_CLEAN_RATE}")
    parser.add_argument('--poison_rate', '-pr', type=float, help=f"Poison rate, default for train: {DEFAULT_POISON_RATE}")
    parser.add_argument('--ext_poison_rate', '-epr', type=float, help=f"Poison rate for extend backdoor dataset, default for train: {DEFAULT_EXTEND_POISON_RATE}")
    parser.add_argument('--trigger', '-tr', type=str, help=f"Trigger pattern, default for train: {DEFAULT_TRIGGER}")
    parser.add_argument('--target', '-ta', type=str, help=f"Target pattern, default for train: {DEFAULT_TARGET}")
    parser.add_argument('--dataset_load_mode', '-dlm', type=str, help=f"Mode of loading dataset, default for train: {DEFAULT_DATASET_LOAD_MODE}", choices=[DatasetLoader.MODE_FIXED, DatasetLoader.MODE_FLEX, DatasetLoader.MODE_EXTEND, DatasetLoader.MODE_NONE])
    parser.add_argument('--solver_type', '-solt', type=str, help=f"Target solver type of backdoor training, default for train: {DEFAULT_SOLVER_TYPE}", choices=['sde', 'ode'])
    parser.add_argument('--sde_type', '-sdet', type=str, help=f"Diffusion model type, default for train: {DEFAULT_SDE_TYPE}", choices=["SDE-VP", "SDE-VE", "SDE-LDM"])
    parser.add_argument('--psi', '-ps', type=float, help=f"Backdoor scheduler type, value between [1, 0], default for train: {DEFAULT_PSI}")
    parser.add_argument('--ve_scale', '-ves', type=float, help=f"Variance Explode correction term scaler, default for train: {DEFAULT_VE_SCALE}")
    parser.add_argument('--vp_scale', '-vps', type=float, help=f"Variance Preserve correction term scaler, default for train: {DEFAULT_VP_SCALE}")
    parser.add_argument('--gpu', '-g', type=str, help=f"GPU usage, default for train/resume: {DEFAULT_GPU}")
    parser.add_argument('--ckpt', '-c', type=str, help=f"Load from the checkpoint, default: {DEFAULT_CKPT}")
    parser.add_argument('--overwrite', '-o', action='store_true', help=f"Overwrite the existed training result or not, default for train/resume: {DEFAULT_CKPT}")
    parser.add_argument('--R_trigger_only', '-trigonly', action='store_true', help="Making poisoned image without clean images")
    parser.add_argument('--postfix', '-p', type=str, help=f"Postfix of the name of the result folder, default for train/resume: {DEFAULT_POSTFIX}")
    parser.add_argument('--fclip', '-fc', type=str, help=f"Force to clip in each step or not during sampling/measure, default for train/resume: {DEFAULT_FCLIP}", choices=['w', 'o'])
    parser.add_argument('--save_image_epochs', '-sie', type=int, help=f"Save sampled image per epochs, default: {DEFAULT_SAVE_IMAGE_EPOCHS}")
    parser.add_argument('--save_model_epochs', '-sme', type=int, help=f"Save model per epochs, default: {DEFAULT_SAVE_MODEL_EPOCHS}")
    parser.add_argument('--is_save_all_model_epochs', '-isame', action='store_true', help=f"")
    parser.add_argument('--sample_ep', '-se', type=int, help=f"Select i-th epoch to sample/measure, if no specify, use the lastest saved model, default: {DEFAULT_SAMPLE_EPOCH}")
    parser.add_argument('--result', '-res', type=str, help=f"Output file path, default: {DEFAULT_RESULT}")
    parser.add_argument('--lora_rank', '-rnk', type=int, help=f"LoRA rank, None for disable LoRA, default for train: {DEFAULT_LORA_RANK}")
    parser.add_argument('--lora_alpha', '-alp', type=int, help=f"LoRA alpha, None for disable LoRA, default for train: {DEFAULT_LORA_ALPHA}")
    

    args = parser.parse_args()
    
    return args

# args = parse_args()

@dataclass
class TrainingConfig:
    # mode = DEFAULT_MODE
    # dataset = args.dataset
    task: str = TASK_GENERATE
    project: str = DEFAULT_PROJECT
    sched: str = DEFAULT_SCHED
    batch: int = DEFAULT_BATCH
    epoch: int = DEFAULT_EPOCH
    ddim_eta: float = DEFAULT_DDIM_ETA
    infer_steps: int = DEFAULT_INFER_STEPS
    infer_start: int = DEFAULT_INFER_START
    inpaint_mul: float = DEFAULT_INPAINT_MUL
    eval_max_batch: int = DEFAULT_EVAL_MAX_BATCH
    learning_rate: float = DEFAULT_LEARNING_RATE
    clean_rate: float = DEFAULT_CLEAN_RATE
    poison_rate: float = DEFAULT_POISON_RATE
    ext_poison_rate: float = DEFAULT_EXTEND_POISON_RATE
    trigger: str = DEFAULT_TRIGGER
    target: str = DEFAULT_TARGET
    dataset_load_mode: str = DEFAULT_DATASET_LOAD_MODE
    solver_type: str = DEFAULT_SOLVER_TYPE
    sde_type: str = DEFAULT_SDE_TYPE
    psi: float = DEFAULT_PSI
    ve_scale: float = DEFAULT_VE_SCALE
    vp_scale: float = DEFAULT_VP_SCALE
    gpu: str = DEFAULT_GPU
    ckpt: str = DEFAULT_CKPT
    overwrite: bool = DEFAULT_OVERWRITE
    postfix: str  = DEFAULT_POSTFIX
    fclip: str = DEFAULT_FCLIP
    save_image_epochs: int = DEFAULT_SAVE_IMAGE_EPOCHS
    save_model_epochs: int = DEFAULT_SAVE_MODEL_EPOCHS
    is_save_all_model_epochs: bool = DEFAULT_IS_SAVE_ALL_MODEL_EPOCHS
    sample_ep: int = DEFAULT_SAMPLE_EPOCH
    result: str = DEFAULT_RESULT
    lora_rank: int = DEFAULT_LORA_RANK
    lora_alpha: int = DEFAULT_LORA_ALPHA
    
    eval_sample_n: int = 16  # how many images to sample during evaluation
    # measure_sample_n: int = 1024
    measure_sample_n: int = 10000
    # measure_sample_n: int = 16
    measure_inpaint_sample_n: int = 1024
    batch_32: int = 128
    batch_256: int = 64
    gradient_accumulation_steps: int = 1
    # learning_rate_32_fine: float = 2e-4
    # learning_rate_256_fine: float = 6e-5
    learning_rate_32_scratch: float = 2e-4
    learning_rate_256_scratch: float = 2e-5
    lr_warmup_steps: int = 500
    # save_image_epochs: int = 1
    # mixed_precision: str = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    is_mixed_precision: bool = False
    mixed_precision: str = 'no'  # `no` for float32, `fp16` for automatic mixed precision

    push_to_hub: bool = False  # whether to upload the saved model to the HF Hub
    hub_private_repo: bool = False  
    overwrite_output_dir: bool = True  # overwrite the old model when re-running the notebook
    seed: int = 0
    dataset_path: str = 'datasets'
    ckpt_dir: str = 'ckpt'
    data_ckpt_dir: str = 'data.ckpt'
    lora_dir: str = 'lora'
    ep_model_dir: str = 'epochs'
    ckpt_path: str = None
    data_ckpt_path: str = None
    lora_path: str = None
    # hub_token = "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"

def naming_fn(config: TrainingConfig):
    add_on: str = ""
    # add_on += "_clip" if config.clip else ""
    add_on += f"_{config.postfix}" if config.postfix else ""
    return f'fres_{config.ckpt}_{config.dataset}_ep{config.epoch}_{config.solver_type}_c{config.clean_rate}_p{config.poison_rate}_epr{config.ext_poison_rate}_{config.trigger}-{config.target}_psi{config.psi}_lr{config.learning_rate}_vp{config.vp_scale}_ve{config.ve_scale}{add_on}'

def read_json(args: argparse.Namespace, file: str):
    with open(os.path.join(args.ckpt, file), "r") as f:
        return json.load(f)

def write_json(content: Dict, config: argparse.Namespace, file: str):
    with open(os.path.join(config.output_dir, file), "w") as f:
        return json.dump(content, f, indent=2)

def setup():
    args_file: str = "args.json"
    config_file: str = "config.json"
    sampling_file: str = "sampling.json"
    measure_file: str = "measure.json"
    
    args: argparse.Namespace = parse_args()
    config: TrainingConfig = TrainingConfig()
    args_data: Dict = {}
    
    # print(f"Argument args: {args.__dict__}")
    if args.mode == MODE_RESUME or args.mode == MODE_SAMPLING or args.mode == MODE_MEASURE:
        with open(os.path.join(args.ckpt, args_file), "r") as f:
            args_data = json.load(f)
            
        # print(f"Argument dataset_load_mode: {config.dataset_load_mode}")
        # print(f"Argument args_data: {args_data}")
        
        for key, value in args_data.items():
            if value != None:
                setattr(config, key, value)
        setattr(config, "output_dir", args.ckpt)
    # print(f"Argument Init: {config.__dict__}")
    # print(f"Argument dataset_load_mode: {config.dataset_load_mode}")
    
    for key, value in args.__dict__.items():
        if args.mode == MODE_TRAIN and (key not in NOT_MODE_TRAIN_OPTS) and value != None:
            setattr(config, key, value)
        elif args.mode == MODE_TRAIN_MEASURE and (key not in NOT_MODE_TRAIN_MEASURE_OPTS) and value != None:
            setattr(config, key, value)
        elif args.mode == MODE_RESUME and key in MODE_RESUME_OPTS and value != None:
            setattr(config, key, value)
        elif args.mode == MODE_SAMPLING and key in MODE_SAMPLING_OPTS and value != None:
            setattr(config, key, value)
        elif args.mode == MODE_MEASURE and key in MODE_MEASURE_OPTS and value != None:
            setattr(config, key, value)
        elif value != None and not (key in IGNORE_ARGS):
            raise NotImplementedError(f"Argument: {key}={value} isn't used in mode: {args.mode}")
        
    # print(f"Argument Override: {config.__dict__}")
    # print(f"Argument dataset_load_mode: {config.dataset_load_mode}")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", config.gpu)

    print(f"PyTorch detected number of availabel devices: {torch.cuda.device_count()}")
    setattr(config, "device_ids", [int(i) for i in range(len(config.gpu.split(',')))])
    # setattr(config, "device_ids", config.gpu)
    
    # sample_ep options
    if isinstance(config.sample_ep, int):
        if config.sample_ep < 0:
            config.sample_ep = None
    
    # Clip option
    if config.fclip == 'w':
        setattr(config, "clip", True)
    elif config.fclip == 'o':
        setattr(config, "clip", False)
    else:
        setattr(config, "clip", None)
        
    # Mixed Precision Options
    if config.is_mixed_precision:
        if config.sde_type == "SDE-VP" or config.sde_type == "SDE-LDM":
            config.mixed_precision = 'fp16'
        elif config.sde_type == "SDE-VE":
            config.mixed_precision = 'no'
    # Determine gradient accumulation & Learning Rate
    bs = 0
    if config.dataset in [DatasetLoader.CIFAR10, DatasetLoader.MNIST, DatasetLoader.CELEBA_HQ_LATENT_PR05, DatasetLoader.CELEBA_HQ_LATENT]:
        bs = config.batch_32
        if config.learning_rate == None:
            if config.ckpt == None:
                config.learning_rate = config.learning_rate_32_scratch
            else:
                config.learning_rate = DEFAULT_LEARNING_RATE_32
    elif config.dataset in [DatasetLoader.CELEBA, DatasetLoader.CELEBA_HQ, DatasetLoader.LSUN_CHURCH, DatasetLoader.LSUN_BEDROOM]:
        bs = config.batch_256
        if config.learning_rate == None:
            if config.ckpt == None:
                config.learning_rate = config.learning_rate_256_scratch
            else:
                config.learning_rate = DEFAULT_LEARNING_RATE_256
    else:
        raise NotImplementedError()
    if bs % config.batch != 0:
        raise ValueError(f"batch size {config.batch} should be divisible to {bs} for dataset {config.dataset}")
    if bs < config.batch:
        raise ValueError(f"batch size {config.batch} should be smaller or equal to {bs} for dataset {config.dataset}")
    config.gradient_accumulation_steps = int(bs // config.batch)
    
    if args.mode == MODE_TRAIN or args.mode == MODE_TRAIN_MEASURE:
        setattr(config, "output_dir", os.path.join(config.result, naming_fn(config=config)))
    
    print(f"MODE: {config.mode}")
    if config.mode == MODE_TRAIN or args.mode == MODE_TRAIN_MEASURE:
        if not config.overwrite and os.path.isdir(config.output_dir):
            raise ValueError(f"Output directory: {config.output_dir} has already been created, please set overwrite flag --overwrite or -o")
        
        os.makedirs(config.output_dir, exist_ok=True)
        
        # with open(os.path.join(config.output_dir, args_file), "w") as f:
        #     json.dump(vars(args), f, indent=2)
        write_json(content=vars(args), config=config, file=args_file)
        write_json(content=config.__dict__, config=config, file=config_file)
    elif config.mode == MODE_SAMPLING:
        write_json(content=config.__dict__, config=config, file=sampling_file)
    elif config.mode == MODE_MEASURE or args.mode == MODE_TRAIN_MEASURE:
        write_json(content=config.__dict__, config=config, file=measure_file)
    elif config.mode == MODE_RESUME:
        pass
    else:
        raise NotImplementedError(f"Mode: {config.mode} isn't defined")
    
    if config.ckpt_path == None:
        config.ckpt_path = os.path.join(config.output_dir, config.ckpt_dir)
        config.data_ckpt_path = os.path.join(config.output_dir, config.data_ckpt_dir)
        os.makedirs(config.ckpt_path, exist_ok=True)
    config.lora_path = os.path.join(config.output_dir, config.lora_dir)
    os.makedirs(config.lora_path, exist_ok=True)
    
    name_id = str(config.output_dir).split('/')[-1]
    wandb.init(project=config.project, name=name_id, id=name_id, settings=wandb.Settings(start_method="fork"))
    print(f"Argument Final: {config.__dict__}")
    # print(f"Argument dataset_load_mode: {config.dataset_load_mode}")
    return config

config = setup()
"""## Config

For convenience, we define a configuration grouping all the training hyperparameters. This would be similar to the arguments used for a [training script](https://github.com/huggingface/diffusers/tree/main/examples).
Here we choose reasonable defaults for hyperparameters like `num_epochs`, `learning_rate`, `lr_warmup_steps`, but feel free to adjust them if you train on your own dataset. For example, `num_epochs` can be increased to 100 for better visual quality.
"""

import numpy as np
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import StructuralSimilarityIndexMeasure
from accelerate import Accelerator
from tqdm.auto import tqdm
import lpips
from datasets import Dataset
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

from fid_score import fid

from diffusers import DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import convert_state_dict_to_diffusers
# from diffusers.utils import check_min_version, convert_state_dict_to_diffusers, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module

from model import DiffuserModelSched, batch_sampling, batch_sampling_save
from util import Samples, MemoryLog, Log, batchify, match_count
from loss import p_losses_diffuser, adaptive_score_loss, LossFn
from fid_score import fid
from util_fourier import filtered_by_freq_all

# torch.multiprocessing.set_start_method('spawn')

def get_accelerator(config: TrainingConfig):
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps, 
        log_with=["tensorboard", "wandb"],
        # log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs")
    )
    return accelerator

def init_tracker(config: TrainingConfig, accelerator: Accelerator):
    tracked_config = {}
    for key, val in config.__dict__.items():
        if isinstance(val, int) or isinstance(val, float) or isinstance(val, str) or isinstance(val, bool) or isinstance(val, torch.Tensor):
            tracked_config[key] = val
    accelerator.init_trackers(config.project, config=tracked_config)

def get_latents_dataloader(config: TrainingConfig, dsl: DatasetLoader, vae):
    # vae = vae.to('cuda')
    # @torch.no_grad
    def encode_latents(x: torch.Tensor, weight_dtype: str=None, scaling_factor: float=None):
        with torch.no_grad():
            x = x.to(vae.device)
            if weight_dtype != None and weight_dtype != "":
                x = x.to(dtype=weight_dtype)
            if scaling_factor != None:
                return (vae.encode(x).latents * scaling_factor).detach().cpu()
            # return vae.encode(x).latents * vae.config.scaling_factor
            return vae.encode(x).latents.detach().cpu()
    
    def encode_by_keys(batch):
        batch[DatasetLoader.TARGET] = encode_latents(x=batch[DatasetLoader.TARGET])
        batch[DatasetLoader.PIXEL_VALUES] = encode_latents(x=batch[DatasetLoader.PIXEL_VALUES])
        return batch
    
    dl: DataLoader  = dsl.get_dataloader(batch_size=200)
    target_latents = [encode_latents(x=batch[DatasetLoader.TARGET]) for batch in tqdm(dl)]
    poisoned_latents = [encode_latents(x=batch[DatasetLoader.PIXEL_VALUES]) for batch in tqdm(dl)]
    image_latents = [encode_latents(x=batch[DatasetLoader.IMAGE]) for batch in tqdm(dl)]
    latent_ds = Dataset.from_dict({DatasetLoader.TARGET: target_latents, DatasetLoader.PIXEL_VALUES: poisoned_latents, DatasetLoader.IMAGE: image_latents})
    
    return DataLoader(latent_ds, batch_size=config.batch, shuffle=True, pin_memory=True, num_workers=8)

def get_data_loader(config: TrainingConfig):
    ds_root: Union[str, os.PathLike] = os.path.join(config.dataset_path)
    
    vmin: float = -1.0 
    vmax: float = 1.0
    if config.sde_type == DiffuserModelSched.SDE_VP or config.sde_type == DiffuserModelSched.SDE_LDM:
        vmin, vmax = -1.0, 1.0
    elif config.sde_type == DiffuserModelSched.SDE_VE:
        vmin, vmax = 0.0, 1.0
    else:
        raise NotImplementedError(f"sde_type: {config.sde_type} isn't implemented")
    
    if hasattr(config, 'R_trigger_only'):
        dsl = DatasetLoader(root=ds_root, name=config.dataset, batch_size=config.batch, vmin=vmin, vmax=vmax).set_poison(trigger_type=config.trigger, target_type=config.target, clean_rate=config.clean_rate, poison_rate=config.poison_rate, ext_poison_rate=config.ext_poison_rate).prepare_dataset(mode=config.dataset_load_mode, R_trigger_only=config.R_trigger_only)
    else:
        dsl = DatasetLoader(root=ds_root, name=config.dataset, batch_size=config.batch, vmin=vmin, vmax=vmax).set_poison(trigger_type=config.trigger, target_type=config.target, clean_rate=config.clean_rate, poison_rate=config.poison_rate, ext_poison_rate=config.ext_poison_rate).prepare_dataset(mode=config.dataset_load_mode)
    print(f"datasetloader len: {len(dsl)}")
    return dsl

def get_repo(config: TrainingConfig, accelerator: Accelerator):
    repo = None
    if accelerator.is_main_process:
        # if config.push_to_hub:
        #     repo = init_git_repo(config, at_init=True)
        # accelerator.init_trackers(config.output_dir, config=config.__dict__)
        init_tracker(config=config, accelerator=accelerator)
    return repo
        
def get_model_optim_sched(config: TrainingConfig, accelerator: Accelerator, dataset_loader: DatasetLoader):
    image_size: int = dataset_loader.image_size
    channel: int = dataset_loader.channel
    if config.ckpt != None:
        if config.sample_ep != None and config.mode in [MODE_MEASURE, MODE_SAMPLING]:
            ep_model_path = get_ep_model_path(config=config, dir=config.ckpt, epoch=config.sample_ep)
            model, vae, noise_sched, get_pipeline, lora_layers = DiffuserModelSched.get_model_sched(ckpt=ep_model_path, clip_sample=config.clip, noise_sched_type=config.sched, sde_type=config.sde_type, rank=config.lora_rank, alpha=config.lora_alpha)
        else:
            model, vae, noise_sched, get_pipeline, lora_layers = DiffuserModelSched.get_model_sched(ckpt=config.ckpt, clip_sample=config.clip, noise_sched_type=config.sched, sde_type=config.sde_type, rank=config.lora_rank, alpha=config.lora_alpha)
        # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    else:
        model, vae, noise_sched, get_pipeline, lora_layers = DiffuserModelSched.get_model_sched(image_size=image_size, channels=channel, ckpt=DiffuserModelSched.MODEL_DEFAULT, noise_sched_type=config.sched, clip_sample=config.clip, sde_type=config.sde_type, rank=config.lora_rank, alpha=config.lora_alpha)
        # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # model, noise_sched, get_pipeline = DiffuserModelSched.get_model_sched(image_size=image_size, channels=channel, ckpt=DiffuserModelSched.NCSNPP_32_DEFAULT, clip_sample=config.clip, noise_sched_type=config.sched, sde_type=config.sde_type)
    print(f"get_model_optim_sched/lora_layers: {lora_layers}")
    # IMPORTANT: Optimizer must be placed after nn.DataParallel because it needs to record parallel weights. If not, it cannot load_state properly.
    model = nn.DataParallel(model, device_ids=config.device_ids)
    if vae is not None:
        vae = vae.to(f'cuda:{config.device_ids[0]}')
    # if vae != None:
    #     vae = nn.DataParallel(vae, device_ids=config.device_ids)
    if lora_layers is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    else:
        print(f"lora_layers: {lora_layers}")
        optimizer = torch.optim.Adam(lora_layers, lr=config.learning_rate)
    lr_sched = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(dataset_loader.num_batch * config.epoch),
    )
    
    cur_epoch = cur_step = 0
    
    accelerator.register_for_checkpointing(model, optimizer, lr_sched)
    if config.mode == MODE_RESUME:
        if config.ckpt == None:
            raise ValueError(f"Argument 'ckpt' shouldn't be None for resume mode")
        accelerator.load_state(config.ckpt_path)
        data_ckpt = torch.load(config.data_ckpt_path)
        cur_epoch = data_ckpt['epoch']
        cur_step = data_ckpt['step']
    
    return model, vae, lora_layers, optimizer, lr_sched, noise_sched, cur_epoch, cur_step, get_pipeline

def get_encode_collate_fn(unet, vae):
    # vae = vae.to('cpu')
    def encode_latents(x: torch.Tensor, weight_dtype: str=None, scaling_factor: float=None):
        x = x.to('cuda')
        if weight_dtype != None and weight_dtype != "":
            x = x.to(dtype=weight_dtype)
        if scaling_factor != None:
            return vae.encode(x).latents * scaling_factor
        # return vae.encode(x).latents * vae.config.scaling_factor
        return vae.encode(x).latents
    
    def encode_by_keys(batch):
        batch[DatasetLoader.TARGET] = encode_latents(x=batch[DatasetLoader.TARGET])
        batch[DatasetLoader.PIXEL_VALUES] = encode_latents(x=batch[DatasetLoader.PIXEL_VALUES])
        return batch
    
    def serialize(batch):
        return {
            DatasetLoader.IMAGE: torch.stack([x[DatasetLoader.IMAGE] for x in batch]),
            DatasetLoader.TARGET: torch.stack([x[DatasetLoader.TARGET] for x in batch]),
            DatasetLoader.PIXEL_VALUES: torch.stack([x[DatasetLoader.PIXEL_VALUES] for x in batch]),
        }
    
    def process(batch):
        batch = serialize(batch)
        # if vae != None:
        #     batch = encode_by_keys(batch)
        batch[DatasetLoader.TARGET] = batch[DatasetLoader.TARGET].to('cuda')
        batch[DatasetLoader.PIXEL_VALUES] = batch[DatasetLoader.PIXEL_VALUES].to('cuda')
        return batch
    
    return process


    

def init_train(config: TrainingConfig, dataset_loader: DatasetLoader):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps, 
        log_with="wandb",
        project_dir=os.path.join(config.output_dir, "logs")
    )
    
    accelerator = get_accelerator(config=config)
    # repo = None
    if accelerator.is_main_process:
        # if config.push_to_hub:
        #     repo = init_git_repo(config, at_init=True)
        tracker_project: str = str(config.output_dir).split('/')[-1]
        # print(f"Config: {config.__dict__}")
        def convert_dict(config):
            ret = {}
            for key in config.__dict__.keys():
                if config.__dict__[key] is None:
                    # print(f"[{key}]: None")
                    ret[key] = str(None)
                elif isinstance(config.__dict__[key], (int, float, bool, str)) or torch.is_tensor(config.__dict__[key]):
                    # print(f"[{key}]: {type(config.__dict__[key])}")
                    ret[key] = config.__dict__[key]
                elif isinstance(config.__dict__[key], list):
                    str_ls: list[str] = []
                    for item in config.__dict__[key]:
                        str_ls.append(str(item))
                    val = ",".join(str_ls)
                    # print(f"[{key}]: {type(val)}")
                    ret[key] = val
            return ret
        accelerator.init_trackers(tracker_project, config=convert_dict(config))
    repo = get_repo(config=config, accelerator=accelerator)
    
    model, vae, lora_layers, optimizer, lr_sched, noise_sched, cur_epoch, cur_step, get_pipeline = get_model_optim_sched(config=config, accelerator=accelerator, dataset_loader=dataset_loader)
    
    # Prepare everything
    # There is no specific order to remember, you just need to unpack the 
    # objects in the same order you gave them to the prepare method.
    
    # if vae == None:
    dataloader = dataset_loader.get_dataloader()
    # else:
    #     dataloader = get_latents_dataloader(config=config, dsl=dataset_loader, vae=vae)
    model, vae, optimizer, dataloader, lr_sched = accelerator.prepare(
        model, vae, optimizer, dataloader, lr_sched
    )
    return accelerator, repo, model, vae, lora_layers, noise_sched, optimizer, dataloader, lr_sched, cur_epoch, cur_step, get_pipeline

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def auto_grid(images: List):
    eval_samples_n = len(images)
    nrow = 1
    ncol = eval_samples_n
    for i in range(ceil(sqrt(eval_samples_n)), 0, -1):
        if eval_samples_n % i == 0:
            nrow = i
            ncol = eval_samples_n // nrow
            break
    return make_grid(images, rows=nrow, cols=ncol)

def sampling(config: TrainingConfig, file_name: Union[int, str], pipeline):
    def gen_samples(init: torch.Tensor, folder: Union[os.PathLike, str], start_from: int=0):
        test_dir = os.path.join(config.output_dir, folder)
        os.makedirs(test_dir, exist_ok=True)
        
        # Sample some images from random noise (this is the backward diffusion process).
        # The default pipeline output type is `List[PIL.Image]`
        print(f"Pipeline: {type(pipeline)}")
        if config.ddim_eta == None:
            pipline_res = pipeline(num_inference_steps=config.infer_steps, start_from=start_from, batch_size=config.eval_sample_n, generator=torch.manual_seed(config.seed), init=init, save_every_step=True, output_type=None)
        else:
            pipline_res = pipeline(num_inference_steps=config.infer_steps, start_from=start_from, eta=config.ddim_eta, batch_size=config.eval_sample_n, generator=torch.manual_seed(config.seed), init=init, save_every_step=True, output_type=None)
        images = pipline_res.images
        movie = pipline_res.movie
        
        # # Because PIL can only accept 2D matrix for gray-scale images, thus, we need to convert the 3D tensors into 2D ones.
        images = [Image.fromarray(image) for image in np.squeeze((images * 255).round().astype("uint8"))]
        init_images = [Image.fromarray(image) for image in np.squeeze((movie[0] * 255).round().astype("uint8"))]
        
        # eval_samples_n = len(images)
        # nrow = 1
        # ncol = eval_samples_n
        # for i in range(ceil(sqrt(eval_samples_n)), 0, -1):
        #     if eval_samples_n % i == 0:
        #         nrow = i
        #         ncol = eval_samples_n // nrow
        #         break

        # # Make a grid out of the images
        image_grid = auto_grid(images)
        init_image_grid = auto_grid(init_images)

        # sam_obj = Samples(samples=np.array(movie), save_dir=test_dir)
        
        clip_opt = "" if config.clip else "_noclip"
        # # Save the images
        if isinstance(file_name, int):
            image_grid.save(f"{test_dir}/{file_name:04d}{clip_opt}.png")
            init_image_grid.save(f"{test_dir}/{file_name:04d}{clip_opt}_sample_t0.png")
            # sam_obj.save(file_path=f"{file_name:04d}{clip_opt}_samples.pkl")
            # sam_obj.plot_series(slice_idx=slice(None), end_point=True, prefix_img_name=f"{file_name:04d}{clip_opt}_sample_t", animate_name=f"{file_name:04d}{clip_opt}_movie", save_mode=Samples.SAVE_FIRST_LAST, show_mode=Samples.SHOW_NONE)
        elif isinstance(file_name, str):
            image_grid.save(f"{test_dir}/{file_name}{clip_opt}.png")
            init_image_grid.save(f"{test_dir}/{file_name}{clip_opt}_sample_t0.png")
            # sam_obj.save(file_path=f"{file_name}{clip_opt}_samples.pkl")
            # sam_obj.plot_series(slice_idx=slice(None), end_point=True, prefix_img_name=f"{file_name}{clip_opt}_sample_t", animate_name=f"{file_name}{clip_opt}_movie", save_mode=Samples.SAVE_FIRST_LAST, show_mode=Samples.SHOW_NONE)
        else:
            raise TypeError(f"Argument 'file_name' should be string nor integer.")
    
    with torch.no_grad():
        if config.lora_alpha is not None and config.lora_rank is not None:
            pipeline = load_lora(pipeline=pipeline, save_path=config.lora_path)
        print(f"Sampling Init Noise -Sample_n: {config.eval_sample_n}, Channel: {pipeline.unet.in_channels}, Sample_size: {pipeline.unet.sample_size}")
        noise = torch.randn(
                    (config.eval_sample_n, pipeline.unet.in_channels, pipeline.unet.sample_size, pipeline.unet.sample_size),
                    generator=torch.manual_seed(config.seed),
                )
        
        if config.task == TASK_GENERATE:
            # Sample Clean Samples
            gen_samples(init=noise, folder="samples", start_from=0)
            # Sample Backdoor Samples
            # init = noise + torch.where(dsl.trigger.unsqueeze(0) == -1.0, 0, 1)
            if hasattr(pipeline, 'encode'):
                init = noise.to(pipeline.device) + pipeline.encode(dsl.trigger.unsqueeze(0).to(pipeline.device))
            else:
                init = noise.to(pipeline.device) + dsl.trigger.unsqueeze(0).to(pipeline.device)
            # print(f"Trigger - (max: {torch.max(dsl.trigger)}, min: {torch.min(dsl.trigger)}) | Noise - (max: {torch.max(noise)}, min: {torch.min(noise)}) | Init - (max: {torch.max(init)}, min: {torch.min(init)})")
            gen_samples(init=init, folder="backdoor_samples", start_from=0)
        else:
            # Special Sampling
            start_from_sp = config.infer_start
            noise_sp = noise * 0.3
            mul = config.inpaint_mul
            imgs = []
            ds = dsl.get_dataset()
            for idx in range(config.eval_sample_n):
                imgs.append(ds[-idx][DatasetLoader.IMAGE])
            imgs = torch.stack(imgs)
            
            poisoned_imgs = pipeline.encode(dsl.get_poisoned(imgs))

            ext = f"_{config.sched}_{config.infer_steps}_st{start_from_sp}_m{mul}"
            
            # imgs = imgs * 0.7
            # poisoned_imgs = poisoned_imgs * 0.7
            
            if config.task == TASK_UNPOISONED_DENOISE:
                # Sample Unpoisoned Noisy Samples
                gen_samples(init=(imgs + noise_sp) * mul, folder=f"unpoisoned_noisy_samples{ext}", start_from=start_from_sp)
            elif config.task == TASK_POISONED_DENOISE:
                # Sample Poisoned Noisy Samples
                gen_samples(init=(poisoned_imgs + noise_sp) * mul, folder=f"poisoned_noisy_samples{ext}", start_from=start_from_sp)
            elif config.task == TASK_UNPOISONED_INPAINT_BOX:
                # Sample Inpainted Box Unpoisoned Samples
                corrupt_imgs = dsl.get_inpainted_by_type(imgs=imgs, inpaint_type=DatasetLoader.INPAINT_BOX)
                gen_samples(init=corrupt_imgs * mul, folder=f"inpaint_box_unpoisoned_samples{ext}", start_from=start_from_sp)
            elif config.task == TASK_POISONED_INPAINT_BOX:
                # Sample Inpainted Box Poisoned Samples
                corrupt_imgs = dsl.get_inpainted_by_type(imgs=poisoned_imgs, inpaint_type=DatasetLoader.INPAINT_BOX)
                gen_samples(init=corrupt_imgs * mul, folder=f"inpaint_box_poisoned_samples{ext}", start_from=start_from_sp)
            elif config.task == TASK_UNPOISONED_INPAINT_LINE:
                # Sample Inpainted Line Unpoisoned Samples
                corrupt_imgs = dsl.get_inpainted_by_type(imgs=imgs, inpaint_type=DatasetLoader.INPAINT_LINE)
                gen_samples(init=corrupt_imgs * mul, folder=f"inpaint_line_unpoisoned_samples{ext}", start_from=start_from_sp)
            elif config.task == TASK_POISONED_INPAINT_LINE:
                # Sample Inpainted Line Poisoned Samples
                corrupt_imgs = dsl.get_inpainted_by_type(imgs=poisoned_imgs, inpaint_type=DatasetLoader.INPAINT_LINE)
                gen_samples(init=corrupt_imgs * mul, folder=f"inpaint_line_poisoned_samples{ext}", start_from=start_from_sp)
            else:
                raise NotImplementedError(f"Sampling task: {config.task} isn't implemented")
        # # Inpainted box
        # half_dim = imgs.shape[-1] // 2
        # up_left = half_dim - half_dim // 3
        # low_right = half_dim + half_dim // 3
        
        
        
        # # Sample Inpainted Unpoisoned Noisy Samples
        # gen_samples(init=inpainted_unpoisoned_imgs + noise_sp, folder="inpainted_unpoisoned_noisy_samples", start_from=start_from_sp)
        
        # # Sample Inpainted Poisoned Samples
        # inpainted_poisoned_imgs = dsl.get_inpainted_boxes(poisoned_imgs, up=up_left, low=low_right, left=up_left, right=low_right)
        # gen_samples(init=inpainted_poisoned_imgs, folder="inpainted_poisoned_samples", start_from=start_from_sp)
        
        # # Sample Inpainted Poisoned Noisy Samples
        # gen_samples(init=inpainted_poisoned_imgs + noise_sp, folder="inpainted_poisoned_noisy_samples", start_from=start_from_sp)
        
        # # Inpainted Rect
        # half_dim = imgs.shape[-1] // 2
        # up = half_dim - half_dim
        # low = half_dim + half_dim
        # left = half_dim - half_dim // 10
        # right = half_dim + half_dim // 20
        
        # # Sample Inpainted Rect Unpoisoned Samples
        # inpainted_unpoisoned_imgs = dsl.get_inpainted_boxes(imgs, up=up, low=low, left=left, right=right)
        # gen_samples(init=inpainted_unpoisoned_imgs, folder="inpainted_rect_unpoisoned_samples", start_from=start_from_sp)
        
        # # Sample Inpainted Rect Unpoisoned Noisy Samples
        # gen_samples(init=inpainted_unpoisoned_imgs + noise_sp, folder="inpainted_rect_unpoisoned_noisy_samples", start_from=start_from_sp)
        
        # # Sample Inpainted Rect Poisoned Samples
        # inpainted_poisoned_imgs = dsl.get_inpainted_boxes(poisoned_imgs, up=up, low=low, left=left, right=right)
        # gen_samples(init=inpainted_poisoned_imgs, folder="inpainted_rect_poisoned_samples", start_from=start_from_sp)
        
        # # Sample Inpainted Rect Poisoned Noisy Samples
        # gen_samples(init=inpainted_poisoned_imgs + noise_sp, folder="inpainted_rect_poisoned_noisy_samples", start_from=start_from_sp)

def save_imgs(imgs: np.ndarray, file_dir: Union[str, os.PathLike], file_name: Union[str, os.PathLike]="") -> None:
    os.makedirs(file_dir, exist_ok=True)
    # Because PIL can only accept 2D matrix for gray-scale images, thus, we need to convert the 3D tensors into 2D ones.
    images = [Image.fromarray(image) for image in np.squeeze((imgs * 255).round().astype("uint8"))]
    for i, img in enumerate(tqdm(images)):
        img.save(os.path.join(file_dir, f"{file_name}{i}.png"))

def update_score_file(config: TrainingConfig, score_file: str, fid_sc: float=None, mse_sc: float=None, ssim_sc: float=None, lpips_sc: float=None) -> Dict:
    # def get_key_gen(config: TrainingConfig, key: str) -> str:
    def get_key(config: TrainingConfig, key: str) -> str:
        res = f"{key}_ep{config.sample_ep}" if config.sample_ep != None else key
        res += "_noclip" if not config.clip else ""
        if config.sched != None:
            res += f"_{config.sched}-{config.infer_steps}" if config.sched != None else ""
        if config.sched == DiffuserModelSched.DDIM_SCHED and config.ddim_eta != DEFAULT_DDIM_ETA:
            res += f"-eta{config.ddim_eta}"
        if config.task != TASK_GENERATE:
            res += f"_{config.measure_inpaint_sample_n}"
            res += f"_{config.task}"
        else:
            res += f"_{config.measure_sample_n}"
        return res
    
    def get_key_non_gen(config: TrainingConfig, key: str) -> str:
        res = f"{key}_ep{config.sample_ep}" if config.sample_ep != None else key
        res += "_noclip" if not config.clip else ""
        if config.sched != None:
            res += f"_{config.sched}-{config.infer_steps}" if config.sched != None else ""
        res += f"_{config.measure_sample_n}"
        return res
    
    def update_dict(data: Dict, key: str, val):
        data[str(key)] = val if val != None else data[str(key)]
        return data
    
    def update_dicts(data: Dict, score: Dict, get_key_fn: Callable[[TrainingConfig, str], str]):
        for key, val in score.items():
            data = update_dict(data=data, key=get_key_fn(config=config, key=key), val=val)
        return data
    
    if config.task == TASK_GENERATE:
        if fid_sc == None or mse_sc == None or ssim_sc == None:
            raise ValueError(f"Task: {config.task} requires FID, MSE, and SSIM scores")
        # get_key_fn = get_key_gen
        scores = {"FID": fid_sc, "MSE": mse_sc, "SSIM": ssim_sc}
    else:
        if lpips_sc == None or mse_sc == None or ssim_sc == None:
            raise ValueError(f"Task: {config.task} requires LPIPS, MSE, and SSIM scores")
        scores = {"LPIPS": lpips_sc, "MSE": mse_sc, "SSIM": ssim_sc}
        # get_key_fn = get_key_non_gen
    get_key_fn = get_key
    sc: Dict = {}
    try:
        with open(os.path.join(config.output_dir, score_file), "r") as f:
            sc = json.load(f)
    except:
        Log.info(f"No existed {score_file}, create new one")
    finally:
        with open(os.path.join(config.output_dir, score_file), "w") as f:
            sc = update_dicts(data=sc, score=scores, get_key_fn=get_key_fn)
            json.dump(sc, f, indent=2, sort_keys=True)
        return sc
    
# def log_score(config: TrainingConfig, accelerator: Accelerator, fid_sc: float, mse_sc: float, ssim_sc: float, scores: Dict):
def log_score(config: TrainingConfig, accelerator: Accelerator, scores: Dict, step: int):
    # def get_key(config: TrainingConfig, key):
    #     res: str = f"{key}_noclip" if not config.clip else key
    #     return res
    
    def parse_ep(key):
        ep_str = ''.join(filter(str.isdigit, key))
        return config.epoch - 1 if ep_str == '' else int(ep_str)
    
    def parse_clip(key):
        return False if "noclip" in key else True
    
    def parse_metric(key):
        return key.split('_')[0]
    
    def get_log_key(key):
        res = parse_metric(key)
        res += "_noclip" if not parse_clip(key) else ""
        return res
        
    def get_log_ep(key):
        return parse_ep(key)
    
    # accelerator.log({get_key(config=config, key="FID"): fid_sc, "epoch": config.sample_ep})
    # accelerator.log({get_key(config=config, key="MSE"): mse_sc, "epoch": config.sample_ep})
    # accelerator.log({get_key(config=config, key="SSIM"): ssim_sc, "epoch": config.sample_ep})
    
    for key, val in scores.items():
        print(f"Log: ({get_log_key(key)}: {val}, epoch: {get_log_ep(key)}, step: {step})")
        accelerator.log({get_log_key(key): val, 'epoch': get_log_ep(key)}, step=step)
        
    accelerator.log(scores)

def measure_subfolder_naming_ext(config: TrainingConfig):
    res = ""
    if config.sched != None:
        res += f"_{config.sched}-{config.infer_steps}" if config.sched != None else ""
    res += f"_{config.measure_sample_n}"
    return res

def get_batch_sizes(sample_n: int, max_batch_n: int):
    if sample_n > max_batch_n:
        replica = sample_n // max_batch_n
        residual = sample_n % max_batch_n
        batch_sizes = [max_batch_n] * (replica) + ([residual] if residual > 0 else [])
    else:
        batch_sizes = [sample_n]
    return batch_sizes
            
def get_bs_inits(sample_n: int, max_batch_n: int, init: torch.Tensor=None):
    if init == None:
        batch_sizes = get_batch_sizes(sample_n=sample_n, max_batch_n=max_batch_n)
    else:
        if sample_n != len(init):
            raise ValueError(f"Argument sample_n should be equal to len(init)")
        init = torch.split(init, max_batch_n)
        batch_sizes = list(map(lambda x: len(x), init))
    return batch_sizes, init

def gen_batch_samples(pipeline, init: torch.Tensor, num_inference_steps: int, start_from: int, batch_size: int, seed: int=1):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    pipeline_res = pipeline(
        num_inference_steps=num_inference_steps,
        start_from=start_from,
        batch_size=batch_size, 
        generator=torch.manual_seed(seed),
        init=init,
        save_every_step=False,
        output_type=None
    )
    return pipeline_res.images

def gen_samples(pipeline, sample_n: int, init: torch.Tensor=None, num_inference_steps: int=1000, start_from: int=0, max_batch_n: int=256, seed: int=1):
    batch_sizes, init = get_bs_inits(sample_n=sample_n, init=init, max_batch_n=max_batch_n)
    
    sample_imgs_ls = []
    for i, batch_sz in enumerate(batch_sizes):
        if init != None:
            samples = gen_batch_samples(pipeline, init=init[i], num_inference_steps=num_inference_steps, start_from=start_from, batch_size=batch_sz, seed=seed)
        else:
            samples = gen_batch_samples(pipeline, init=None, num_inference_steps=num_inference_steps, start_from=start_from, batch_size=batch_sz, seed=seed)
        sample_imgs_ls.append(samples)
        del samples
        
    sample_imgs = np.vstack(sample_imgs_ls)
    return sample_imgs

@dataclass
class InpaintScore:
    mse: float
    ssim: float
    lpips: float

def measure_inpaint(config: TrainingConfig, pipeline, target_imgs: torch.Tensor, corrupt_imgs: torch.Tensor):
    if len(target_imgs) != len(corrupt_imgs):
        raise ValueError

    start_from_sp: float = config.infer_start
    mul = config.inpaint_mul

    device = torch.device(config.device_ids[0])
    pipeline, target_imgs, corrupt_imgs = pipeline.to(device), target_imgs.to(device), corrupt_imgs.to(device)
    recover_imgs = gen_samples(pipeline=pipeline, sample_n=len(target_imgs), init=corrupt_imgs * mul, num_inference_steps=config.infer_steps, start_from=start_from_sp, max_batch_n=config.eval_max_batch, seed=config.seed)
    recover_imgs = torch.Tensor(recover_imgs).permute(0, 3, 1, 2).to(device)
    
    print(f"recover_imgs: {recover_imgs.shape}, target_imgs: {target_imgs.shape}")
    # mse_sc = float(nn.MSELoss(reduction='mean')(recover_imgs, target_imgs))
    # ssim_sc = float(StructuralSimilarityIndexMeasure(data_range=1.0).to(device)(recover_imgs, target_imgs))
    mse_sc = Metric.mse_batch(a=recover_imgs, b=target_imgs, max_batch_n=config.eval_max_batch)
    ssim_sc = Metric.ssim_batch(a=recover_imgs, b=target_imgs, max_batch_n=config.eval_max_batch, device=device)
    lpips_sc = float(torch.mean(lpips.LPIPS(net='alex').to(device)(recover_imgs, target_imgs)))
    return mse_sc, ssim_sc, lpips_sc
@dataclass
class InpaintTaskScore:
    poisoned_noisy: InpaintScore
    unpoisoned_noisy: InpaintScore

def measure_inpaints(config: TrainingConfig, pipeline, dsl: DatasetLoader) -> Tuple[float, float, float]:
    device = torch.device(config.device_ids[0])
    noise = torch.randn(
                (config.measure_inpaint_sample_n, pipeline.unet.in_channels, pipeline.unet.sample_size, pipeline.unet.sample_size),
                generator=torch.manual_seed(config.seed),
            )
    # Special Sampling
    noise_sp = noise * 0.3
    imgs = []
    ds = dsl.get_dataset()
    for idx in range(config.measure_inpaint_sample_n):
        imgs.append(ds[-idx][DatasetLoader.IMAGE])
    imgs = torch.stack(imgs)
    
    # Target
    reps = ([config.measure_inpaint_sample_n] + ([1] * (len(dsl.target.shape))))
    
    if config.sde_type == DiffuserModelSched.SDE_VE:
        backdoor_targets = torch.squeeze((dsl.target.repeat(*reps)).clamp(0, 1)).to(device)
    else:
        backdoor_targets = torch.squeeze((dsl.target.repeat(*reps) / 2 + 0.5).clamp(0, 1)).to(device)
    # backdoor_targets = torch.squeeze((dsl.target.repeat(*reps) / 2 + 0.5).clamp(0, 1)).to(config.device_ids[0])
    # Sample Poisoned Samples
    poisoned_imgs = dsl.get_poisoned(imgs)
    
    if config.task == TASK_UNPOISONED_DENOISE:
        # Inpaint Unpoisoned noisy images
        sc = measure_inpaint(config=config, pipeline=pipeline, target_imgs=imgs, corrupt_imgs=imgs + noise_sp)
    elif config.task == TASK_POISONED_DENOISE:
        # Inpaint Poisoned noisy images
        sc = measure_inpaint(config=config, pipeline=pipeline, target_imgs=backdoor_targets, corrupt_imgs=poisoned_imgs + noise_sp)
    elif config.task == TASK_UNPOISONED_INPAINT_LINE:
        # Inpaint unpoisoned Inpaint Line images
        corrupt_imgs = dsl.get_inpainted_by_type(imgs=imgs, inpaint_type=DatasetLoader.INPAINT_LINE)
        sc = measure_inpaint(config=config, pipeline=pipeline, target_imgs=imgs, corrupt_imgs=corrupt_imgs)
    elif config.task == TASK_POISONED_INPAINT_LINE:
        # Inpaint Poisoned Inpaint Line images
        corrupt_imgs = dsl.get_inpainted_by_type(imgs=poisoned_imgs, inpaint_type=DatasetLoader.INPAINT_LINE)
        sc = measure_inpaint(config=config, pipeline=pipeline, target_imgs=backdoor_targets, corrupt_imgs=corrupt_imgs)
    elif config.task == TASK_UNPOISONED_INPAINT_BOX:
        # Inpaint unpoisoned Inpaint Box images
        corrupt_imgs = dsl.get_inpainted_by_type(imgs=imgs, inpaint_type=DatasetLoader.INPAINT_BOX)
        sc = measure_inpaint(config=config, pipeline=pipeline, target_imgs=imgs, corrupt_imgs=corrupt_imgs)
    elif config.task == TASK_POISONED_INPAINT_BOX:
        # Inpaint Poisoned Inpaint Box images
        corrupt_imgs = dsl.get_inpainted_by_type(imgs=poisoned_imgs, inpaint_type=DatasetLoader.INPAINT_BOX)
        sc = measure_inpaint(config=config, pipeline=pipeline, target_imgs=backdoor_targets, corrupt_imgs=corrupt_imgs)
    else:
        raise NotImplementedError(f"Measurement task: {config.task} isn't implemented")
    
    return sc

class Metric:
    @staticmethod
    def batch_metric(a: torch.Tensor, b: torch.Tensor, max_batch_n: int, fn: callable):
        a_batchs = batchify(xs=a, max_batch_n=max_batch_n)
        b_batchs = batchify(xs=b, max_batch_n=max_batch_n)
        scores: List[torch.Tensor] = [fn(a, b) for a, b in zip(a_batchs, b_batchs)]
        if len(scores) == 1:
            return scores[0].mean()
        return torch.cat(scores, dim=0).mean()
    
    # @staticmethod
    # def batch_object_metric(a: object, b: object, max_batch_n: int, fn: callable):
    #     a_batchs = batchify_generator(xs=a, max_batch_n=max_batch_n)
    #     b_batchs = batchify_generator(xs=b, max_batch_n=max_batch_n)
    #     scores: List[torch.Tensor] = [fn(a, b) for a, b in zip(a_batchs, b_batchs)]
    #     if len(scores) == 1:
    #         return scores.mean()
    #     return torch.cat(scores, dim=0).mean()
    
    @staticmethod
    def get_batch_operator(a: torch.Tensor, b: torch.Tensor):
        batch_operator: callable = None
        if torch.is_tensor(a) and torch.is_tensor(b):
            batch_operator = Metric.batch_metric
        elif (torch.is_tensor(a) and not torch.is_tensor(b)) or (not torch.is_tensor(a) and torch.is_tensor(b)):
            raise TypeError(f"Both arguement a {type(a)} and b {type(b)} should have the same type")
        else:
            raise NotImplementedError
            # batch_operator = Metric.batch_object_metric
        return batch_operator
    
    @staticmethod
    def mse_batch(a: torch.Tensor, b: torch.Tensor, max_batch_n: int):
        Log.critical("COMPUTING MSE")
        batch_operator: callable = Metric.get_batch_operator(a=a, b=b)
        def metric(x, y):
            mse: torch.Tensor = nn.MSELoss(reduction='none')(x, y).mean(dim=[i for i in range(1, len(x.shape))])
            # print(f"MSE: {mse.shape}")
            return mse
        return  float(batch_operator(a=a, b=b, max_batch_n=max_batch_n, fn=metric))
    
    @staticmethod
    def mse_thres_batch(a: torch.Tensor, b: torch.Tensor, thres: float, max_batch_n: int):
        Log.critical("COMPUTING MSE-THRESHOLD")
        batch_operator: callable = Metric.get_batch_operator(a=a, b=b)
        def metric(x, y):
            # print(f"x: {x.shape}, y: {y.shape}")
            # print(f"Mean Dims: {[i for i in range(1, len(x))]}")
            probs: torch.Tensor = nn.MSELoss(reduction='none')(x, y).mean(dim=[i for i in range(1, len(x.shape))])
            mse_thres: torch.Tensor = torch.where(probs < thres, 1.0, 0.0)
            # print(f"MSE Threshold: {mse_thres.shape}")
            return mse_thres
        return  float(batch_operator(a=a, b=b, max_batch_n=max_batch_n, fn=metric))
    
    @staticmethod
    def ssim_batch(a: torch.Tensor, b: torch.Tensor, device: str, max_batch_n: int):
        Log.critical("COMPUTING SSIM")
        batch_operator: callable = Metric.get_batch_operator(a=a, b=b)
        def metric(x, y):
            ssim: torch.Tensor = StructuralSimilarityIndexMeasure(data_range=1.0, reduction='none').to(device)(x, y)
            if len(ssim.shape) < 1:
                ssim = ssim.unsqueeze(dim=0)
            # print(f"SSIM: {ssim.shape}")
            return ssim
        return  float(batch_operator(a=a, b=b, max_batch_n=max_batch_n, fn=metric))

def measure(config: TrainingConfig, accelerator: Accelerator, dataset_loader: DatasetLoader, folder_name: Union[int, str], pipeline, resample: bool=False, recomp: bool=True):
    score_file = "score.json"
    
    fid_sc = mse_sc = ssim_sc = lpips_sc = None
    re_comp_clean_metric = False
    re_comp_backdoor_metric = False
    
    # Random Number Generator
    rng = torch.Generator()
    rng.manual_seed(config.seed)
    
    # Dataset samples
    step = dataset_loader.num_batch * (config.sample_ep + 1 if config.sample_ep != None else config.epoch)
    
    # Folders
    dataset_img_dir = os.path.join(folder_name, config.dataset)
    folder_path_ls = [config.output_dir, folder_name]
    if config.sample_ep != None:
        folder_path_ls += [f"ep{config.sample_ep}"]
    clean_folder = "clean" + ("_noclip" if not config.clip else "") + measure_subfolder_naming_ext(config=config)
    backdoor_folder = "backdoor" + ("_noclip" if not config.clip else "") + measure_subfolder_naming_ext(config=config)
    clean_path = os.path.join(*folder_path_ls, clean_folder)
    backdoor_path = os.path.join(*folder_path_ls, backdoor_folder)
    
    # if not os.path.isdir(dataset_img_dir) or resample:
    #     ds = dataset_loader.get_dataset().shuffle(seed=config.seed)
    #     os.makedirs(dataset_img_dir, exist_ok=True)
    #     # dataset_loader.show_sample(img=ds[0][DatasetLoader.IMAGE], is_show=False, file_name=os.path.join(clean_measure_dir, f"0.png"))
    #     for i, img in enumerate(tqdm(ds[:config.measure_sample_n][DatasetLoader.IMAGE])):
    #         dataset_loader.show_sample(img=img, is_show=False, file_name=os.path.join(dataset_img_dir, f"{i}.png"))
    #     re_comp_clean_metric = True
    
    # Init noise
    noise = torch.randn(
                (config.measure_sample_n, pipeline.unet.in_channels, pipeline.unet.sample_size, pipeline.unet.sample_size),
                generator=torch.manual_seed(config.seed),
            )
    backdoor_noise = noise + pipeline.encode(dataset_loader.trigger.unsqueeze(0)).to(noise.device)
    
    if config.lora_alpha is not None and config.lora_rank is not None:
        pipeline = load_lora(pipeline=pipeline, save_path=config.output_dir)
    
    if config.task != TASK_GENERATE:
        mse_sc, ssim_sc, lpips_sc = measure_inpaints(config=config, pipeline=pipeline, dsl=dataset_loader)
        print(f"{config.task} - LPIPS: {lpips_sc}, MSE: {mse_sc}, SSIM: {ssim_sc}")
        sc = update_score_file(config=config, score_file=score_file, lpips_sc=lpips_sc, mse_sc=mse_sc, ssim_sc=ssim_sc)
    else:
        # Sampling
        if not os.path.isdir(clean_path) or match_count(dir=clean_path) < config.measure_sample_n or resample:
            batch_sampling_save(sample_n=config.measure_sample_n, num_inference_steps=config.infer_steps, ddim_eta=config.ddim_eta, pipeline=pipeline, path=clean_path, init=noise, max_batch_n=config.eval_max_batch, rng=rng)
            re_comp_clean_metric = True

        if not os.path.isdir(backdoor_path) or match_count(dir=backdoor_path) < config.measure_sample_n or resample:
            batch_sampling_save(sample_n=config.measure_sample_n, num_inference_steps=config.infer_steps, ddim_eta=config.ddim_eta, pipeline=pipeline, path=backdoor_path, init=backdoor_noise,  max_batch_n=config.eval_max_batch, rng=rng)
            re_comp_backdoor_metric = True
        
        # Compute Score
        if re_comp_clean_metric or recomp:
            fid_sc = float(fid(path=[dataset_img_dir, clean_path], device=config.device_ids[0], num_workers=4, batch_size=config.eval_max_batch))
        
        if re_comp_backdoor_metric or recomp:
            device = torch.device(config.device_ids[0])
            # gen_backdoor_target = torch.from_numpy(backdoor_sample_imgs)
            # print(f"backdoor_sample_imgs shape: {backdoor_sample_imgs.shape}")
            gen_backdoor_target = ImagePathDataset(path=backdoor_path)[:].to(device)
            
            reps = ([len(gen_backdoor_target)] + ([1] * (len(dsl.target.shape))))
            if config.sde_type == DiffuserModelSched.SDE_VE:
                backdoor_target = torch.squeeze((dsl.target.repeat(*reps)).clamp(0, 1)).to(device)
            else:
                backdoor_target = torch.squeeze((dsl.target.repeat(*reps) / 2 + 0.5).clamp(0, 1)).to(device)
            # backdoor_target = torch.squeeze((dsl.target.repeat(*reps) / 2 + 0.5).clamp(0, 1)).to(device)
            
            print(f"gen_backdoor_target: {gen_backdoor_target.shape}, vmax: {torch.max(gen_backdoor_target)}, vmin: {torch.min(backdoor_target)} | backdoor_target: {backdoor_target.shape}, vmax: {torch.max(backdoor_target)}, vmin: {torch.min(backdoor_target)}")
            # mse_sc = float(nn.MSELoss(reduction='mean')(gen_backdoor_target, backdoor_target))
            # ssim_sc = float(StructuralSimilarityIndexMeasure(data_range=1.0).to(device)(gen_backdoor_target, backdoor_target))
            mse_sc = Metric.mse_batch(a=gen_backdoor_target, b=backdoor_target, max_batch_n=config.eval_max_batch)
            ssim_sc = Metric.ssim_batch(a=gen_backdoor_target, b=backdoor_target, max_batch_n=config.eval_max_batch, device=device)
        print(f"[{config.sample_ep}] FID: {fid_sc}, MSE: {mse_sc}, SSIM: {ssim_sc}")
        sc = update_score_file(config=config, score_file=score_file, fid_sc=fid_sc, mse_sc=mse_sc, ssim_sc=ssim_sc)
        
    # accelerator.log(sc)
    log_score(config=config, accelerator=accelerator, scores=sc, step=step)

"""With this in end, we can group all together and write our training function. This just wraps the training step we saw in the previous section in a loop, using Accelerate for easy TensorBoard logging, gradient accumulation, mixed precision training and multi-GPUs or TPU training."""

def get_ep_model_path(config: TrainingConfig, dir: Union[str, os.PathLike], epoch: int):
    return os.path.join(dir, config.ep_model_dir, f"ep{epoch}")


def unwrap_model(model):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def save_lora(pipeline, save_path: str, unet=None):
    if unet is not None:
        unwrapped_unet = unwrap_model(unet)
    else:
        unwrapped_unet = unwrap_model(pipeline.unet)
    unet_lora_state_dict = convert_state_dict_to_diffusers(
        get_peft_model_state_dict(unwrapped_unet)
    )

    pipeline.save_lora_weights(
        save_directory=save_path,
        unet_lora_layers=unet_lora_state_dict,
        safe_serialization=True,
    )
    
def load_lora(pipeline, save_path: str, none_ok: bool=True):
    if len(os.listdir(save_path)) > 0:
        pipeline.load_lora_weights(save_path)
    elif not none_ok:
        raise ValueError(f"Cannot load LoRA module under the path, {save_path}")
        
    return pipeline

def checkpoint(config: TrainingConfig, accelerator: Accelerator, pipeline, cur_epoch: int, cur_step: int, repo=None, commit_msg: str=None, unet=None):
    accelerator.save_state(config.ckpt_path)
    accelerator.save({'epoch': cur_epoch, 'step': cur_step}, config.data_ckpt_path)
    # if config.push_to_hub:
    #     push_to_hub(config, pipeline, repo, commit_message=commit_msg, blocking=True)
    # else:
    pipeline.save_pretrained(config.output_dir)
    if config.lora_alpha is not None and config.lora_rank is not None:
        save_lora(pipeline=pipeline, save_path=config.lora_path, unet=unet)
        
    if config.is_save_all_model_epochs:
        # ep_model_path = os.path.join(config.output_dir, config.ep_model_dir, f"ep{cur_epoch}")
        ep_model_path = get_ep_model_path(config=config, dir=config.output_dir, epoch=cur_epoch)
        os.makedirs(ep_model_path, exist_ok=True)
        pipeline.save_pretrained(ep_model_path)

def train_loop(config: TrainingConfig, accelerator: Accelerator, repo, model: nn.Module, get_pipeline, noise_sched, optimizer: torch.optim, loader, lr_sched, vae=None, lora_layers=None, start_epoch: int=0, start_step: int=0):
    weight_dtype: str = None
    scaling_factor: float = 1.0
    
    try:
        # memlog = MemoryLog('memlog.log')
        cur_step = start_step
        epoch = start_epoch
        
        loss_fn = LossFn(noise_sched=noise_sched, sde_type=config.sde_type, loss_type="l2", psi=config.psi, solver_type=config.solver_type, vp_scale=config.vp_scale, ve_scale=config.ve_scale)
        
        # Test evaluate
        # pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)        
        pipeline = get_pipeline(accelerator, model, vae, noise_sched)
        sampling(config, 0, pipeline)
        
        checkpoint(config=config, accelerator=accelerator, pipeline=pipeline, cur_epoch=epoch, cur_step=cur_step, repo=repo, commit_msg=f"Epoch {epoch}", unet=model)

        # clean_model = copy.deepcopy(model).eval()
        # Now you train the model
        for epoch in range(int(start_epoch), int(config.epoch)):
            progress_bar = tqdm(total=len(loader), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(loader):
                clean_images = batch['pixel_values']
                
                # clean_images = batch['pixel_values'].to(model.device_ids[0])
                # target_images = batch["target"].to(model.device_ids[0])
                # Sample noise to add to the images
                # noise = torch.randn(clean_images.shape).to(clean_images.device)
                bs = clean_images.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_sched.config.num_train_timesteps, (bs,), device=clean_images.device).long()

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                
                with accelerator.accumulate(model):
                    # Predict the noise residual
                    # loss = p_losses_diffuser(noise_sched, model=model, sde_type=config.sde_type, x_start=target_images, R=clean_images, timesteps=timesteps, noise=noise, loss_type="l2", psi=config.psi, solver_type=config.solver_type, vp_scale=config.vp_scale, ve_scale=config.ve_scale)
                    # loss = loss_fn.p_loss_by_keys(batch=batch, model=model, vae=None, target_latent_key="target", poison_latent_key="pixel_values", timesteps=timesteps, noise=None, weight_dtype=weight_dtype, scaling_factor=scaling_factor)
                    loss = loss_fn.p_loss_fourier_by_keys(batch=batch, model=model, vae=None, target_latent_key="target", poison_latent_key="pixel_values", timesteps=timesteps, noise=None, weight_dtype=weight_dtype, scaling_factor=scaling_factor)
                    
                    # loss = loss_fn.p_loss(model=model, x_start=target_images, R=clean_images, timesteps=timesteps, noise=noise)
                    # loss = adaptive_score_loss(noise_sched, backdoor_model=model, clean_model=clean_model, x_start=target_images, R=clean_images, timesteps=timesteps, noise=noise, loss_type="l2")
                    accelerator.backward(loss)
                    
                    # clip_grad_norm_: https://huggingface.co/docs/accelerate/v0.13.2/en/package_reference/accelerator#accelerate.Accelerator.clip_grad_norm_
                    if accelerator.sync_gradients:
                        if lora_layers is None:
                            accelerator.clip_grad_norm_(model.parameters(), 1.0)
                        else:
                            accelerator.clip_grad_norm_(lora_layers, 1.0)
                    optimizer.step()
                    lr_sched.step()
                    optimizer.zero_grad()
                # memlog.append()
                
                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "lr": lr_sched.get_last_lr()[0], "epoch": epoch, "step": cur_step}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=cur_step)
                cur_step += 1

            # After each epoch you optionally sample some demo images with evaluate() and save the model
            if accelerator.is_main_process:
                # pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)
                pipeline = get_pipeline(accelerator, model, vae, noise_sched)

                if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.epoch - 1:
                    sampling(config, epoch, pipeline)

                if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.epoch - 1:
                    checkpoint(config=config, accelerator=accelerator, pipeline=pipeline, cur_epoch=epoch, cur_step=cur_step, repo=repo, commit_msg=f"Epoch {epoch}", unet=model)
            # memlog.append()
    except:
        Log.error("Training process is interrupted by an error")
        print(traceback.format_exc())
    finally:
        pass
        # Interrupt in finally block will corrupt the checkpoint
        # Log.info("Save model and sample images")
        pipeline = get_pipeline(accelerator, model, vae, noise_sched)
        if accelerator.is_main_process:
            checkpoint(config=config, accelerator=accelerator, pipeline=pipeline, cur_epoch=epoch, cur_step=cur_step, repo=repo, commit_msg=f"Epoch {epoch}", unet=model)
            sampling(config, 'final', pipeline)
    return get_pipeline(accelerator, model, vae, noise_sched)

"""## Let's train!

Let's launch the training (including multi-GPU training) from the notebook using Accelerate's `notebook_launcher` function:
"""
dsl = get_data_loader(config=config)
accelerator, repo, model, vae, lora_layers, noise_sched, optimizer, dataloader, lr_sched, cur_epoch, cur_step, get_pipeline = init_train(config=config, dataset_loader=dsl)

if config.mode == MODE_TRAIN or config.mode == MODE_RESUME or config.mode == MODE_TRAIN_MEASURE:
    pipeline = train_loop(config, accelerator, repo, model, get_pipeline, noise_sched, optimizer, dataloader, lr_sched, vae=vae, start_epoch=cur_epoch, start_step=cur_step)

    if config.mode == MODE_TRAIN_MEASURE and accelerator.is_main_process:
        accelerator.free_memory()
        accelerator.clear()
        measure(config=config, accelerator=accelerator, dataset_loader=dsl, folder_name='measure', pipeline=pipeline)
elif config.mode == MODE_SAMPLING:
    # pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)
    pipeline = get_pipeline(accelerator, model, vae, noise_sched)
    if config.sample_ep != None:
        sampling(config=config, file_name=int(config.sample_ep), pipeline=pipeline)
    else:
        sampling(config=config, file_name="final", pipeline=pipeline)
elif config.mode == MODE_MEASURE:
    # pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)
    pipeline = get_pipeline(accelerator, model, vae, noise_sched)
    measure(config=config, accelerator=accelerator, dataset_loader=dsl, folder_name='measure', pipeline=pipeline)
    if config.sample_ep != None:
        sampling(config=config, file_name=int(config.sample_ep), pipeline=pipeline)
    else:
        sampling(config=config, file_name="final", pipeline=pipeline)
else:
    raise NotImplementedError()

accelerator.end_training()
