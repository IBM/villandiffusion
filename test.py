import os

import torch 

from diffusers import UNet2DModel, VQModel, DDPMScheduler, DDIMScheduler, DPMSolverMultistepScheduler, UniPCMultistepScheduler, PNDMScheduler, DEISMultistepScheduler, HeunDiscreteScheduler, LMSDiscreteScheduler, ScoreSdeVeScheduler, KarrasVeScheduler, DiffusionPipeline, DDPMPipeline, DDIMPipeline, PNDMPipeline, ScoreSdeVePipeline, LDMPipeline, KarrasVePipeline
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.utils.torch_utils import is_compiled_module
from accelerate import Accelerator
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

class TrainingConfig:
    mixed_precision: str = 'fp16'
    gradient_accumulation_steps: int = 1
    output_dir: str = 'test'

def get_accelerator(config: TrainingConfig):
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps, 
        log_with=["tensorboard", "wandb"],
        # log_with="tensorboard",
        # project_dir=os.path.join(config.output_dir, "logs")
    )
    return accelerator

def __get_lora_layers(unet, rank: int, alpha: int):
    for param in unet.parameters():
        param.requires_grad_(False)
        
    unet_lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    unet.add_adapter(unet_lora_config)
    lora_layers = list(filter(lambda p: p.requires_grad, unet.parameters()))
    print(f"Added LoRA Layers")
    print(f"unet.parameters(): {unet.parameters()}")
    for i, p in enumerate(unet.parameters()):
        print(f"[{i}] {p.requires_grad}, {p}")
    return unet, lora_layers

def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def save_lora(pipeline, unet, accelerator, save_path: str):
    unwrapped_unet = unwrap_model(pipeline.unet, accelerator)
    unet_lora_state_dict = convert_state_dict_to_diffusers(
        get_peft_model_state_dict(unwrapped_unet)
    )

    DDPMPipeline.save_lora_weights(
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

if __name__ == "__main__":
    ckpt_id: str = "google/ddpm-ema-celebahq-256"
    rank: int = 128
    alpha: int = 128
    save_path: str = "test"
    
    config = TrainingConfig()
    accelerator = get_accelerator(config=config)
    
    pipeline: DDPMPipeline = DDPMPipeline.from_pretrained(ckpt_id)
    
    unet = pipeline.unet
    
    lora_layers = None
    unet, lora_layers = __get_lora_layers(unet=unet, rank=rank, alpha=alpha)
    cast_training_params(unet, dtype=torch.float32)
    
    unet = torch.nn.DataParallel(unet, device_ids=[1,2])
    
    unet = accelerator.prepare(unet)
    os.makedirs(save_path, exist_ok=True)
    save_lora(pipeline=pipeline, unet=unet, accelerator=accelerator, save_path=save_path)
    
    print(f"lora_layers {lora_layers}")
    
    optimizer = torch.optim.Adam(lora_layers, lr=0.001)