# %%
import copy
from functools import partial
from os import terminal_size
from sched import scheduler
from typing import Callable, Dict, List, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from dataset import Backdoor, DEFAULT_VMIN, DEFAULT_VMAX
from model import DiffuserModelSched
# from tmp_loss_sde import q_sample_diffuser_alt_half
from util_fourier import filtered_by_freq

"""## Defining the forward diffusion process

The forward diffusion process gradually adds noise to an image from the real distribution, in a number of time steps $T$. This happens according to a **variance schedule**. The original DDPM authors employed a linear schedule:

> We set the forward process variances to constants
increasing linearly from $\beta_1 = 10^{−4}$
to $\beta_T = 0.02$.

However, it was shown in ([Nichol et al., 2021](https://arxiv.org/abs/2102.09672)) that better results can be achieved when employing a cosine schedule. 

Below, we define various schedules for the $T$ timesteps, as well as corresponding variables which we'll need, such as cumulative variances.
"""

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

class NoiseScheduler():
    SCHED_COSINE = "SC_COS"
    SCHED_LINEAR = "SC_LIN"
    SCHED_QUADRATIC = "SC_QUAD"
    SCHED_SIGMOID = "SC_SIGM"
    def __init__(self, timesteps: int, scheduler: str, s: float=0.008):
        self.__timesteps = int(timesteps)
        self.__s = float(s)
        self.__scheduler = scheduler
        
        # define beta schedule
        if self.__scheduler == self.SCHED_COSINE:
            self.__betas = NoiseScheduler.cosine_beta_schedule(timesteps=self.__timesteps, s=self.__s)
        elif self.__scheduler == self.SCHED_LINEAR:
            self.__betas = NoiseScheduler.linear_beta_schedule(timesteps=self.__timesteps)
            self.__derivative_beta = 1 / self.__timesteps
            self.__derivative_alpha = - 1 / self.__timesteps
        elif self.__scheduler == self.SCHED_QUADRATIC:
            self.__betas = NoiseScheduler.quadratic_beta_schedule(timesteps=self.__timesteps)
        elif self.__scheduler == self.SCHED_SIGMOID:
            self.__betas = NoiseScheduler.sigmoid_beta_schedule(timesteps=self.__timesteps)
        else:
            raise ImportError(f"Undefined scheduler: {self.__scheduler}")
            
        # define alphas 
        self.__alphas = 1. - self.betas
        self.__alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.__alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.__sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # Calculations for backdoor
        self.__sqrt_alphas = torch.sqrt(self.alphas)
        self.__one_minus_sqrt_alphas = 1 - self.sqrt_alphas
        self.__one_minus_alphas = 1 - self.alphas

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.__sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.__sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.__R_coef = self.one_minus_sqrt_alphas * self.sqrt_one_minus_alphas_cumprod / self.one_minus_alphas

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.__posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
    @staticmethod
    def cosine_beta_schedule(timesteps, s=0.008):
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    @staticmethod
    def linear_beta_schedule(timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps)
    @staticmethod
    def quadratic_beta_schedule(timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2
    @staticmethod
    def sigmoid_beta_schedule(timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(-6, 6, timesteps)
        return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
    @property
    def betas(self):
        return self.__betas
    @property
    def alphas(self):
        return self.__alphas
    @property
    def alphas_cumprod(self):
        return self.__alphas_cumprod
    @property
    def alphas_cumprod_prev(self):
        return self.__alphas_cumprod_prev
    @property
    def sqrt_recip_alphas(self):
        return self.__sqrt_recip_alphas
    @property
    def sqrt_alphas(self):
        return self.__sqrt_alphas
    @property
    def one_minus_sqrt_alphas(self):
        return self.__one_minus_sqrt_alphas
    @property
    def one_minus_alphas(self):
        return self.__one_minus_alphas
    @property
    def sqrt_alphas_cumprod(self):
        return self.__sqrt_alphas_cumprod
    @property
    def sqrt_one_minus_alphas_cumprod(self):
        return self.__sqrt_one_minus_alphas_cumprod
    @property
    def R_coef(self):
        return self.__R_coef
    @property
    def posterior_variance(self):
        return self.__posterior_variance

"""<img src="https://drive.google.com/uc?id=1QifsBnYiijwTqru6gur9C0qKkFYrm-lN" width="800" />
    
This means that we can now define the loss function given the model as follows:
"""

# forward diffusion
def q_sample_clean(noise_sched, x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(noise_sched.sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        noise_sched.sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise, noise

def q_sample_backdoor(noise_sched, x_start, R, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(noise_sched.sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        noise_sched.sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )
    R_coef_t = extract(noise_sched.R_coef, t, x_start.shape)

    return sqrt_alphas_cumprod_t * x_start + (1 - sqrt_alphas_cumprod_t) * R + sqrt_one_minus_alphas_cumprod_t * noise, R_coef_t * R + noise 

"""
<img src="https://drive.google.com/uc?id=1QifsBnYiijwTqru6gur9C0qKkFYrm-lN" width="800" />    
This means that we can now define the loss function given the model as follows:
"""

def p_losses_clean(noise_sched, denoise_model, x_start, t, noise=None, loss_type="l2"):
    if len(x_start) == 0: 
        return 0
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy, target = q_sample_clean(noise_sched=noise_sched, x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(target, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(target, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(target, predicted_noise)
    else:
        raise NotImplementedError()

    return loss

def p_losses_backdoor(noise_sched, denoise_model, x_start, R, t, noise=None, loss_type="l2"):
    if len(x_start) == 0: 
        return 0
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy, target = q_sample_backdoor(noise_sched=noise_sched, x_start=x_start, R=R, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(target, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(target, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(target, predicted_noise)
    else:
        raise NotImplementedError()

    return loss

def p_losses(noise_sched, denoise_model, x_start, R, is_clean, t, noise=None, loss_type="l2"):
    is_not_clean = torch.where(is_clean, False, True)
    if noise != None:
        noise_clean = noise[is_clean]
        noise_backdoor = noise[is_not_clean]
    else:
        noise_clean = noise_backdoor = noise
    loss_clean = p_losses_clean(noise_sched=noise_sched, denoise_model=denoise_model, x_start=x_start[is_clean], t=t[is_clean], noise=noise_clean, loss_type=loss_type)
    loss_backdoor = p_losses_backdoor(noise_sched=noise_sched, denoise_model=denoise_model, x_start=x_start[is_not_clean], R=R[is_not_clean], t=t[is_not_clean], noise=noise_backdoor, loss_type=loss_type)

    return (loss_clean + loss_backdoor) / 2


# ==================================================
class LossSampler():
    def __init__(self, noise_sched: NoiseScheduler):
        self.__noise_sched = noise_sched

    def get_fn(self):
        return partial(p_losses_backdoor, self.__noise_sched), partial(q_sample_backdoor, self.__noise_sched)
    
def plot(x, title: str, log_scale: bool=False):
    plt.plot(x)
    plt.title(title)
    if log_scale:
        plt.yscale("log")
    plt.show()
    
def get_derivative(x: torch.Tensor, t: int):
    
    if t + 1 < len(x):
        return x[t + 1] - x[t]
    return x[t] - x[t - 1]

def get_derivatives(x: torch.Tensor):
    x_delta_t = torch.roll(x, -1, 0)
    x_delta_t[-1] = x_delta_t[-2]
    x[-1] = x[-2]
    return x_delta_t - x

def central_derivative(fn, x, stop_thres: float=1e-5, stop_iter_n: int=50, delta: float=1e-2, divisor: float=10.0):
    der = lambda d: (fn(x + d) - fn(x - d)) / (2 * d)
    iter_n = 0
    res = der(delta)
    last_res = 0
    while (abs(res - last_res) > stop_thres or iter_n < 1) and iter_n < stop_iter_n:
        last_res = res
        delta = delta / divisor
        res = der(delta)
        iter_n = iter_n + 1
    return res

def get_alpha_beta_fn_linear(beta_start: float, beta_end: float, timesteps: int):
    def beta_fn(t):
        return float(beta_start) + (float(beta_end) - float(beta_start)) * t / (float(timesteps) - 1.0)
    def alpha_fn(t):
        return 1.0 - beta_fn(t)
    return alpha_fn, beta_fn

def integral(fn: Callable[[Union[int, float]], Union[int, float]], interval_low: float, interval_up: float, div: int=100):
    lin_space = torch.linspace(interval_low, interval_up, div, dtype=torch.float32)
    res = fn(lin_space[:-1])
    return torch.sum(res, dim=0) * (interval_up - interval_low) / div

def prod_integral(xs: torch.Tensor, x_fn: Callable[[Union[int, float]], Union[int, float]], div: int=200):
    def log_x_fn(x):
        return torch.log(x_fn(x).double()).double()
    def integral_fn(x):
        return (torch.trapezoid(log_x_fn(torch.linspace(0, x, div * int(x)).to('cpu').double())) / div).double()
    def exp_integral_fn(x):
        return torch.exp(integral_fn(x)).double()
    return torch.linspace(start=0, end=len(xs)-1, steps=len(xs)).to('cpu').double().apply_(exp_integral_fn).float()

def get_alphas_cumprod_derivative(alphas: torch.Tensor, alpha_fn: Callable[[Union[int, float]], Union[int, float]]):
    div = 200
    def log_alpha_fn(x):
        return torch.log(alpha_fn(x).double()).double()
    def integral_fn(x):
        return (torch.trapezoid(log_alpha_fn(torch.linspace(0, x, div * int(x)).to('cpu').double())) / div).double()
    def exp_integral_fn(x):
        return torch.exp(integral_fn(x)).double()
    def der_fn(x):
        return central_derivative(exp_integral_fn, x, stop_thres=1e-3, stop_iter_n=2, delta=1e-2, divisor=10.0)
    def coef_fn(x):
        return (exp_integral_fn(x) * torch.log(alpha_fn(torch.Tensor([x]).double()))).double()
    
    # fn_int = torch.linspace(start=0, end=len(alphas)-1, steps=len(alphas)).double().apply_(integral_fn)
    # fn_prod_int = torch.linspace(start=0, end=len(alphas)-1, steps=len(alphas)).double().apply_(exp_integral_fn)
    # for i in range(len(fn_prod_int[:20])):
    #     print(f"Time: {i} - Alpha Fn Product Integral Analytic: {fn_prod_int[i]}")
    # plot(fn_prod_int, title="Alpha Fn Product Integral", log_scale=True)
    # print(f"fn_int: {fn_int[:20]}")
    # plot(fn_int, title="Alpha Fn Integral")
    
    res = torch.linspace(start=0, end=len(alphas)-1, steps=len(alphas)).to('cpu').float().apply_(coef_fn).double()
    return res
    # return torch.exp(integral_res) * (torch.log(alphas[-1]) - torch.log(alphas[0]))

def get_alphas_hat_derivative(alphas_cumprod: torch.Tensor, alphas: torch.Tensor, alpha_fn: Callable[[Union[int, float]], Union[int, float]]):
    return get_alphas_cumprod_derivative(alphas=alphas, alpha_fn=alpha_fn).to(alphas_cumprod.device) / 2 * (alphas_cumprod ** 0.5)

def get_sigmas_hat_derivative(alphas_cumprod: torch.Tensor, alphas: torch.Tensor, alpha_fn: Callable[[Union[int, float]], Union[int, float]]):
    return - get_alphas_cumprod_derivative(alphas=alphas, alpha_fn=alpha_fn).to(alphas_cumprod.device) / 2 * ((1 - alphas_cumprod) ** 0.5)

def sci(x: float):
    return "{:.2e}".format(x)

def get_R_coef_alt(alphas_cumprod: torch.Tensor, alphas: torch.Tensor, alpha_fn: Callable[[Union[int, float]], Union[int, float]], psi: float=1, solver_type: str='sde'):
    one_minus_alphas_cumprod = 1 - alphas_cumprod
    
    # Fokker-Planck: g^2(t) = derivative of \hat{\beta}^2(t)
    # coef =  psi * (torch.sqrt(one_minus_alphas_cumprod / alphas_cumprod)) + (1 - psi)
    
    # g^2(t) = \frac{d \hat{\beta}^2(t)}{dt} - 2 * \frac{d \log \hat{\alpha}(t)}{dt} * \hat{\beta}^2(t)
    coef = (psi * (torch.sqrt(one_minus_alphas_cumprod / alphas_cumprod)) + (1 - psi)) / (1 + (one_minus_alphas_cumprod / alphas_cumprod))
    
    # Simplified
    # coef = torch.ones_like(alphas_cumprod)
    
    if str(solver_type).lower() == 'ode':
        return coef
    elif str(solver_type).lower() == 'sde':
        return 0.5 * coef
    else:
        raise NotImplementedError(f"Coefficient solver_type: {solver_type} isn't implemented")
    
def get_R_coef_variational(alphas_cumprod: torch.Tensor, psi: float=1, solver_type: str='sde'):
    coef = psi * (1 - alphas_cumprod ** 0.5) / (1 - alphas_cumprod) ** 0.5 + (1 - psi)
    
    if str(solver_type).lower() == 'ode':
        return 2 * coef
    elif str(solver_type).lower() == 'sde':
        return coef
    else:
        raise NotImplementedError(f"Coefficient solver_type: {solver_type} isn't implemented")

# def get_R_coef_baddiff(alphas_cumprod: torch.Tensor, psi: float=1, solver_type: str='sde'):
#     coef = psi * (1 - alphas_cumprod ** 0.5) / (1 - alphas_cumprod) ** 0.5 + (1 - psi)
    
#     if str(solver_type).lower() == 'ode':
#         return 2 * coef
#     elif str(solver_type).lower() == 'sde':
#         return coef
#     else:
#         raise NotImplementedError(f"Coefficient solver_type: {solver_type} isn't implemented")

def get_R_coef(alphas_cumprod: torch.Tensor, alphas: torch.Tensor, alpha_fn: Callable[[Union[int, float]], Union[int, float]], psi: float=1):    
    alphas_hat = (alphas_cumprod ** 0.5).double()
    sigmas_hat = ((1 - alphas_cumprod) ** 0.5).double()
    alphas_hat_derivative = get_alphas_hat_derivative(alphas_cumprod=alphas_cumprod, alphas=alphas, alpha_fn=alpha_fn).double()
    sigmas_hat_derivative = get_sigmas_hat_derivative(alphas_cumprod=alphas_cumprod, alphas=alphas, alpha_fn=alpha_fn).double()
    
    alt_r = 0.5 * alphas_hat / (alphas_hat + sigmas_hat)
    # plot(alt_r, title="Alternate R", log_scale=True)
    
    a = (- psi * alphas_hat_derivative + (1 - psi) * sigmas_hat_derivative).double()
    b = (psi * (1 - alphas_hat) + (1 - psi) * sigmas_hat).double()
    c = (2 * sigmas_hat * sigmas_hat_derivative - 2 * (alphas_hat_derivative / alphas_hat) * (sigmas_hat ** 2)).double()
    
    # plot(alpha_fn(torch.linspace(0, 999, 1000).float()), title="Alpha Fn", log_scale=True)
    # fn_cumprod = torch.cumprod(alpha_fn(torch.linspace(0, 999, 1000).float()), dim=0)
    # for i in range(len(fn_cumprod[:20])):
    #     print(f"Time: {i} - Alpha Fn Cumprod: {fn_cumprod[i]}")
    # plot(fn_cumprod, title="Alpha Fn Cumprod", log_scale=True)
    # plot(alphas, title="Alpha")
    # for i in range(len(alphas_cumprod[:20])):
    #     print(f"Time: {i} - Alpha Cumprod: {alphas_cumprod[i]}")
    # plot(alphas_cumprod, title="Alpha Cumprod", log_scale=True)
    
    # plot(get_alphas_cumprod_derivative(alphas=alphas, alpha_fn=alpha_fn), title="Alpha Cumprod Derivative Anlytic")
    # plot(get_derivatives(x=alphas_cumprod)[:-1], title="Alpha Cumprod Derivative Numeric")
    # plot(alphas_hat, title="Alpha Hat", log_scale=True)
    # plot(sigmas_hat, title="Beta Hat", log_scale=True)
    # plot(alphas_hat_derivative, title="Alpha Hat Derivative")
    # plot(sigmas_hat_derivative, title="Sigma Hat Derivative")
    # plot(a, title="Rho Derivative")
    # plot(b, title="Rho")
    # plot(c, title="G^2", log_scale=True)
    
    # plot(alphas_hat_derivative / alphas_hat, title="f(t)")
    
    coef = (sigmas_hat * a / (c)).double()
    
    # for i in range(len(sigmas_hat[:20])):
    #     print(f"Time: {i} - R: {sci(coef[i])} beta_hat: {sci(sigmas_hat[i])}, rho_deriv: {sci(a[i])}, G^2: {sci(c[i])}")
        
    if torch.isnan(sigmas_hat).any():
        print(f"sigmas_hat - Nan: {sigmas_hat[torch.isnan(sigmas_hat).nonzero()]}")
    if torch.isnan(a).any():
        print(f"Rho Derivative - Nan: {a[torch.isnan(a).nonzero()]}")
    if torch.isnan(b).any():
        print(f"Rho - Nan: {b[torch.isnan(b).nonzero()]}")
    if torch.isnan(c).any():
        print(f"G^2 - Nan: {c[torch.isnan(c).nonzero()]}")
    # return torch.clamp(coef, min=None, max=1)
    # return coef
    return alt_r

def get_ks(alphas_hat: torch.Tensor) -> torch.Tensor:
    prev_alphas_hat = torch.roll(alphas_hat, 1, 0)
    prev_alphas_hat[0] = 1
    return alphas_hat / prev_alphas_hat

def get_ws(betas_hat: torch.Tensor, ks: torch.Tensor) -> torch.Tensor:
    ws = [betas_hat[0]]
    residuals = [0]
    for i, beta_hat_i in enumerate(betas_hat):
        if i < 1:
            continue
        residuals.append((ks[i] ** 2) * (ws[i - 1] ** 2 + residuals[i - 1]))
        ws.append((beta_hat_i ** 2 - residuals[i]) ** 0.5)
    return torch.Tensor(ws)

def get_hs(rhos_hat: torch.Tensor, ks: torch.Tensor) -> torch.Tensor:
    hs = [rhos_hat[0]]
    residuals = [0]
    for i, rho_hat_i in enumerate(rhos_hat):
        if i < 1:
            continue
        residuals.append(ks[i] * (hs[i - 1]  + residuals[i - 1]))
        hs.append(rho_hat_i - residuals[i])
    return torch.Tensor(hs)

def get_ws_ve(sigmas: torch.Tensor) -> torch.Tensor:
    ws = [sigmas[0]]
    residuals = [0]
    for i, sigma_i in enumerate(sigmas):
        if i < 1:
            continue
        residuals.append(ws[i - 1] ** 2 + residuals[i - 1])
        ws.append((sigma_i ** 2 - residuals[i]) ** 0.5)
    return torch.Tensor(ws)

def get_hs_ve(rhos_hat: torch.Tensor) -> torch.Tensor:
    hs = [rhos_hat[0]]
    residuals = [0]
    for i, rho_hat_i in enumerate(rhos_hat):
        if i < 1:
            continue
        residuals.append(hs[i - 1]  + residuals[i - 1])
        hs.append(rho_hat_i - residuals[i])
    return torch.Tensor(hs)

def get_R_coef_gen_ve(sigmas: torch.Tensor, rhos_hat: torch.Tensor, 
                      ws: torch.Tensor, hs: torch.Tensor, psi: float=1, 
                      solver_type: str='sde', vp_scale: float=1.0, 
                      ve_scale: float=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    # BadDiffusion style correction term, None
    if psi != 0:
        raise NotImplementedError(f"Variance Explode model doesn't support BadDiffusion style correction term")
    
    # TrojDiff style correction term
    if hs == None:
        raise ValueError(f"Arguement hs shouldn't be {hs} when psi is {psi}")
    
    prev_rhos_hat = torch.roll(rhos_hat, 1, 0)
    prev_rhos_hat[0] = 0
    
    prev_sigmas = torch.roll(sigmas, 1, 0)
    prev_sigmas[0] = 0
    
    trojdiff_step = rhos_hat
    trojdiff_coef = ve_scale * (ws ** 2 * (rhos_hat - prev_rhos_hat) + hs * prev_sigmas) / (ws ** 2 * sigmas)
    # print(f"trojdiff_coef isnan: {torch.isnan(trojdiff_coef)}")
    
    # Coefficients & Steps
    step = trojdiff_step
    coef = trojdiff_coef
    
    if str(solver_type).lower() == 'ode':
        return step, 2 * coef
    elif str(solver_type).lower() == 'sde':
        return step, coef
    else:
        raise NotImplementedError(f"Coefficient solver_type: {solver_type} isn't implemented")
    
def get_R_coef_gen_ve_reduce(sigmas: torch.Tensor, hs: torch.Tensor, rhos_hat_w: float=1.0, psi: float=1, 
                            solver_type: str='sde', vp_scale: float=1.0,
                            ve_scale: float=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    # BadDiffusion style correction term, None
    if psi != 0:
        raise NotImplementedError(f"Variance Explode model doesn't support BadDiffusion style correction term")
    
    # TrojDiff style correction term
    if hs == None:
        raise ValueError(f"Arguement hs shouldn't be {hs} when psi is {psi}")
    
    # prev_rhos_hat = torch.roll(rhos_hat, 1, 0)
    # prev_rhos_hat[0] = 0
    
    prev_sigmas = torch.roll(sigmas, 1, 0)
    prev_sigmas[0] = 0
    
    trojdiff_step = rhos_hat_w * sigmas
    trojdiff_coef = ve_scale * (sigmas * rhos_hat_w / (sigmas + prev_sigmas))
    # print(f"trojdiff_coef isnan: {torch.isnan(trojdiff_coef)}")
    
    # Coefficients & Steps
    step = trojdiff_step
    coef = trojdiff_coef
    
    if str(solver_type).lower() == 'ode':
        return step, 2 * coef
    elif str(solver_type).lower() == 'sde':
        return step, coef
    else:
        raise NotImplementedError(f"Coefficient solver_type: {solver_type} isn't implemented")

def get_hs_vp(alphas: torch.Tensor, alphas_cumprod: torch.Tensor) -> torch.Tensor:
    hs = [(1 - alphas_cumprod[0]) ** 0.5]
    residuals = [0]
    for i, (alphas_cumprod_i, alphas_i) in enumerate(zip(alphas_cumprod, alphas)):
        if i < 1:
            continue
        residuals.append((alphas_i ** 0.5) * (hs[i - 1] + residuals[i - 1]))
        hs.append((1 - alphas_cumprod_i) ** 0.5 - residuals[i])
    return torch.Tensor(hs)

def get_R_coef_gen_vp(alphas_cumprod: torch.Tensor, alphas: torch.Tensor, 
                      hs: torch.Tensor=None, psi: float=1, solver_type: str='sde', 
                      vp_scale: float=1.0, ve_scale: float=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    # BadDiffusion style correction term
    baddiff_step = 1 - alphas_cumprod ** 0.5
    baddiff_coef = vp_scale * (1 - alphas ** 0.5) * (1 - alphas_cumprod) ** 0.5 / (1 - alphas)
    
    # TrojDiff style correction term
    if psi != 1:
        if hs == None:
            raise ValueError(f"Arhuement hs shouldn't be {hs} when psi is {psi}")
        trojdiff_step = (1 - alphas_cumprod) ** 0.5
        trojdiff_coef = - ve_scale * ((alphas ** 0.5 - 1) * (1 - alphas_cumprod) ** 0.5 * (1 - alphas) - hs * (alphas - alphas_cumprod)) / (1 - alphas)
    
        # Coefficients & Steps
        step = psi * baddiff_step + (1 - psi) * trojdiff_step
        coef = psi * baddiff_coef + (1 - psi) * trojdiff_coef
    else:
        # Coefficients & Steps
        step = baddiff_step
        coef = baddiff_coef
    
    if str(solver_type).lower() == 'ode':
        return step, 2 * coef
    elif str(solver_type).lower() == 'sde':
        return step, coef
    else:
        raise NotImplementedError(f"Coefficient solver_type: {solver_type} isn't implemented")
    
def get_R_coef_elbo_gen(noise_sched, sde_type: str="vp", psi: float=1, solver_type: str='sde', 
                        vp_scale: float=1.0, ve_scale: float=1.0, device=None, dtype=None, 
                        rhos_hat_w: float=1.0, rhos_hat_b: float=0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    if sde_type == DiffuserModelSched.SDE_VP or sde_type == DiffuserModelSched.SDE_LDM:
        if device == None:
            device = noise_sched.alphas.device
        if dtype == None:
            dtype = noise_sched.alphas.dtype
            
        alphas: torch.Tensor = noise_sched.alphas.to(device=device, dtype=dtype)
        alphas_cumprod: torch.Tensor = noise_sched.alphas_cumprod.to(device=device, dtype=dtype)
        
        # hs
        if get_R_coef_elbo_gen.hs_vp == None:
            get_R_coef_elbo_gen.hs_vp = get_hs_vp(alphas=alphas, alphas_cumprod=alphas_cumprod)
        hs: torch.Tensors = get_R_coef_elbo_gen.hs_vp.to(device=device, dtype=dtype)
        
        step, R_coef = get_R_coef_gen_vp(alphas_cumprod=alphas_cumprod, alphas=alphas, hs=hs, psi=psi, solver_type=solver_type, vp_scale=vp_scale, ve_scale=ve_scale)
    elif sde_type == DiffuserModelSched.SDE_VE:
        if device == None:
            device = noise_sched.sigmas.device
        if dtype == None:
            dtype = noise_sched.sigmas.dtype
            
        sigmas: torch.Tensor = noise_sched.sigmas.to(device=device, dtype=dtype).flip(dims=[0])
        rhos_hat: torch.Tensor = rhos_hat_w * sigmas + rhos_hat_b
        
        # ws
        if get_R_coef_elbo_gen.ws_ve == None:
            get_R_coef_elbo_gen.ws_ve = get_ws_ve(sigmas=sigmas)
        ws: torch.Tensor = get_R_coef_elbo_gen.ws_ve.to(device=device, dtype=dtype)
        # print(f"sigmas: {sigmas}")
        # print(f"sigmas isnan: {torch.isnan(sigmas).any()}: {torch.isnan(sigmas)}")
        # print(f"ws isnan: {torch.isnan(ws).any()}: {torch.isnan(ws)}")
        
        # hs
        if get_R_coef_elbo_gen.hs_ve == None:
            get_R_coef_elbo_gen.hs_ve = get_hs_ve(rhos_hat=rhos_hat)
        hs: torch.Tensor = get_R_coef_elbo_gen.hs_ve.to(device=device, dtype=dtype)
        # print(f"hs isnan: {torch.isnan(hs).any()}: {torch.isnan(hs)}")
        
        step, R_coef = get_R_coef_gen_ve(sigmas=sigmas, rhos_hat=rhos_hat, ws=ws, hs=hs, psi=psi, solver_type=solver_type, vp_scale=vp_scale, ve_scale=ve_scale)
        # R_coef = - R_coef / sigmas
        step, R_coef = step.flip(dims=[0]), R_coef.flip(dims=[0])
        # print(f"step: {torch.isnan(step).any()}, Min: {step.min()}, Max: {step.max()}: {step}")
        # print(f"R_coef: {torch.isnan(R_coef).any()}, Min: {R_coef.min()}, Max: {R_coef.max()}: {R_coef}")
    else:
        raise NotImplementedError(f"sde_type: {sde_type} isn't implemented")
    
    return step, R_coef
    
get_R_coef_elbo_gen.hs_vp: torch.Tensor = None
get_R_coef_elbo_gen.ws_ve: torch.Tensor = None
get_R_coef_elbo_gen.hs_ve: torch.Tensor = None
    
def get_R_coef_continuous(alphas_cumprod: torch.Tensor, alphas: torch.Tensor, hs: torch.Tensor=None, psi: float=1, solver_type: str='sde', vp_scale: float=1.0, ve_scale: float=1.0):
    # Variance Preserve
    vp_step = 1 - alphas_cumprod ** 0.5
    vp_coef = vp_scale * (1 - alphas_cumprod) ** 0.5 / (1 - alphas_cumprod)
    
    # Variance Explode
    if psi != 1:
        if hs == None:
            raise ValueError(f"Arhuement hs shouldn't be {hs} when psi is {psi}")
        ve_step = (1 - alphas_cumprod) ** 0.5
        ve_coef = ve_scale * 0.5
    
        # Coefficients & Steps
        step = psi * vp_step + (1 - psi) * ve_step
        coef = psi * vp_coef + (1 - psi) * ve_coef
    else:
        # Coefficients & Steps
        step = vp_step
        coef = vp_coef
    
    if str(solver_type).lower() == 'ode':
        return step, 2 * coef
    elif str(solver_type).lower() == 'sde':
        return step, coef
    else:
        raise NotImplementedError(f"Coefficient solver_type: {solver_type} isn't implemented")

def q_sample_diffuser_alt(noise_sched, sde_type: str, x_start: torch.Tensor, 
                          R: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor=None, 
                          psi: float=1, solver_type: str="sde", vp_scale: float=1.0, 
                          ve_scale: float=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    if noise is None:
        noise = torch.randn_like(x_start)
        
    def unqueeze_n(x):
        return x.reshape(len(x_start), *([1] * len(x_start.shape[1:])))  
    
    # alphas = noise_sched.alphas.to(device=x_start.device, dtype=x_start.dtype)
    # betas = noise_sched.betas.to(device=x_start.device, dtype=x_start.dtype)
    timesteps = timesteps.to(x_start.device)
    
    # Alphas Cumprod
    # if q_sample_diffuser_alt.alphas_cumprod == None:
    #     alpha_fn, beta_fn = get_alpha_beta_fn_linear(beta_start=float(betas[0]), beta_end=float(betas[-1]), timesteps=float(len(betas)))
    #     q_sample_diffuser_alt.alphas_cumprod = prod_integral(xs=alphas, x_fn=alpha_fn).to(device=x_start.device, dtype=x_start.dtype)
    # alphas_cumprod = q_sample_diffuser_alt.alphas_cumprod
    # alphas_cumprod = noise_sched.alphas_cumprod.to(device=x_start.device, dtype=x_start.dtype)
    
    # sqrt_alphas_cumprod = alphas_cumprod ** 0.5
    
    # hs
    # if q_sample_diffuser_alt.hs == None:
    #     q_sample_diffuser_alt.hs = get_hs_vp(alphas=alphas, alphas_cumprod=alphas_cumprod)
    # hs = q_sample_diffuser_alt.hs.to(device=x_start.device, dtype=x_start.dtype)
    
    # BadDiffusion
    # R_coef = (1 - alphas ** 0.5) * (1 - alphas_cumprod) ** 0.5 / (1 - alphas)
    step, R_coef = get_R_coef_elbo_gen(noise_sched=noise_sched, sde_type=sde_type, psi=psi, solver_type=solver_type, vp_scale=vp_scale, ve_scale=ve_scale, device=x_start.device, dtype=x_start.dtype)
    # step, R_coef = get_R_coef_gen_vp(alphas_cumprod=alphas_cumprod, alphas=alphas, hs=hs, psi=psi, solver_type=solver_type, vp_scale=vp_scale, ve_scale=ve_scale)
    # step, R_coef = get_R_coef_continuous(alphas_cumprod=alphas_cumprod, alphas=alphas, hs=hs, psi=psi, solver_type=solver_type, vp_scale=vp_scale, ve_scale=ve_scale)
    # plot(R_coef, title="R Coef Discrete")
    
    # Generalized
    # alpha_fn, beta_fn = get_alpha_beta_fn_linear(beta_start=float(betas[0]), beta_end=float(betas[-1]), timesteps=float(len(betas)))
    # R_coef = get_R_coef_alt(alphas_cumprod=alphas_cumprod, alphas=alphas, alpha_fn=alpha_fn, psi=psi, solver_type=solver_type)
    # plot(R_coef, title="R Coef Continuous")
    
    # Unsqueeze & Select
    R_coef_t = unqueeze_n(R_coef[timesteps])
    step_t = unqueeze_n(step[timesteps])    
    if sde_type == DiffuserModelSched.SDE_VP or sde_type == DiffuserModelSched.SDE_LDM:
        noisy_images = noise_sched.add_noise(x_start, noise, timesteps)
        return noisy_images + step_t * R, R_coef_t * R + noise 
    elif sde_type == DiffuserModelSched.SDE_VE:
        sigma_t = unqueeze_n(noise_sched.sigmas.to(timesteps.device)[timesteps])
        noisy_images = x_start + sigma_t * noise
        # noisy_images = x_start
        print(f"noisy_images: {noisy_images.shape}, {torch.isnan(noisy_images).any()}, Min: {noisy_images.min()}, Max: {noisy_images.max()}")
        print(f"R: {torch.isnan(R).any()}, Min: {R.min()}, Max: {R.max()}")
        print(f"sigma_t: {sigma_t.shape}, {torch.isnan(sigma_t).any()}, Min: {sigma_t.min()}, Max: {sigma_t.max()}")
        # return noisy_images + step_t * R, - (R_coef_t * R + noise) / sigma_t
        # return noisy_images, - (noise) / sigma_t
        return noisy_images, noise
    else:
        raise NotImplementedError(f"sde_type: {sde_type} isn't implemented")
q_sample_diffuser_alt.alphas_cumprod = None
q_sample_diffuser_alt.hs = None
    
def q_sample_diffuser(noise_sched, x_start: torch.Tensor, R: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor=None) -> torch.Tensor:
    if noise is None:
        noise = torch.randn_like(x_start)
        
    def unqueeze_n(x):
        return x.reshape(len(x_start), *([1] * len(x_start.shape[1:])))

    alphas_cumprod = noise_sched.alphas_cumprod.to(device=x_start.device, dtype=x_start.dtype)
    alphas = noise_sched.alphas.to(device=x_start.device, dtype=x_start.dtype)
    betas = noise_sched.betas.to(device=x_start.device, dtype=x_start.dtype)
    timesteps = timesteps.to(x_start.device)

    sqrt_alphas_cumprod_t = alphas_cumprod[timesteps] ** 0.5
    sqrt_one_minus_alphas_cumprod_t = (1 - alphas_cumprod[timesteps]) ** 0.5
    R_coef_t = (1 - alphas[timesteps] ** 0.5) * sqrt_one_minus_alphas_cumprod_t / (1 - alphas[timesteps])
    
    sqrt_alphas_cumprod_t = unqueeze_n(sqrt_alphas_cumprod_t)
    
    # NOTE: BadDiffusion
    # R_coef = (1 - alphas ** 0.5) * (1 - alphas_cumprod) ** 0.5 / (1 - alphas)
    # plot(R_coef, title="R Coef", log_scale=True)
    R_coef_t = unqueeze_n(R_coef_t)
    
    noisy_images = noise_sched.add_noise(x_start, noise, timesteps)
    
    # if q_sample_diffuser.R_coef == None:
    #     # NOTE: Generalized BadDiffusion
    #     alpha_fn, beta_fn = get_alpha_beta_fn_linear(beta_start=float(betas[0]), beta_end=float(betas[-1]), timesteps=float(len(betas)))
    #     # q_sample_diffuser.R_coef = torch.flip(get_R_coef(alphas_cumprod=alphas_cumprod, alphas=alphas, alpha_fn=alpha_fn, psi=1), dims=(0,))
    #     q_sample_diffuser.R_coef = get_R_coef_alt(alphas_cumprod=alphas_cumprod, alphas=alphas, alpha_fn=alpha_fn, psi=1).float()
    # R_coef_t = unqueeze_n(q_sample_diffuser.R_coef[timesteps])
    # # plot(q_sample_diffuser.R_coef, title="R Coef", log_scale=True)
    # if torch.isnan(R_coef_t).any():
    #     print(f"Nan: {timesteps[torch.isnan(R_coef_t).nonzero()]}")
    return noisy_images + (1 - sqrt_alphas_cumprod_t) * R, R_coef_t * R + noise 

q_sample_diffuser.R_coef = None

def p_losses_diffuser(noise_sched, model: nn.Module, sde_type: str, x_start: torch.Tensor, R: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor=None, loss_type: str="l2", psi: float=1, solver_type: str="sde", vp_scale: float=1.0, ve_scale: float=1.0) -> torch.Tensor:
    if len(x_start) == 0: 
        return 0
    if noise is None:
        noise = torch.randn_like(x_start)
    noise = noise.clamp(-2, 2)
        
    def unqueeze_n(x):
        return x.reshape(len(x_start), *([1] * len(x_start.shape[1:])))  
    
    # if sde_type == DiffuserModelSched.SDE_VE:
    #     x_start = x_start / 2 + 0.5
    #     R = R / 2 + 0.5

    # Main loss function
    x_noisy, target = q_sample_diffuser_alt(noise_sched=noise_sched, sde_type=sde_type, x_start=x_start, R=R, timesteps=timesteps, noise=noise, psi=psi, solver_type=solver_type, vp_scale=vp_scale, ve_scale=ve_scale)
    
    # Additiolnal loss function
    # x_noisy_half, target_half = q_sample_diffuser_alt_half(noise_sched=noise_sched, x_start=x_start, R=R, timesteps=timesteps, noise=noise)
    # predicted_noise_half = model(x_noisy_half.contiguous(), timesteps.contiguous(), return_dict=False)[0]
    
    if sde_type == DiffuserModelSched.SDE_VP or sde_type == DiffuserModelSched.SDE_LDM:
        predicted_noise = model(x_noisy.contiguous(), timesteps.contiguous(), return_dict=False)[0]
        print(f"x_noisy: {x_noisy.shape}, {torch.isnan(x_noisy).any()}, min: {x_noisy.min()}, max: {x_noisy.max()}")
        print(f"predicted_noise: {predicted_noise.shape}, {torch.isnan(predicted_noise).any()}, min: {predicted_noise.min()}, max: {predicted_noise.max()}")
        
        if loss_type == 'l1':
            loss: torch.Tensor = F.l1_loss(target, predicted_noise, reduction='none')
        elif loss_type == 'l2':
            loss = F.mse_loss(target, predicted_noise, reduction='none')
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(target, predicted_noise, reduction='none')
        else:
            raise NotImplementedError()
        return loss.mean()
    elif sde_type == DiffuserModelSched.SDE_VE:
        sigma_t = noise_sched.sigmas.unsqueeze(0).to(timesteps.device)[timesteps]
        predicted_noise = model(x_noisy.contiguous(), sigma_t.contiguous(), return_dict=False)[0]
        print(f"x_noisy: {x_noisy.shape}, {torch.isnan(x_noisy).any()}, min: {x_noisy.min()}, max: {x_noisy.max()}")
        print(f"predicted_noise: {predicted_noise.shape}, {torch.isnan(predicted_noise).any()}, min: {predicted_noise.min()}, max: {predicted_noise.max()}")
        
        if loss_type == 'l1':
            loss: torch.Tensor = F.l1_loss(target, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(target, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(target, predicted_noise)
        else:
            raise NotImplementedError()
        # return (loss * unqueeze_n(noise_sched.sigmas.to(timesteps.device)[timesteps]) ** 2).mean()
        return loss
    else:
        raise NotImplementedError(f"sde_type: {sde_type} isn't implemented")

class LossFn:
    RANDN_BOUND: float = 2.5
    def __init__(self, noise_sched, sde_type: str, loss_type: str="l2", psi: float=1, solver_type: str="sde", vp_scale: float=1.0, ve_scale: float=1.0, rhos_hat_w: float=1.0, rhos_hat_b: float=0.0):
        self.__noise_sched = noise_sched
        if sde_type == DiffuserModelSched.SDE_VP or sde_type == DiffuserModelSched.SDE_LDM:
            self.__alphas: torch.Tensor = self.__noise_sched.alphas
            self.__alphas_cumprod: torch.Tensor = self.__noise_sched.alphas_cumprod
            self.__betas: torch.Tensor = self.__noise_sched.betas
        if sde_type == DiffuserModelSched.SDE_VE:
            self.__sigmas: torch.Tensor = self.__noise_sched.sigmas.flip([0])
            
        self.__sde_type = sde_type
        self.__loss_type = loss_type
        self.__psi = psi
        self.__solver_type = solver_type
        self.__vp_scale = vp_scale
        self.__ve_scale = ve_scale
        self.__rhos_hat_w = rhos_hat_w
        self.__rhos_hat_b = rhos_hat_b
        
        self.__hs_vp: torch.Tensor = None
        self.__ws_ve: torch.Tensor = None
        self.__hs_ve: torch.Tensor = None
        
    def __norm(self):
        reduction = 'none'
        if self.__loss_type == 'l1':
            return partial(F.l1_loss, reduction=reduction)
        elif self.__loss_type == 'l2':
            return partial(F.mse_loss, reduction=reduction)
        elif self.__loss_type == "huber":
            return partial(F.smooth_l1_loss, reduction=reduction)
        else:
            raise NotImplementedError()
        
    def __get_R_step_coef(self, device=None, dtype=None):
        if self.__sde_type == DiffuserModelSched.SDE_VP or self.__sde_type == DiffuserModelSched.SDE_LDM:
            if device == None:
                device = self.__alphas.device
            if dtype == None:
                dtype = self.__alphas.dtype
            
            alphas: torch.Tensor = self.__alphas.to(device=device, dtype=dtype)
            alphas_cumprod: torch.Tensor = self.__alphas_cumprod.to(device=device, dtype=dtype)
            betas: torch.Tensor = self.__betas.to(device=device, dtype=dtype)
            
            # hs
            if self.__hs_vp == None:
                self.__hs_vp = get_hs_vp(alphas=alphas, alphas_cumprod=alphas_cumprod)
            hs: torch.Tensors = self.__hs_vp.to(device=device, dtype=dtype)
            
            step, R_coef = get_R_coef_gen_vp(alphas_cumprod=alphas_cumprod, alphas=alphas, hs=hs, psi=self.__psi, solver_type=self.__solver_type, vp_scale=self.__vp_scale, ve_scale=self.__ve_scale)
        elif self.__sde_type == DiffuserModelSched.SDE_VE:
            if device == None:
                device = self.__sigmas.device
            if dtype == None:
                dtype = self.__sigmas.dtype
                
            sigmas: torch.Tensor = self.__sigmas.to(device=device, dtype=dtype)
            rhos_hat: torch.Tensor = self.__rhos_hat_w * sigmas + self.__rhos_hat_b
            
            # ws
            if self.__ws_ve == None:
                self.__ws_ve = get_ws_ve(sigmas=sigmas)
            ws: torch.Tensor = self.__ws_ve.to(device=device, dtype=dtype)
            # print(f"sigmas: {sigmas}")
            # print(f"sigmas isnan: {torch.isnan(sigmas).any()}: {torch.isnan(sigmas)}")
            # print(f"ws isnan: {torch.isnan(ws).any()}: {torch.isnan(ws)}")
            
            # hs
            if self.__hs_ve == None:
                self.__hs_ve = get_hs_ve(rhos_hat=rhos_hat)
            hs: torch.Tensor = self.__hs_ve.to(device=device, dtype=dtype)
            # print(f"hs isnan: {torch.isnan(hs).any()}: {torch.isnan(hs)}")
            
            # step, R_coef = get_R_coef_gen_ve(sigmas=sigmas, rhos_hat=rhos_hat, ws=ws, hs=hs, psi=self.__psi, solver_type=self.__solver_type, vp_scale=self.__vp_scale, ve_scale=self.__ve_scale)
            step, R_coef = get_R_coef_gen_ve_reduce(sigmas=sigmas, hs=hs, rhos_hat_w=self.__rhos_hat_w, psi=self.__psi, solver_type=self.__solver_type, vp_scale=self.__vp_scale, ve_scale=self.__ve_scale)
            # print(f"step: {torch.isnan(step).any()}, Min: {step.min()}, Max: {step.max()}: {step}")
            # print(f"R_coef: {torch.isnan(R_coef).any()}, Min: {R_coef.min()}, Max: {R_coef.max()}: {R_coef}")
        else:
            raise NotImplementedError(f"sde_type: {self.__sde_type} isn't implemented")
        
        return step, R_coef
        
    def __get_inputs_targets(self, x_start: torch.Tensor, R: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor):
        # if noise is None:
        #     noise = torch.randn_like(x_start)
            
        def unqueeze_n(x):
            return x.reshape(len(x_start), *([1] * len(x_start.shape[1:])))  
        
        timesteps = timesteps.to(x_start.device)
        step, R_coef = self.__get_R_step_coef(device=x_start.device, dtype=x_start.dtype)
        
        # Unsqueeze & Select
        R_coef_t = unqueeze_n(R_coef[timesteps])
        step_t = unqueeze_n(step[timesteps])
        
        if self.__sde_type == DiffuserModelSched.SDE_VP or self.__sde_type == DiffuserModelSched.SDE_LDM:
            noisy_images = self.__noise_sched.add_noise(x_start, noise, timesteps)
            return noisy_images + step_t * R, R_coef_t * R + noise 
        elif self.__sde_type == DiffuserModelSched.SDE_VE:
            sigma_t = unqueeze_n(self.__sigmas.to(timesteps.device)[timesteps])
            noisy_images = x_start + sigma_t * noise
            # noisy_images = x_start
            # print(f"step_t: {step_t.shape}, Min: {step_t.min()}, Max: {step_t.max()}")
            # print(f"R_coef_t: {R_coef_t.shape}, Min: {R_coef_t.min()}, Max: {R_coef_t.max()}")
            return noisy_images + step_t * R, R_coef_t * R + noise
            
            # print(f"noisy_images: {noisy_images.shape}, {torch.isnan(noisy_images).any()}, Min: {noisy_images.min()}, Max: {noisy_images.max()}")
            # print(f"R: {torch.isnan(R).any()}, Min: {R.min()}, Max: {R.max()}")
            # No likelihood_weighting
            # return noisy_images, noise
        else:
            raise NotImplementedError(f"sde_type: {self.__sde_type} isn't implemented")
        
    @staticmethod
    def __encode_latents(vae, x: torch.Tensor, weight_dtype: str=None, scaling_factor: float=None):
        vae = vae.eval()
        with torch.no_grad():
            x = x.to(vae.device)
            if weight_dtype != None and weight_dtype != "":
                x = x.to(dtype=weight_dtype)
            if scaling_factor != None:
                return (vae.encode(x).latents * scaling_factor).clone().detach()
            # return vae.encode(x).latents * vae.config.scaling_factor
            return vae.encode(x).latents.clone().detach()
    @staticmethod
    def __decode_latents(vae, x: torch.Tensor, weight_dtype: str=None, scaling_factor: float=None):
        vae = vae.eval()
        with torch.no_grad():
            x = x.to(vae.device)
            if weight_dtype != None and weight_dtype != "":
                x = x.to(dtype=weight_dtype)
            if scaling_factor != None:
                return (vae.decode(x).sample / scaling_factor).clone().detach()
            # return vae.decode(x).sample / vae.config.scaling_factor
            return (vae.decode(x).sample).clone().detach()
    @staticmethod
    def __get_latent(batch, key: str, vae=None, weight_dtype: str=None, scaling_factor: float=None) -> torch.Tensor:
        if vae == None:
            return batch[key]
        return LossFn.__encode_latents(vae=vae, x=batch[key], weight_dtype=weight_dtype, scaling_factor=scaling_factor)
    @staticmethod
    def __get_latents(batch, keys: List[str], vae=None, weight_dtype: str=None, scaling_factor: float=None) -> List[torch.Tensor]:
        return [LossFn.__get_latent(batch=batch, vae=vae, key=key, weight_dtype=weight_dtype, scaling_factor=scaling_factor) for key in keys]
    
    def p_loss_by_keys(self, batch, model: nn.Module, target_latent_key: torch.Tensor, poison_latent_key: torch.Tensor, 
                       timesteps: torch.Tensor, vae=None, noise: torch.Tensor=None, weight_dtype: str=None, scaling_factor: float=None) -> torch.Tensor:        
        target_latents, poison_latents = LossFn.__get_latents(batch=batch, keys=[target_latent_key, poison_latent_key], vae=vae, weight_dtype=weight_dtype, scaling_factor=scaling_factor)
        
        return self.p_loss(model=model, x_start=target_latents, R=poison_latents, timesteps=timesteps, noise=noise)
        
    def p_loss(self, model: nn.Module, x_start: torch.Tensor, R: torch.Tensor, 
               timesteps: torch.Tensor, noise: torch.Tensor=None) -> torch.Tensor:
        if len(x_start) == 0: 
            return 0
        if noise is None:
            noise = torch.randn_like(x_start)
        # noise = noise.clamp(-LossFn.RANDN_BOUND, LossFn.RANDN_BOUND)
            
        def unqueeze_n(x):
            return x.reshape(len(x_start), *([1] * len(x_start.shape[1:])))  
            
        # Main loss function
        x_noisy, target = self.__get_inputs_targets(x_start=x_start, R=R, timesteps=timesteps, noise=noise)
        
        if self.__sde_type == DiffuserModelSched.SDE_VP or self.__sde_type == DiffuserModelSched.SDE_LDM:
            predicted_noise = model(x_noisy.contiguous(), timesteps.contiguous(), return_dict=False)[0]
            loss: torch.Tensor = self.__norm()(target=target, input=predicted_noise)
            return loss.mean()
        elif self.__sde_type == DiffuserModelSched.SDE_VE:
            sigmas_t: torch.Tensor = self.__sigmas.to(timesteps.device)[timesteps]
            predicted_noise = model(x_noisy.contiguous(), sigmas_t.contiguous(), return_dict=False)[0]
            
            # print(f"x_noisy: {x_noisy.shape}, {torch.isnan(x_noisy).any()}, min: {x_noisy.min()}, max: {x_noisy.max()}")
            # print(f"predicted_noise: {predicted_noise.shape}, {torch.isnan(predicted_noise).any()}, min: {predicted_noise.min()}, max: {predicted_noise.max()}")
            
            loss: torch.Tensor = self.__norm()(target=target, input=- predicted_noise * unqueeze_n(sigmas_t))
            return loss.mean()
        else:
            raise NotImplementedError(f"sde_type: {self.__sde_type} isn't implemented")
        
    def p_loss_fourier_by_keys(self, batch, model: nn.Module, target_latent_key: torch.Tensor, poison_latent_key: torch.Tensor, timesteps: torch.Tensor, vae=None, noise: torch.Tensor=None, 
                               weight_dtype: str=None, scaling_factor: float=None, thres: Union[int, float]=0.3, pass_type: str='high') -> torch.Tensor:
        target_latents, poison_latents = LossFn.__get_latents(batch=batch, keys=[target_latent_key, poison_latent_key], vae=vae, weight_dtype=weight_dtype, scaling_factor=scaling_factor)
        
        return self.p_loss_fourier(model=model, x_start=target_latents, R=poison_latents, timesteps=timesteps, noise=noise, thres=thres, pass_type=pass_type)
        
    def p_loss_fourier(self, model: nn.Module, x_start: torch.Tensor, R: torch.Tensor, 
               timesteps: torch.Tensor, noise: torch.Tensor=None, thres: Union[int, float]=0.3, pass_type: str='high') -> torch.Tensor:
        if len(x_start) == 0: 
            return 0
        if noise is None:
            noise = torch.randn_like(x_start)
        # noise = noise.clamp(-LossFn.RANDN_BOUND, LossFn.RANDN_BOUND)
            
        def unqueeze_n(x):
            return x.reshape(len(x_start), *([1] * len(x_start.shape[1:])))  
            
        
        # Main loss function
        x_noisy, target = self.__get_inputs_targets(x_start=x_start, R=R, timesteps=timesteps, noise=noise)
        
        p = partial(filtered_by_freq, dim=list(range(len(x_noisy.shape)))[-2:], in_diamiter=0, out_diamiter=x_noisy.shape[-2] * thres)
        
        if self.__sde_type == DiffuserModelSched.SDE_VP or self.__sde_type == DiffuserModelSched.SDE_LDM:
            predicted_noise = model(x_noisy.contiguous(), timesteps.contiguous(), return_dict=False)[0]
            loss: torch.Tensor = self.__norm()(target=p(target), input=p(predicted_noise))
            return loss.mean()
        elif self.__sde_type == DiffuserModelSched.SDE_VE:
            sigmas_t: torch.Tensor = self.__sigmas.to(timesteps.device)[timesteps]
            predicted_noise = model(x_noisy.contiguous(), sigmas_t.contiguous(), return_dict=False)[0]
            
            # print(f"x_noisy: {x_noisy.shape}, {torch.isnan(x_noisy).any()}, min: {x_noisy.min()}, max: {x_noisy.max()}")
            # print(f"predicted_noise: {predicted_noise.shape}, {torch.isnan(predicted_noise).any()}, min: {predicted_noise.min()}, max: {predicted_noise.max()}")
            
            loss: torch.Tensor = self.__norm()(target=p(target), input=p(- predicted_noise * unqueeze_n(sigmas_t)))
            return loss.mean()
        else:
            raise NotImplementedError(f"sde_type: {self.__sde_type} isn't implemented")

def adaptive_score_loss(noise_sched, backdoor_model: nn.Module, clean_model: torch.nn.Module, x_start: torch.Tensor, R: torch.Tensor, timesteps: torch.Tensor, psi: float=0, noise: torch.Tensor=None, loss_type: str="l2", backprop_depth: int=2, timesteps_num: int=1000) -> torch.Tensor:
    if timesteps_num - backprop_depth < 0:
        raise ValueError(f"backprop_depth should <= timesteps_num")
    if noise is None:
        noise = torch.randn_like(x_start)
        
    def unqueeze_n(x):
        return x.reshape(len(x), *([1] * len(x_start.shape[1:])))
    
    # Set up model mode
    backdoor_model = backdoor_model.train()
    clean_model = clean_model.eval()
    
    alphas = noise_sched.alphas.to(device=x_start.device, dtype=x_start.dtype)
    betas = noise_sched.betas.to(device=x_start.device, dtype=x_start.dtype)
    timesteps = torch.clamp(timesteps, max=timesteps_num - backprop_depth - 1).to(x_start.device)
    
    if adaptive_score_loss.alphas_cumprod_derivative == None:
        alpha_fn, beta_fn = get_alpha_beta_fn_linear(beta_start=float(betas[0]), beta_end=float(betas[-1]), timesteps=float(len(betas)))
        adaptive_score_loss.alphas_cumprod_derivative = get_alphas_cumprod_derivative(alphas=alphas, alpha_fn=alpha_fn).to(device=x_start.device, dtype=x_start.dtype)
        adaptive_score_loss.alphas_cumprod = noise_sched.alphas_cumprod.to(device=x_start.device, dtype=x_start.dtype)
        # adaptive_score_loss.alphas_cumprod = prod_integral(xs=alphas, x_fn=alpha_fn).to(device=x_start.device, dtype=x_start.dtype)
        
    alphas_cumprod_derivative = adaptive_score_loss.alphas_cumprod_derivative
    alphas_cumprod = adaptive_score_loss.alphas_cumprod
    
    def ode_x_k_t(model: torch.nn.Module, xs: torch.Tensor, rs: torch.Tensor, k: int, ts: torch.Tensor, f_func, h_func, g_square_func, sigma_func, delta: float=1e-6) -> torch.Tensor:
        # with torch.no_grad():
        if k == 0:
            return xs
        prev_ode_x_k_t: torch.Tensor = ode_x_k_t(model=model, xs=xs, rs=rs, k=k - 1, ts=ts - 1, f_func=f_func, h_func=h_func, g_square_func=g_square_func, sigma_func=sigma_func)
        pred = model(prev_ode_x_k_t.contiguous(), (ts - 1).contiguous(), return_dict=False)[0]
        
        if torch.isnan(xs).any():
            print(f"[{k}] xs: Nan")
        if torch.isnan(pred).any():
            print(f"[{k}] ode pred: Nan")
        if torch.isnan(prev_ode_x_k_t).any():
            print(f"[{k}] prev_ode_x_k_t: Nan")
            
        return prev_ode_x_k_t - (f_func[k] * prev_ode_x_k_t + h_func[k] * rs + g_square_func[k] / (2 * sigma_func[k] + delta) * pred)
    
    def sde_x_k_t(model: torch.nn.Module, xs: torch.Tensor, rs: torch.Tensor, k: int, u: float, ts: torch.Tensor, f_func, h_func, g_square_func, sigma_func, rand: bool=True, delta: float=1e-6) -> torch.Tensor:
        if k == 0:
            return xs
        prev_sde_x_k_t: torch.Tensor = sde_x_k_t(model=model, xs=xs, rs=rs, k=k - 1, u=u, ts=ts - 1, f_func=f_func, h_func=h_func, g_square_func=g_square_func, sigma_func=sigma_func, rand=True)
        pred = model(prev_sde_x_k_t.contiguous(), (ts - 1).contiguous(), return_dict=True)[0]
        
        if torch.isnan(xs).any():
            print(f"[{k}] xs: Nan")
        if torch.isnan(pred).any():
            print(f"[{k}] sde pred: Nan")
        if torch.isnan(prev_sde_x_k_t).any():
            print(f"[{k}] prev_sde_x_k_t: Nan")
            
        if rand:
            return prev_sde_x_k_t - (f_func[k] * prev_sde_x_k_t + g_square_func[k] * (u + 1) / (2 * sigma_func[k] + delta) * pred + torch.sqrt(g_square_func[k]) * u * torch.randn_like(xs))
        else:
            return prev_sde_x_k_t - (f_func[k] * prev_sde_x_k_t + g_square_func[k] * (u + 1) / (2 * sigma_func[k] + delta) * pred)
        
    def func_t_dict_gen(func: torch.Tensor, k: int, timesteps: torch.Tensor):
        funcs: Dict[int, torch.Tensor] = {}
        for i in range(1, k + 1):
            funcs[k - i + 1] = unqueeze_n(func[timesteps + i])
        return funcs
        
    # Functions used in the expansion
    f_func= 1 / (2 * alphas_cumprod) * alphas_cumprod_derivative
    # g_square_func = - alphas_cumprod_derivative
    g_square_func = - alphas_cumprod_derivative / alphas_cumprod
    sigma_func = torch.sqrt(1 - alphas_cumprod)
    h_func = - psi * (alphas_cumprod_derivative / (2 * torch.sqrt(alphas_cumprod))) - (1 - psi) * (alphas_cumprod_derivative / (2 * torch.sqrt(1 - alphas_cumprod)))
    
    # print(f"sigma_func min: {sigma_func.min()}")
    if torch.isnan(f_func).any():
        print(f"f_func: Nan")
    if torch.isnan(g_square_func).any():
        print(f"g_square_func: Nan")
    if torch.isnan(sigma_func).any():
        print(f"sigma_func: Nan")
    if torch.isnan(h_func).any():
        print(f"h_func: Nan")
    
    f_func_dict = func_t_dict_gen(f_func, k=backprop_depth, timesteps=timesteps)
    g_square_func_dict = func_t_dict_gen(g_square_func, k=backprop_depth, timesteps=timesteps)
    sigma_func_dict = func_t_dict_gen(sigma_func, k=backprop_depth, timesteps=timesteps)
    h_func_dict = func_t_dict_gen(h_func, k=backprop_depth, timesteps=timesteps)
    
    # ODE ground truth and SDE prediction
    x_noisy = noise_sched.add_noise(x_start, noise, timesteps + backprop_depth)
    target_x_k_t = ode_x_k_t(model=clean_model, xs=x_noisy, rs=R, k=backprop_depth, ts=timesteps, f_func=f_func_dict, h_func=h_func_dict, g_square_func=g_square_func_dict, sigma_func=sigma_func_dict)
    pred_x_k_t = sde_x_k_t(model=backdoor_model, xs=x_noisy, rs=R, k=backprop_depth, u=1, ts=timesteps, f_func=f_func_dict, h_func=h_func_dict, g_square_func=g_square_func_dict, sigma_func=sigma_func_dict, rand=False)
    
    if torch.isnan(x_start).any():
        print(f"x_start: Nan")
    if torch.isnan(R).any():
        print(f"R: Nan")
    if torch.isnan(target_x_k_t).any():
        print(f"target_x_k_t: Nan")
    if torch.isnan(pred_x_k_t).any():
        print(f"pred_x_k_t: Nan")
    
    if loss_type == 'l1':
        loss = F.l1_loss(target_x_k_t, pred_x_k_t)
    elif loss_type == 'l2':
        loss = F.mse_loss(target_x_k_t, pred_x_k_t)
        if torch.isnan(loss):
            print(f"loss: Nan")
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(target_x_k_t, pred_x_k_t)
    else:
        raise NotImplementedError()

    return loss
adaptive_score_loss.alphas_cumprod_derivative = None
adaptive_score_loss.alphas_cumprod = None

# %%
if __name__ == '__main__':
    import os
    
    from diffusers import DDPMScheduler
    
    from dataset import DatasetLoader
    from model import DiffuserModelSched
    
    time_step = 95
    num_train_timesteps = 100
    # time_step = 140
    # num_train_timesteps = 150
    ds_root = os.path.join('datasets')
    dsl = DatasetLoader(root=ds_root, name=DatasetLoader.CELEBA_HQ).set_poison(trigger_type=Backdoor.TRIGGER_GLASSES, target_type=Backdoor.TARGET_CAT, clean_rate=1, poison_rate=0.2).prepare_dataset()
    print(f"Full Dataset Len: {len(dsl)}")
    image_size = dsl.image_size
    channels = dsl.channel
    ds = dsl.get_dataset()
    
    """
    # CIFAR10
    # sample = ds[50000]
    # MNIST
    # sample = ds[60000]
    # CelebA-HQ
    # sample = ds[10000]
    sample = ds[24000]
    
    target = torch.unsqueeze(sample[DatasetLoader.TARGET], dim=0)
    source = torch.unsqueeze(sample[DatasetLoader.PIXEL_VALUES], dim=0)
    bs = len(source)
    model, noise_sched = DiffuserModelSched.get_model_sched(image_size=image_size, channels=channels, model_type=DiffuserModelSched.MODEL_DEFAULT)
    
    print(f"bs: {bs}")
    
    # Sample a random timestep for each image
    # mx_timestep = noise_sched.num_train_timesteps
    # timesteps = torch.randint(0, mx_timestep, (bs,), device=source.device).long()
    timesteps = torch.tensor([time_step] * bs, device=source.device).long()
    
    
    print(f"target Shape: {target.shape}")
    dsl.show_sample(img=target[0])
    print(f"source Shape: {source.shape}")
    dsl.show_sample(img=source[0])
    
    noise = torch.randn_like(target)
    noise_sched = DDPMScheduler(num_train_timesteps=num_train_timesteps)
    noisy_images = noise_sched.add_noise(source, noise, timesteps)
    
    noisy_x, target_x = q_sample_diffuser(noise_sched, x_start=target, R=source, timesteps=timesteps, noise=noise)
    
    # noise_sched_b = NoiseScheduler(timesteps=num_train_timesteps, scheduler=NoiseScheduler.SCHED_LINEAR)
    # noisy_x_b, target_x_b = q_sample_backdoor(noise_sched_b, x_start=target, R=source, t=timesteps, noise=noise)
    
    # print(f"noisy_x == noisy_x: {torch.all(noisy_x == noisy_x_b)}")
    # print(f"target_x == target_x_b: {torch.all(target_x == target_x_b)}")
    
    # print(f"noisy_x_b Shape: {noisy_x_b.shape}")
    # dsl.show_sample(img=noisy_x_b[0], vmin=torch.min(noisy_x_b), vmax=torch.max(noisy_x_b))
    # print(f"target_x_b Shape: {target_x_b.shape}")
    # dsl.show_sample(img=target_x_b[0], vmin=torch.min(target_x_b), vmax=torch.max(target_x_b))
    
    print(f"target_x Shape: {target_x.shape}")
    dsl.show_sample(img=target_x[0], vmin=torch.min(target_x), vmax=torch.max(target_x))
    print(f"noisy_x Shape: {noisy_x.shape}")
    dsl.show_sample(img=noisy_x[0], vmin=torch.min(noisy_x), vmax=torch.max(noisy_x))
    print(f"source Shape: {source.shape}")
    dsl.show_sample(img=source[0], vmin=torch.min(source), vmax=torch.max(source))
    diff = (noisy_x - source)
    print(f"noisy_x - source Shape: {diff.shape}")
    dsl.show_sample(img=diff[0], vmin=torch.min(diff), vmax=torch.max(diff))
    diff = (target_x - noise)
    print(f"target_x - noise Shape: {diff.shape}")
    dsl.show_sample(img=diff[0], vmin=torch.min(diff), vmax=torch.max(diff))
    
    print(f"noisy_images Shape: {noisy_images.shape}")
    dsl.show_sample(img=noisy_images[0], vmin=torch.min(noisy_images), vmax=torch.max(noisy_images))
    diff_x = noisy_x - noisy_images
    print(f"noisy_x - noisy_images Shape: {diff_x.shape}")
    dsl.show_sample(img=diff_x[0], vmin=torch.min(diff_x), vmax=torch.max(diff_x))
    
    diff_x = noisy_x - target_x
    print(f"noisy_x - target_x Shape: {diff_x.shape}")
    dsl.show_sample(img=diff_x[0], vmin=torch.min(diff_x), vmax=torch.max(diff_x))# %%
    """

    noise_sched = DDPMScheduler(num_train_timesteps=1000)
    sample = ds[24000:25000]
    target = sample[DatasetLoader.TARGET]
    source = sample[DatasetLoader.PIXEL_VALUES]
    noise = torch.randn_like(target)
    print(f"target shape: {target.shape}, source shape: {source.shape}")
    timesteps = torch.linspace(0, 999, 1000, dtype=torch.long)
    noisy_x, target_x = q_sample_diffuser_alt(noise_sched, x_start=target, R=source, timesteps=timesteps, noise=noise)
    loss = torch.sum((noisy_x - target_x) ** 2, dim=(1, 2, 3))
    print(f"Loss shape: {loss.shape}")
    
    plt.plot(loss)
    plt.yscale("log")
    plt.show()
# %%
