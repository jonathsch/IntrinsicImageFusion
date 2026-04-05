import os
from iif.utils.logging import init_logger
import torch
from torch.optim import Optimizer
from torch.utils.cpp_extension import load, _import_module_from_library


def _load(name, sources):
    module_path = os.path.dirname(__file__)
    build_directory = os.path.join(module_path, "build", name)
    os.makedirs(build_directory, exist_ok=True)

    try:
        module = _import_module_from_library(name, build_directory, True)
    except ImportError:
        sources = [os.path.join(module_path, source) for source in sources]  
        module = load(
            name,
            sources,
            build_directory=build_directory,
            verbose=False,
            with_cuda=True,
            extra_cflags=["-O3"],
        )
    return module

sources=['cuda/adam_upd.cpp', 'cuda/adam_upd_kernel.cu']
adam_upd_cuda = _load('adam_upd_cuda', sources)


''' Extend Adam optimizer
masked update (ignore zero grad)
'''
class MaskedAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, **kwargs):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(MaskedAdam, self).__init__(params, defaults)

        self.module_logger = init_logger()
        if len(kwargs) > 0:
            self.module_logger.warning(f"Unrecognized arguments passed to MaskedAdam: {list(kwargs.keys())}")

    def __setstate__(self, state):
        super(MaskedAdam, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            skip_zero_grad = True
            # skip_zero_grad = group['skip_zero_grad']

            for param in group['params']:
                if param.grad is not None:
                    state = self.state[param]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)

                    state['step'] += 1

                    if skip_zero_grad:
                        adam_upd_cuda.masked_adam_upd(
                                param, param.grad, state['exp_avg'], state['exp_avg_sq'],
                                state['step'], beta1, beta2, lr, eps)
                    else:
                        adam_upd_cuda.adam_upd(
                                param, param.grad, state['exp_avg'], state['exp_avg_sq'],
                                state['step'], beta1, beta2, lr, eps)
        return loss

