import math
import torch
from torch.optim.optimizer import Optimizer
import torch.distributed as dist


def print_rank_0(msg):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(msg)


class SlimAdamW(Optimizer):
    """Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_

    .. _Adam: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(
        self, params, model_object, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, foreach=False
    ):
        assert not amsgrad
        assert not foreach
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

        super(SlimAdamW, self).__init__(params, defaults)
        self.linear_params = set()
        for module in model_object.modules():
            if isinstance(module, torch.nn.Linear):
                if module.weight.requires_grad:
                    self.linear_params.add(module.weight)

    def __setstate__(self, state):
        super(SlimAdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data.mul_(1 - group["lr"] * group["weight_decay"])
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")
                amsgrad = group["amsgrad"]

                state = self.state[p]
                is_linear = p in self.linear_params

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    if not is_linear:
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(p.data)
                    else:
                        state["exp_avg_sq"] = torch.zeros((p.shape[0],), device=p.device, dtype=p.dtype)
                        print_rank_0(
                            f"SlimAdamW: Detected linear module of shape {p.shape}, only maintaining second moment of shape {p.shape[0]}"
                        )

                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        raise NotImplementedError
                        state["max_exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # if group['weight_decay'] != 0:
                #     grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                exp_avg.mul_(beta1).add_(alpha=1 - beta1, other=grad)
                if not is_linear:
                    exp_avg_sq.mul_(beta2).addcmul_(value=1 - beta2, tensor1=grad, tensor2=grad)
                else:
                    sq_grad_1d = torch.mean(grad**2, dim=1)
                    exp_avg_sq.mul_(beta2).add_(alpha=1 - beta2, other=sq_grad_1d)
                    del sq_grad_1d

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                if not is_linear:
                    p.data.addcdiv_(value=-step_size, tensor1=exp_avg, tensor2=denom)
                else:
                    p.data.addcdiv_(value=-step_size, tensor1=exp_avg, tensor2=denom.unsqueeze(1))

        return loss
