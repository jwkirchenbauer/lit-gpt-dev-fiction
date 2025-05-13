# type: ignore
import torch
from torch.optim import Optimizer
from torch import Tensor


from typing import List, Optional, Tuple, Union, Dict
from math import sqrt
from functools import partial
import torch.distributed as dist

from .slim_adamw import SlimAdamW


def get_param_groups(named_parameters, no_weight_decay_for_bias_and_norm_params=True, no_wd_on_embedding=False):
    param_groups = []
    if no_weight_decay_for_bias_and_norm_params or no_wd_on_embedding:
        wd_params = []
        no_wd_params = []

        for name, param in named_parameters:
            no_wd = False
            if no_weight_decay_for_bias_and_norm_params and ("norm" in name.lower() or "bias" in name.lower()):
                no_wd = True
            if no_wd_on_embedding and ("wte" in name.lower() or "embedding" in name.lower()):
                no_wd = True

            if no_wd:
                no_wd_params.append(param)
            else:
                wd_params.append(param)

        if wd_params:
            param_groups.append({"params": wd_params})
        if no_wd_params:
            param_groups.append({"params": no_wd_params, "weight_decay": 0.0})
    else:
        param_groups.append({"params": [p for _, p in named_parameters]})
    return param_groups


def get_optimizer(
    optimizer_name,
    model=None,
    pytorch_optimizer_sharding: bool = False,
    allow_fusion: bool = True,
    use_apex_adamw: bool = False,
):
    if hasattr(torch.optim, optimizer_name):
        optim_class = getattr(torch.optim, optimizer_name)  # read all torch optimizers
    elif optimizer_name == "LionW":
        optim_class = LionW
    elif optimizer_name == "SophiaG":
        optim_class = SophiaG
    elif optimizer_name == "ELLISAdam":
        optim_class = ELLISAdam
    elif optimizer_name == "SlimAdamW":
        optim_class = partial(SlimAdamW, model_object=model)
    else:
        raise ValueError(f"Invalid optimizer {optimizer_name} requested.")

    if optimizer_name == "AdamW" and use_apex_adamw:
        from apex.optimizers import FusedAdam

        optim_class = FusedAdam
        print("Using apex.optimizers.FusedAdam")

    if allow_fusion:
        import inspect

        if "fused" in inspect.signature(optim_class).parameters:
            # llm.c trick to fish for fused implementations
            optim_class = partial(optim_class, fused=True)

    if pytorch_optimizer_sharding and torch.distributed.is_initialized():
        # Zero-1 is technically unsupported in modern pytorch, test with caution ...
        # we could also replace this with an explicit torch.compile(optimizer) pass, but that would not shard afaik
        # this mode also does not interact nicely with --model_telemetry=True
        from torch.distributed.optim import ZeroRedundancyOptimizer

        return partial(ZeroRedundancyOptimizer, optimizer_class=optim_class, overlap_with_ddp=False)
    else:
        return optim_class


class LionW(Optimizer):
    """
    Adapted from https://github.com/google/automl/blob/master/lion/lion_pytorch.py
    and further modified from https://github.com/allenai/OLMo/blob/829f1d69d001b67a8a9845cc75c9a5edc8432d29/olmo/optim.py
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        **kwargs,
    ):
        assert lr > 0.0
        assert all(0.0 <= beta <= 1.0 for beta in betas)
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
        for group in self.param_groups:
            group["initial_lr"] = group["lr"]
        self._update_total_dot_prod: Optional[torch.Tensor] = None
        self._update_total_norm: Optional[torch.Tensor] = None
        self._signed_update_total_norm: Optional[torch.Tensor] = None

    def get_post_step_metrics(
        self, module: torch.nn.Module, process_group: Optional[dist.ProcessGroup] = None
    ) -> Dict[str, torch.Tensor]:
        update_total_dot_prod = self._update_total_dot_prod
        update_total_norm = self._update_total_norm
        signed_update_total_norm = self._signed_update_total_norm
        if update_total_dot_prod is None or update_total_norm is None or signed_update_total_norm is None:
            return {}

        # if is_distributed() and isinstance(module, FullyShardedDataParallel):
        #     # Reduce total dot prod and norms across all ranks.
        #     update_total_norm = update_total_norm**2.0
        #     signed_update_total_norm = signed_update_total_norm**2.0
        #     # Reduce all together to avoid multiple communication calls.
        #     all_together = torch.stack([update_total_dot_prod, update_total_norm, signed_update_total_norm])
        #     # Only need the final result on rank0, since that's where we log from.
        #     dist.reduce(
        #         all_together,
        #         0 if process_group is None else dist.get_global_rank(process_group, 0),
        #         group=process_group,
        #     )
        #     update_total_dot_prod, update_total_norm, signed_update_total_norm = all_together
        #     update_total_norm = update_total_norm**0.5
        #     signed_update_total_norm = signed_update_total_norm**0.5

        update_cos_sim = update_total_dot_prod / torch.max(
            update_total_norm * signed_update_total_norm, torch.tensor(1e-8, device=torch.device("cuda"))
        )
        return {"update_cos_sim": update_cos_sim}

    @torch.no_grad()
    def step(self, closure=None) -> None:
        if closure is not None:
            with torch.enable_grad():
                closure()

        update_total_dot_prod = torch.tensor(0.0, dtype=torch.float32)
        update_norms = []
        signed_update_norms = []

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Perform step weight decay
                p.data.mul_(1 - group["lr"] * group["weight_decay"])

                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]

                # Weight update
                update = exp_avg * beta1 + grad * (1 - beta1)
                signed_update = torch.sign(update)
                p.add_(signed_update, alpha=-group["lr"])

                # Decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

                # Track dot product and norms of update vs signed update in order to calculate
                # their cosine similarity.
                update_total_dot_prod = update_total_dot_prod.to(update.device)
                update_total_dot_prod += torch.tensordot(update, signed_update, dims=len(update.shape))
                update_norms.append(torch.linalg.vector_norm(update, 2.0, dtype=torch.float32))
                signed_update_norms.append(torch.linalg.vector_norm(signed_update, 2.0, dtype=torch.float32))

        # Compute cosine similarity between update and signed update.
        self._update_total_dot_prod = update_total_dot_prod.to(torch.device("cuda"))
        self._update_total_norm = torch.linalg.vector_norm(
            torch.stack(update_norms),
            2.0,
            dtype=torch.float32,
        ).to(torch.device("cuda"))
        self._signed_update_total_norm = torch.linalg.vector_norm(
            torch.stack(signed_update_norms),
            2.0,
            dtype=torch.float32,
        ).to(torch.device("cuda"))


"""
# Sophia would require this training loop
for epoch in range(epochs):
    for X, Y in data_loader:
        # standard training code
        logits, loss = model(X, Y)
        loss.backward()
        optimizer.step(bs=bs)
        optimizer.zero_grad(set_to_none=True)
        iter_num += 1

        if iter_num % k != k - 1:
            continue
        else:
            # update hessian EMA
            logits, _ = model(X, None)
            samp_dist = torch.distributions.Categorical(logits=logits)
            y_sample = samp_dist.sample()
            loss_sampled = F.cross_entropy(logits.view(-1, logits.size(-1)), y_sample.view(-1), ignore_index=-1)
            loss_sampled.backward()
            optimizer.update_hessian()
            optimizer.zero_grad(set_to_none=True)
            model.zero_grad()
"""


# stolen from https://github.com/Liuhong99/Sophia/blob/main/sophia.py
class SophiaG(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-4,
        betas=(0.965, 0.99),
        rho=0.04,
        weight_decay=1e-1,
        *,
        maximize: bool = False,
        capturable: bool = False,
        **kwargs,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= rho:
            raise ValueError("Invalid rho parameter at index 1: {}".format(rho))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(
            lr=lr, betas=betas, rho=rho, weight_decay=weight_decay, maximize=maximize, capturable=capturable
        )
        super(SophiaG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("capturable", False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]["step"])
        if not step_is_tensor:
            for s in state_values:
                s["step"] = torch.tensor(float(s["step"]))

    @torch.no_grad()
    def update_hessian(self):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = (
                        torch.zeros((1,), dtype=torch.float, device=p.device)
                        if self.defaults["capturable"]
                        else torch.tensor(0.0)
                    )
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["hessian"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                if "hessian" not in state.keys():
                    state["hessian"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state["hessian"].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

    @torch.no_grad()
    def step(self, closure=None, bs=5120):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            state_steps = []
            hessian = []
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                params_with_grad.append(p)

                if p.grad.is_sparse:
                    raise RuntimeError("Hero does not support sparse gradients")
                grads.append(p.grad)
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = (
                        torch.zeros((1,), dtype=torch.float, device=p.device)
                        if self.defaults["capturable"]
                        else torch.tensor(0.0)
                    )
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["hessian"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                if "hessian" not in state.keys():
                    state["hessian"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state["exp_avg"])
                state_steps.append(state["step"])
                hessian.append(state["hessian"])

                if self.defaults["capturable"]:
                    bs = torch.ones((1,), dtype=torch.float, device=p.device) * bs

            sophiag(
                params_with_grad,
                grads,
                exp_avgs,
                hessian,
                state_steps,
                bs=bs,
                beta1=beta1,
                beta2=beta2,
                rho=group["rho"],
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                maximize=group["maximize"],
                capturable=group["capturable"],
            )

        return loss


def sophiag(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    hessian: List[Tensor],
    state_steps: List[Tensor],
    capturable: bool = False,
    *,
    bs: int,
    beta1: float,
    beta2: float,
    rho: float,
    lr: float,
    weight_decay: float,
    maximize: bool,
    **kwargs,
):

    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

    func = _single_tensor_sophiag

    func(
        params,
        grads,
        exp_avgs,
        hessian,
        state_steps,
        bs=bs,
        beta1=beta1,
        beta2=beta2,
        rho=rho,
        lr=lr,
        weight_decay=weight_decay,
        maximize=maximize,
        capturable=capturable,
    )


def _single_tensor_sophiag(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    hessian: List[Tensor],
    state_steps: List[Tensor],
    *,
    bs: int,
    beta1: float,
    beta2: float,
    rho: float,
    lr: float,
    weight_decay: float,
    maximize: bool,
    capturable: bool,
):

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        hess = hessian[i]
        step_t = state_steps[i]

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            hess = torch.view_as_real(hess)
            param = torch.view_as_real(param)

        # update step
        step_t += 1

        # Perform stepweight decay
        param.mul_(1 - lr * weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

        if capturable:
            step_size = lr
            step_size_neg = step_size.neg()

            ratio = (exp_avg.abs() / (rho * bs * hess + 1e-15)).clamp(None, 1)
            param.addcmul_(exp_avg.sign(), ratio, value=step_size_neg)
        else:
            step_size_neg = -lr

            ratio = (exp_avg.abs() / (rho * bs * hess + 1e-15)).clamp(None, 1)
            param.addcmul_(exp_avg.sign(), ratio, value=step_size_neg)


def _parse_str_to_dtype(string_rep: str):
    if "bf16" in string_rep:
        return torch.bfloat16
    elif "f16" in string_rep or "fp16" in string_rep:
        return torch.float16
    else:
        return torch.float32


# an apple cobbler of many sources
class ELLISAdam(Optimizer):
    def __init__(
        self,
        params,
        lr: Union[float, Tensor] = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-6,
        weight_decay: float = 1e-2,
        *,
        foreach: Optional[bool] = None,
        nesterov: bool = False,
        eps_adjustment: bool = False,
        update_clipping: bool = False,
        kahan_sum_compensation: bool = False,
        buffer_dtype: Optional[Union[torch.dtype, str]] = None,  # can be torch.float16 / torch.bfloat16
        running_init: bool = False,
        tensor_wise_finite_check: bool = False,
        tensor_wise_gradient_normalization: bool = False,
        adafactor_like_beta_corrections: bool = False,
        atan_adam: bool = False,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            foreach=foreach,
            nesterov=nesterov,
            eps_adjustment=eps_adjustment,
            update_clipping=update_clipping,
            kahan_sum_compensation=kahan_sum_compensation,
            buffer_dtype=_parse_str_to_dtype(buffer_dtype) if isinstance(buffer_dtype, str) else buffer_dtype,
            running_init=running_init,
            tensor_wise_finite_check=tensor_wise_finite_check,
            tensor_wise_gradient_normalization=tensor_wise_gradient_normalization,
            adafactor_like_beta_corrections=adafactor_like_beta_corrections,
            atan_adam=atan_adam,
        )
        if foreach:
            raise ValueError("Todo: reinstate a foreach version, minimizing additional mem alloc")
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("foreach", None)
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):
                    step_val = float(p_state["step"])
                    p_state["step"] = torch.tensor(step_val, dtype=torch.float32)

    @torch.no_grad()
    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        kahan_comps,
        running_init: bool = False,
        buffer_dtype=None,
        kahan_sum_compensation: bool = False,
        tensor_wise_gradient_normalization: bool = False,
    ):
        for p in group["params"]:
            if p.grad is None:
                continue
            params_with_grad.append(p)
            grads.append(p.grad)

            state = self.state[p]
            _tensor_constructors = dict(memory_format=torch.preserve_format)
            if buffer_dtype is not None:
                _tensor_constructors["dtype"] = buffer_dtype

            # State initialization
            if len(state) == 0:
                # note(crcrpar): Deliberately host `step` on CPU if both capturable and fused are off.
                # This is because kernel launches are costly on CUDA and XLA.
                state["step"] = 0  # torch.tensor(0.0, dtype=torch.float32)

                if kahan_sum_compensation:
                    state["kahan_comps"] = torch.zeros_like(p, **_tensor_constructors)
                else:
                    state["kahan_comps"] = None
                if running_init:
                    grad = p.grad if not tensor_wise_gradient_normalization else p.grad / p.grad.norm()
                    # Exponential moving average of gradient values
                    state["exp_avg"] = grad.clone().to(**_tensor_constructors)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = grad.pow(2).clone().to(**_tensor_constructors)
                else:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p, **_tensor_constructors)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p, **_tensor_constructors)

            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            state_steps.append(state["step"])
            kahan_comps.append(state["kahan_comps"])

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            kahan_comps = []
            beta1, beta2 = group["betas"]

            self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                kahan_comps,
                running_init=group["running_init"],
                kahan_sum_compensation=group["kahan_sum_compensation"],
                buffer_dtype=group["buffer_dtype"],
                tensor_wise_gradient_normalization=group["tensor_wise_gradient_normalization"],
            )
            _single_tensor_modded_adamw(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                kahan_comps,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                nesterov=group["nesterov"],
                eps_adjustment=group["eps_adjustment"],
                update_clipping=group["update_clipping"],
                kahan_sum_compensation=group["kahan_sum_compensation"],
                buffer_dtype=group["buffer_dtype"],
                tensor_wise_finite_check=group["tensor_wise_finite_check"],
                tensor_wise_gradient_normalization=group["tensor_wise_gradient_normalization"],
                adafactor_like_beta_corrections=group["adafactor_like_beta_corrections"],
                atan_adam=group["atan_adam"],
            )

        return loss


def _single_tensor_modded_adamw(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    kahan_comps: List[Tensor],
    *,
    beta1: float,
    beta2: float,
    lr: Union[Tensor, float],
    weight_decay: float,
    eps: float,
    nesterov: bool = False,
    eps_adjustment: bool = False,
    update_clipping: bool = False,
    kahan_sum_compensation: bool = False,
    buffer_dtype=Optional[torch.dtype],
    tensor_wise_finite_check: bool = False,
    tensor_wise_gradient_normalization: bool = False,
    adafactor_like_beta_corrections: bool = False,
    atan_adam: bool = False,
):
    if adafactor_like_beta_corrections:
        # update group step
        step_t = state_steps[0]  # crime
        step_t += 1
        beta1 = (beta1**step_t - beta1) / (beta1**step_t - 1)
        beta2 = (beta2**step_t - beta2) / (beta2**step_t - 1)

    if nesterov:
        alpha = 2 * (1 - beta1) - (1 - beta1) ** 2  # only for nesterov to fuse the two lerps

    for i, param in enumerate(params):
        grad = grads[i].to(buffer_dtype)
        if tensor_wise_finite_check:
            if (~torch.isfinite(grad)).sum() > 0:
                continue

        if tensor_wise_gradient_normalization:
            grad = grad / grad.norm()
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]
        kahan_comp = kahan_comps[i]

        # Decay the first and second moment running average coefficient
        if nesterov:
            # Only difference between NAdamW and AdamW in this implementation.
            # The official PyTorch implementation of NAdam uses a different algorithm.
            # We undo these ops later on, which could cause numerical issues but saves
            # us from having to make an extra copy of the gradients.
            exp_avg.lerp_(grad, alpha)
        else:
            exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        step_size = lr.clone() if isinstance(lr, torch.Tensor) else lr

        if update_clipping:
            rms = grad.pow(2).div_(exp_avg_sq.clamp_(min=eps**2)).mean().sqrt()  # impl like optimi
            step_size = step_size / rms.clamp(min=1.0)

        if not adafactor_like_beta_corrections:
            step_t += 1
            bias_correction1 = 1 - beta1**step_t
            bias_correction2 = 1 - beta2**step_t
            bias_correction2_sqrt = sqrt(bias_correction2)

            step_size = step_size / bias_correction1
        else:
            bias_correction2 = 1.0
            bias_correction2_sqrt = 1.0

        # Actual adam step
        if kahan_sum_compensation:
            # Perform stepweight decay
            kahan_comp.mul_(1 - lr * weight_decay)
            if atan_adam:
                # a = b = 1
                kahan_comp.add_(torch.atan2(exp_avg, exp_avg_sq.sqrt().div_(bias_correction2_sqrt)), alpha=-step_size)
            elif eps_adjustment:
                kahan_comp.addcdiv_(exp_avg, exp_avg_sq.div(bias_correction2).add_(eps**2).sqrt(), value=-step_size)
            else:
                kahan_comp.addcdiv_(exp_avg, exp_avg_sq.sqrt().div_(bias_correction2_sqrt).add_(eps), value=-step_size)
            # update weights with kahan compensation using grad as temp buffer
            grad.copy_(param.detach())
            param.add_(kahan_comp)
            # save error back to kahan compensation for next iteration
            kahan_comp.add_(grad.sub_(param))
        else:
            # Perform stepweight decay
            param.mul_(1 - lr * weight_decay)
            if atan_adam:
                param.add_(torch.atan2(exp_avg, exp_avg_sq.sqrt().div_(bias_correction2_sqrt)), alpha=-step_size)
            elif eps_adjustment:
                param.addcdiv_(exp_avg, exp_avg_sq.div(bias_correction2).add_(eps**2).sqrt(), value=-step_size)
            else:
                param.addcdiv_(exp_avg, exp_avg_sq.sqrt().div_(bias_correction2_sqrt).add_(eps), value=-step_size)

        # undo nadam
        if nesterov:
            exp_avg.lerp_(grad, 1 - 1 / beta1)
