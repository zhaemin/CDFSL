import torch
import math
from torch.optim.lr_scheduler import _LRScheduler

'''
last_epoch -> -1에서부터 시작해서 step 호출마다 1씩 증가 / -1인 경우 현재 optimizer의 각 param_group의 'lr' 값을 initial_lr로 사용 -> 특정 epoch부터 다시 시작하고 싶을 때 사용자가 지정 가능
_step_count -> step()이 불린 횟수, 사용자가 지정 못함
'''

class WarmupCosineAnnealingScheduler(_LRScheduler):
    def __init__(self, optimizer: torch.optim, warmup_steps: int, base_lr: float, last_epoch: int = -1, T_max=100, eta_min=1e-3):
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.T_max = T_max
        self.eta_min = eta_min
        super(WarmupCosineAnnealingScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Warm up
            return [self.base_lr * (self.last_epoch + 1) / self.warmup_steps for _ in self.optimizer.param_groups]
        else:
            # cosine annealing warmup - pytorch
            if self.last_epoch == self.warmup_steps:
                return [group["lr"] for group in self.optimizer.param_groups]
            
            # 특정 epoch에서 초기화
            elif self._step_count == 1 and self.last_epoch > 0:
                return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos((self.last_epoch) * math.pi / self.T_max)) / 2 for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)]
            
            # T_max 반복 주기를 다 돌았다면
            elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
                return [group["lr"] + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2 for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)]
            
            return [(1 + math.cos(math.pi * self.last_epoch / self.T_max)) / (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) * (group["lr"] - self.eta_min) + self.eta_min for group in self.optimizer.param_groups]
