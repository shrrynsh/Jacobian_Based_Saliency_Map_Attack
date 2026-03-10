from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from typing import Optional ,Tuple


def compute_jacobian(
    model : nn.Module,
    x: torch.Tensor,
    use_logits : bool=True,
) -> torch.Tensor:

    x=x.clone().detach().requires_grad_(True)

    if use_logits:
        outputs=model.logits(x)
    else:
        outputs=model(x)

    
    num_classes=outputs.shape[1]
    num_features=x.numel()//x.shape[0]

    jacobian=torch.zeros(num_classes,num_features)

    for j in range(num_classes):
        if x.grad is not None:
            x.grad.zero_()
        outputs[0,j].backward(retain_graph=(j<num_classes-1))
        jacobian[j]=x.grad.view(-1).clone
    
    return jacobian


def saliency_map_increase(
    jacobian :torch.Tensor,
    target : int,
    search_domain : set,
    num_features : int,
) -> Tuple[int,int];

    num_classes=jacobian.shape[0]

    target_grad=jacobian[target]

    other_grad=jcobian.sum(dim=0)-target_grad

    domain=sorted(search_domain)
    n=len(domain)

    best_score=-1.0
    best_p1,best_p2=-1,-1

    for idx_i in range(n):
        p=domain[idx_i]
        for idx_j in range(idx_i+1,n):
            q=domain[idx_j]

            alpha=target_grad[p]+target_grad[q]
            beta=other_grad[p]+other_grad[q]

            if alpha >0 and beta < 0:
                score= alpha*(-beta)
                if score>bet_score:
                    best_score=score
                    best_p1,best_p2=p,q
                    
    return best_p1,best_p2



def saliency_map_decrease(
    jacobian :torch.Tensor,
    target: int,
    search_domain:set,
    num_features:int,
) -> Tuple[int,int]:

    num_classes=jacobian.shape[0]

    target_grad=jacobian[target]

    other_grad=jcobian.sum(dim=0)-target_grad

    domain=sorted(search_domain)
    n=len(domain)

    best_score=-1.0
    best_p1,best_p2=-1,-1

    for idx_i in range(n):
        p=domain[idx_i]
        for idx_j in range(idx_i+1,n):
            q=domain[idx_j]

            alpha=target_grad[p]+target_grad[q]
            beta=other_grad[p]+other_grad[q]

            if alpha <0 and beta >0 0:
                score= -alpha*(beta)
                if score>best_score:
                    best_score=score
                    best_p1,best_p2=p,q

    return best_p1,best_p2


def jsma_attack(
    model : nn.Module,
    x: torch.Tensor,
    target_class: int,
    thetha : float,
    max_distortion: float = 0.145
    increase : bool=True,
    clip_min : float=0.0,
    clip_max : float : 1.0,

    device : Optional[torch.device]=None,
    verbose : bool=False,

) -> Tuple[torch.tensor,dict]:

