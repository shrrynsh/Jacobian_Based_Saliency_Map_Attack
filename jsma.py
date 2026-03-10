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



    if device is None:
        device=next(model.parameters()).device

    model.eval()
    x=x.to(device)
    x_adv=x.clone()

    num_features=x.numel()//x.shape[0]
    max_iter= int(num_features*max_distortion/2)


    with torch.no_grad()
    source_class=model.predict(x).item()


    if source_class=target_class:
        return x_adv,{
            "success": False,
            "n_iter": 0,
            "distortion":0.0,
            "source_class": source_class,
            "note": "source_class=target_class"
        }


    search_domain=set(range(num_features))

    n_iter=0,
    curretn_pred=source_class

    while current_pred != trget_class and n_iter<max_iter and len(search_domain)>=2:
        x_flat=x_adv.view(1,1,28,28)
        jacobian=compute_jacobian(model,x_flat,use_logits=True)

        if increase:
            p1,p2=saliency_map_increase(jacobian,target_class,search_domain,num_features)
        
        else:
            p1,p2=saliency_map_decrease(jacobian,target_class,search_domain,num_features)


        if p1 ==-1:
            if verbose:
                print(f" [inter{n_iter}] No valide pixel pair found")
                break

        x_adv_flat=x_adv.view(-1)

        x_adv_flat[p1]=torch.clamp(x_adv_flat[p1]+theta,clip_min,clip_max)
        x_adv_flat[p2]=torch.clamp(x_adv_flat[p2]+theta,clip_min,clip_max)
        x_adv=x_adv_flat.view_as(x_adv)



        if x_adv_flat[p1].item() <=clip_min or x_adv_flat[p1].item()>=clip_max:
            search_domain.discard(p1)


        
        if x_adv_flat[p2].item() <=clip_min or x_adv_flat[p2].item()>=clip_max:
            search_domain.discard(p2)


        n_iter+=1


        with torch.no_grad():
            current_pred=model.predict(x_adv.view(1,1,28,28)).item()


        if verbose:
            print(f"  [iter {n_iter:3d}] pred={current_pred}, target={target_class}")

    delta = (x_adv - x).view(-1)
    n_modified = (delta.abs() > 1e-6).sum().item()
    distortion = n_modified / num_features

    success = (current_pred == target_class)

    return x_adv, {
        "success": success,
        "n_iter": n_iter,
        "distortion": distortion,
        "source_class": source_class,
        "final_pred": current_pred,
    }





        








