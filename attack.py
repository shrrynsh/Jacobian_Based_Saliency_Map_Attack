import argparse
import os 
import time
import json 
import numpy as np 
import torch
from torch.utils.data import DataLoader,Subset
from torchvision import dataset,transforms
from tqdm import tqdm

from model import LenNet5, load_model
from jsma import JSMAAttack

def get_args():
    p = argparse.ArgumentParser(description="Run JSMA attack on MNIST")
    p.add_argument("--model_path",     type=str,   default="./checkpoints/lenet_mnist.pth")
    p.add_argument("--data_dir",       type=str,   default="./data")
    p.add_argument("--save_dir",       type=str,   default="./results")
    p.add_argument("--n_samples",      type=int,   default=100,
                   help="Number of source samples to attack")
    p.add_argument("--max_distortion", type=float, default=0.145,
                   help="Max distortion Υ (paper: 0.145 = 14.5%%)")
    p.add_argument("--theta",          type=float, default=1.0,
                   help="Pixel intensity change per iteration (paper: +1)")
    p.add_argument("--strategy",       type=str,   default="increase",
                   choices=["increase", "decrease"],
                   help="Saliency map strategy")
    p.add_argument("--source_class",   type=int,   default=None,
                   help="Attack only this source class (default: all)")
    p.add_argument("--target_class",   type=int,   default=None,
                   help="Attack only this target class (default: all)")
    p.add_argument("--device",         type=str,   default=None)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--verbose",        action="store_true")
    return p.parse_args()



def load_test_data(data_dir: str, n_samples : int, source_class: int=None,seed: int=42):
    transform=transofrms.Compose([transforms.toTensor()])
    dataset=datsets.MNIST(root=data_dir,train=False,download=True,transform=transform)

    if source_class is not None:
        indices=[i for i,(_,y) in enumerate(dataset) if y==source_class]
        np.random.seed(seed)
        indices=np.random.choice(indices,min(n_samples,len(indices)),replace=False).tolist()

    else:
        np.random.seed(seed)
        indices=np.random.choice(len(dataset),min(n_smaples,len(dataset)),replace=False).tolist()


    subset=Subset(dataset,indices)
    loader=DataLoader(subset,batch_size=1,shuffle=False)
    return loader



class AttackResults:
    def __init__(self):
        self.records=[]

    def add(self,source:int,target:int,success:bool,distortion:float,n_iter:int):
        self.records.append({
            "source":source,
            "target":target,
            "success":success,
            "distortion":distortion,
            "n_iter":n_iter,

        })
        

    def summary(self) -> dict:
        if not self.records:
            return {}

        total=len(self.records)
        successes=[r for r in self.records if r['success']]
        n_success=len(successes)

        success_rate=100.0*n_success/total
        avg_distortion_all=100.0*np.mean([r['distortion'] for r in self.records])
        avg_distortion_success=(
            100.0*np.mean([r['distortion'] for r in success])
            if successes else 0.0
        )
        

        return {
            "total_attacks": total,
            "n_success":n_success,
            "success_rate_pct":round(success_rate,2)
            "avg_distortion_all_pct":round(avg_distortion_pct,2)
            "avg_distortion_success_pct":round(avg_distortion_success,2)
        }


    def per_class_summary(self) -> dict:
        """Compute success rate and avg distortion for each (source, target) pair."""
        from collections import defaultdict
        pairs = defaultdict(list)
        for r in self.records:
            pairs[(r["source"], r["target"])].append(r)

        result = {}
        for (s, t), recs in pairs.items():
            successes = [r for r in recs if r["success"]]
            result[(s, t)] = {
                "n": len(recs),
                "n_success": len(successes),
                "success_rate": len(successes) / len(recs) if recs else 0,
                "avg_distortion_success": (
                    np.mean([r["distortion"] for r in successes])
                    if successes else 0.0
                ),
            }
        return result

    def to_numpy_matrices(self, num_classes: int = 10):
        """
        Build success rate and distortion matrices (num_classes × num_classes).
        Matches Figure 12 and Figure 13 in the paper.
        """
        pair_data = self.per_class_summary()
        success_matrix = np.zeros((num_classes, num_classes))
        distortion_matrix = np.zeros((num_classes, num_classes))

        for (s, t), data in pair_data.items():
            success_matrix[s, t] = data["success_rate"]
            distortion_matrix[s, t] = data["avg_distortion_success"] * 100  # percent

        return success_matrix, distortion_matrix

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "records": self.records,
                "summary": self.summary(),
            }, f, indent=2)
        print(f"Results saved to {path}")

    @classmethod
    def load(cls, path: str) -> "AttackResults":
        with open(path) as f:
            data = json.load(f)
        obj = cls()
        obj.records = data["records"]
        return obj


def run_attack(args):

    if args.device:
        device=torch.device(args.device)

    else:
        device=torch.device("cuda" if torch.cuda.is_available else "cpu")
    print(f"Device :{device}")


    print{f"Loading model from {args.model_path}"}
    model=load_model(args.model_path,device)


    transform=transforms.Compose([transforms.ToTensor()])
    test_data=datasets.MNIST(root=args.data_dir,train=False,doenload=True,transform=transform)
    loader_check=DataLoader(Subset(test_data,list(range(1000))),batch_size=500)
    correct=0
    total=0


    with torch.no_grad():
        for imgs,labels in loader_check:
            imgs,labels=imgs.to(deivce),labels.to(device)
            preds=model.predict(imgs)
            correct+=preds.eq(labels).sum().item()
            total+=imgs.size(0)

    
    print(f"Model accuracy (1k samples) : {100*correct/total:.2f}%")


    attack=JSMAAttack(
        model=model,
        theta=args.theta,
        max_distortion=args.max_distortion,
        increase=(args.strategy=="increase"),
        device=device,
        
    )

    loader=load_test_data(args.data_dir,args.n_samples,args.source_class,args.seed)
    num_classes=10


    results=AttackResults()
    os.makedirs(args.save_dir,exist_ok=True)

    target_classes=(
        [args.target_class] if args.target_class is not None
        else list(range(num_classes))
        )


    total_attacks=len(loader)*len([t for t in target_classes])
    print(f"\nRunning JSMA attack:")
    print(f"  Strategy:       {args.strategy} (θ={args.theta})")
    print(f"  Max distortion: {args.max_distortion*100:.1f}%")
    print(f"  Source samples: {len(loader.dataset)}")
    print(f"  Target classes: {target_classes}")
    print(f"  Total attacks: {total_attacks:,}")



    start_time=time.time()
    attack_count=0

    pbar=tqdm(total=total_attacks,desc='Crafting adversarial samples')

    for imgs,labels in loader:
        img=imgs[0].to(device)
        sorce=labels[0].item()

        pred_source=model.predict(img.unsueeze(0)).item()
        if pred_source!= source:
            for t in target_classes:
                pbar.update(1)
            continue

        for target in target_classes:
            if target==source:
                pbar.update(1)
                continue

            x_adv,stats=attack.craft(img.unsqueeze(0),target_class=target,verbose=args.verbose)

            results.add(
                source=source,
                target=target,
                success=stats["success"],
                distortion=stats['distortion'],
                n_iter=stats["n_iter"]
            )

            attck_count+=1
            pbar.update(1)

            if args.verbose:
                status='success' if stats['success'] else "fail"
                print(
                    f" {status} {source} ->{target}:"
                    f"distortion ={stats['distortion']*100:.1f}%,"
                    f"iters={stats['n_iter']}"
                    
                )
    pbar.close()

    elapsed = time.time() - start_time
    summary = results.summary()

    # Print results
    print(f"\n{'='*60}")
    print("ATTACK RESULTS")
    print(f"{'='*60}")
    print(f"Total attacks:          {summary['total_attacks']:,}")
    print(f"Successful attacks:     {summary['n_success']:,}")
    print(f"Success rate (τ):       {summary['success_rate_pct']:.2f}%")
    print(f"Avg distortion (all):   {summary['avg_distortion_all_pct']:.2f}%")
    print(f"Avg distortion (ε):     {summary['avg_distortion_success_pct']:.2f}%")
    print(f"Time elapsed:           {elapsed:.1f}s")
    print(f"Time per attack:        {elapsed/max(attack_count,1):.2f}s")
    print()
    print("Paper targets: τ=97.10%, ε=4.02%")
    print(f"{'='*60}")

    # Save results
    results_path = os.path.join(
        args.save_dir,
        f"results_{args.strategy}_{args.n_samples}samples.json"
    )
    results.save(results_path)

    # Print per-class breakdown
    print("\nPer-class success rates (rows=source, cols=target):")
    success_matrix, distortion_matrix = results.to_numpy_matrices()
    np.save(os.path.join(args.save_dir, "success_matrix.npy"), success_matrix)
    np.save(os.path.join(args.save_dir, "distortion_matrix.npy"), distortion_matrix)

    # Print success matrix
    header = "    " + "  ".join(f"{t:4d}" for t in range(10))
    print(header)
    for s in range(10):
        row = f"{s:2d}  " + "  ".join(
            f"{success_matrix[s,t]*100:3.0f}%" if s != t else "  --  "
            for t in range(10)
        )
        print(row)

    return results


if __name__ == "__main__":
    args = get_args()
    run_attack(args)





            
