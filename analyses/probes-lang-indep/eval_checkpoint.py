import argparse
import itertools

import datasets
datasets.logging.disable_progress_bar()
import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import tqdm
import transformers, peft


parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('--n_layers', type=int)
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--eval_every', type=int, default=100)
parser.add_argument('--device', type=torch.device, default=torch.device('cpu'))
args = parser.parse_args()

assert args.n_layers >= 0

dataset = datasets.load_from_disk(args.dataset)
dim_model = len(dataset['train'][0]['emb'])

def torch_rankdata(all_obs):
    s = all_obs[all_obs.argsort()]
    indices = torch.arange(1, s.numel() + 1, device=args.device)
    indices = indices.masked_select(
        torch.cat([
            torch.tensor([True], device=args.device), 
            s[1:] != s[:-1]
        ])
    )
    _, inverse, indices_hi = torch.unique(all_obs, return_counts=True, return_inverse=True)
    indices  = indices + (indices_hi - 1)/2
    return indices.gather(-1, inverse)    

def do_eval(dataset, model=nn.Identity()):
    pos, neg = [], []
    n_items = len(dataset['test'])
    test_data = dataset['test'].map(lambda example: {'emb': model(example['emb'].to(args.device))})
    n_items = (n_items * (n_items - 1)) // 2
    with tqdm.trange(n_items) as pbar:
        for datapoint_i, datapoint_j in itertools.combinations(test_data, 2):
            emb_i = datapoint_i['emb']
            emb_j = datapoint_j['emb']
            storage = pos if datapoint_i['lang'] == datapoint_j['lang'] else neg
            distance = torch.norm(emb_i - emb_j, p=2)
            storage.append(distance)
            pbar.update()
    pos, neg = torch.tensor(pos, device=args.device), torch.tensor(neg, device=args.device)
    """U_stat_manual = torch.tensor(0., dtype=torch.double, device=args.device)
    start = 0
    chunksize = 250
    for end in tqdm.trange(chunksize, len(pos) + chunksize, chunksize):
        a = pos[start:end]
        start = end
        U_stat_manual += (neg.unsqueeze(0) < a.unsqueeze(1)).sum()
        U_stat_manual += 0.5 * (neg.unsqueeze(0) == a.unsqueeze(1)).sum()
    mean_U = U_stat_manual /  (pos.numel() * neg.numel())
    U_var = torch.tensor(0., dtype=torch.double, device=args.device)
    win, tie, lose = (1 - mean_U) ** 2, (0.5 - mean_U) ** 2, (0 - mean_U) ** 2
    start = 0
    for end in tqdm.trange(chunksize, len(pos) + chunksize, chunksize):
        a = pos[start:end]
        start = end
        U_var += torch.where(neg.unsqueeze(0) < a.unsqueeze(1), win, 0.).sum()
        U_var += torch.where(neg.unsqueeze(0) == a.unsqueeze(1), tie, 0.).sum()
        U_var += torch.where(neg.unsqueeze(0) > a.unsqueeze(1), lose, 0.).sum()
    U_var = U_var /  (pos.numel() * neg.numel() - 1)
    
    # mess from scipy
    x, y, xy = scipy.stats._mannwhitneyu._broadcast_concatenate(pos.cpu().numpy(), neg.cpu().numpy(), 0)
    n1, n2 = x.shape[-1], y.shape[-1]
    ranks = scipy.stats.rankdata(xy, axis=-1)  # method 2, step 1
    R1 = ranks[..., :n1].sum(axis=-1)    # method 2, step 2
    U1 = R1 - n1*(n1+1)/2                # method 2, step 3
    U2 = n1 * n2 - U1                    # as U1 + U2 = n1 * n2"""
    
    bootstraps = 10_000
    effect_sizes = []
    torch_rankdata(torch.concat([pos, neg]))
    for _ in tqdm.trange(bootstraps):
        sample_pos = torch.gather(
            pos, 
            0, 
            torch.randint_like(pos, pos.numel(), dtype=torch.long),
        )
        sample_neg = torch.gather(
            neg, 
            0, 
            torch.randint_like(neg, neg.numel(), dtype=torch.long),
        )
        ranks = torch_rankdata(torch.concat([sample_pos, sample_neg]))
        R1 = ranks[:pos.numel()].to(dtype=torch.double).sum()
        U = R1 - pos.numel()*(pos.numel()+1)/2
        effect_sizes.append(U.item() / (pos.numel() * neg.numel()))  
    effect_sizes = torch.tensor(effect_sizes, device=args.device)
    U, p = scipy.stats.mannwhitneyu(pos.cpu().numpy(), neg.cpu().numpy())
    f = U / (pos.numel() * neg.numel())
    tqdm.tqdm.write(f"{args.dataset} same: {pos.mean().item():.5f}, different: {neg.mean().item():.5f}")
    tqdm.tqdm.write(f'direct: {U} {p} {f}')
    tqdm.tqdm.write(f'bootstrapped effect size f ({bootstraps} bootstraps): {effect_sizes.mean().item()} ± {effect_sizes.std().item()}')

if args.n_layers == 0:
    do_eval(dataset)
    import sys; sys.exit(0)
# else

from optim import AdaFactorFairSeq

assert dim_model == 1024, 'I was lied to!'
layers = [nn.Dropout(p=0.1), nn.Linear(dim_model, dim_model), nn.LayerNorm(dim_model, elementwise_affine=False), nn.ReLU()] * args.n_layers
layers = layers[:-1]
model = nn.Sequential(*layers).to(args.device)
optimizer = AdaFactorFairSeq(model.parameters())
criterion = nn.TripletMarginLoss(margin=45.243786) # I can explain
print(model)
train_data = dataset['train'].to_pandas()

def get_pos(lang):
    return train_data[train_data.lang == lang].sample(n=1)['emb'].iloc[0]
def get_neg(lang):
    return train_data[train_data.lang != lang].sample(n=1)['emb'].iloc[0]


for epoch in tqdm.trange(args.n_epochs, desc='Epochs'):
    train_data = train_data.sample(frac=1.).reset_index(drop=True)
    start = 0
    for end in tqdm.trange(
        args.eval_every, 
        len(train_data) + args.eval_every, 
        args.eval_every, 
        desc=f'E{epoch}',
    ):
        batch = train_data.loc[start:end]
        start = end
        optimizer.zero_grad()
        pos = torch.tensor(np.array(batch.lang.apply(get_pos).to_list()), device=args.device)
        neg = torch.tensor(np.array(batch.lang.apply(get_neg).to_list()), device=args.device)
        anc = torch.tensor(np.array(batch.emb.to_list()), device=args.device)
        anc = model(anc)
        pos = model(pos) 
        neg = model(neg)
        loss = criterion(anc, pos, neg)
        loss.backward()
        optimizer.step()
        model.eval()
        do_eval(dataset, model=model)
        model.train()
        


