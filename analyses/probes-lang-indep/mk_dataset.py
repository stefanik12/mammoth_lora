import argparse
import itertools
import pathlib

import datasets
import torch
import torch.nn as nn
import transformers
import peft


parser = argparse.ArgumentParser()
parser.add_argument('checkpoint')
parser.add_argument('savedir')
parser.add_argument('--meanpooled', action='store_true')
parser.add_argument('--dropfirst', action='store_true')
parser.add_argument('--languages', nargs='+')
parser.add_argument('--device', type=torch.device, default=torch.device('cpu'))
args = parser.parse_args()

base_model_enc = transformers.AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint + "/base_model").model.encoder.to(args.device)

dataset = datasets.load_dataset("facebook/flores", 'all', split='dev', trust_remote_code=True)
dataset = dataset.train_test_split(shuffle=True, seed=632, test_size=0.1)

@torch.no_grad()
def get_embeddings(examples):
    return_dict = {'lang': [], 'emb': []}
    for language in args.languages:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.checkpoint + "/base_model", 
            src_lang=language,
        )
        for idx in range(len(examples[f"sentence_{language}"])):            
            inputs = tokenizer(examples[f"sentence_{language}"][idx], return_tensors='pt')
            if args.dropfirst:
                for key in inputs.keys():
                    inputs[key] = inputs[key][:,1:,...]
            outputs = base_model_enc(**inputs.to(args.device))
            embeddings = outputs.last_hidden_state.squeeze(0)
            if args.meanpooled:
                emb = embeddings.mean(0)
                return_dict['emb'].append(emb.cpu())
                return_dict['lang'].append(language)
            else:
                for emb in embeddings:
                    return_dict['emb'].append(emb.cpu())
                    return_dict['lang'].append(language)
    return return_dict

dataset = dataset.map(get_embeddings, batched=True, remove_columns=dataset['train'].column_names, batch_size=100)
dataset.set_format('torch')
dataset = dataset.save_to_disk(args.savedir)
