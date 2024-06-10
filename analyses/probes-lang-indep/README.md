## usage

step 1., create a dataset.

E.g., this will pick sentence-level datapoints for three langs (French, English, Ukranian) and remove the language token:

```sh
python3 mk_dataset.py  checkpoint-31000/base_model xx2en/ckpt-31000/fra-eng-ukr.sent --languages fra_Latn eng_Latn ukr_Cyrl --device cuda:0 --dropfirst --meanpool
```

step 2., evaluate without probe:

```sh
python3 eval_checkpoint.py xx2en/ckpt-31000/fra-eng-ukr.sent --n_layers 0 --device cuda:0
```

or with a probe:

```sh
python3 eval_checkpoint.py xx2en/ckpt-31000/fra-eng-ukr.sent --n_layers 1 --device cuda:0
```

There are a few controls you can tweak.

**NB:** token-level embs will take a fucking long time.


## todos
- [ ] decomposition from hell
- [ ] logging to file would be nice
