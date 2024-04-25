import itertools
import os.path

from adaptor.utils import AdaptationDataset

from training.data_utils import find_aligned_by_index
from training.langs import nllb_eng_src_in_tatoeba

# training_langs = nllb_eng_src_in_tatoeba
training_langs = "fao,fon"
eval_langs = "fur"
base_data_dir = "data/example_data_dir"

training_langs = training_langs.split(",")
eval_langs = eval_langs.split(",")

file_template = base_data_dir + "/eng-%s/test.%s"

already_evaled = set()

for train_lang in training_langs:
    if not os.path.exists(file_template % (train_lang, "src")):
        continue
    train_src_all = list(AdaptationDataset.iter_text_file_per_line(file_template % (train_lang, "src")))
    train_tgt_all = list(AdaptationDataset.iter_text_file_per_line(file_template % (train_lang, "trg")))

    for eval_lang in eval_langs:
        if not os.path.exists(file_template % (eval_lang, "src")):
            continue
        if "%s-%s" % (train_lang, eval_lang) in already_evaled:
            continue
        eval_src_all = list(AdaptationDataset.iter_text_file_per_line(file_template % (eval_lang, "src")))
        eval_tgt_all = list(AdaptationDataset.iter_text_file_per_line(file_template % (eval_lang, "trg")))

        train_tgt, eval_tgt = find_aligned_by_index(train_src_all, train_tgt_all, eval_src_all, eval_tgt_all)
        num_shared = len(train_tgt)

        print("%s and %s share %s test samples" % (train_lang, eval_lang, num_shared))
        already_evaled.add("%s-%s" % (train_lang, eval_lang))
