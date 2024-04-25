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

for train_lang, eval_lang in itertools.product(training_langs, eval_langs):
    if os.path.exists(file_template % (train_lang, "src")):
        train_tgt, eval_tgt = find_aligned_by_index(
                list(AdaptationDataset.iter_text_file_per_line(file_template % (train_lang, "src"))),
                list(AdaptationDataset.iter_text_file_per_line(file_template % (train_lang, "trg"))),
                list(AdaptationDataset.iter_text_file_per_line(file_template % (eval_lang, "src"))),
                list(AdaptationDataset.iter_text_file_per_line(file_template % (eval_lang, "trg")))
        )
        num_shared = len(train_tgt)
    else:
        num_shared = 0

    print("%s and %s share %s test samples" % (train_lang, eval_lang, num_shared))
