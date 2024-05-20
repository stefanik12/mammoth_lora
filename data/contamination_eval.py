import argparse
import itertools
import os
import gzip
import io
from functools import partial
from typing import Iterable, Any, Iterator
from tqdm import tqdm

from datasets import load_dataset

from training.langs import nllb_eng_src_in_tatoeba, flores200_langs

parser = argparse.ArgumentParser()

TRAIN_FNAME = "train.%s.gz"
TEST_FNAME = "test.%s"

parser.add_argument("--train_dataset_dir", help="Training dataset directory in Tatoeba format.",
                    required=True, type=str)
parser.add_argument("--new_train_dataset_dir", help="Training dataset directory for deduplicated data.",
                    type=str, default="")

parser.add_argument("--test_dataset_name", help="Test set to evaluate contamination against."
                                                "One of: `tatoeba`, `flores200`, `all`.", default='all')
parser.add_argument("--src_tatoeba_langs", help="Source Tatoeba languages to evaluate data leakage with.",
                    type=str, default="eng")
parser.add_argument("--tgt_tatoeba_langs", help="Target Tatoeba languages to evaluate data leakage with.",
                    type=str, default=",".join(nllb_eng_src_in_tatoeba))

args = parser.parse_args()


def read_file(path: str) -> Iterable[str]:
    """
    Iterate over the lines of a file on a given path.
    At this point, `path` is checked to be of a supported format.
    :param path: file path
    """
    if path.endswith(".gz"):
        with io.TextIOWrapper(io.BufferedReader(gzip.open(path))) as file:
            for line in file:
                yield line.strip()
    else:
        # assumes plain, newline-separated text file
        with open(path) as f:
            for line in f:
                yield line.strip()


def write_gz_file(lines: Iterable[str], path: str) -> None:
    with gzip.open(path, "wb") as out_f:
        out_f.writelines((line+"\n").encode('utf-8') for line in lines)


def tokenize(text: str) -> str:
    return text.lower().replace(".", "").replace(",", "")


class Counter:

    i = 0

    def increment(self):
        self.i += 1


def count_and_iter(counter: Counter, iterator: Iterable[Any]) -> Iterator[Any]:
    for sample in iterator:
        counter.increment()
        yield sample


# test pairs collection
all_test_pairs = []

for src_lang, tgt_lang in itertools.product(args.src_tatoeba_langs.split(","), args.tgt_tatoeba_langs.split(",")):
    print("Loading %s->%s test sets" % (src_lang, tgt_lang))
    if args.test_dataset_name in ("flores200", "all"):
        fl_src_lang = next(fl_lang for fl_lang in flores200_langs if fl_lang.startswith(src_lang))
        fl_tgt_lang = next(fl_lang for fl_lang in flores200_langs if fl_lang.startswith(tgt_lang))

        flores_dataset = load_dataset("Muennighoff/flores200", "%s-%s" % (fl_src_lang, fl_tgt_lang),
                                      trust_remote_code=True)
        for k in flores_dataset.keys():
            test_src = set(map(tokenize, flores_dataset[k]['sentence_%s' % fl_src_lang]))
            test_tgt = set(map(tokenize, flores_dataset[k]['sentence_%s' % fl_tgt_lang]))

            all_test_pairs.extend(zip(test_src, test_tgt))

    elif args.test_dataset_name in ("tatoeba", "all"):
        lang_dir = os.path.join(args.train_dataset_dir, "%s-%s" % (src_lang, tgt_lang))
        test_src = set(map(tokenize, read_file(os.path.join(lang_dir, TEST_FNAME % "src"))))
        test_tgt = set(map(tokenize, read_file(os.path.join(lang_dir, TEST_FNAME % "trg"))))

        all_test_pairs.extend(zip(test_src, test_tgt))

# train pairs: stream filtering
all_pairs = itertools.product(args.src_tatoeba_langs.split(","), args.tgt_tatoeba_langs.split(","))

for lang_pair_i, (src_lang, tgt_lang) in enumerate(all_pairs):
    # slurm support: skip pair if it's not assigned to this process
    # from https://hpcc.umd.edu/hpcc/help/slurmenv.html
    # process_i = os.environ.get("SLURM_PROCID", None) # Nope
    process_i = os.environ.get("SLURM_LOCALID", None) # Nope
    num_processes = os.environ.get("SLURM_NTASKS", None) # Yes: this returns #SBATCH --ntasks
    is_slurm_process = process_i is not None and num_processes is not None
    if is_slurm_process:
        print("Running in parallel: process rank %s, total processes: %s" % (process_i, num_processes))
        if lang_pair_i % int(num_processes) != int(process_i):
            continue

    lang_dir = os.path.join(args.train_dataset_dir, "%s-%s" % (src_lang, tgt_lang))

    print("Reading data from %s->%s train set" % (src_lang, tgt_lang))
    train_src = read_file(os.path.join(lang_dir, TRAIN_FNAME % "src"))
    train_tgt = read_file(os.path.join(lang_dir, TRAIN_FNAME % "trg"))

    orig_counter = Counter()
    orig_counter_wrap = partial(count_and_iter, orig_counter)
    new_counter = Counter()
    new_counter_wrap = partial(count_and_iter, new_counter)

    print("Deduplicating %s->%s" % (src_lang, tgt_lang))
    src_tgt_pairs_iter = orig_counter_wrap(zip(train_src, train_tgt))
    # src_tgt_pairs_iter = tqdm(orig_counter_wrap(zip(train_src, train_tgt)),
    #                           desc="Deduplicating %s->%s" % (src_lang, tgt_lang),
    #                           position=0 if not is_slurm_process else int(process_i))
    src_stream_out, tgt_stream_out = zip(*(src_tgt for src_tgt in src_tgt_pairs_iter
                                           if (tokenize(src_tgt[0]), tokenize(src_tgt[1])) not in all_test_pairs))
    if args.new_train_dataset_dir:
        os.makedirs(args.new_train_dataset_dir, exist_ok=True)
        new_lang_dir = os.path.join(args.new_train_dataset_dir, "%s-%s" % (src_lang, tgt_lang))
        os.makedirs(new_lang_dir, exist_ok=True)
        
        write_gz_file(new_counter_wrap(src_stream_out), os.path.join(new_lang_dir, TRAIN_FNAME % "src"))
        write_gz_file(src_stream_out, os.path.join(new_lang_dir, TRAIN_FNAME % "trg"))
    else:
        # we need to reiterate new data to trigger counter
        _ = [0 for _ in new_counter_wrap(src_stream_out)]

    print("Pair %s->%s original samples: %s, new samples: %s" % (src_lang, tgt_lang, orig_counter.i, new_counter.i))
