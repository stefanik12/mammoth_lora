import argparse
import itertools
import os

from adaptor.adapter import Adapter
from adaptor.evaluators.generative import BLEU
from adaptor.lang_module import LangModule
from adaptor.objectives.objective_base import Objective
from adaptor.schedules import ParallelSchedule
from adaptor.utils import AdaptationArguments, StoppingStrategy, SavingStrategy, AdaptationDataset
from peft import LoraConfig, TaskType
from tqdm import tqdm
from lora_lang_objective import LoraLangObjective, Sequence2SequenceBaseline

parser = argparse.ArgumentParser()
parser.add_argument("--base_data_dir", help="A path containing bitexts in `{src-tgt}/train.src.gz`"
                                            "and `{src-tgt}/test.src` format.", required=True, type=str)
parser.add_argument("--base_model", help="A pre-trained model to initialize "
                                         "the training with", required=True, type=str)
parser.add_argument("--reset_weights", help="Whether to reset the base model's weights",
                    type=bool, default=False)
parser.add_argument("--target_langs", help="Coma-separated list of target languages. E.g: "
                                           "`sgn,tah`. Defaults to the NLLB's target languages.", default="")
parser.add_argument("--resume_training", help="Whether this is a continued training."
                                              "Defaults to False", default="False", type=str)
parser.add_argument("--baseline_training", help="Whether this is a training of the monolithic baseline."
                                                "Defaults to False", default="False", type=str)
parser.add_argument("--use_language_prefixes", help="Whether to prefix expected outputs with language_id."
                                                    "Defaults to True", default="True", type=str)
parser.add_argument("--local_run", default="False", type=str)
parser.add_argument("--firstn", type=int)

args = parser.parse_args()
args.resume_training = args.resume_training.lower() != "false"
args.baseline_training = args.baseline_training.lower() != "false"
args.use_language_prefixes = args.use_language_prefixes.lower() != "false"
args.local_run = args.local_run.lower() != "false"

print("Running with arguments: %s" % args)

if args.resume_training:
    # remove the checkpoint-X part of path
    checkpoint_dir = args.base_model.split("/checkpoint-")[0]
else:
    checkpoint_dir = "/scratch/project_462000447/members/mstefani/"
    checkpoint_dir += "checkpoints" if not args.baseline_training else "checkpoints-baseline"
    checkpoint_dir += "-all_langs" if not args.target_langs else "-%s_langs" % len(args.target_langs.split(","))
    if not args.local_run and os.environ.get("LOCAL_RANK", 0) == 0:
        import wandb
        wandb.init(project="mammoth-lora")
        checkpoint_dir = checkpoint_dir + "-" + wandb.run.name

print("Checkpoint will be saved to '{}'".format(checkpoint_dir))

lang_module = LangModule(args.base_model)

if args.reset_weights:
    lang_module.reinit_base_model()

nllb_eng_src_in_tatoeba = ['epo', 'est', 'eus', 'ewe', 'fao', 'fij', 'fin',
                           'fon', 'fra', 'fur', 'gla', 'gle', 'glg', 'grn', 'guj',
                           'hat', 'hau', 'heb', 'hin', 'hne', 'hun', 'hye', 'ibo',
                           'ilo', 'isl', 'ita', 'jav', 'jpn', 'kab', 'kac', 'kam',
                           'kan', 'kas', 'kas', 'kat', 'kaz', 'kbp', 'kea', 'khm',
                           'kik', 'kin', 'kir', 'kmb', 'kon', 'kor', 'lao', 'lij',
                           'lim', 'lin', 'lit', 'lmo', 'ltz', 'lua', 'lug', 'luo',
                           'lus', 'mag', 'mai', 'mal', 'mar', 'mkd', 'mlt', 'mni',
                           'mos', 'mri', 'mya', 'nld', 'nso', 'nus', 'nya', 'oci',
                           'pag', 'pan', 'pap', 'pol', 'por', 'ron', 'run', 'rus',
                           'sag', 'san', 'sat', 'scn', 'shn', 'sin', 'slk', 'slv',
                           'smo', 'sna', 'snd', 'som', 'sot', 'spa', 'srd', 'ssw',
                           'sun', 'swe', 'szl', 'tam', 'tat', 'tel', 'tgk', 'tgl',
                           'tha', 'tir', 'tpi', 'tsn', 'tso', 'tuk', 'tum', 'tur',
                           'tzm', 'uig', 'ukr', 'umb', 'urd', 'vec', 'vie', 'war',
                           'wol', 'xho', 'yor', 'zho', 'zho', 'zul']

if not args.target_langs:
    target_langs = nllb_eng_src_in_tatoeba
else:
    target_langs = args.target_langs.split(",")

per_model_target_modules = {"facebook/nllb-200-distilled-600M": ["q_proj", "v_proj"]}

peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,
        target_modules=per_model_target_modules.get(lang_module.model_name_or_path, None)  # default for other models
)


def texts_from_path(path: str):
    texts_iter = AdaptationDataset.iter_text_file_per_line(path)
    firstn_texts = itertools.islice(texts_iter, args.firstn)
    return [t for t in tqdm(firstn_texts, desc="Loading texts from %s" % path)]


class CustomBLEU(BLEU):

    def evaluate_str(self, expected_list, actual_list) -> float:
        expected_nonempty = [e for e, a in zip(expected_list, actual_list) if e and a]
        actual_nonempty = [a for e, a in zip(expected_list, actual_list) if e and a]
        return super().evaluate_str(expected_nonempty, actual_nonempty)


def init_objective(src_lang: str,
                   tgt_lang: str,
                   base_data_dir="data/example_data_dir") -> Objective:
    lang_dir = os.path.join(base_data_dir, "%s-%s" % (src_lang, tgt_lang))

    # evaluation
    model_spec_generation_kwargs = {}
    source_texts_prefix_fn = None
    if args.use_language_prefixes:
        if hasattr(lang_module.tokenizer, "lang_code_to_id"):
            model_bos_token_id = next(v for k, v in lang_module.tokenizer.lang_code_to_id.items() if tgt_lang in k)
            model_spec_generation_kwargs = {"forced_bos_token_id": model_bos_token_id}
        else:
            # prefix source texts with prompt
            source_texts_prefix_fn = lambda src_text, lang: "Translate to %s: %s" % (lang, src_text)

    general_kwargs = {"max_length": 128}

    evaluators = [CustomBLEU(decides_convergence=True,
                             generation_kwargs={**model_spec_generation_kwargs, **general_kwargs})]

    shared_args = [lang_module]
    shared_kwargs = {
        "texts_or_path": texts_from_path(os.path.join(lang_dir, "train.src.gz")),
        "labels_or_path": texts_from_path(os.path.join(lang_dir, "train.trg.gz")),
        "source_lang_id": src_lang,
        "target_lang_id": tgt_lang,
        "batch_size": 2,
        "val_evaluators": evaluators,
        "objective_id": tgt_lang,
        "source_texts_prefix_fn": source_texts_prefix_fn,
        "max_samples_per_eval_log": 20,
    }

    try:
        shared_kwargs["val_texts_or_path"] = texts_from_path(os.path.join(lang_dir, "test.src"))
        shared_kwargs["val_labels_or_path"] = texts_from_path(os.path.join(lang_dir, "test.trg"))

        if args.baseline_training:
            objective = Sequence2SequenceBaseline(*shared_args, **shared_kwargs)
        else:
            objective = LoraLangObjective(*shared_args, peft_config=peft_config, **shared_kwargs)
    except FileNotFoundError as e:
        # test split does not exist
        print("Test split of %s-%s not found. We will not perform evaluation on this pair." % (src_lang, tgt_lang))
        shared_kwargs = {k: v for k, v in shared_kwargs.items() if "val" not in k}
        if args.baseline_training:
            objective = Sequence2SequenceBaseline(*shared_args, **shared_kwargs)
        else:
            objective = LoraLangObjective(*shared_args, peft_config=peft_config, **shared_kwargs)

    return objective


objectives = [init_objective("eng", tgt_lang, args.base_data_dir) for tgt_lang in tqdm(target_langs,
                                                                                       desc="Loading objectives...")]
saving_strategy = SavingStrategy.FIRST_OBJECTIVE if args.baseline_training else SavingStrategy.ALL_OBJECTIVES

training_arguments = AdaptationArguments(output_dir=checkpoint_dir,
                                         learning_rate=2e-5,
                                         stopping_strategy=StoppingStrategy.ALL_OBJECTIVES_CONVERGED,
                                         saving_strategy=saving_strategy,
                                         stopping_patience=5,
                                         do_train=True,
                                         do_eval=True,
                                         warmup_steps=5000,
                                         gradient_accumulation_steps=len(target_langs),
                                         logging_steps=50,
                                         eval_steps=1000,
                                         save_steps=1000,
                                         num_train_epochs=10,
                                         evaluation_strategy="steps",
                                         # no_cuda=True,
                                         save_peft_base_model=True,
                                         local_rank=os.environ.get("LOCAL_RANK", 0),
                                         save_total_limit=6,
                                         bf16=True,
                                         )

schedule = ParallelSchedule(objectives=objectives, args=training_arguments)

adapter = Adapter(lang_module, schedule, args=training_arguments)

adapter.train()
