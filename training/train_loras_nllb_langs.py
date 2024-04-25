import argparse
import itertools
import os

import torch
from adaptor.adapter import Adapter
from adaptor.lang_module import LangModule
from adaptor.objectives.seq2seq import Sequence2Sequence
from adaptor.schedules import ParallelSchedule
from adaptor.utils import AdaptationArguments, StoppingStrategy, SavingStrategy
from datasets import load_dataset
from peft import LoraConfig, TaskType
from tqdm import tqdm

from training.evaluators import LangGradients, FloresBLEU
from training.langs import nllb_eng_src_in_tatoeba, flores200_langs
from training.lora_lang_objective import LoraLangObjective, Sequence2SequenceBaseline
from training.strided_schedule import StridedSchedule

torch.multiprocessing.set_start_method('spawn')

parser = argparse.ArgumentParser()
parser.add_argument("--base_data_dir", help="A path containing bitexts in `{src-tgt}/train.src.gz`"
                                            "and `{src-tgt}/test.src` format.", required=True, type=str)
parser.add_argument("--base_model", help="A pre-trained model to initialize "
                                         "the training with", required=True, type=str)
parser.add_argument("--checkpoint_dir", help="A base folder where to store the training checkpoints."
                                             "Ignored in continued training.", type=str, default=".")
parser.add_argument("--base_model_type", help="A type of the model to decide on LoRA adapters mapping. ",
                    default=None, type=str)
parser.add_argument("--reset_weights", help="Whether to reset the base model's weights",
                    type=bool, default=False)
parser.add_argument("--target_langs", help="Coma-separated list of target languages. E.g: "
                                           "`sgn,tah`. Defaults to the NLLB's target languages.", default="")
parser.add_argument("--extra_eval_langs", help="Coma-separated list extra languages for evaluation in training. "
                                               "E.g: `sgn,tah`. Defaults to empty.", default="")
parser.add_argument("--pair_evaluation_langs", help="Language pairs on which to perform pair evaluations"
                                                    "(GradientDotProduct eval). Format: 'fur,tah;epo,est'", default="")
parser.add_argument("--samples_per_lang", help="Number of batches to sample in training from single lang. Default (1)"
                                               " means sample training batch from all languages", default=1, type=int)
parser.add_argument("--eval_batches", default=20, type=int)
parser.add_argument("--eval_steps", default=500, type=int)
parser.add_argument("--save_steps", default=500, type=int)
parser.add_argument("--resume_from_checkpoint", help="Whether this is a continued training."
                                                     "Defaults to False", default="False", type=str)
parser.add_argument("--baseline_training", help="Whether this is a training of the monolithic baseline."
                                                "Defaults to False", default="False", type=str)
parser.add_argument("--freeze_shared_params", help="Whether to avoid training of the modules' "
                                                   "shared parameters. Defaults to False", default="False", type=str)
parser.add_argument("--use_language_prefixes", help="Whether to prefix expected outputs with language_id."
                                                    "Defaults to True", default="True", type=str)
parser.add_argument("--allow_unseen_langs", help="Whether the language_id must be included in the model's"
                                                 "vocab. Note that if not, then models using `language_prefixes` in the"
                                                 " decoder might be prefixed with an unknown token.",
                    default="False", type=str)
parser.add_argument("--local_run", default="False", type=str)
parser.add_argument("--eval_run", default="False", type=str)

args = parser.parse_args()
args.resume_from_checkpoint = args.resume_from_checkpoint.lower() != "false"
args.baseline_training = args.baseline_training.lower() != "false"
args.use_language_prefixes = args.use_language_prefixes.lower() != "false"
args.local_run = args.local_run.lower() != "false"
args.eval_run = args.eval_run.lower() != "false"
args.freeze_shared_params = args.freeze_shared_params.lower() != "false"
args.allow_unseen_langs = args.allow_unseen_langs.lower() != "false"

print("Running with arguments: %s" % args)

if args.resume_from_checkpoint:
    # remove the checkpoint-X part of path
    checkpoint_dir = args.base_model.split("/checkpoint-")[0]
else:
    checkpoint_dir = os.path.join(args.checkpoint_dir, ("checkpoints" if not args.baseline_training
                                                        else "checkpoints-baseline"))
    if not args.local_run and os.environ.get("LOCAL_RANK", 0) == 0:
        import wandb

        wandb.init(project="mammoth-lora")
        checkpoint_dir = checkpoint_dir + "-" + wandb.run.name

print("Checkpoint will be saved to '{}'".format(checkpoint_dir))

lang_module = LangModule(args.base_model)

if args.reset_weights:
    lang_module.reinit_base_model()

if not args.target_langs:
    target_langs = nllb_eng_src_in_tatoeba
else:
    target_langs = args.target_langs.split(",")

model_type = args.base_model_type if args.base_model_type else args.base_model
print("Model type: %s" % model_type)
per_model_target_modules = {"facebook/nllb-200-distilled-600M": ["q_proj", "v_proj"]}

peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,
        target_modules=per_model_target_modules.get(model_type, None)  # default for other models
)


def init_objective(src_lang: str,
                   tgt_lang: str,
                   base_data_dir="data/example_data_dir",
                   is_eval_objective: bool = False) -> Sequence2Sequence:
    lang_dir = os.path.join(base_data_dir, "%s-%s" % (src_lang, tgt_lang))

    # evaluation
    model_spec_generation_kwargs = {}
    source_texts_prefix_fn = None
    specific_src_lang = None
    specific_tgt_lang = None
    if args.use_language_prefixes:
        if hasattr(lang_module.tokenizer, "lang_code_to_id"):
            specific_src_lang = next(k for k, v in lang_module.tokenizer.lang_code_to_id.items()
                                     if k.startswith(src_lang))
            try:
                specific_tgt_lang, tgt_token_id = next((k, v) for k, v in lang_module.tokenizer.lang_code_to_id.items()
                                                       if tgt_lang in k)
                specific_tgt_lang = next(k for k, v in lang_module.tokenizer.lang_code_to_id.items() if k.startswith(tgt_lang))
                model_spec_generation_kwargs = {"forced_bos_token_id": tgt_token_id}
            except StopIteration as e:
                if args.allow_unseen_langs:
                    tgt_token_id = next(t_id for t, t_id in lang_module.tokenizer.vocab.items() if tgt_lang == t)
                    model_spec_generation_kwargs = {"forced_bos_token_id": tgt_token_id}
                else:
                    raise e
        else:
            # prefix source texts with prompt
            source_texts_prefix_fn = lambda src_text, lang: "Translate to %s: %s" % (lang, src_text)

    general_kwargs = {"max_length": 128}

    evaluators = [FloresBLEU(decides_convergence=True,
                             generation_kwargs={**model_spec_generation_kwargs, **general_kwargs})]

    shared_args = [lang_module]
    # resolve training path
    if is_eval_objective:
        src, tgt = [], []
    else:
        src = os.path.join(lang_dir, "train.src.gz")
        if not os.path.exists(src):
            src = os.path.join(lang_dir, "train.src")
        assert os.path.exists(src), "Could not find %s to initialize %s objective." % (src, tgt_lang)

        tgt = os.path.join(lang_dir, "train.trg.gz")
        if not os.path.exists(tgt):
            tgt = os.path.join(lang_dir, "train.trg")
        assert os.path.exists(tgt), "Could not find %s to initialize %s objective." % (tgt, tgt_lang)

    # with a priority, load aligned dev sets for all the languages from FLORES
    try:
        # find the matching language pair in the flores list of langs
        fl_src_lang = next(fl_lang for fl_lang in flores200_langs if fl_lang.startswith(src_lang))
        fl_tgt_lang = next(fl_lang for fl_lang in flores200_langs if fl_lang.startswith(tgt_lang))
        flores_dataset = load_dataset("Muennighoff/flores200", "%s-%s" % (fl_src_lang, fl_tgt_lang), split="dev")
        val_src = flores_dataset['sentence_%s' % fl_src_lang]
        val_tgt = flores_dataset['sentence_%s' % fl_tgt_lang]

    except StopIteration:
        # ValueError: BuilderConfig 'hun_Latn-est_adf' not found.
        # resort do a test split from Tatoeba training resources
        val_src = os.path.join(lang_dir, "test.src")
        val_tgt = os.path.join(lang_dir, "test.trg")

    shared_kwargs = {
        "texts_or_path": src,
        "labels_or_path": tgt,
        "val_texts_or_path": val_src,
        "val_labels_or_path": val_tgt,
        "source_lang_id": specific_src_lang if specific_src_lang is not None else src_lang,
        "target_lang_id": specific_tgt_lang if specific_tgt_lang is not None else tgt_lang,
        "batch_size": 1,
        "val_evaluators": evaluators,
        "objective_id": tgt_lang,
        "source_texts_prefix_fn": source_texts_prefix_fn,
        "max_samples_per_eval_log": args.eval_batches,
    }

    try:
        if args.baseline_training:
            objective = Sequence2SequenceBaseline(*shared_args, **shared_kwargs)
        else:
            objective = LoraLangObjective(*shared_args, peft_config=peft_config,
                                          freeze_shared_params=args.freeze_shared_params, **shared_kwargs)
    except FileNotFoundError as e:
        # test split does not exist
        print("Test split of %s-%s not found. We will not perform evaluation on this pair." % (src_lang, tgt_lang))
        shared_kwargs = {k: v for k, v in shared_kwargs.items() if "val" not in k}
        if args.baseline_training:
            objective = Sequence2SequenceBaseline(*shared_args, **shared_kwargs)
        else:
            objective = LoraLangObjective(*shared_args, peft_config=peft_config,
                                          freeze_shared_params=args.freeze_shared_params, **shared_kwargs)

    return objective


if args.extra_eval_langs:
    eval_objectives = [init_objective("eng", tgt_lang, args.base_data_dir, is_eval_objective=True)
                       for tgt_lang in tqdm(args.extra_eval_langs.split(","), desc="Loading eval objectives...")]
else:
    eval_objectives = []

objectives = [init_objective("eng", tgt_lang, args.base_data_dir, is_eval_objective=args.eval_run)
              for tgt_lang in tqdm(target_langs, desc="Loading objectives...")]


if args.pair_evaluation_langs and not args.freeze_shared_params:
    # when we freeze_shared_params, we can not compute and compare their gradients
    eval_compared_lang_pairs = [tuple(pair.split(",")) for pair in args.pair_evaluation_langs.split(";")]

    all_objectives = objectives + eval_objectives

    eval_objective_pairs = [(ref_o, comp_o) for ref_o, comp_o in itertools.product(all_objectives, repeat=2)
                            if (ref_o.given_id, comp_o.given_id) in eval_compared_lang_pairs]
    print("Performing comparative evaluation on the following language pairs: %s"
          % [(ref_o.target_lang_id, comp_o.target_lang_id) for ref_o, comp_o in eval_objective_pairs])

    pair_evaluators = [LangGradients(*pair) for pair in eval_objective_pairs]

    objectives[0].evaluators["eval"] += pair_evaluators

saving_strategy = SavingStrategy.FIRST_OBJECTIVE if args.baseline_training else SavingStrategy.ALL_OBJECTIVES

training_arguments = AdaptationArguments(output_dir=checkpoint_dir,
                                         learning_rate=2e-5,
                                         stopping_strategy=StoppingStrategy.ALL_OBJECTIVES_CONVERGED,
                                         saving_strategy=saving_strategy,
                                         stopping_patience=5,
                                         do_train=True,
                                         do_eval=True,
                                         warmup_steps=5000,
                                         gradient_accumulation_steps=32 if not args.local_run else 1,
                                         logging_steps=50,
                                         eval_steps=args.eval_steps,
                                         save_steps=args.save_steps,
                                         num_train_epochs=10,
                                         evaluation_strategy="steps",
                                         no_cuda=True if args.local_run else False,
                                         save_peft_base_model=True,
                                         local_rank=os.environ.get("LOCAL_RANK", 0),
                                         save_total_limit=6 if not args.resume_from_checkpoint else None,
                                         bf16=True,  # TODO: comment for puhti
                                         report_to="all" if args.local_run else "wandb",
                                         )

scheduler_args = {"objectives": objectives, "args": training_arguments, "extra_eval_objectives": eval_objectives}

if args.samples_per_lang == 1:
    schedule = ParallelSchedule(**scheduler_args)
else:
    schedule = StridedSchedule(**scheduler_args, num_batches_per_objective=args.samples_per_lang)

adapter = Adapter(lang_module, schedule, args=training_arguments)

# mt5 version: getattr(lang_module.trainable_models['140207366527152'].base_model.model.base_model.encoder.block[2].layer[0].SelfAttention.q, "fao-LoraLangObjective_lora_B").default.weight
# nllb version: getattr(lang_module.trainable_models['140344449624864'].base_model.model.base_model.model.model.encoder.layers[2].self_attn.v_proj, "fao-LoraLangObjective_lora_A").default.weight

if not args.eval_run:
    adapter.train()
else:
    evaluation = adapter.evaluate()
    print("Evaluation results: %s" % evaluation)
print("Job done. Terminating.")
