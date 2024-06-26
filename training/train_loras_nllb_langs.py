import argparse
import itertools
import os
from typing import Optional, List

import torch
import wandb

torch.manual_seed(4321)

import random
random.seed(4321)

from adaptor.adapter import Adapter
from adaptor.objectives.seq2seq import Sequence2Sequence
from adaptor.schedules import ParallelSchedule
from adaptor.utils import AdaptationArguments, StoppingStrategy
from datasets import load_dataset
from peft import LoraConfig, TaskType
from tqdm import tqdm

from training.custom_langmodule import OutputReturningLangModule
from training.evaluators import LangGradients, FloresBLEU
from training.langs import nllb_eng_src_in_tatoeba, flores200_langs
from training.lora_lang_objective import LoraLangObjective, Sequence2SequenceBaseline, LangIndependenceRegularizer
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
parser.add_argument("--translation_direction", help="Direction of translation to train on. One of: `from-eng`, "
                                                    "`to-eng`, `both`  Defaults to `from-eng`.", default="from-eng")
parser.add_argument("--extra_eval_langs", help="Coma-separated list extra languages for evaluation in training. "
                                               "E.g: `sgn,tah`. Defaults to empty.", default="")
parser.add_argument("--pair_evaluation_langs", help="Language pairs on which to perform pair evaluations"
                                                    "(GradientDotProduct eval). Format: 'fur,tah;epo,est'", default="")
parser.add_argument("--samples_per_lang", help="Number of batches to sample in training from single lang. Default (1)"
                                               " means sample training batch from all languages", default=1, type=int)
parser.add_argument("--eval_batches", default=20, type=int)
parser.add_argument("--eval_steps", default=500, type=int)
parser.add_argument("--batch_size", default=2, type=int)
parser.add_argument("--eval_on_flores", default="True", type=str)
parser.add_argument("--save_steps", default=500, type=int)
parser.add_argument("--resume_from_checkpoint", help="Whether this is a continued training."
                                                     "Defaults to False", default="False", type=str)
parser.add_argument("--baseline_training", help="Whether this is a training of the monolithic baseline."
                                                "Defaults to False", default="False", type=str)
parser.add_argument("--freeze_shared_params", help="Whether to avoid training of the modules' "
                                                   "shared parameters. Defaults to False", default="False", type=str)
parser.add_argument("--use_language_prefixes", help="Whether to prefix expected outputs with language_id."
                                                    "Defaults to True", default="True", type=str)
parser.add_argument("--lang_margin_loss_weight", help="Whether to also train in the inverse language translation."
                                                      "Needed for enforcing language independence regularization.",
                    default=0., type=float)
parser.add_argument("--learning_rate", help="Learning rate used with all objectives. Defaults to `2e-5`.",
                    default=2e-5, type=float)
parser.add_argument("--lang_margin", help="Expected margin between the distance of equivalent texts "
                                          "in different langs and non-equivalent texts in the same language. Used only "
                                          "for lang independence regularization, i.e. when lang_margin_loss_weight!=0",
                    default=1., type=float)
parser.add_argument("--allow_unseen_langs", help="Whether the language_id must be included in the model's"
                                                 "vocab. Note that if not, then models using `language_prefixes` in the"
                                                 " decoder might be prefixed with an unknown token.",
                    default="False", type=str)
parser.add_argument("--modularization", help="In which way to modularize languages. "
                                             "One of: `per-lang-pair`, `per-enc-dec-lang`.",
                    default="per-lang-pair", type=str)
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
args.eval_on_flores = args.eval_on_flores.lower() != "false"

print("Running with arguments: %s" % args)
print("Training World size: %s" % int(os.environ.get("WORLD_SIZE", 1)))

wandb.init(project="mammoth-lora")
wandb.log({"slurm_id": os.environ.get("SLURM_JOB_ID", -1)}, commit=False)

if args.resume_from_checkpoint:
    # remove the checkpoint-X part of path
    checkpoint_dir = args.base_model.split("/checkpoint-")[0]
else:
    checkpoint_dir = os.path.join(args.checkpoint_dir, ("checkpoints" if not args.baseline_training
                                                        else "checkpoints-baseline"))
    if not args.local_run and os.environ.get("LOCAL_RANK", 0) == 0:
        checkpoint_dir = checkpoint_dir + "-" + wandb.run.name

print("Checkpoint will be saved to '{}'".format(checkpoint_dir))

lang_module = OutputReturningLangModule(args.base_model)

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
        task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=32, lora_alpha=32, lora_dropout=0.1,
        target_modules=per_model_target_modules.get(model_type, None)  # default for other models
)


def init_objective(src_lang: str,
                   tgt_lang: str,
                   base_data_dir="data/example_data_dir",
                   is_eval_objective: bool = False,
                   is_baseline_objective: bool = False,
                   inverse_lang_direction: bool = False,
                   peft_target_modules: Optional[List[str]] = None,
                   objective_module: Optional[torch.nn.Module] = None,
                   save_baseline_obj_module: bool = True) -> Sequence2Sequence:
    lang_dir = os.path.join(base_data_dir, "%s-%s" % (src_lang, tgt_lang))
    # TODO: we need to extend languages: now we iterate only through the "e"+ target langs

    # evaluation
    model_spec_generation_kwargs = {}
    source_texts_prefix_fn = None
    specific_src_lang = None
    specific_tgt_lang = None
    if args.use_language_prefixes:
        prefix_src_lang = src_lang if not inverse_lang_direction else tgt_lang
        prefix_tgt_lang = tgt_lang if not inverse_lang_direction else src_lang
        if hasattr(lang_module.tokenizer, "lang_code_to_id"):
            try:
                specific_src_lang = next(k for k, v in lang_module.tokenizer.lang_code_to_id.items()
                                         if k.startswith(prefix_src_lang))
            except StopIteration as e:
                if args.allow_unseen_langs:
                    specific_src_lang = next(t_id for t, t_id in lang_module.tokenizer.vocab.items()
                                             if prefix_src_lang == t)
                else:
                    raise e
            try:
                specific_tgt_lang, tgt_token_id = next((k, v) for k, v in lang_module.tokenizer.lang_code_to_id.items()
                                                       if k.startswith(prefix_tgt_lang))
                model_spec_generation_kwargs = {"forced_bos_token_id": tgt_token_id}
            except StopIteration as e:
                if args.allow_unseen_langs:
                    tgt_token_id = next(t_id for t, t_id in lang_module.tokenizer.vocab.items() if prefix_tgt_lang == t)
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
    if args.eval_on_flores:
        try:
            if src_lang != tgt_lang:
                # find the matching language pair in the flores list of langs
                fl_src_lang = next(fl_lang for fl_lang in flores200_langs if fl_lang.startswith(src_lang))
                fl_tgt_lang = next(fl_lang for fl_lang in flores200_langs if fl_lang.startswith(tgt_lang))
                flores_dataset = load_dataset("Muennighoff/flores200", "%s-%s" % (fl_src_lang, fl_tgt_lang), split="dev")
                val_src = flores_dataset['sentence_%s' % fl_src_lang]
                val_tgt = flores_dataset['sentence_%s' % fl_tgt_lang]
            else:
                # special case: init of objective solely to register the module -- no training or eval dataset is used
                val_src, val_tgt = None, None

        except StopIteration:
            # ValueError: BuilderConfig 'hun_Latn-est_adf' not found.
            # resort do a test split from Tatoeba training resources
            val_src = os.path.join(lang_dir, "test.src")
            val_tgt = os.path.join(lang_dir, "test.trg")
    else:
        val_src = os.path.join(lang_dir, "test.src")
        val_tgt = os.path.join(lang_dir, "test.trg")
    if isinstance(val_src, str) and not os.path.exists(val_src):
        # if this is inferred path, check that it exists, or omit it from passing to validation (which would fail)
        print("Test split of %s-%s not found. We will not perform evaluation on this pair." % (src_lang, tgt_lang))
        val_src, val_tgt = None, None

    obj_id = ("%s-%s" % (src_lang, tgt_lang)) if not inverse_lang_direction else ("%s-%s" % (tgt_lang, src_lang))

    objective_src_lang = specific_src_lang if specific_src_lang is not None else (src_lang if not inverse_lang_direction else tgt_lang)
    objective_tgt_lang = specific_tgt_lang if specific_tgt_lang is not None else (tgt_lang if not inverse_lang_direction else src_lang)
    # if inverse_lang_direction:
    #     objective_src_lang, objective_tgt_lang = objective_tgt_lang, objective_src_lang

    shared_kwargs = {
        "texts_or_path": src,
        "labels_or_path": tgt,
        "val_texts_or_path": val_src,
        "val_labels_or_path": val_tgt,
        "source_lang_id": objective_src_lang,
        "target_lang_id": objective_tgt_lang,
        "batch_size": args.batch_size,
        "val_evaluators": evaluators,
        "objective_id": obj_id,
        "source_texts_prefix_fn": source_texts_prefix_fn,
        "max_samples_per_eval_log": args.eval_batches,
        "inverse_direction": inverse_lang_direction,
        "objective_module": objective_module,
        "merge_objective_module": objective_module is None,
    }

    if is_baseline_objective:
        objective = Sequence2SequenceBaseline(*shared_args,
                                              save_objective_module=save_baseline_obj_module,
                                              **shared_kwargs)
    else:
        objective_peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=32, lora_alpha=32, lora_dropout=0.1,
                target_modules=peft_target_modules
        )
        objective = LoraLangObjective(*shared_args, peft_config=objective_peft_config,
                                      freeze_shared_params=args.freeze_shared_params, **shared_kwargs)
    return objective


objectives = []

# based on args.modularization, we filter the default target_modules for PEFT, or we use them as-is
target_modules = per_model_target_modules.get(model_type, None)

# if we filter out which parameters to modularize, we need to preload the model to get a list of all params
if args.modularization == "per-enc-dec-lang":
    print("Loading model %s to get a list of its parameters" % lang_module.model_name_or_path)
    english_objective = init_objective("eng", "eng",
                                       is_eval_objective=True,
                                       is_baseline_objective=False,
                                       peft_target_modules=target_modules)
    all_modules = [n for n, _ in english_objective.compatible_head_model.base_model.model.named_modules()]


fwd_objective = None

for i, tgt_lang in tqdm(enumerate(target_langs), desc="Loading objectives...", total=len(target_langs)):
    # distributed loading of objectives
    # if i % int(os.environ.get("WORLD_SIZE", 1)) != int(os.environ.get("RANK", 0)):
    #     TODO: this will not work: we can not average gradients over different models! We need the same model
    #     continue
    peft_modules = target_modules

    # support for continued baseline training: reload module from the first-loaded objective
    if args.baseline_training and fwd_objective is not None:
        preloaded_module = fwd_objective.compatible_head_model
    else:
        preloaded_module = None

    if args.translation_direction in ("from-eng", "both"):
        if args.modularization == "per-enc-dec-lang":
            # for "from-eng" direction, we parametrize (target) language in decoder
            peft_modules = [m for m in all_modules if "decoder." in m and any(m.endswith(p) for p in target_modules)]

        fwd_objective = init_objective("eng", tgt_lang, args.base_data_dir,
                                       peft_target_modules=peft_modules,
                                       is_eval_objective=args.eval_run,
                                       is_baseline_objective=args.baseline_training,
                                       objective_module=preloaded_module,
                                       save_baseline_obj_module=preloaded_module is not None,
                                       )
        if args.modularization == "per-enc-dec-lang":
            # replace initialized encoder with the english one
            fwd_objective.compatible_head_model.base_model.model.model.encoder \
                = english_objective.compatible_head_model.base_model.model.model.encoder

        objectives.append(fwd_objective)
    if args.translation_direction in ("to-eng", "both"):
        if args.modularization == "per-enc-dec-lang":
            # for "to-eng" direction, we parametrize (source) language in encoder
            peft_modules = [m for m in all_modules if "encoder." in m and any(m.endswith(p) for p in target_modules)]

        bwd_objective = init_objective("eng", tgt_lang, args.base_data_dir,
                                       peft_target_modules=peft_modules,
                                       is_eval_objective=args.eval_run,
                                       is_baseline_objective=args.baseline_training,
                                       inverse_lang_direction=True,
                                       objective_module=preloaded_module,
                                       save_baseline_obj_module=preloaded_module is not None,
                                       )
        if args.modularization == "per-enc-dec-lang":
            # replace initialized encoder with the english one
            bwd_objective.compatible_head_model.base_model.model.model.decoder \
                = english_objective.compatible_head_model.base_model.model.model.decoder

        objectives.append(bwd_objective)
    if args.translation_direction == "both" and args.lang_margin_loss_weight:
        # model's forward() should define the routing
        fwd_model = fwd_objective.compatible_head_model
        obj_model = fwd_model.base_model.model if not args.baseline_training else fwd_model  # differs for lora vs. base
        reg_objective = LangIndependenceRegularizer(lang_module,
                                                    objective_module=obj_model,  # if given, LangModule does not merge
                                                    merge_objective_module=False,
                                                    texts_or_path=[],
                                                    objectives=(fwd_objective, bwd_objective),
                                                    max_samples_per_eval_log=args.eval_batches,
                                                    loss_weight=args.lang_margin_loss_weight,
                                                    # semantic_over_lang_sim_margin=45.24  # non-normalized default
                                                    semantic_over_lang_sim_margin=args.lang_margin,
                                                    save_objective_module=False,
                                                    )
        # if fwd_objective.val_texts is not None or fwd_objective.val_texts_path is not None:
        #     # This fails because apparently we have languages with a single evaluation sample :)
        #     x = next(iter(reg_objective.get_dataset("eval")))
        #     outputs = obj_model(**{k: v for k, v in x.items() if k != "oid"})
        #     print("Objective %s first eval loss: %s" % (reg_objective, outputs.loss.item()))
        objectives.append(reg_objective)

if args.extra_eval_langs:
    # eval objectives are always parametrized by the base model -- we do not train with them
    fwd_model = objectives[0].compatible_head_model
    base_model = fwd_model.base_model.model if not args.baseline_training else fwd_model

    eval_objectives = [init_objective("eng", tgt_lang, args.base_data_dir,
                                      is_eval_objective=True,
                                      is_baseline_objective=True,
                                      objective_module=base_model)
                       for tgt_lang in tqdm(args.extra_eval_langs.split(","), desc="Loading eval objectives...")]
else:
    eval_objectives = []


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

accum_steps = (32 // int(os.environ.get("WORLD_SIZE", 1))) if not args.local_run else 3

training_arguments = AdaptationArguments(output_dir=checkpoint_dir,
                                         learning_rate=args.learning_rate,
                                         stopping_strategy=StoppingStrategy.ALL_OBJECTIVES_CONVERGED,
                                         stopping_patience=20,
                                         do_train=True,
                                         do_eval=True,
                                         warmup_steps=500,
                                         gradient_accumulation_steps=accum_steps,
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
    concurrently_sampled_objs = 1 if not args.translation_direction == "both" \
                                  else 2 if not args.lang_margin_loss_weight \
                                  else 3
    schedule = StridedSchedule(**scheduler_args, num_batches_per_objective=args.samples_per_lang,
                               coupled_objs=concurrently_sampled_objs)

adapter = Adapter(lang_module, schedule, args=training_arguments)

# mt5 version: getattr(lang_module.trainable_models['140207366527152'].base_model.model.base_model.encoder.block[2].layer[0].SelfAttention.q, "fao-LoraLangObjective_lora_B").default.weight
# nllb version: getattr(lang_module.trainable_models['140344449624864'].base_model.model.base_model.model.model.encoder.layers[2].self_attn.v_proj, "fao-LoraLangObjective_lora_A").default.weight

# TODO: these do not match now:
# [lang_module.tokenizer.decode(x["input_ids"][1]) for x in itertools.islice(fwd_objective._get_inputs_iterator("eval"), 10)]
# [lang_module.tokenizer.decode([l for l in x["labels"][1] if l > 0]) for x in itertools.islice(bwd_objective._get_inputs_iterator("eval"), 10)]
# TODO: check encoded inputs for NLLB

if not args.eval_run:
    adapter.train()
else:
    evaluation = adapter.evaluate()
    print("Evaluation results: %s" % evaluation)
print("Job done. Terminating.")
