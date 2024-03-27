import argparse
import os
from tqdm import tqdm

from adaptor.adapter import Adapter
from adaptor.evaluators.generative import BLEU
from adaptor.lang_module import LangModule
from adaptor.schedules import ParallelSchedule
from adaptor.utils import AdaptationArguments, StoppingStrategy
from peft import LoraConfig, TaskType

from lora_lang_objective import LoraLangObjective

parser = argparse.ArgumentParser()
parser.add_argument("--base_data_dir", help="A path containing bitexts in `{src-tgt}/train.src.gz`"
                                            "and `{src-tgt}/test.src` format.", required=True, type=str)
parser.add_argument("--base_model", help="A pre-trained model to initialize "
                                         "the training with", required=True, type=str)
parser.add_argument("--reset_weights", help="Whether to reset the base model's weights",
                    type=bool, default=False)
parser.add_argument("--target_langs", help="Coma-separated list of target languages. E.g: "
                                           "`sgn,tah`. Defaults to the NLLB's target languages.", default="")
args = parser.parse_args()

lang_module = LangModule(args.base_model)

if args.reset_weights:
    lang_module.reinit_base_model()

evaluators = [BLEU(additional_sep_char="▁", decides_convergence=True)]

all_nllb_langs = ['ace', 'ace', 'acm', 'acq', 'aeb', 'afr', 'ajp', 'aka', 'amh', 'apc', 'arb', 'ars', 'ary', 'arz',
                  'asm',
                  'ast', 'awa', 'ayr', 'azb', 'azj', 'bak', 'bam', 'ban', 'bem', 'ben', 'bho', 'bjn', 'bjn', 'bod',
                  'bos',
                  'bug', 'bul', 'cat', 'ceb', 'ces', 'cjk', 'ckb', 'crh', 'cym', 'dan', 'deu', 'dik', 'dyu', 'dzo',
                  'ell',
                  'eng', 'epo', 'est', 'eus', 'ewe', 'fao', 'pes', 'fij', 'fin', 'fon', 'fra', 'fur', 'fuv', 'gla',
                  'gle',
                  'glg', 'grn', 'guj', 'hat', 'hau', 'heb', 'hin', 'hne', 'hrv', 'hun', 'hye', 'ibo', 'ilo', 'ind',
                  'isl',
                  'ita', 'jav', 'jpn', 'kab', 'kac', 'kam', 'kan', 'kas', 'kas', 'kat', 'knc', 'knc', 'kaz', 'kbp',
                  'kea',
                  'khm', 'kik', 'kin', 'kir', 'kmb', 'kon', 'kor', 'kmr', 'lao', 'lvs', 'lij', 'lim', 'lin', 'lit',
                  'lmo',
                  'ltg', 'ltz', 'lua', 'lug', 'luo', 'lus', 'mag', 'mai', 'mal', 'mar', 'min', 'mkd', 'plt', 'mlt',
                  'mni',
                  'khk', 'mos', 'mri', 'zsm', 'mya', 'nld', 'nno', 'nob', 'npi', 'nso', 'nus', 'nya', 'oci', 'gaz',
                  'ory',
                  'pag', 'pan', 'pap', 'pol', 'por', 'prs', 'pbt', 'quy', 'ron', 'run', 'rus', 'sag', 'san', 'sat',
                  'scn',
                  'shn', 'sin', 'slk', 'slv', 'smo', 'sna', 'snd', 'som', 'sot', 'spa', 'als', 'srd', 'srp', 'ssw',
                  'sun',
                  'swe', 'swh', 'szl', 'tam', 'tat', 'tel', 'tgk', 'tgl', 'tha', 'tir', 'taq', 'taq', 'tpi', 'tsn',
                  'tso',
                  'tuk', 'tum', 'tur', 'twi', 'tzm', 'uig', 'ukr', 'umb', 'urd', 'uzn', 'vec', 'vie', 'war', 'wol',
                  'xho',
                  'ydd', 'yor', 'yue', 'zho', 'zho', 'zul']

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

peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)


def init_objective(src_lang: str, tgt_lang: str, base_data_dir="data/example_data_dir") -> LoraLangObjective:
    lang_dir = os.path.join(base_data_dir, "%s-%s" % (src_lang, tgt_lang))
    try:
        objective = LoraLangObjective(lang_module,
                                      peft_config=peft_config,
                                      texts_or_path=os.path.join(lang_dir, "train.src.gz"),
                                      labels_or_path=os.path.join(lang_dir, "train.trg.gz"),
                                      val_texts_or_path=os.path.join(lang_dir, "test.src"),
                                      val_labels_or_path=os.path.join(lang_dir, "test.trg"),
                                      source_lang_id=src_lang,
                                      target_lang_id=tgt_lang,
                                      batch_size=4,
                                      val_evaluators=evaluators,
                                      objective_id=tgt_lang,
                                      max_samples_per_eval_log=20)
    except FileNotFoundError as e:
        # test split does not exist
        print("Test split of %s-%s not found. We will not perform evaluation on this pair." % (src_lang, tgt_lang))
        objective = LoraLangObjective(lang_module,
                                      peft_config=peft_config,
                                      texts_or_path=os.path.join(lang_dir, "train.src.gz"),
                                      labels_or_path=os.path.join(lang_dir, "train.trg.gz"),
                                      source_lang_id=src_lang,
                                      target_lang_id=tgt_lang,
                                      batch_size=4,
                                      objective_id=tgt_lang,
                                      max_samples_per_eval_log=20)
    return objective


objectives = [init_objective("eng", tgt_lang, args.base_data_dir) for tgt_lang in tqdm(target_langs,
                                                                                       desc="Loading objectives...")]

# lang-specific merge checks:
# assert id(getattr(list(lang_module.trainable_models.values())[0].base_model.encoder.block[1].layer[0].SelfAttention.q, "sgn-LoraLangObjective_lora_A").default.weight) \
#     != id(getattr(list(lang_module.trainable_models.values())[1].base_model.encoder.block[1].layer[0].SelfAttention.q, "tah-LoraLangObjective_lora_A").default.weight)
# assert id(list(lang_module.trainable_models.values())[0].base_model.encoder.block[1].layer[0].SelfAttention.q.base_layer.weight) \
#     == id(list(lang_module.trainable_models.values())[1].base_model.encoder.block[1].layer[0].SelfAttention.q.base_layer.weight)
# In the case of objective-specific reference to lora modules:
# assert id(getattr(list(lang_module.trainable_models.values())[0].base_model.encoder.block[1].layer[0].SelfAttention.q, "sgn-LoraLangObjective_lora_A").default.weight) \
#     == id(getattr(list(lang_module.trainable_models.values())[0].base_model.encoder.block[1].layer[0].SelfAttention.q, "lora_A").default.weight)

training_arguments = AdaptationArguments(output_dir="checkpoints",
                                         learning_rate=4e-5,
                                         stopping_strategy=StoppingStrategy.ALL_OBJECTIVES_CONVERGED,
                                         stopping_patience=5,
                                         do_train=True,
                                         do_eval=True,
                                         warmup_steps=5000,
                                         gradient_accumulation_steps=8,
                                         logging_steps=7,
                                         eval_steps=500,
                                         save_steps=1000,
                                         num_train_epochs=10,
                                         evaluation_strategy="steps",
                                         # no_cuda=True
                                         )

schedule = ParallelSchedule(objectives=objectives, args=training_arguments)

adapter = Adapter(lang_module, schedule, args=training_arguments)

adapter.train()
