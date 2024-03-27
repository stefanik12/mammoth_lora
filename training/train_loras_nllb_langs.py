from adaptor.adapter import Adapter
from adaptor.evaluators.generative import BLEU
from adaptor.lang_module import LangModule
from adaptor.schedules import ParallelSchedule
from adaptor.utils import AdaptationArguments, StoppingStrategy
from peft import LoraConfig, TaskType

from training.lora_lang_objective import LoraLangObjective

lang_module = LangModule("google/flan-t5-small")

evaluators = [BLEU(additional_sep_char="▁", decides_convergence=True)]

nllb_langs = ['ace', 'ace', 'acm', 'acq', 'aeb', 'afr', 'ajp', 'aka', 'amh', 'apc', 'arb', 'ars', 'ary', 'arz', 'asm',
              'ast', 'awa', 'ayr', 'azb', 'azj', 'bak', 'bam', 'ban', 'bem', 'ben', 'bho', 'bjn', 'bjn', 'bod', 'bos',
              'bug', 'bul', 'cat', 'ceb', 'ces', 'cjk', 'ckb', 'crh', 'cym', 'dan', 'deu', 'dik', 'dyu', 'dzo', 'ell',
              'eng', 'epo', 'est', 'eus', 'ewe', 'fao', 'pes', 'fij', 'fin', 'fon', 'fra', 'fur', 'fuv', 'gla', 'gle',
              'glg', 'grn', 'guj', 'hat', 'hau', 'heb', 'hin', 'hne', 'hrv', 'hun', 'hye', 'ibo', 'ilo', 'ind', 'isl',
              'ita', 'jav', 'jpn', 'kab', 'kac', 'kam', 'kan', 'kas', 'kas', 'kat', 'knc', 'knc', 'kaz', 'kbp', 'kea',
              'khm', 'kik', 'kin', 'kir', 'kmb', 'kon', 'kor', 'kmr', 'lao', 'lvs', 'lij', 'lim', 'lin', 'lit', 'lmo',
              'ltg', 'ltz', 'lua', 'lug', 'luo', 'lus', 'mag', 'mai', 'mal', 'mar', 'min', 'mkd', 'plt', 'mlt', 'mni',
              'khk', 'mos', 'mri', 'zsm', 'mya', 'nld', 'nno', 'nob', 'npi', 'nso', 'nus', 'nya', 'oci', 'gaz', 'ory',
              'pag', 'pan', 'pap', 'pol', 'por', 'prs', 'pbt', 'quy', 'ron', 'run', 'rus', 'sag', 'san', 'sat', 'scn',
              'shn', 'sin', 'slk', 'slv', 'smo', 'sna', 'snd', 'som', 'sot', 'spa', 'als', 'srd', 'srp', 'ssw', 'sun',
              'swe', 'swh', 'szl', 'tam', 'tat', 'tel', 'tgk', 'tgl', 'tha', 'tir', 'taq', 'taq', 'tpi', 'tsn', 'tso',
              'tuk', 'tum', 'tur', 'twi', 'tzm', 'uig', 'ukr', 'umb', 'urd', 'uzn', 'vec', 'vie', 'war', 'wol', 'xho',
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

nllb_eng_src_in_tatoeba = ['sgn', 'tah']

peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)


def init_objective(src_lang: str, tgt_lang: str, base_data_dir="data/example_data_dir/%s-%s") -> LoraLangObjective:
    try:
        objective = LoraLangObjective(lang_module,
                                      peft_config=peft_config,
                                      texts_or_path=(base_data_dir % (src_lang, tgt_lang)) + "/train.src.gz",
                                      labels_or_path=(base_data_dir % (src_lang, tgt_lang)) + "/train.trg.gz",
                                      val_texts_or_path=(base_data_dir % (src_lang, tgt_lang)) + "/test.src",
                                      val_labels_or_path=(base_data_dir % (src_lang, tgt_lang)) + "/test.trg",
                                      source_lang_id=src_lang,
                                      target_lang_id=tgt_lang,
                                      batch_size=2,
                                      val_evaluators=evaluators,
                                      objective_id=tgt_lang,
                                      max_samples_per_eval_log=9)
    except FileNotFoundError:
        # test split does not exist
        print("Test split of %s-%s not found. We will not perform evaluation on this pair." % (src_lang, tgt_lang))
        objective = LoraLangObjective(lang_module,
                                      peft_config=peft_config,
                                      texts_or_path=(base_data_dir % (src_lang, tgt_lang)) + "/train.src.gz",
                                      labels_or_path=(base_data_dir % (src_lang, tgt_lang)) + "/train.trg.gz",
                                      source_lang_id=src_lang,
                                      target_lang_id=tgt_lang,
                                      batch_size=2,
                                      objective_id=tgt_lang,
                                      max_samples_per_eval_log=9)
    return objective


objectives = [init_objective("eng", tgt_lang) for tgt_lang in nllb_eng_src_in_tatoeba]

# lang-specific merge checks:
# assert id(getattr(list(lang_module.trainable_models.values())[0].base_model.encoder.block[1].layer[0].SelfAttention.q, "sgn-LoraLangObjective_lora_A").default.weight) \
#     != id(getattr(list(lang_module.trainable_models.values())[1].base_model.encoder.block[1].layer[0].SelfAttention.q, "tah-LoraLangObjective_lora_A").default.weight)
# assert id(list(lang_module.trainable_models.values())[0].base_model.encoder.block[1].layer[0].SelfAttention.q.base_layer.weight) \
#     == id(list(lang_module.trainable_models.values())[1].base_model.encoder.block[1].layer[0].SelfAttention.q.base_layer.weight)
# In the case of objective-specific reference to lora modules:
# assert id(getattr(list(lang_module.trainable_models.values())[0].base_model.encoder.block[1].layer[0].SelfAttention.q, "sgn-LoraLangObjective_lora_A").default.weight) \
#     == id(getattr(list(lang_module.trainable_models.values())[0].base_model.encoder.block[1].layer[0].SelfAttention.q, "lora_A").default.weight)

training_arguments = AdaptationArguments(output_dir="experiments",
                                         learning_rate=2e-5,
                                         stopping_strategy=StoppingStrategy.ALL_OBJECTIVES_CONVERGED,
                                         stopping_patience=5,
                                         do_train=True,
                                         do_eval=True,
                                         warmup_steps=5000,
                                         gradient_accumulation_steps=4,
                                         logging_steps=7,
                                         eval_steps=2,
                                         save_steps=1000,
                                         num_train_epochs=10,
                                         evaluation_strategy="steps",
                                         no_cuda=True)

schedule = ParallelSchedule(objectives=objectives, args=training_arguments)

adapter = Adapter(lang_module, schedule, args=training_arguments)

adapter.train()
