from adaptor.adapter import Adapter
from adaptor.evaluators.generative import BLEU
from adaptor.lang_module import LangModule
from adaptor.schedules import ParallelSchedule
from adaptor.utils import AdaptationArguments, StoppingStrategy
from peft import LoraConfig, TaskType

from training.lora_lang_objective import LoraLangObjective

lang_module = LangModule("google/flan-t5-small")

evaluators = [BLEU(additional_sep_char="▁", decides_convergence=True)]

base_data_dir = "data/example_data_dir/eng-%s"

target_langs = ["sgn", "tah"]

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)


objectives = [LoraLangObjective(lang_module,
                                peft_config=peft_config,
                                texts_or_path=(base_data_dir % tgt_lang) + "/train.src.gz",
                                labels_or_path=(base_data_dir % tgt_lang) + "/train.trg.gz",
                                val_texts_or_path=(base_data_dir % tgt_lang) + "/test.src",
                                val_labels_or_path=(base_data_dir % tgt_lang) + "/test.trg",
                                source_lang_id="eng",
                                target_lang_id=tgt_lang,
                                batch_size=2,
                                val_evaluators=evaluators,
                                objective_id=tgt_lang,
                                max_samples_per_eval_log=9)
              for tgt_lang in target_langs]

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
