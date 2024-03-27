import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING


model1 = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
model2 = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)

model1 = get_peft_model(model1, peft_config)
model2 = get_peft_model(model2, peft_config)

assert getattr(model1.base_model.encoder.block[1].layer[0].SelfAttention.q, "lora_A").default.weight \
 == getattr(model2.base_model.encoder.block[1].layer[0].SelfAttention.q, "lora_A").default.weight

assert getattr(model1.base_model.encoder.block[1].layer[0].SelfAttention.q, "lora_B").default.weight \
 == getattr(model2.base_model.encoder.block[1].layer[0].SelfAttention.q, "lora_B").default.weight

def mark_all_params_as_trainable(model: torch.nn.Module) -> None:
    for n, p in model.named_parameters():
        p.requires_grad = True


mark_all_params_as_trainable(model)

model.print_trainable_parameters()
