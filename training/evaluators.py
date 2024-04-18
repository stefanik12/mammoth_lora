import abc
import itertools
import re
from typing import List, Dict, Iterable, Sequence, Union

import torch
from adaptor.evaluators.evaluator_base import EvaluatorBase
from adaptor.evaluators.generative import BLEU
from adaptor.objectives.objective_base import Objective
from adaptor.objectives.seq2seq import Sequence2Sequence
from adaptor.utils import AdaptationDataset
from peft.mixed_model import PEFT_TYPE_TO_MODEL_MAPPING
from transformers import PreTrainedTokenizer


class CustomBLEU(BLEU):

    def evaluate_str(self, expected_list, actual_list) -> float:
        expected_nonempty = [e for e, a in zip(expected_list, actual_list) if e and a]
        actual_nonempty = [a for e, a in zip(expected_list, actual_list) if e and a]
        return super().evaluate_str(expected_nonempty, actual_nonempty)

    def __str__(self) -> str:
        return super().__str__()


class ContrastiveEvaluator(EvaluatorBase, abc.ABC):
    def __init__(self,
                 ref_objective: Sequence2Sequence,
                 other_objective: Sequence2Sequence,
                 decides_convergence: bool = False):
        super().__init__(decides_convergence)

        self.ref_objective = ref_objective
        self.other_objective = other_objective


class LangGradients(ContrastiveEvaluator):

    def __init__(self, *args, grouping: str = "per-layer", **kwargs):
        super().__init__(*args, **kwargs)
        self.grouping = grouping

    @staticmethod
    def _gradients_for_objective(objective: Objective,
                                 param_names: Iterable[str]) -> Dict[str, torch.Tensor]:
        encoded_dataset = objective.get_dataset("eval", 0, add_oid=False, is_training_dataset=False)
        model = objective.compatible_head_model  # objectives keep models with correctly set routing for this dataset

        agg_grads = None
        norm = 2  # normalization in the running average
        for batch in encoded_dataset:
            outputs = model(**batch)  # forward also takes care of model-specific shifting decoder_input_ids

            # Optimisation: for custom objectives, make sure to compute loss with objective.compute_loss():
            # loss = objective.compute_loss(outputs.logits, batch["labels"], batch["input_ids"], "eval")
            outputs.loss.backward()  # computes and retains grads for all params
            if agg_grads is None:
                agg_grads = {name: model.get_parameter(name).grad for name in param_names}
            else:
                # running average
                new_grads_weight = (1 / norm)
                agg_weight = (1 - (1 / norm))
                agg_grads = {name: (agg_weight * agg_grads[name]) + (new_grads_weight * model.get_parameter(name).grad)
                             for name in param_names}
                norm += 1
        return agg_grads

    def __call__(self,
                 model: torch.nn.Module,
                 tokenizer: PreTrainedTokenizer,
                 dataset: AdaptationDataset) -> Dict[str, float]:
        own_lang_params = set(n for n, p in self.ref_objective.compatible_head_model.named_parameters())
        other_lang_params = set(n for n, p in self.other_objective.compatible_head_model.named_parameters())
        if hasattr(self.ref_objective, "peft_config"):
            peft_modules_prefix = PEFT_TYPE_TO_MODEL_MAPPING[self.ref_objective.peft_config.peft_type].prefix
            shared_params = [p for p in own_lang_params if not "." + peft_modules_prefix in p]
            own_params = [p for p in own_lang_params if "." + peft_modules_prefix in p]
        else:
            shared_params = own_lang_params & other_lang_params
            own_params = shared_params

        # 1. compute gradients on the objective's dataset
        own_gradients = self._gradients_for_objective(self.ref_objective, shared_params)
        # 2. compute gradients on other objectives' datasets
        other_gradients = self._gradients_for_objective(self.other_objective, shared_params)

        # 3. compare this objective's dataset with the behaviour of other datasets
        out_dict = {}
        if self.grouping == "per-layer":

            groups = [re.findall(r"(?:encoder|decoder).(?:layers|block).[0-9]+", param_name) for param_name in own_gradients.keys()]
            for layer_group in sorted(dict.fromkeys(itertools.chain(*groups))):  # consistently sorted for logging
                group_params = [p for p in shared_params if layer_group in p]
                group_cos = [torch.cosine_similarity(own_gradients[p].flatten(), other_gradients[p].flatten(), dim=0)
                             for p in group_params]
                group_cos_avg = torch.mean(torch.stack(group_cos))
                group_dot_products = [own_gradients[p].flatten().dot(other_gradients[p].flatten()) / own_gradients[p].numel()
                                      for p in group_params]
                group_dot_product_avg = torch.mean(torch.stack(group_dot_products))
                layer_idx = layer_group.replace("block.", "layers.")
                out_dict["%s-%s-%s" % (str(self), "cos_sim", layer_idx)] = group_cos_avg.item()
                out_dict["%s-%s-%s" % (str(self), "dot_prod", layer_idx)] = group_dot_product_avg.item()
        else:
            all_cos = [torch.cosine_similarity(own_gradients[p].flatten(), other_gradients[p].flatten(), dim=0)
                       for p in own_gradients.keys()]
            all_dot_products = [own_gradients[p].flatten().dot(other_gradients[p].flatten()) / own_gradients[p].numel()
                                for p in own_gradients.keys()]
            out_dict["%s-%s" % (str(self), "cos_sim")] = torch.mean(torch.stack(all_cos)).item()
            out_dict["%s-%s" % (str(self), "dot_prod")] = torch.mean(torch.stack(all_dot_products)).item()

        shared_gradients = torch.hstack([own_gradients[p].flatten() for p in shared_params]).mean()
        out_dict["%s-%s" % (str(self), "shared-mean")] = shared_gradients
        # logging of gradients for modules (applicable only for the modular training)
        if own_params:
            module_gradients = torch.hstack([own_gradients[p].flatten() for p in shared_params]).mean()
            out_dict["%s-%s" % (str(self), "modules-mean")] = module_gradients

        return out_dict

    def __str__(self):
        return super().__str__() + "_%s-%s" % (self.ref_objective.target_lang_id, self.other_objective.target_lang_id)
