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


class GradientsDotProduct(ContrastiveEvaluator):

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
            outputs = model(**{k: v for k, v in batch.items() if k != "labels"})  # drop labels to avoid computing loss
            loss = objective.compute_loss(outputs.logits, batch["labels"], batch["input_ids"], "eval")
            loss.backward()  # computes and retains grads for all params
            if agg_grads is None:
                agg_grads = {name: model.get_parameter(name).grad for name in param_names}
            else:
                # running aggregation
                new_grads_weight = (1 / norm)
                agg_weight = (1 - (1 / norm))
                agg_grads = {name: (agg_weight * agg_grads[name]) + (new_grads_weight * model.get_parameter(name).grad)
                             for name in param_names}
                norm += 1
        return agg_grads

    def __call__(self,
                 model: torch.nn.Module,
                 tokenizer: PreTrainedTokenizer,
                 dataset: AdaptationDataset) -> Union[float, Dict[str, float]]:
        own_lang_params = set(n for n, p in self.ref_objective.compatible_head_model.named_parameters())
        other_lang_params = set(n for n, p in self.other_objective.compatible_head_model.named_parameters())
        shared_params = own_lang_params & other_lang_params

        # 1. compute gradients on the objective's dataset
        own_gradients = self._gradients_for_objective(self.ref_objective, shared_params)
        # 2. compute gradients on other objectives' datasets
        other_gradients = self._gradients_for_objective(self.other_objective, shared_params)

        # 3. compare this objective's dataset with the behaviour of other datasets
        if self.grouping == "per-layer":
            out_dict = {}
            groups = [re.findall(r"block.[0-9]+", param_name) for param_name in own_gradients.keys()]
            for layer_group in sorted(dict.fromkeys(itertools.chain(*groups))):  # consistently sorted for logging
                group_params = [p for p in shared_params if layer_group in p]
                group_dot_products = [own_gradients[p].flatten().dot(other_gradients[p].flatten()) / own_gradients[p].numel()
                                      for p in group_params]
                group_dot_product_avg = torch.mean(torch.stack(group_dot_products))
                key = str(self) + "-" + layer_group.replace("block.", "layer.")
                out_dict[key] = group_dot_product_avg.item()
            return out_dict
        else:
            all_dot_products = [own_gradients[p].flatten().dot(other_gradients[p].flatten()) / own_gradients[p].numel()
                                for p in own_gradients.keys()]
            return torch.mean(torch.stack(all_dot_products)).item()

    def __str__(self):
        return super().__str__() + "_%s-%s" % (self.ref_objective.target_lang_id, self.other_objective.target_lang_id)
