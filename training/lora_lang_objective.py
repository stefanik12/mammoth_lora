import copy
import functools
import logging
from typing import Any, List, Union, Dict, Optional, Tuple, Iterable, Callable

import torch
from adaptor.objectives.seq2seq import Sequence2Sequence
from peft import PeftConfig, get_peft_model
from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
from transformers import DataCollatorForSeq2Seq

logger = logging.getLogger()


class Sequence2SequenceBaseline(Sequence2Sequence):

    def __init__(self, *args, source_texts_prefix_fn: Optional[Callable[[str, str], str]] = None, **kwargs):
        self.source_texts_prefix_fn = source_texts_prefix_fn
        super().__init__(*args, **kwargs)

        self.collator = DataCollatorForSeq2Seq(self.tokenizer, self.compatible_head_model,
                                               pad_to_multiple_of=8, max_length=256)

    def per_objective_log(self, split: str) -> Dict[str, float]:
        """
        Generates a log of this objective for a given split, using Evaluators of selected split.
        :param split: Split of the log. Either `train` or `eval`.
        :return: Dict of the format {split + objective_name + evaluator_name: evaluator_value}
        """
        out_logs = {}
        if split == "eval" and self.val_texts is None and self.val_texts_path is None:
            logger.warning("Skipping evaluation for %s" % self)
            return out_logs
        # aggregate per-progress_bar-steps, or per-evaluation-steps, keep the results of unprocessed evaluations
        loss_history = self.loss_history[split][-self.max_samples_per_log[split]:]
        mean_loss = sum(loss_history) / len(loss_history) if len(loss_history) else float("inf")
        self.evaluations_history[split]["loss"].append(mean_loss)

        out_logs["%s_%s_loss" % (split, self)] = mean_loss
        out_logs["%s_%s_num_batches" % (split, self)] = len(loss_history)

        for evaluator in self.evaluators[split]:
            dataset = self.get_dataset(split, 0, self.compatible_head_model.device,
                                       add_oid=False,
                                       is_training_dataset=False)
            # evaluator should already return an aggregated value, so unlike loss, we don't average it
            try:
                evaluator_value = evaluator(self.compatible_head_model, self.tokenizer, dataset)
            except IndexError:
                logger.error("Error decoding sources of %s in %s", evaluator, self)
                evaluator_value = float('inf') if evaluator.smaller_is_better else 0
            try:
                self.evaluations_history[split][evaluator].append(evaluator_value)
            except KeyError:
                self.evaluations_history[split][evaluator] = [evaluator_value]

            out_logs["%s_%s_%s" % (split, self, evaluator)] = evaluator_value

        return out_logs

    def _per_split_iterators(self, split: str) -> Union[Tuple[Iterable[str], Iterable[str]],
                                                        Tuple[Iterable[str], Iterable[str], Iterable[str]]]:
        sources_iter, targets_iter = super()._per_split_iterators(split)
        if self.source_texts_prefix_fn is not None:
            sources_iter = map(lambda src_text: self.source_texts_prefix_fn(src_text, self.target_lang_id), sources_iter)
        return sources_iter, targets_iter


class LoraLangObjective(Sequence2SequenceBaseline):

    def __init__(self, *args, peft_config: PeftConfig, freeze_shared_params: bool = False, **kwargs):
        self.peft_config = peft_config
        self.freeze_shared_params = freeze_shared_params
        assert "peft_objective" not in kwargs, "LoraLangObjective is enforced to be loaded as peft_objective=True."
        assert "objective_args_for_head_config" not in kwargs

        super().__init__(*args, objective_args_for_head_config={"peft_config": peft_config},
                         peft_objective=True, **kwargs)

    @staticmethod
    def mark_all_objective_params_as_trainable(model: torch.nn.Module) -> None:
        for n, p in model.named_parameters():
            p.requires_grad = True

    def rename_objective_peft_params(self, model: torch.nn.Module) -> torch.nn.Module:
        # from https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
        def rgetattr(obj: torch.nn.Module, attr: str, *args: List[Any]):
            def _getattr(obj, attr):
                return getattr(obj, attr, *args)

            return functools.reduce(_getattr, [obj] + attr.split('.'))

        def rsetattr(obj: torch.nn.Module, attr: str, val: torch.nn.Parameter):
            pre, _, post = attr.rpartition('.')

            return setattr(rgetattr(obj, pre) if pre else obj, post, val)

        for orig_param_name, orig_param_value in model.named_parameters():
            if PEFT_TYPE_TO_MODEL_MAPPING[self.peft_config.peft_type].prefix in orig_param_name:
                # orig_param_value = rgetattr(model, orig_param_name)
                new_param_name = "%s_%s_%s" % (self.source_lang_id, self.target_lang_id, orig_param_name)
                rsetattr(model, new_param_name, orig_param_value)
        return model

    def rename_peft_modules2(self,
                             orig_submodule: torch.nn.Module,
                             iter_submodule: Union[torch.nn.Module, None],
                             orig_name_prefix: str,
                             new_name_prefix: str,
                             recursive_root=False) -> Union[torch.nn.Module, None]:
        if recursive_root:
            iter_submodule = copy.deepcopy(orig_submodule)  # a copy of the original model used only for interation

        for name, _ in iter_submodule.named_children():
            orig_param = getattr(orig_submodule, name)
            iter_param = getattr(iter_submodule, name)

            if name.startswith(orig_name_prefix):
                orig_val = getattr(orig_submodule, name)
                setattr(orig_submodule, "%s_%s" % (new_name_prefix, name), orig_val)
                # delattr(orig_submodule, name)  # for peft compatibility, both attributes will reference the same keys
                if not hasattr(orig_submodule, "adapter_layer_names"):
                    raise ValueError()
                orig_submodule.adapter_layer_names = ("%s_%s" % (new_name_prefix, name))
            self.rename_peft_modules2(orig_param, iter_param, orig_name_prefix, new_name_prefix)

        if recursive_root:
            del iter_submodule

    def register_compatible_head_model(self, *args, **kwargs) -> torch.nn.Module:
        # first, get the new module for this target lang: registration method will perform parameter merging
        super(Sequence2SequenceBaseline, self).register_compatible_head_model(*args, **kwargs)
        lang_module = args[0]
        # rename PEFT components with lang_src-lang_tgt id, so that they are not merged within other modules
        peft_modules_prefix = PEFT_TYPE_TO_MODEL_MAPPING[self.peft_config.peft_type].prefix
        self.rename_peft_modules2(lang_module.trainable_models[str(id(self))], None,
                                  peft_modules_prefix, str(self), recursive_root=True)

        # re-run the merging to the first module with renamed PEFT components
        if len(lang_module.trainable_models) > 1:
            unmatched_modules = lang_module._partially_merge_models(list(lang_module.trainable_models.values())[0],
                                                                    lang_module.trainable_models[str(id(self))],
                                                                    no_merge_keys_containing=peft_modules_prefix)
            # TODO: find out if the second merging run has any sense
            logger.warning("These layers of %s objective are not merged: %s" % (self.target_lang_id, unmatched_modules))

        if not self.freeze_shared_params:
            # enable back training of the original model's parameters, disabled by PEFT
            self.mark_all_objective_params_as_trainable(lang_module.trainable_models[str(id(self))])
        return lang_module.trainable_models[str(id(self))]
