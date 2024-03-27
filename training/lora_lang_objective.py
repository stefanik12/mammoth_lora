import copy
import functools
import logging
from typing import Any, List, Union

import torch
from adaptor.objectives.seq2seq import Sequence2Sequence
from peft import PeftConfig, get_peft_model
from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING

logger = logging.getLogger()


class LoraLangObjective(Sequence2Sequence):

    def __init__(self, *args, peft_config: PeftConfig, **kwargs):
        self.peft_config = peft_config
        super().__init__(*args, **kwargs)

    def mark_all_objective_params_as_trainable(self, model: torch.nn.Module) -> None:
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
        # first, get the new module for this target lang: registration method will perform parameter merging,
        # but as it relies on shared modules names and PEFT renames the modules, we will need to re-run the merging
        super().register_compatible_head_model(*args, **kwargs)
        lang_module = args[0]
        # then, extend it with PEFT modules
        lang_module.trainable_models[str(id(self))] = get_peft_model(lang_module.trainable_models[str(id(self))],
                                                                     self.peft_config)
        # rename PEFT components with lang_src-lang_tgt id, so that they are not merged within other modules
        peft_modules_prefix = PEFT_TYPE_TO_MODEL_MAPPING[self.peft_config.peft_type].prefix
        self.rename_peft_modules2(lang_module.trainable_models[str(id(self))], None,
                                  peft_modules_prefix, str(self), recursive_root=True)

        # re-run the merging to the first module with renamed PEFT components
        if len(lang_module.trainable_models) > 1:
            unmatched_modules = lang_module._partially_merge_models(list(lang_module.trainable_models.values())[0],
                                                                    lang_module.trainable_models[str(id(self))],
                                                                    no_merge_keys_containing=peft_modules_prefix)
            logger.warning("These layers of %s objective are not merged: %s" % (self.target_lang_id, unmatched_modules))

        # reset PEFT disabling of some parameters' training
        self.mark_all_objective_params_as_trainable(lang_module.trainable_models[str(id(self))])
        return lang_module.trainable_models[str(id(self))]
