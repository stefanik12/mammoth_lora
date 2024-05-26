import copy
import functools
import itertools
import logging
from typing import Any, List, Union, Dict, Optional, Tuple, Iterable, Callable

import torch
from adaptor.objectives.objective_base import UnsupervisedObjective, Objective
from adaptor.objectives.seq2seq import Sequence2Sequence, Sequence2SequenceMixin, SequentialMixin
from peft import PeftConfig
from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
from transformers import DataCollatorForSeq2Seq, BatchEncoding
from transformers.modeling_outputs import Seq2SeqLMOutput

logger = logging.getLogger()


class Sequence2SequenceBaseline(Sequence2Sequence):

    def __init__(self, *args,
                 source_texts_prefix_fn: Optional[Callable[[str, str], str]] = None,
                 inverse_direction: bool = False,
                 **kwargs):
        self.source_texts_prefix_fn = source_texts_prefix_fn
        self.inverse_direction = inverse_direction
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
        if self.inverse_direction:
            logger.warning("Changing translation direction (to %s->%s) for objective %s",
                           self.target_lang_id, self.source_lang_id, self.given_id)
            sources_iter, targets_iter = targets_iter, sources_iter

        if self.source_texts_prefix_fn is not None:
            objective_lang = self.target_lang_id if not self.inverse_direction else self.source_lang_id
            sources_iter = map(lambda src_text: self.source_texts_prefix_fn(src_text, objective_lang), sources_iter)
        return sources_iter, targets_iter

    def _compute_loss(self,
                      model_outputs: Seq2SeqLMOutput,
                      labels: torch.LongTensor,
                      inputs: Optional[Union[BatchEncoding, Dict[str, torch.Tensor]]] = None) -> torch.Tensor:
        # override for a compatibility with OutputReturningLangModule needed for regularization in LoraLangObjective
        return super()._compute_loss(model_outputs.logits, labels, inputs)


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


class LangIndependenceRegularizer(UnsupervisedObjective, Sequence2SequenceMixin):

    def __init__(self, *args,
                 objectives: Tuple[Objective, Objective],
                 semantic_over_lang_sim_margin: float = 0.05,
                 embeddings_pooling_strategy: str = "mean",
                 **kwargs):
        assert "peft_objective" not in kwargs, "We don't want to regularize on a new module, but the original model"

        self.objectives = objectives  # must precede remaining initialization
        super().__init__(*args, peft_objective=False, batch_size=1, **kwargs)
        self.val_texts = []  # overrides "is None" condition that skips this objective from evaluation (loss)
        self.dataset_length = self.objectives[0].dataset_length  # both objectives' datasets have identical length
        # TODO: check that self.compatible_head_model points to the base model

        self.semantic_over_lang_sim_margin = semantic_over_lang_sim_margin
        self.embeddings_pooling_strategy = embeddings_pooling_strategy

        device = self.compatible_head_model.device
        self.batch_size = self.objectives[0].batch_size
        self.src_embeddings_mask = torch.arange(start=0, end=self.batch_size).to(device)
        self.tgt_embeddings_mask = torch.arange(start=self.batch_size, end=2*self.batch_size).to(device)

    def _get_inputs_iterator(self, split: str) -> Iterable[Union[BatchEncoding, Dict[str, torch.Tensor]]]:
        fwd_iter, bwd_iter = (o._get_inputs_iterator(split) for o in self.objectives)
        if self.max_samples_per_log[split] < self.objectives[0].dataset_length[split]:
            # cut only the selected number of batches from both iterators
            fwd_iter = itertools.islice(fwd_iter, self.max_samples_per_log["eval"])
            bwd_iter = itertools.islice(bwd_iter, self.max_samples_per_log["eval"])

        # interleave two iterators: samples of both should be compatible with the base model
        pad_token_t = torch.tensor(self.tokenizer.pad_token_id, device=self.compatible_head_model.device)
        zero_token_t = torch.tensor(0, device=self.compatible_head_model.device)

        def pad(encoding: BatchEncoding, key: str, expected_length: int, token_t: torch.Tensor) -> BatchEncoding:
            tensor = encoding[key]
            actual_length = len(tensor)
            if actual_length < expected_length:
                tensor = torch.concat([tensor, token_t.expand((expected_length - actual_length))])
                encoding[key] = tensor
            return encoding

        def concat_batches(iter1: Iterable[BatchEncoding], iter2: Iterable[BatchEncoding]) -> Iterable[BatchEncoding]:
            for batch1, batch2 in zip(iter1, iter2):
                new_batch = ([{k: batch1[k][i] for k in batch1.keys()} for i in range(len(batch1["input_ids"]))] +
                             [{k: batch2[k][i] for k in batch1.keys()} for i in range(len(batch2["input_ids"]))])

                longest_input_ids = max(len(x["input_ids"]) for x in new_batch)
                new_batch = [pad(x, "input_ids", longest_input_ids, pad_token_t) for x in new_batch]
                if "attention_mask" in batch1:
                    new_batch = [pad(x, "attention_mask", longest_input_ids, zero_token_t) for x in new_batch]

                longest_labels = max(len(x["labels"]) for x in new_batch)
                new_batch = [pad(x, "labels", longest_labels, zero_token_t) for x in new_batch]
                if "decoder_input_ids" in batch1:
                    longest_decoder_ids = max(len(x["decoder_input_ids"]) for x in new_batch)
                    new_batch = [pad(x, "decoder_input_ids", longest_decoder_ids, pad_token_t) for x in new_batch]

                yield self.collator(new_batch)

        return concat_batches(fwd_iter, bwd_iter)

    def _hidden_states_from_model_output(self, model_outputs: Seq2SeqLMOutput) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.compatible_head_model.config.is_encoder_decoder:
            encoder_hidden: torch.FloatTensor = torch.stack(model_outputs.encoder_hidden_states, dim=2)
            decoder_hidden: torch.FloatTensor = torch.stack(model_outputs.decoder_hidden_states, dim=2)
            assert encoder_hidden is not None and decoder_hidden is not None, \
                "Trained model does not seem to return hidden states."
            all_hidden = torch.hstack([encoder_hidden, decoder_hidden])
            return all_hidden[self.src_embeddings_mask], all_hidden[self.tgt_embeddings_mask]
        else:
            raise ValueError("This objective is now implemented only for encoder-decoder architectures.")

    def _compute_loss(self,
                      model_outputs: Seq2SeqLMOutput,
                      labels: torch.LongTensor,
                      inputs: Optional[Union[BatchEncoding, Dict[str, torch.Tensor]]] = None) -> torch.Tensor:
        if self.embeddings_pooling_strategy == "mean":
            langs, other_langs = ("src", "tgt"), ("tgt", "src")
            hidden = dict(zip(langs, self._hidden_states_from_model_output(model_outputs)))
            out_loss = torch.tensor(0., device=self.compatible_head_model.device)
            for anchor_lang, other_lang in zip(langs, other_langs):
                anchor_hidden = hidden[anchor_lang].mean(dim=(1, 2))
                positive_hidden = hidden[other_lang].mean(dim=(1, 2))  # TypeError: unsupported operand type(s) for -: 'set' and 'Tensor'
                negative_hidden = anchor_hidden.roll(shifts=1, dims=0)

                loss_fn = torch.nn.TripletMarginLoss(margin=self.semantic_over_lang_sim_margin)
                # Arguments format:       (anchor, positive, negative)
                out_loss += loss_fn(anchor_hidden, positive_hidden, negative_hidden) / anchor_hidden.numel()

            return out_loss
        else:
            # TODO: implement best-matching pooling here and compare with mean
            raise ValueError("Not implemented pooling strategy: %s" % self.embeddings_pooling_strategy)

    def __str__(self) -> str:
        if all(isinstance(o, SequentialMixin) for o in self.objectives):
            all_langs = set([o.source_lang_id for o in self.objectives] + [o.target_lang_id for o in self.objectives])
            return "-".join(sorted(all_langs)) + "-" + super().__str__()
        else:
            return super().__str__()
