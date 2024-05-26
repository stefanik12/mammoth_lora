import inspect

import torch
from adaptor.lang_module import LangModule


class OutputReturningLangModule(LangModule):

    def forward(self, return_loss: bool = True, **inputs) -> torch.LongTensor:
        """
        Performs forward pass over the head identified by the sample's `oid`.
        :param inputs: given head input arguments with corresponding values.
        :return: HF model class-specific outputs
        """
        try:
            selected_head_model = self.trainable_models[str(inputs["oid"].item())]
        except KeyError:
            raise ValueError("Requesting inference with the objective having no registered head."
                             "If you are using `extra_eval_objectives`, "
                             "do not forget to fill in their `share_other_objective_head`.")
        # include only correct inputs for a specific model
        list_of_model_specific_inputs = inspect.getfullargspec(selected_head_model.forward).args
        model_specific_inputs = {k: v for k, v in inputs.items() if k in list_of_model_specific_inputs}

        # including labels cause the loss to be computed twice - by objective + by HF models forward()
        # but labels are also used to infer decoder_input_ids of some models, so we need to pass it
        selected_head_output = selected_head_model(**model_specific_inputs, output_hidden_states=True)

        return selected_head_output
