import itertools
import logging
import os
from typing import Iterator, Iterable, Dict, Any

from adaptor.objectives.objective_base import Objective
from adaptor.schedules import Schedule, SequentialSchedule

logger = logging.getLogger()


class StridedSchedule(Schedule):

    label = "sequential"

    def __init__(self, *args, num_batches_per_objective: int, paired: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_batches_per_objective = num_batches_per_objective
        self.paired = paired
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.process_rank = int(os.environ.get("RANK", 0))
        logger.warning("Initializing Schedule with rank %s", self.process_rank)

    def _sample_objectives(self, split: str) -> Iterator[Objective]:
        """
        Sample objectives in a sequential order - each objective is sampled for its `dataset_length` steps.

        :param split: data split to iterate. `train` or `eval`. Currently, Schedule base uses only "train".
        :return: Iterator over the references to objectives.
        """
        # infinite loop - termination is determined by _should_stop() + _combine_datasets()
        if self.world_size >= len(self.objectives[split]):
            # if this training runs with more processes than objectives, each process samples exclusive objectives
            objectives_loop = itertools.cycle([o for i, o in enumerate(self.objectives[split].values())
                                               if i % self.world_size == self.process_rank])
        else:
            objectives_loop = itertools.cycle(self.objectives[split].values())

        while True:
            if not self.paired:
                current_objective = next(objectives_loop)
                logger.info("Starting sampling from %s objective", str(current_objective))
                for _ in range(self.num_batches_per_objective):
                    yield current_objective
            else:
                # pairs of objectives following each other are guaranteed to have semantically aligned inputs
                # alignment is necessary for language independence regularization in RegularizedLoraLangObjective
                first_objective = next(objectives_loop)
                second_objective = next(objectives_loop)
                for _ in range(self.num_batches_per_objective):
                    yield first_objective
                    yield second_objective

    def _combine_datasets(self, split: str) -> Iterable[Dict[str, Any]]:
        """
        This overrides sequential sampling in evaluation -- that does not work with regularization,
        which requires paired sampling.
        """
        if split == "train" or self.paired:  # Update: with OR for self.paired condition
            objective_sampler = self._sample_objectives(split)
        else:
            # evaluation split uses simple, sequential evaluation over objectives
            objective_sampler = SequentialSchedule.single_iteration_eval_sampling(self.objectives["eval"].values())

        objectives_data_samplers = {obj: self._sample_objective_dataset(obj, obj_i, split)
                                    for obj_i, obj in enumerate(self.objectives[split].values())}
        for i, objective in enumerate(objective_sampler):
            try:
                yield next(objectives_data_samplers[objective])
            except StopIteration:
                continue
            # stop on next requested batch, if we're in the should_stop state from on_log event
            if self.should_stop:
                return
