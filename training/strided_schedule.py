import itertools
import logging
import os
from typing import Iterator

from adaptor.objectives.objective_base import Objective
from adaptor.schedules import Schedule

logger = logging.getLogger()


class StridedSchedule(Schedule):

    label = "sequential"

    def __init__(self, *args, num_batches_per_objective: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_batches_per_objective = num_batches_per_objective
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
            current_objective = next(objectives_loop)
            logger.info("Starting sampling from %s objective", str(current_objective))
            for _ in range(self.num_batches_per_objective):
                yield current_objective
