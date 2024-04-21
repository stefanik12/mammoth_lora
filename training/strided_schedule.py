import itertools
import logging
from typing import Iterator

from adaptor.objectives.objective_base import Objective
from adaptor.schedules import Schedule


logger = logging.getLogger()


class StridedSchedule(Schedule):

    label = "sequential"

    def __init__(self, *args, num_batches_per_objective: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_batches_per_objective = num_batches_per_objective

    def _sample_objectives(self, split: str) -> Iterator[Objective]:
        """
        Sample objectives in a sequential order - each objective is sampled for its `dataset_length` steps.

        :param split: data split to iterate. `train` or `eval`. Currently, Schedule base uses only "train".
        :return: Iterator over the references to objectives.
        """
        # infinite loop - termination is determined by _should_stop() + _combine_datasets()
        objectives_loop = itertools.cycle(self.objectives[split].values())

        while True:
            current_objective = next(objectives_loop)
            logger.info("Starting sampling from %s objective", str(current_objective))
            for _ in range(self.num_batches_per_objective):
                yield current_objective
