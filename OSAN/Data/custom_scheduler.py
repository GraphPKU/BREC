from typing import List
from copy import deepcopy
import random


class BaseScheduler:
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def __call__(self):
        return self.value


class MultiStepScheduler(BaseScheduler):
    def __init__(self, value: float, milestone: List[int], gamma: float):
        """
        Decay at the given milestones

        :param value:
        :param milestone:
        :param gamma:
        """
        super(MultiStepScheduler, self).__init__(value)
        self.milestone = deepcopy(milestone)
        self.counter = 0
        self.gamma = gamma

    def __call__(self):
        if not self.milestone:
            return self.value

        self.counter += 1
        if self.counter == self.milestone[0]:
            self.value *= self.gamma
            self.milestone.pop(0)

        return self.value


class StepScheduler(BaseScheduler):
    def __init__(self, value: float, gamma: float = 0.999):
        """
        Decay every step

        :param value:
        :param gamma:
        """
        super(StepScheduler, self).__init__(value)
        self.gamma = gamma

    def __call__(self):
        self.value *= self.gamma
        return self.value


class RandomScheduler(BaseScheduler):
    def __init__(self, value: float, upper: float = 1., lower: float = 0.):
        """

        :param value:
        :param upper:
        :param lower:
        """
        super(RandomScheduler, self).__init__(value)
        assert 0. < lower < upper
        self.upper = upper
        self.lower = lower

    def __call__(self):
        self.value = random.uniform(self.lower, self.upper)
        return self.value
