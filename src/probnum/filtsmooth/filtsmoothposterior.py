"""Abstract Base Class for posteriors over states after applying filtering/smoothing"""
from abc import ABC, abstractmethod


class FiltSmoothPosterior(ABC):
    @abstractmethod
    def __call__(self, location):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError
