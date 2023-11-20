from abc import ABC, abstractmethod

class LLMModel(ABC):

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name

    @abstractmethod
    def try_retrieve_model(self):
       pass
