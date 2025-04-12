from abc import ABC, abstractmethod
from arklex.utils.graph_state import MessageState


def register_worker(cls):
    """Decorator to register a worker."""
    cls.name = cls.__name__  # Automatically set name to the class name
    return cls


class BaseWorker(ABC):
    
    description = None

    def __str__(self):
        return f"{self.__class__.__name__}"

    def __repr__(self):
        return f"{self.__class__.__name__}"
    
    @abstractmethod
    def _execute(self, msg_state: MessageState):
        pass

    def execute(self, msg_state: MessageState):
        response_return = self._execute(msg_state)
        response_state = MessageState.model_validate(response_return)
        if response_state.trajectory and response_state.trajectory[-1] and response_state.trajectory[-1][-1]:
            response_state.trajectory[-1][-1].output = response_state.response if response_state.response else response_state.message_flow
        return response_state