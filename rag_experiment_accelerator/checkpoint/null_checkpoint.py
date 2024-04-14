from rag_experiment_accelerator.checkpoint.checkpoint import Checkpoint


class NullCheckpoint(Checkpoint):
    def __init__(self):
        pass

    def exists(self):
        return False

    def load(self):
        pass

    def save(self, data):
        pass
