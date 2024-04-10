from rag_experiment_accelerator.checkpoint.checkpoint import Checkpoint


class NullCheckpoint(Checkpoint):
    def exists(self):
        pass

    def load(self):
        pass

    def save(self, data):
        pass
