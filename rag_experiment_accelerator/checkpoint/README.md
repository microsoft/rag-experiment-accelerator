# Checkpoints

## What is a checkpoint?
Checkpoints are used to skip the processing of data that has already been processed in previous runs.
A checkpoint object is used to wrap methods, so when the method is called with an ID that was called before, instead of executing the method, the checkpoint will return the result of the previous execution.

## Usage

### 1. Initialize the checkpoint object:
```python
init_checkpoint(checkpoint_name, config)
```

### 2. Wrap the method you want to cache with the checkpoint decorator:
```python
@cache_with_checkpoint(id="arg2.id")
def method(arg1, arg2):
    pass
```

or wrap the method using the checkpoint object:
```python
 get_checkpoint().load_or_run(method, arg2.id, arg1, arg2)
```

(arg2.id is the ID that uniquely identifies the call in this example)

This call will check if the provided method has previously been executed with the given ID, If it has, it returns the cached result, otherwise, it executes the method with the given arguments and caches the result for future calls.

## Checkpoint types

### Checkpoint
The base class for all checkpoints. It provides the basic functionality for initializing and retrieving the checkpoint instance.

A Checkpoint object is a singleton, meaning, only one checkpoint instance exists at a time.
To create a new checkpoint instance (or to override the existing instance), use the `init_checkpoint` method, this method will create a checkpoint object according to the provided configuration.

To get the current checkpoint instance, use the `get_checkpoint` method.

A checkpoint object is created for every step and index name, so if you change the index name data will not be saved and loaded from the checkpoint of the previous index.

### LocalStorageCheckpoint
Checkpoint implementation for the local executions of the pipeline (i.e. the developer's machine), uses the `pickle` library for serializing and persisting the method results to the local storage.
The checkpoint data is saved in the `artifacts/checkpoint` directory.

### NullCheckpoint
Checkpoint implementation that does not cache any data. This is useful when you want to disable the checkpointing mechanism.

## Deleting Checkpoint data
To delete the checkpoint data, simply run the following `Make` command:
```bash
make clear_checkpoints
```