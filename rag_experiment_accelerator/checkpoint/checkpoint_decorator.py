from rag_experiment_accelerator.checkpoint.checkpoint import Checkpoint


def cache_with_checkpoint(key: str = None):
    """
    A decorator that can be used to cache the results of a method call using the globally initialized Checkpoint object.
    An key must be provided to the decorator, which is used to identify the cached result.
    If the method is called with the same id again, the cached result is returned instead of executing the method.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            if key is None:
                raise ValueError(
                    "'key' must be provided to the cache_with_checkpoint decorator"
                )

            eval_context = {**globals(), **locals(), **kwargs}
            arg_dict = {
                param: value
                for param, value in zip(
                    func.__code__.co_varnames[: func.__code__.co_argcount], args
                )
            }
            eval_context.update(arg_dict)

            try:
                evaluated_id = eval(key, eval_context)
            except Exception as e:
                raise ValueError(
                    f"Failed to evaluate the provided expression: {key}"
                ) from e

            checkpoint = Checkpoint.get_instance()

            return checkpoint.load_or_run(func, evaluated_id, *args, **kwargs)

        return wrapper

    return decorator
