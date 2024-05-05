from rag_experiment_accelerator.checkpoint.checkpoint import get_checkpoint


def run_with_checkpoint(id=None):
    """
    A decorator that can be used to cache the results of a method call using a Checkpoint object.
    An id must be provided to the decorator, which is used to identify the cached result.
    If the method is called with the same id again, the cached result is returned instead of executing the method again.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            if id is None:
                raise ValueError(
                    "'id' must be provided to the run_with_checkpoint decorator"
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
                evaluated_id = eval(id, eval_context)
            except Exception as e:
                raise ValueError(
                    f"Failed to evaluate the provided expression: {id}"
                ) from e

            checkpoint = get_checkpoint()
            return checkpoint.load_or_run(func, evaluated_id, *args, **kwargs)

        return wrapper

    return decorator
