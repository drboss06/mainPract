def round_decorator(metric_func):
    """
    Decorator that applies round() function to
    pandas.Series.
    """

    def wrapper(cls, *args, **kwargs):
        pd_series = metric_func(cls, *args, **kwargs)
        if cls._round is not None:
            pd_series = pd_series.round(cls._round)
        return pd_series

    return wrapper
