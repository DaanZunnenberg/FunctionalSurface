"""Shared utilities for the funcgarch package."""

__all__ = ['ResultContainer']

_ANSI_PURPLE: str = '\033[95m'
_ANSI_RESET:  str = '\033[0m'


class ResultContainer:
    """Generic container that stores arbitrary key-value pairs as attributes.

    Used to wrap scipy OptimizeResult and other structured outputs.
    """

    def __init__(self, **kwargs) -> None:
        for key, val in kwargs.items():
            self.__setattr__(key, val)

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        lines = []
        for key, val in self.__dict__.items():
            addr = f'{_ANSI_PURPLE}0x{id(val):x}{_ANSI_RESET}'
            lines.append(f'{key}: [{type(val).__name__} @ {addr}] = {val!r}')
        return '\n'.join(lines)


if __name__ == '__main__':
    import scipy.optimize

    fun = lambda x: (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2
    cons = (
        {'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2},
        {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
        {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2},
    )
    res = scipy.optimize.minimize(
        fun, (2, 0), method='SLSQP', bounds=((0, None), (0, None)),
        constraints=cons, options={'gtol': 1e-6, 'disp': True},
    )
    container = ResultContainer(**{key: res[key] for key in res.__dir__()})
    print(container)
