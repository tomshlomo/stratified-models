from typing import Iterable, Tuple, TypeVar, Union

T = TypeVar("T")
NestedT = Union[T, Tuple["NestedT", T]]


def flatten(nested: NestedT) -> Iterable[T]:
    if isinstance(nested, tuple):
        yield from flatten(nested[0])
        yield from flatten(nested[1])
    else:
        yield nested


def nest(flat: Iterable[T]) -> NestedT:
    flat = iter(flat)
    nested = next(flat)
    for t in flat:
        nested = (nested, t)
    return nested
