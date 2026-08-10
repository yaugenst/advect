"""NumPy-owned random-state policy during abstract staging."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import advect as ad
import advect.numpy._stage_lifecycle as stage_lifecycle
from advect.core._array_api.profiles import LATEST_ARRAY_API_VERSION

if TYPE_CHECKING:
    from collections.abc import Callable

_DIRECT_GENERATOR_BIT_GENERATOR = np.random.PCG64(7)


def test_rng_tripwire_uses_numpy_public_inventory() -> None:
    public_callables = tuple(
        name for name in np.random.__all__ if callable(getattr(np.random, name))
    )
    assert public_callables == stage_lifecycle._RNG_NAMES


def test_rng_tripwire_reuses_precomputed_canonical_guards() -> None:
    original_random = np.random.random

    with stage_lifecycle._ambient_rng_tripwire():
        first_guard = np.random.random
    assert np.random.random is original_random

    with stage_lifecycle._ambient_rng_tripwire():
        second_guard = np.random.random
    assert np.random.random is original_random
    assert second_guard is first_guard


def test_rng_tripwire_guards_and_restores_external_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, ...]] = []

    def external_random(shape: tuple[int, ...]) -> np.ndarray[Any, Any]:
        calls.append(shape)
        return np.ones(shape, dtype=np.float64)

    monkeypatch.setattr(np.random, "random", external_random)

    with stage_lifecycle._ambient_rng_tripwire():
        guarded_random = np.random.random
        assert guarded_random is not external_random
        np.testing.assert_array_equal(
            guarded_random((2,)),
            np.ones((2,), dtype=np.float64),
        )

    assert calls == [(2,)]
    assert np.random.random is external_random


@pytest.mark.parametrize(
    "rng",
    [
        pytest.param(np.random.default_rng(7), id="generator"),
        pytest.param(np.random.RandomState(7), id="random-state"),
        pytest.param(np.random.PCG64(7), id="bit-generator"),
    ],
)
def test_stage_rejects_captured_numpy_rng_instances(rng: object) -> None:
    def random_offset(x: object) -> object:
        _ = rng
        return x

    with pytest.raises(RuntimeError, match="ambient mutable random state"):
        ad.stage(random_offset, specs=(ad.ArraySpec((2,), "float32"),))


@pytest.mark.parametrize(
    "entrypoint",
    [
        pytest.param(np.random.Generator, id="generator-alias"),
        pytest.param(np.random.RandomState, id="random-state-alias"),
        pytest.param(np.random.default_rng, id="default-rng-alias"),
    ],
)
def test_stage_rejects_captured_numpy_rng_entrypoint_aliases(entrypoint: object) -> None:
    def uses_alias(x: object) -> object:
        _ = entrypoint
        return x

    with pytest.raises(RuntimeError, match="ambient mutable random state"):
        ad.stage(uses_alias, specs=(ad.ArraySpec((2,), "float32"),))


@pytest.mark.parametrize(
    "random_call",
    [
        pytest.param(lambda shape: np.random.random(shape), id="legacy-draw"),  # noqa: NPY002
        pytest.param(
            lambda shape: np.random.laplace(size=shape),  # noqa: NPY002
            id="previously-omitted-draw",
        ),
        pytest.param(lambda _shape: np.random.seed(7), id="global-seed"),  # noqa: NPY002
        pytest.param(lambda _shape: np.random.default_rng(), id="generator-construction"),
        pytest.param(
            lambda _shape, bit_generator=_DIRECT_GENERATOR_BIT_GENERATOR: np.random.Generator(
                bit_generator
            ),
            id="direct-generator-construction",
        ),
        pytest.param(lambda _shape: np.random.RandomState(), id="random-state-construction"),
    ],
)
def test_stage_rejects_ambient_numpy_rng_calls(
    random_call: Callable[[tuple[int, ...]], object],
) -> None:
    def random_offset(x: Any) -> Any:
        random_call(x.shape)
        return x

    with pytest.raises(RuntimeError, match="random-number generation"):
        ad.stage(random_offset, specs=(ad.ArraySpec((2,), "float32"),))


def test_stage_allows_deterministic_nested_autodiff() -> None:
    def loss(x: Any) -> Any:
        return np.sum(x * x)

    def nested_gradient(x: Any) -> Any:
        return ad.grad(loss)(x)

    program = ad.stage(
        nested_gradient,
        specs=(ad.ArraySpec((2,), "float32"),),
        array_api_version=min(np.__array_api_version__, LATEST_ARRAY_API_VERSION),
    )

    np.testing.assert_array_equal(
        program(np.array([1.0, 2.0], dtype=np.float32)),
        np.array([2.0, 4.0], dtype=np.float32),
    )


def test_stage_rejects_ambient_rng_inside_nested_autodiff() -> None:
    def random_loss(x: Any) -> Any:
        return np.sum(x * np.random.random(x.shape))  # noqa: NPY002 - tripwire target

    def nested_gradient(x: Any) -> Any:
        return ad.grad(random_loss)(x)

    with pytest.raises(RuntimeError, match="random-number generation"):
        ad.stage(
            nested_gradient,
            specs=(ad.ArraySpec((2,), "float32"),),
        )


def test_staging_rng_tripwire_restores_every_entry_point_after_exception() -> None:
    originals = {
        name: getattr(np.random, name)
        for name in stage_lifecycle._RNG_NAMES
        if hasattr(np.random, name)
    }

    class StageError(Exception):
        pass

    def fail_during_stage(_x: object) -> object:
        raise StageError

    with pytest.raises(StageError):
        ad.stage(fail_during_stage, specs=(ad.ArraySpec((2,), "float32"),))

    assert {name: getattr(np.random, name) for name in originals} == originals
    assert stage_lifecycle._RNG_PATCH_DEPTH == 0
    assert stage_lifecycle._RNG_ORIGINALS == {}


def test_staging_rng_tripwire_does_not_affect_other_threads() -> None:
    entered = Event()
    release = Event()

    def blocking_identity(x: Any) -> Any:
        entered.set()
        assert release.wait(timeout=5)
        return x

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            ad.stage,
            blocking_identity,
            specs=(ad.ArraySpec((2,), "float32"),),
            array_api_version=min(np.__array_api_version__, LATEST_ARRAY_API_VERSION),
        )
        assert entered.wait(timeout=5)
        try:
            generator = np.random.default_rng(7)
            assert np.random.default_rng(generator) is generator
            assert isinstance(generator, np.random.Generator)
            assert issubclass(type(generator), np.random.Generator)
            legacy = np.random.RandomState(7)
            assert isinstance(legacy, np.random.RandomState)
            sample = float(generator.random())
        finally:
            release.set()
        program: Any = future.result(timeout=5)

    assert isinstance(sample, float)
    np.testing.assert_array_equal(
        program(np.array([1.0, 2.0], dtype=np.float32)),
        np.array([1.0, 2.0], dtype=np.float32),
    )


def test_overlapping_staging_scopes_share_and_restore_the_rng_tripwire() -> None:
    rendezvous = Barrier(2)
    original_random = np.random.random

    def random_after_rendezvous(x: Any) -> Any:
        rendezvous.wait(timeout=5)
        return x + np.random.random(x.shape)  # noqa: NPY002 - tripwire target

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                ad.stage,
                random_after_rendezvous,
                specs=(ad.ArraySpec((2,), "float32"),),
            )
            for _index in range(2)
        ]
        for future in futures:
            with pytest.raises(RuntimeError, match="random-number generation"):
                future.result(timeout=5)

    assert np.random.random is original_random
    assert stage_lifecycle._RNG_PATCH_DEPTH == 0
    assert stage_lifecycle._RNG_ORIGINALS == {}
