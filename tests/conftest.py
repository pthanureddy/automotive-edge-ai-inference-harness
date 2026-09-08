from __future__ import annotations

import pytest

from edge_ai.synthetic import generate_windows
from edge_ai.training import fit_bundle


@pytest.fixture(scope="session")
def fitted_bundle():
    return fit_bundle(generate_windows(cycles_per_class=10, windows_per_cycle=3, seed=81))
