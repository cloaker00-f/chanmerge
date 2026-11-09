"""chanmerge 2.0: merged, exposure-corrected Chandra flux images for dust scattering halo work."""
__version__ = "2.0.0.dev0"

from .plan import RunPlan, Observation, Category, Group, RowSet, Band, ChipSet, ChipChoice  # noqa: F401
from .check import check_plan  # noqa: F401
from .tables import rebuild_index, load_index  # noqa: F401


def run(plan, **kw):
    # imported here so the light parts work without ciao installed
    from .run import run as _run
    return _run(plan, **kw)
