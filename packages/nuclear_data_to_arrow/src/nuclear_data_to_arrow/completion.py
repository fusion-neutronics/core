"""What makes a converted .arrow directory complete, and how to say so.

``version.json`` is the completion marker: the writers put it down as the very
last thing they do, so its presence means the conversion ran all the way
through. The converters' resume path reads it back to decide what to skip.

Both halves live here on purpose. They used to disagree, and the way that
surfaced was bad: the writers wrote ``version.json`` first, so every directory
an interrupted run left behind looked finished to the skip check. A rebuild
killed partway through U238 left two of its eight tables on disk, and a plain
re-run skipped it and called the library complete.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

MARKER = "version.json"

FORMAT_VERSION = 1

# Tables that every complete directory has, whatever the evaluation contains.
# The rest (urr, total_nu, fission_photon, subshells, compton, bremsstrahlung)
# are written only when the evaluation has that data, so their absence proves
# nothing. This is the mandatory set validate_arrow.py reads back.
REQUIRED_TABLES = {
    "neutron": ("nuclide.arrow", "fast_xs.arrow", "reactions.arrow"),
    "photon": ("element.arrow",),
}


def write_completion_marker(path, library="", data_version=""):
    """Write ``version.json``. Call this last, once every table is on disk.

    Written to a temporary name and renamed, so a process killed mid-write
    leaves either no marker or a complete one, never a half-written file that
    still satisfies ``is_file()``.

    ``data_version`` identifies the *published release* of the data, and is the
    field yamc compares a cached copy against (yamc issue #366). It is not
    ``converter_version``, which identifies the code: two rebuilds from the same
    converter produce the same ``converter_version`` and are still different
    data, which is exactly the case that has to invalidate a cache. The hosted
    objects are overwritten in place on a re-publish, so the URL and the cache
    key are identical before and after and this stamp is the only thing that
    differs.

    Supplied by whoever runs the build rather than derived here: only they know
    whether a run is a new release or a resumed one, and a value derived from
    the clock would make every resumed nuclide claim a different release from
    its siblings. Left empty it is written as an empty string, which yamc treats
    the same as absent.
    """
    from . import __version__

    path = Path(path)
    info = {
        "format_version": FORMAT_VERSION,
        "library": library,
        "data_version": data_version,
        "converter_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    tmp = path / f"{MARKER}.tmp"
    tmp.write_text(json.dumps(info, indent=2))
    os.replace(tmp, path / MARKER)


def is_complete(path, particle):
    """Is *path* a finished conversion, safe for a resume to skip?

    The marker alone would be enough for anything written since it started
    being written last. The table check is what catches the directories the
    old ordering already left on disk, where the marker is present and the
    tables are not.
    """
    path = Path(path)
    if not (path / MARKER).is_file():
        return False
    return all((path / table).is_file()
               for table in REQUIRED_TABLES[particle])
