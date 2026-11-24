"""MkDocs hook that stops mkdocs-exporter corrupting inline images.

The exporter rewrites every ``src`` in a page as a path under the site
directory before printing it, and passes the result through
``os.path.abspath``. A ``data:`` URI goes through the same rewrite, and
``abspath`` collapses the ``//`` runs inside the base64, so the plots the
pages embed inline print as a strip of noise. A ``data:`` URI needs no
resolving, so this leaves it as it is.
"""

from mkdocs_exporter.preprocessor import Preprocessor

_resolve_link = Preprocessor._resolve_link


def _keep_data_uris(self, url, base, root=None):
    if url.startswith("data:"):
        return url
    return _resolve_link(self, url, base, root)


Preprocessor._resolve_link = _keep_data_uris
