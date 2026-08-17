# Contributing to itksnap-dls

`itksnap-dls` is the deep-learning segmentation server for
[ITK-SNAP](https://github.com/pyushkevich/itksnap). It is part of the ITK-SNAP project and
follows the same governance and community standards:

- **[ITK-SNAP Governance](https://github.com/pyushkevich/itksnap/blob/master/GOVERNANCE.md)**
  — how the project is run and who maintains it.
- **[ITK-SNAP Code of Conduct](https://github.com/pyushkevich/itksnap/blob/master/CODE_OF_CONDUCT.md)**
  — expectations for participation, which apply here too.

This file covers what is specific to this repository. For an explanation of how the server
is put together, see the [Developer Guide](docs/developer.md).

## Ways to contribute

**Report a bug** at <https://github.com/pyushkevich/itksnap-dls/issues>. Please include:

- the `itksnap-dls` version (`pip show itksnap-dls`) and your Python version;
- your platform and device (`cpu`, `cuda`, `mps`), and the GPU model if relevant;
- the server's console output, which logs model loading and request handling;
- whether the problem appears when calling the REST API directly (for example with
  `curl` or the interactive docs at `http://localhost:8911/docs`) or only from within
  ITK-SNAP. This distinction is the single most useful thing you can determine, because it
  separates a server bug from an ITK-SNAP client bug.

**Ask a usage question** on the
[ITK-SNAP mailing lists](http://www.itksnap.org/pmwiki/pmwiki.php?n=MailingLists) rather
than the issue tracker.

**Contribute code.** Open an issue first for anything beyond a small fix — particularly
for new model wrappers or changes to the REST API, since ITK-SNAP releases are built
against a specific server API.

## Development setup

```bash
git clone https://github.com/pyushkevich/itksnap-dls.git
cd itksnap-dls
python -m venv .venv && source .venv/bin/activate    # Python 3.10 or newer
pip install -e ".[test]"
```

Run the server:

```bash
python -m itksnap_dls --port 8911
```

Run the tests:

```bash
pytest                              # fast tests only
pytest -m integration_nni           # requires a real nnInteractive model download
pytest -m integration_sam2          # requires a real SAM2 model download
```

The integration markers are declared in `pyproject.toml`. They are slow and download model
weights from Hugging Face, so they are not part of the default run — but **do** run the
marker relevant to any model code you change, since the fast tests exercise the API surface
rather than inference itself.

Build the documentation:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

## Pull requests

1. Fork the repository and create a topic branch off `main`.
2. Make your change, with tests where the behavior is testable.
3. Push to your fork and open a pull request against `main`.

**Commit messages** — a recommendation, not a requirement. The maintainers are trying out
the same convention as ITK-SNAP first: prefix the subject with `BUG:`, `ENH:`, `DOC:`,
`PERF:`, `STYLE:`, or `WIP:`, keep it under 72 characters, write it in the imperative mood,
and use the body to explain what and why. A message that clearly explains the change matters
much more than one that matches the format.

**Things that make a pull request easy to review.** These are suggestions; none is a
precondition for opening one, and a maintainer can help with any of them during review.

- The server still starts and `GET /status` responds.
- `pytest` passes.
- If you added or changed a route, the [Developer Guide](docs/developer.md) REST API
  section is updated to match.
- If you added a dependency, it is declared in `pyproject.toml` — and consider whether it
  belongs in the base dependencies or in an optional extra, since the base install already
  pulls in PyTorch and is large.

## Compatibility with ITK-SNAP

ITK-SNAP talks to this server over HTTP, and released versions of ITK-SNAP are in the field
talking to servers users installed separately. Treat the REST API as a published interface:

- The `/v2/...` routes are the current API; the unprefixed routes are retained for older
  ITK-SNAP clients. **Do not remove or change the meaning of an existing route** without
  discussion — it will break installed copies of ITK-SNAP.
- Add new capabilities as new routes, or as new fields in an existing response that older
  clients can ignore.

## Licensing

<!-- TODO(review): This repository's license needs reconciling before this file is
     merged. LICENSE.txt contains the GNU GPL v3 text, but pyproject.toml declares
     "License :: OSI Approved :: MIT License" in its classifiers. The statement below
     follows LICENSE.txt; if MIT is the intent, both this section and LICENSE.txt need
     to change instead. -->

itksnap-dls is distributed under the **GNU General Public License, version 3** (see
[`LICENSE.txt`](LICENSE.txt)). By submitting a contribution you agree that it may be
distributed under that license.
