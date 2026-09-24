# ITK-SNAP Deep Learning Segmentation (DLS) Plugins
This package contains Python scripts that enable deep learning segmentation models to be run from ITK-SNAP. The documentation is here:

* [itksnap-dls.readthedocs.io](https://itksnap-dls.readthedocs.io)

## Running with Docker (ADPKD)

The server can run in a container next to the adpkd-net segmentation service:

```bash
docker compose -f docker/compose.yml up --build
```

Then connect ITK-SNAP to `<host>:8911` and choose the `ADPKD` model. The segmentation starts when the image is uploaded; the first interaction waits for it (minutes on CPU).

The ADPKD connection is configured with environment variables (or a `.env` file):

| Variable | Default | Meaning |
|---|---|---|
| `ADPKD_URL` | `http://adpkd:9000` | adpkd-net job API |
| `ADPKD_SHARED_INPUT_DIR` | `/data/adpkd/inputs/itksnap` | Where input images are written (shared with adpkd-net) |
| `ADPKD_SHARED_OUTPUT_DIR` | `/data/adpkd/outputs` | Where adpkd-net writes `<job_id>/seg.nii.gz` |
| `ADPKD_RUN_SMALL` | `false` | Use the single, faster model |
| `ADPKD_RUN_CPU` | `false` | Run adpkd-net without a GPU |
| `ADPKD_POLL_INTERVAL` | `2.0` | Seconds between job status checks |
| `ADPKD_TIMEOUT` | `3600` | Seconds before a job is considered failed |

Both containers must mount the shared data volume at the same path, because file paths are passed between them. See the checklist at the top of `docker/compose.yml` for adding the adpkd-net service.

## For developers

* [Contributing Guidelines](CONTRIBUTING.md) — how to report bugs and submit pull requests.
* [Developer Guide](docs/developer.md) — architecture, the REST API, and how to add a model.
* [ITK-SNAP Governance](https://github.com/pyushkevich/itksnap/blob/master/GOVERNANCE.md) — how the project is run and who maintains it.
* [ITK-SNAP Code of Conduct](https://github.com/pyushkevich/itksnap/blob/master/CODE_OF_CONDUCT.md) — expectations for participation.
