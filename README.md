# ITK-SNAP Deep Learning Segmentation (DLS) Plugins
This package contains Python scripts that enable deep learning segmentation models to be run from ITK-SNAP. The documentation is here:

* [itksnap-dls.readthedocs.io](https://itksnap-dls.readthedocs.io)

## Running with Docker (ADPKD)

The server can run in a container next to the adpkd-net segmentation service. By default it runs on the CPU, so it starts on any machine:

```bash
docker compose -f docker/compose.yml up --build
```

To use an NVIDIA GPU, add the GPU override file:

```bash
docker compose -f docker/compose.yml -f docker/compose.gpu.yml up --build
```

The GPU variant needs the host to pass the GPU into containers:

* Windows: a current NVIDIA driver and Docker Desktop with the WSL2 backend.
* Linux: the NVIDIA driver and the NVIDIA Container Toolkit, set up with `sudo nvidia-ctk runtime configure --runtime=docker`.
* Check with `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`, which should list the GPU.

The server picks the GPU automatically when it is visible and logs `Using GPU 0: <name>`; otherwise it logs `No GPU available, using CPU.`. To always include the GPU override, set `COMPOSE_FILE=compose.yml;compose.gpu.yml` in `docker/.env` (use `:` instead of `;` on Linux and macOS).

On CPU-only machines:

* Set `ADPKD_RUN_SMALL=true`. ADPKD takes roughly 10–30× longer on the CPU, and the single model shortens that.
* Raise `ITKSNAP_DLS_CPU_THREADS` towards the number of CPU cores to speed up nnInteractive.
* `ADPKD_RUN_CPU` is not needed: adpkd-net falls back to the CPU on its own. It only forces the CPU on a GPU machine.

Then connect ITK-SNAP to `<host>:8911` and choose the `ADPKD` model. The segmentation starts when the image is uploaded; the first interaction waits for it (minutes on CPU).

The server is configured with environment variables (in the shell or `docker/.env`):

| Variable | Default | Meaning |
|---|---|---|
| `ADPKD_URL` | `http://adpkd:9000` | adpkd-net job API |
| `ADPKD_SHARED_INPUT_DIR` | `/data/adpkd/inputs/itksnap` | Where input images are written (shared with adpkd-net) |
| `ADPKD_SHARED_OUTPUT_DIR` | `/data/adpkd/outputs` | Where adpkd-net writes `<job_id>/seg.nii.gz` |
| `ADPKD_RUN_SMALL` | `false` | Use the single, faster model |
| `ADPKD_RUN_CPU` | `false` | Force adpkd-net onto the CPU even when a GPU is available |
| `ADPKD_POLL_INTERVAL` | `2.0` | Seconds between job status checks |
| `ADPKD_TIMEOUT` | `3600` | Seconds before a job is considered failed |
| `ITKSNAP_DLS_CPU_THREADS` | `2` | CPU threads for nnInteractive (also `--cpu-threads`) |

Both containers must mount the shared data volume at the same path, because file paths are passed between them. See the checklist at the top of `docker/compose.yml` for adding the adpkd-net service.

## For developers

* [Contributing Guidelines](CONTRIBUTING.md) — how to report bugs and submit pull requests.
* [Developer Guide](docs/developer.md) — architecture, the REST API, and how to add a model.
* [ITK-SNAP Governance](https://github.com/pyushkevich/itksnap/blob/master/GOVERNANCE.md) — how the project is run and who maintains it.
* [ITK-SNAP Code of Conduct](https://github.com/pyushkevich/itksnap/blob/master/CODE_OF_CONDUCT.md) — expectations for participation.
