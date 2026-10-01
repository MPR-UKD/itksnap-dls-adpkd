# ITK-SNAP Deep Learning Segmentation (DLS) Plugins
This package contains Python scripts that enable deep learning segmentation models to be run from ITK-SNAP. The documentation is here:

* [itksnap-dls.readthedocs.io](https://itksnap-dls.readthedocs.io)

## Running with Docker (ADPKD)

`docker/compose.yml` runs two containers: the ITK-SNAP DLS server (`itksnap-dls`, models nnInteractive, SAM2 and ADPKD) and the ADPKD segmentation service (`adpkd`, built from the [adpkd-net](https://github.com/MPR-UKD/woznicki-adpkd) submodule). ITK-SNAP only talks to `itksnap-dls`; the `adpkd` service is reachable inside the compose network only.

### 1. Get the code with the submodule

```bash
git clone --recurse-submodules <this-repo-url>
# or, in an existing clone:
git submodule update --init
```

The submodule `libs/woznicki-adpkd` tracks its `http-api` branch, which contains the job API.

### 2. Configure

```bash
cp docker/.env.example docker/.env
```

Edit `docker/.env`; every variable is documented in the file. Relative paths are resolved from the `docker/` directory, so the defaults use `data/` and `models/` in the repository root:

```text
data/                     DATA_DIR, mounted at /data in both containers
  adpkd/inputs/itksnap/   images sent for segmentation (deleted after each run)
  adpkd/outputs/<job_id>/ adpkd-net results (seg.nii.gz, results.json)
models/                   MODELS_DIR
  itksnap/                nnInteractive and SAM2 weights, downloaded on first use
  adpkd/                  adpkd-net trained models (see below)
```

Both folders are gitignored and excluded from the Docker build. `data/` holds patient images, so keep it out of cloud-synced folders.

### 3. Add the ADPKD model weights

Download the `trained_models` folder from the [adpkd-net Google Drive](https://drive.google.com/drive/folders/1D2glVKAKcAdQmmqct964RZoxHCpyDqgc?usp=sharing) and copy its **contents** into `models/adpkd/`, so that `models/adpkd/nnUNet/` and `models/adpkd/preprocessing/` exist. Without them, ADPKD jobs fail; nnInteractive and SAM2 still work.

### 4. Start

On any machine (CPU):

```bash
docker compose -f docker/compose.yml up --build
```

With an NVIDIA GPU, add the GPU override file:

```bash
docker compose -f docker/compose.yml -f docker/compose.gpu.yml up --build
```

Then connect ITK-SNAP to `<host>:8911` (or the `PORT` from `docker/.env`) and choose the `ADPKD` model. The segmentation starts when the image is uploaded; the first interaction waits for it (minutes on the CPU).

ADPKD segments the kidneys (both in one structure) and the liver. Each click picks one of them into the active ITK-SNAP label, so they can go into separate labels: select a label, click on a kidney, then select another label and click on the liver. The right mouse button removes the organ again; scribbles and polygons pick the organ they cover most. All of this reuses the one segmentation per image.

### GPU

The GPU override needs the host to pass the GPU into containers:

* Windows: a current NVIDIA driver and Docker Desktop with the WSL2 backend.
* Linux: the NVIDIA driver and the NVIDIA Container Toolkit, set up with `sudo nvidia-ctk runtime configure --runtime=docker`.
* Check with `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`, which should list the GPU. The DLS server uses CUDA 12.8, so a recent driver (R570 or newer) is recommended.

Both services pick the GPU automatically when it is visible; the DLS server logs `Using GPU 0: <name>`, otherwise `No GPU available, using CPU.`. To include the GPU override by default, uncomment `COMPOSE_FILE` in `docker/.env`. This only applies when running `docker compose up` from inside the `docker/` directory, because Compose reads `docker/.env` only there or with `-f docker/compose.yml`.

On CPU-only machines:

* Set `ADPKD_RUN_SMALL=true`. ADPKD takes roughly 10–30× longer on the CPU, and the single model shortens that.
* Raise `ITKSNAP_DLS_CPU_THREADS` towards the number of CPU cores to speed up nnInteractive.

### Settings

`docker/.env` sets these variables for the compose file:

| Variable | Default | Meaning |
|---|---|---|
| `DATA_DIR` | `../data` | Shared data folder on the host, mounted at `/data` |
| `MODELS_DIR` | `../models` | Model weights on the host (`itksnap/`, `adpkd/`) |
| `PORT` | `8911` | Host port ITK-SNAP connects to |
| `ADPKD_RUN_SMALL` | `false` | Use the single, faster ADPKD model |
| `ITKSNAP_DLS_CPU_THREADS` | `2` | CPU threads for nnInteractive (also `--cpu-threads`) |

The DLS server also reads these ADPKD settings; the compose file already sets them to match the container layout, so they only need changing outside Docker:

| Variable | Default | Meaning |
|---|---|---|
| `ADPKD_URL` | `http://adpkd:9000` | adpkd-net job API |
| `ADPKD_SHARED_INPUT_DIR` | `/data/adpkd/inputs/itksnap` | Where input images are written (shared with adpkd-net) |
| `ADPKD_SHARED_OUTPUT_DIR` | `/data/adpkd/outputs` | Where adpkd-net writes `<job_id>/seg.nii.gz` |
| `ADPKD_RUN_CPU` | `false` | Force adpkd-net onto the CPU even when a GPU is available |
| `ADPKD_POLL_INTERVAL` | `2.0` | Seconds between job status checks |
| `ADPKD_TIMEOUT` | `3600` | Seconds before a job is considered failed |

Both containers must see the shared data at the same path (`/data`), because absolute file paths are passed between them.

## For developers

* [Contributing Guidelines](CONTRIBUTING.md) — how to report bugs and submit pull requests.
* [Developer Guide](docs/developer.md) — architecture, the REST API, and how to add a model.
* [ITK-SNAP Governance](https://github.com/pyushkevich/itksnap/blob/master/GOVERNANCE.md) — how the project is run and who maintains it.
* [ITK-SNAP Code of Conduct](https://github.com/pyushkevich/itksnap/blob/master/CODE_OF_CONDUCT.md) — expectations for participation.
