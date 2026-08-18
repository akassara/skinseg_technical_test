# SkinSeg Technical Test

## Docker

Build the image:

```bash
docker compose build
```

The container expects nnU-Net data under `./nnunet_data` and mounts the repository dataset from `./data`. Start the default 10-epoch GPU training with:

```bash
docker compose run --rm skinseg
```

The launcher uses these defaults:

- `DATASET_ID=1`
- `CONFIGURATION=2d`
- `FOLDS="0 1 2"`
- `NNUNET_NUM_SPLITS=3`
- `TRAINER=nnUNetTrainer_10epochs`
- `DEVICE=cuda`

Override them at runtime, for example:

```bash
TRAINER=nnUNetTrainer_10epochs FOLDS="1" docker compose run --rm skinseg
```

For a CPU run, use:

```bash
DEVICE=cpu docker compose run --rm skinseg
```

The raw, preprocessed, and results directories are persisted on the host at:

```text
./nnunet_data/nnunet_raw
./nnunet_data/nnunet_preprocessed
./nnunet_data/nnunet_results
```

The preprocessing commands in `scripts/launch.sh` are currently disabled. Enable them when the raw dataset is mounted and needs to be prepared inside the container.
