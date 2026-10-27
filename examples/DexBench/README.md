# Finetuning RBY1-Wuji2 Model

This guide shows how to finetune GR00T N1.7 on data collected with the RBY1-M
robot and Wuji Hand 2. It provides absolute and arm-relative action configurations
for DexBench tasks.

This repository's `main` is upstream Isaac-GR00T's `n1.7-release` tag (commit
`23ace64f17aa5015259b8609d371eb61a357c776`) plus these DexBench examples. Follow the
[installation guide](../../README.md#installation) before running the commands
below from the repository root.

## Dataset

Prepare your data in [GR00T-flavored LeRobot v2.1 format](../../getting_started/data_preparation.md).
The `action` and `observation.state` columns contain absolute joint positions in
radians, using the following order:

| Group | Slice | Joint order |
| --- | --- | --- |
| `left_arm_joints` | `0:7` | `left_arm_0` through `left_arm_6` |
| `right_arm_joints` | `7:14` | `right_arm_0` through `right_arm_6` |
| `head_joints` | `14:16` | `head_0`, `head_1` |
| `hands_joints` | `16:56` | Joint stage, then left/right, then thumb/index/middle/ring/pinky |
| `torso_joints` | `56:62` | `torso_0` through `torso_5` |

Slices exclude the end index. The 62 policy joints exclude the four wheels.
The four stages are `cmc_flex`, `cmc_abd`, `mcp`, `ip` for the thumb, and
`mcp_flex`, `mcp_abd`, `pip`, `dip` for the other fingers. For each stage, list
the five left-hand joints followed by the five right-hand joints.

Use one current RGB frame from each selected camera. The default is the two
ego cameras; see [Camera Selection](#camera-selection) for the wrist cameras.

| Camera | LeRobot feature |
| --- | --- |
| `camera_ego_left` | `observation.images.camera_ego_left` |
| `camera_ego_right` | `observation.images.camera_ego_right` |
| `camera_wrist_left` | `observation.images.camera_wrist_left` |
| `camera_wrist_right` | `observation.images.camera_wrist_right` |

The DexBench-EA LeRobot sets for GR00T (`lerobot/gr00t_62d_320x180` in each task's dataset) use
320×180 frames downscaled from the simulator's 640×360. The official
`nvidia/GR00T-N1.7-3B` checkpoint supplies the image preprocessing settings:
shortest edge 256 and crop fraction 0.95. No image overrides are needed.
Use the same input preparation during evaluation.

## Handling the Dataset

Prepare the data columns in the 62-joint layout above. If the dataset does not
already contain a matching `meta/modality.json`, copy the example below.
Copying this file does not reorder a raw 66-joint export.

```bash
cp examples/DexBench/modality.json /path/to/dataset/meta/modality.json
```

The language instruction is read from `meta/tasks.jsonl` through `task_index`.
For arm-relative training, generate statistics with the following command:

```bash
export RBY1_ACTION_MODE=arm_rel
uv run python -m gr00t.data.stats \
  --dataset-path /path/to/dataset \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/DexBench/rby1_wuji2_config.py
```

For absolute training, set `RBY1_ACTION_MODE=abs` instead. A dataset with
both absolute and arm-relative statistics can be used for either mode. The arm
entries in `meta/relative_stats.json` have shape `[40, 7]`. Regenerate these
entries after changing the dataset; the upstream tool reuses existing entries.

## Camera Selection

Select the cameras with `RBY1_CAMERAS`, a comma-separated list of the camera
names above. The default is `camera_ego_left,camera_ego_right`. To train with
the left ego camera and both wrist cameras:

```bash
export RBY1_CAMERAS=camera_ego_left,camera_wrist_left,camera_wrist_right
```

The list order is the image order in the model input. The checkpoint records
it, so `RBY1_CAMERAS` is not needed for inference.

Every selected camera must appear in the dataset's `meta/info.json` features
and in the `video` section of `meta/modality.json`. The example `modality.json`
lists all four cameras; entries that are not selected are ignored. Selected
cameras must share one frame size, because their images are stacked after
preprocessing. Evaluation must send frames at the size used for training.

Statistics do not depend on the cameras, so existing statistics can be reused.
Each image adds vision tokens, so a third camera increases the step time.

## Action Configuration

Use [rby1_wuji2_config.py](rby1_wuji2_config.py) for both modes. Select the mode
with `RBY1_ACTION_MODE` before generating statistics or starting training.
The default is `abs`.

| `RBY1_ACTION_MODE` | Arms | Hands, head, torso |
| --- | --- | --- |
| `abs` | Absolute target positions | Absolute target positions |
| `arm_rel` | Target positions relative to the current observation | Absolute target positions |

Both configurations predict 40 actions from the current observation. Arm-relative
targets use the observation at the start of the chunk as the reference for all
40 steps. The native policy converts the predictions back to absolute positions.
The evaluation client selects how many actions to execute from each chunk.

## Finetuning

Run [train.sh](train.sh) from the repository root. It uses the upstream finetune
launcher with the RBY1-Wuji2 modality configuration, batch size 256,
30,000 steps, learning rate 1e-4,
cosine decay, 5% warmup, and state dropout 0.2. The visual encoder and LLM are
frozen. The projector and diffusion model are trained.

Run arm-relative finetuning on four GPUs:

```bash
RBY1_ACTION_MODE=arm_rel NUM_GPUS=4 uv run bash examples/DexBench/train.sh \
  --dataset-path /path/to/dataset
```

For absolute actions, set `RBY1_ACTION_MODE=abs` (the default). Checkpoints go to
`outputs/rby1-wuji2-abs` or `outputs/rby1-wuji2-arm_rel`. When `RBY1_CAMERAS` is
set, the camera names are appended, such as
`outputs/rby1-wuji2-arm_rel-ego_left-wrist_left-wrist_right`. Training resumes from
the latest checkpoint in the output directory, so keep a separate directory for
each camera set. Use the same mode when generating statistics and finetuning.
Set `NUM_GPUS` to change the GPU count.

Run arm-relative finetuning with the left ego camera and both wrist cameras:

```bash
RBY1_ACTION_MODE=arm_rel RBY1_CAMERAS=camera_ego_left,camera_wrist_left,camera_wrist_right \
  NUM_GPUS=4 uv run bash examples/DexBench/train.sh --dataset-path /path/to/dataset
```

Additional arguments are forwarded to the upstream launcher. For example,
append `--output-dir /path/to/run --max-steps 10000` to override the defaults.

To enable Weights & Biases logging, set `WANDB_ENTITY` and add `--use-wandb`.

## Inference Server

Run [serve.sh](serve.sh) from the repository root on the machine used for model
inference. It starts the upstream server with `NEW_EMBODIMENT` and forwards
the remaining arguments:

```bash
CUDA_VISIBLE_DEVICES=0 uv run bash examples/DexBench/serve.sh \
  --model-path outputs/rby1-wuji2-arm_rel/checkpoint-30000 \
  --device cuda \
  --host 0.0.0.0 \
  --port 5555
```

For absolute actions, use the checkpoint from `outputs/rby1-wuji2-abs`.
`RBY1_ACTION_MODE` is not needed for inference.
The checkpoint contains the modality configuration, normalization statistics,
and image preprocessing settings. Both modes return absolute joint targets;
the policy converts arm-relative predictions using the input state.

Connect the DexBench evaluation client to this machine's address and port.
Keep simulator joint and camera bindings, task settings, seeds, and execution
horizon in DexBench. The client must send every camera the checkpoint was
trained with; a wrist camera requires wrist rendering in the simulator. These examples predict 40 actions; the client selects
how many to execute before requesting another chunk, such as 8, 16, or 40.
The server's `--execution-horizon` option is for dataset replay only.

See [Policy API](../../getting_started/policy.md#using-server-client-architecture-for-remote-inference)
for the upstream client interface.
