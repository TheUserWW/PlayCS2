# CS2 Model and GSI Interface

This project is dedicated to CS2 and provides the model architecture and data interfaces. The trainer is not yet implemented, and there are no trained CS2 weights.

## Model Pipeline

```text
Single-frame RGB → CNN → visual features preserving a 3×5 spatial grid ───┐
Keyboard/mouse actions actually executed in the previous observation interval → action encoding ───────┤
Local player GSI → values, field availability, data age ──────┼→ per-timestep fusion
Screenshot timestamps → time deltas and elapsed time within the clip ──────────┘       ↓
                                             Causal Transformer
                                                     ↓
                    Forward/back movement / left/right movement / concurrent buttons / weapon slot / mouse X, Y
                                                     + value estimate
```

The implementation lives in `Transformer/game_policy.py`, using the `Block` from `Transformer/model.py` and its fused-Q/K/V `CausalSelfAttention`; the configuration lives in `Transformer/config.py`.
The original text model, tokenizer, legacy checkpoint buffers, and import fallback branches have all been removed.
The CNN encodes single-frame RGB; the model only uses CS2 state and action formats.
The default Transformer is 384-dimensional with 6 heads, 8 layers, and 128 timesteps; each step is fused into a single token.
The current default model has 15,565,403 parameters in total; the state vector is 94-dimensional and the historical action vector is 23-dimensional.
Input RGB is resized inside the model to 224×128 and normalized. This resolution is a configurable starting point, and its effect on long-range target recognition has not yet been validated.

## Inputs and Timing Relationships

`CS2Policy.forward()` accepts the following PyTorch tensors, all placed on the model's device:

| Parameter | Shape | Convention |
|---|---|---|
| `images` | `[B,T,3,H,W]` | uint8 RGB, 0–255 |
| `previous_actions` | `[B,T,ACTION_DIM]` | float32, produced by `encode_previous_action()` |
| `state_values` | `[B,T,STATE_DIM]` | float32, produced by `encode_gsi()` |
| `state_available` | `[B,T,STATE_DIM]` | bool, whether the field is actually available |
| `gsi_age` | `[B,T]` | float32, seconds since local reception |
| `gsi_fresh` | `[B,T]` | bool, whether unexpired GSI has been received |
| `timestamps` | `[B,T]` | float64 recommended; monotonic clock time of the screenshot |
| `valid` | `[B,T]` | optional bool; only right-side padding is supported; at least one frame per sequence |
| `episode_starts` | `[B,T]` | optional bool; first step of a new round / respawn / new clip |

**The input at time t is the current frame plus the action executed over the previous observation interval, and the output is the action to execute next.**
Training data must shift the action labels according to this convention; the action to be predicted must not be used as a same-step input.
At the start of a clip, use `encode_previous_action(None)`; the model also masks action inputs that are explicitly marked as episode starts.
The causal mask prohibits attending to the future; the episode mask prohibits attention across rounds; padding does not take part in computing valid timesteps.
The training loss must use the returned `valid` mask.

`encode_frames()` + `forward_features()` can be used to reuse visual features. Cache them only while the CNN is frozen or during inference;
once the CNN has been updated by training, old caches must be invalidated. The model does not implicitly retain history across training batches.

## Action Outputs

The return value is a dictionary of tensors, with an output at every timestep:

- `forward_logits`, `strafe_logits`: 3 classes each, namely none/forward/back and none/left/right, so opposite directions are never pressed at the same time.
- `button_logits`: 8 concurrent binary classifications; the order is defined by `cs2.actions.BUTTONS`.
- `weapon_logits`: 6 classes; 0 means keep the current weapon, 1–5 mean weapon slots.
- `mouse_x_logits`, `mouse_y_logits`: 19 relative mouse displacement buckets each, including fine displacements such as 0, ±1, ±2.
- `value`: one value estimate per step, reserved for later reinforcement learning.
- `hidden`: the temporal decision representation at each step; `valid`: the valid timesteps.

`decode_action()` performs deterministic decoding and returns a `CS2Action`; **it does not execute any keyboard or mouse operations**.
Displacement units are relative mouse input counts; the actual angle turned depends on settings such as game sensitivity, which need to be recorded and kept consistent.
In the first version, all output branches share the temporal representation; there is no explicit autoregressive dependency between actions, and no separately trained classifier for rotating or holding angles.

## GSI Reception

`cs2/gsi.py` uses the Python standard library to receive CS2 HTTP POSTs on the local machine. The default address matches the configuration already in the repository:
`http://127.0.0.1:3000/`. The receiver starts only when `start()` is called explicitly or when it is entered via `with`.

Copy the repository's `gamestate_integration_ai.cfg` into `game/csgo/cfg/` under the CS2 installation directory, then restart the game. The game installation directory is not modified automatically here.
The original 0.1-second buffer/throttle is retained, with the heartbeat changed to 1 second to avoid idle states being misjudged as long disconnections.
This is an asynchronous state update mechanism, **not a guarantee of 10 Hz sampling of frames, coordinates, or view angles**.

```powershell
python -m cs2.gsi
```

Optional authentication: add `"auth" { "token" "the value you set" }` to the cfg, and start the receiving end with
`python -m cs2.gsi --token 'the same value'`. When not set, data is received using the existing unauthenticated configuration; it listens on the local machine only.
The viewer and the recorder each occupy port 3000; do not start them at the same time.

The receiver checks CS2's `provider.appid = 730`, limits request size, and provides thread-safe historical snapshots.
Each POST replaces the current snapshot; `previously` / `added` are not wrongly merged into the current state; the auth token is not stored.
`GSIStore.at_or_before(captured_at)` returns only data received before the screenshot time.
GSI's server/game timestamps are not used for direct comparison with local screenshots; the local `time.monotonic()` reception time is used.
The reception time cannot eliminate game-side buffering and transmission latency; the data age expresses "how long since local reception".

## Availability Conventions

The state vector is defined by `cs2/state.py` and includes health, armor, economy, flash level, ammunition, current weapon,
team, round state, and more, with an availability mask for every feature. Field order is stored using a schema version and
`FEATURE_NAMES`; subsequent checkpoints and datasets must record the corresponding version together with the model configuration.

An actually observed zero (for example, zero health) is different from a missing field. Data that is missing, not yet received, stale for more than 2 seconds,
or later than the screenshot cannot serve as valid state. If `player.steamid` differs from `provider.steamid`,
that player's state is not treated as local player input. Unknown weapons have a dedicated unknown class.

GSI for regular players does not guarantee XYZ coordinates, pitch/yaw, or enemy positions; some spatial fields are available only when spectating.
The default configuration does not request spectator data, the model does not consume `allplayers`, and there are no fabricated coordinates or memory-reading fallback paths.
Screenshots and keyboard/mouse records remain independent sources; GSI only supplements game state.

Interface references:
- [Valve GSI documentation](https://developer.valvesoftware.com/wiki/Counter-Strike:_Global_Offensive_Game_State_Integration)
- [CounterStrike2GSI project documentation and configuration examples](https://github.com/antonpup/CounterStrike2GSI)

## Online Timeline API

```python
import time
import torch
from Transformer import CS2Policy
from cs2.gsi import GSIReceiver
from cs2.timeline import PolicyTimeline

policy = CS2Policy().eval()  # Randomly initialized; its outputs have no game-playing ability until it is connected to training.
with GSIReceiver() as receiver:
    timeline = PolicyTimeline(policy, receiver.store)
    captured_at = time.monotonic()
    rgb_frame = torch.zeros(3, 480, 640, dtype=torch.uint8)  # Replace with a real screenshot
    proposal, tensors = timeline.observe(
        rgb_frame, captured_at=captured_at, previous_action=None,
    )
    # Once an executor is connected in the future, the next observe call should pass the action actually executed, not a copy of the proposal.
```

`PolicyTimeline` caches at most `block_size` CNN features and recomputes a bounded Transformer context on each call.
There is no KV cache, which avoids positional-encoding mismatches in the sliding window. **The caller must call `timeline.reset()` when the map changes, the round changes, the player dies and respawns,
the capture timeline is interrupted, or the model weights or device change**; the first frame after a restart passes None as the action.
Inference must use `eval()`. No screenshot loop or action executor is connected yet in this work.

## Recording and Project Entry Points

```powershell
python record_demo.py
```

Recording requires OpenCV, mss, and NumPy from `requirements-cs2.txt`.
The recorder now saves RGB, key presses, the GSI snapshot received before the screenshot, normalized features, availability, and reception times;
the data schema is `cs2-screen-gsi-v1`.
The existing desktop cursor delta has been explicitly renamed as a diagnostic field; it is not the game's raw relative mouse input and cannot be used as an aiming label.
Raw mouse action capture, the CS2 action executor, and the trainer are left for later work.

`python main.py` starts the CS2 GSI viewer, with the same arguments as `python -m cs2.gsi`.
The HL1 environment, navigation rewards, old training scripts, audio/depth-specific code, game binding configurations, and the
`GameMemoryReader` placeholder interface have been deleted; the existing data and asset directories are retained.

## Checks

A working PyTorch is required; GSI itself needs no third-party dependencies:

```powershell
python -m unittest discover -s tests -v
```

Coverage includes real local HTTP POSTs, optional authentication, malformed data, snapshot replacement, stale/missing/spectator information handling,
causality, isolation across clips, padding, gradients for all modalities, consistency between the sliding timeline and batched inference,
and checks of CUDA optimizer updates and clean exit in a separate child process.
These checks are not a substitute for live integration testing in CS2 or for evaluating in-game performance after training.

## Python / PyTorch Environment

The current Windows Python 3.15 previously had `torch 2.14.1+cu126` installed; a minimal backpropagation example completed its computation and then crashed with an access violation (-1073741819) on exit. The official issue record explains the discrepancy between the Windows cp315 packages and the early Python 3.15 binary interface, and the fix is included in 2.15. See
[PyTorch #200779](https://github.com/pytorch/pytorch/issues/200779).

The environment fix adopts the official stable `torch 2.15.0+cu132` and syncs the currently installed torchvision to the matching version.
The actual current versions are Python 3.15.1, torch 2.15.0+cu132, torchvision 0.29.2+cu132, CUDA 13.2.
The model itself does not depend on torchvision. `requirements-cuda.txt` pins this GPU build, and
`requirements-cs2.txt` constrains the model dependency to `torch>=2.15,<2.16`.

`tests/test_runtime_exit.py` runs CPU/GPU and worker-thread backpropagation in a separate Python process,
while checking both the output markers and the real process exit code, so that the earlier situation of "assertions pass but the process crashes on exit" is not missed.
The tests are not passed by force-exiting, skipping backpropagation, or swallowing error codes.

### Verification Results After the Fix

- All 13 tests pass in full, including CPU/GPU backpropagation on the main thread and worker threads, and child-process exit checks;
  the test runner exits normally with exit code 0.
- The default 15,565,403-parameter model completes a 128-step forward pass, backpropagation,
  an AdamW parameter update, and a weight save/load check on an RTX 3060 Laptop GPU; all gradients are finite, and the process exit code is 0.
- CUDA NMS operations in the installed torchvision work correctly; `python -m pip check` reports no dependency conflicts, with exit code 0.
- Static compilation, and the command-line help checks for the GSI viewer and recorder, are all normal.

Live GSI integration testing with CS2, real screen recording, and evaluation of in-game performance still need to be carried out later.
