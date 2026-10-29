# PlayCS2

An AI player model built specifically for CS2: it fuses per-frame CNN features, historical keyboard/mouse actions, and official GSI state, and outputs concurrent keyboard/mouse actions through a causal Transformer. The model, data interfaces, and recorder are currently provided; the trainer is not yet implemented.

Interface and timeline conventions are described in [CS2_ARCHITECTURE.md](CS2_ARCHITECTURE.md).
The complete pretraining pipeline design for professional demos is in [PRETRAINING_DESIGN.md](PRETRAINING_DESIGN.md); the modules and commands described there are a plan to be implemented.

## Modules

| File | Purpose |
|---|---|
| `Transformer/config.py` | CS2 model configuration |
| `Transformer/model.py` | Causal multi-head self-attention with fused Q/K/V, and residual blocks |
| `Transformer/game_policy.py` | CNN, per-timestep fusion, multi-branch action heads and a value head |
| `cs2/actions.py` | CS2 action definitions, historical action encoding and decoding |
| `cs2/state.py` | GSI state, field availability, and staleness handling |
| `cs2/gsi.py` | Local GSI HTTP receiver and state viewer |
| `cs2/timeline.py` | Online timeline of screenshots, actually executed actions, and GSI |
| `record_demo.py` | Recording of RGB, key presses, and frame-aligned GSI |
| `tests/` | Checks for the model, GSI, GPU updates, and clean process exit |

The HL1 environment, coordinate-navigation rewards, the old training entry point, the memory-reading placeholder interface, and the text-model entry point have been removed.
Existing recordings, weights, logs, and asset data have not been deleted. The new model only accepts CS2 state and action formats.

## Installation

The current Windows / NVIDIA environment uses the stable PyTorch 2.15 release, which fixes the Python 3.15 ABI issue:

```powershell
python -m pip install -r requirements-cuda.txt -r requirements-cs2.txt
```

The model depends only on PyTorch, and the GSI receiver uses only the Python standard library; screen recording additionally requires OpenCV, mss, and NumPy.
The project does not depend on torchvision; to keep the packages already installed in the current Python environment consistent, the environment fix also upgraded its installed version in step.

## GSI

Copy `gamestate_integration_ai.cfg` into the `game/csgo/cfg/` folder of the CS2 installation directory, restart the game, and run:

```powershell
python main.py
```

The equivalent module entry point is `python -m cs2.gsi`. The default listening address is `http://127.0.0.1:3000/`.
GSI only supplements game state; screenshots and keyboard/mouse action records are each captured separately. GSI for regular players does not guarantee coordinates, view angles, or enemy positions.

Recording runs `python record_demo.py`, which starts its own GSI receiver and cannot occupy port 3000 at the same time as the viewer.
The desktop cursor delta is only a diagnostic field; raw relative mouse capture, which is what aiming training needs, is left for a later implementation.

## Model API

```python
from Transformer import CS2Policy, CS2PolicyConfig

policy = CS2Policy(CS2PolicyConfig()).eval()
```

The model is randomly initialized and has no trained CS2 playing ability. Input shapes and time-alignment requirements are covered in the architecture document.

## Verification

```powershell
python -m unittest discover -s tests -v
```

The tests include CPU/GPU backpropagation, CUDA parameter updates, tensor operations inside threads, and clean-exit checks in separate processes.
The parent process checks the child process's exit code; even if the backpropagation assertions pass, an access violation at exit still fails the test.
