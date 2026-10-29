# CS2 Professional Demo Pretraining Program Design

Status: design draft. This document defines the modules, data contracts, workflow, and acceptance requirements of the program to be implemented; the entry points, commands, configurations, and new action format described here are not yet implemented. This document does not modify any model or training code.

## 1. Goals and Scope

The first version trains a CS2 behavior policy that receives first-person frames, local player state, and action history. The main task is to imitate the professional player's next actions, jointly training the per-frame CNN, the causal Transformer, and the action output heads.

Existing foundation: a model with roughly 15.56 million parameters, 384 dimensions, 6 heads, 8 layers, and 128-step context; a 94-dimensional GSI state; an RTX 3060 Laptop with 6 GB; Python 3.15.1 and PyTorch 2.15.0+cu132. The current asset inventory is 17 `.dem` files, about 5.54 GB, in 7 match groups; content integrity and renderability have not yet been audited.

The deliverables of the first version are a reproducibly generated demonstration dataset, pretrained weights, an offline evaluation report, and an inference export package. It does not include reinforcement learning, buy-menu UI control, voice communication, audio understanding, long-term tactical planning over full rounds, or automatically playing real matches.

The first version's control interval at 20 Hz is 50 ms, and 128 steps cover at most about 6.4 seconds. It serves to establish basic control and short-term decision-making; this precision is not to be advertised as reproducing the full subtick actions of professional players. Raw commands and subtick data should be preserved so that 40/64 Hz can be evaluated later.

## 2. Overall Workflow and Dependencies

Raw demo → registration and group splitting → offline parsing → action quality audit → fixed-player first-person rendering → frame and command alignment → unified state and action labels → shard packing → short-sequence training → long-sequence joint training → validation and export.

The program is divided into independently retryable stages. `prepare` orchestrates the preparation steps, and `train` only reads data that has already passed checks; it must not launch the game or parse an entire match inside the training loop.

Raw demos are read-only and are never patched or overwritten. Every derived result records the source file hash, parser version, game build number, configuration hash, schema version, and generation time.

## 3. Program Modules and Directory Layout

The following is a proposed directory layout, not modules that currently exist:

| Module | Responsibility |
|---|---|
| `pretraining/cli.py` | Subcommands, arguments, status, and exit codes |
| `pretraining/catalog.py` | Demo registration, deduplication, match grouping, split manifests |
| `pretraining/parsers/demoparser.py` | Python parsing backend and unified intermediate format |
| `pretraining/audit.py` | Command integrity, change rates, anomalies, and semantic verification |
| `pretraining/render/plan.py` | Rendering tasks for rounds, players, and time spans |
| `pretraining/render/capture.py` | CS2 replay capture, frame manifests, task resumption |
| `pretraining/alignment.py` | Alignment of frames, ticks, commands, and events |
| `pretraining/labels.py` | Actions v2, masks, confidence, and provenance |
| `pretraining/state_adapter.py` | Local player state mapping and GSI timeline simulation |
| `pretraining/packing.py` | Indexing, sharding, verification, and atomic commits |
| `pretraining/dataset.py` | Trajectories, windows, right-side padding, and batching |
| `pretraining/losses.py` | Branch supervision, confidence, and normalization over valid labels |
| `pretraining/trainer.py` | Optimization, precision, gradient accumulation, saving and resuming |
| `pretraining/evaluation.py` | Natural-distribution evaluation, baselines, ablations, and reports |
| `pretraining/checkpoint.py` | Complete training snapshots, strict validation, inference export |
| `configs/pretraining/` | Three configurations: smoke, short, and sequence |

Data is stored in layers:

| Path | Contents |
|---|---|
| `demos/` | Raw `.dem` files, kept read-only |
| `datasets/cs2/catalog/` | File registration, provenance, and match grouping |
| `datasets/cs2/parsed/` | Commands, states, and events at native time precision |
| `datasets/cs2/rendered/` | POV frames and per-frame time manifests |
| `datasets/cs2/labels/` | States, actions, masks, and quality metrics |
| `datasets/cs2/shards/` | Image shards and columnar indexes for training |
| `datasets/cs2/manifests/` | Dataset versions, splits, statistics, and check results |
| `runs/pretrain/<run_id>/` | Configuration copies, training logs, reports, and checkpoints |
| `models/cs2/<export_id>/` | Model parameters, schema, preprocessing, and deployment calibration contracts |

Existing recordings, weights, and assets are neither automatically included in training nor automatically deleted. The training entry point explicitly receives one fixed manifest.

## 4. Stage One: Registration, Grouping, and Asset Audit

### 4.1 File Registration

Each demo records: file path, SHA-256, size, header information, map, recording build number, duration, tick range, recording type, teams, player list, source, and parsing status. The type is determined from the file contents; a demo must not be classified as GOTV or POV based only on the match name.

Cases such as unreadable headers, obvious truncation, or mismatched map assets go into an error list. For recoverable parts, only complete rounds that pass validation are exported, and the original error locations are retained.

### 4.2 Data Splitting

Splitting happens before sampling, statistics, and training. Splits are grouped by the same match/BO3, guaranteeing that all maps, player perspectives, and duplicate downloads of one match stay in the same split.

The current 7 match groups start from a 5-group training / 1-group validation / 1-group final test division, with a fixed manifest and random seed; maps are balanced as far as possible, but a match is never broken up for the sake of balance. Cross-map and cross-team generalization is an independent evaluation task and is not mixed with ordinary same-distribution validation.

All class weights, training sampling statistics, threshold choices, and any fitted normalization are derived from the training split only. The test set is not involved in hyperparameter tuning or checkpoint selection.

### 4.3 Parsing Backend

The first choice is the Python `demoparser2`, because its documentation includes view angles, states, weapons, events, and usercmd fields. Installability under Python 3.15 and parsing results on samples are verified first, and only then are the exact version and package hash pinned.

Pinning an old version number alone does not count as compatibility: official issue #340 recorded a case where delta usercmd decoding omissions left events and positions normal but action data stale, and the fix landed in 0.42.0. The first version requires a release that contains this fix and has been validated on this project's samples, and it also checks whether the required subtick/time fields are actually exposed.

If the public interface cannot preserve command times, update provenance, or complete subticks, old values expanded per tick must not be disguised as new commands. In that case the parsing adapter layer should be extended, or the full UserCmd mode of `demoinfocs-golang` evaluated; it is an alternative parsing backend, not another model format. The backend is never switched automatically without validation.

### 4.4 Data Quality Report

Reported separately per demo, round, player, and action branch:

- Whether each requested column actually exists, its non-null rate, plausible range, and number of state changes.
- Full/delta command recovery status, deltas without a baseline, undecodable records, and duplicated or non-advancing command numbers; diagnostics not exposed by the backend are explicitly marked as unknown.
- Command time coverage, anomalous intervals, subtick coverage, and command-sequence reset positions.
- The relationship between view-angle changes and command changes; auxiliary consistency between attack state and firing events.
- Anomalies where movement commands stay unchanged for a long time while position/velocity clearly changes; long-term invariance alone must not be judged an error.
- For each action branch, the number of frames and duration finally usable for supervision, their provenance, and rejection reasons.

Firing events, displacement, and ammunition changes are diagnostic clues, not absolute criteria in one-to-one correspondence with button states; checks must take cooldowns, reloads, weapon types, inertia, and collisions into account.

The following cases must not pass the quality gate automatically: commands that are almost entirely default values, command times that are unknown and cannot be aligned, undeterminable player identity, and state wrongly inherited across rounds. The program must display the rejection reasons, not merely print "parsing complete".

## 5. Stage Two: First-Person Frame Generation

### 5.1 Rendering Strategy

Rendering uses the installed CS2 replay feature to generate frames from each player's fixed first-person perspective; frames are never substituted by process-memory reads, DLL injection, or rendering from omniscient state. Offline demo parsing does not change the convention that actual running still uses screenshots and official GSI.

The initial sample selects one Dust2 demo, 2 players, and 3–5 complete rounds; coverage expands to all players only after validation. Ten player perspectives can add trajectories, but they still come from the same match and cannot be counted as 10 independent matches.

Rendering tasks are sharded into continuous round segments per player. A single game instance executes capture serially; parsing can run in parallel on the CPU, while game rendering and GPU training run separately to reduce contention and timing jitter.

### 5.2 Frame Contract

Source frames are unified to 1280×720, 16:9. The model side keeps scaling configurable; the first version still uses the existing 224×128, and actual running uses the same transformation. Spot checks must judge whether distant targets are erased by the scaling; if necessary, a higher resolution is evaluated under a new preprocessing version rather than silently changing the dataset.

First-person perspective, FOV/scope-state interpretation, HUD layout, and frame preprocessing are fixed. X-ray and spectator panels are turned off. Full-map enemy radar, spectator identities, and extra state that ordinary players would not have must never enter the input; regions that cannot be reliably reconstructed use a uniform mask, and actual running uses the same mask.

Crosshair style may be standardized, but its position and the fidelity of scoped views must be checked. Real visual changes such as flashes, smoke, scoping, being hit, and recoil are retained; difficult frames are not all filtered out.

Valve's TrueView improves playback by re-running client prediction, but timing differences in effects may still exist, and it may not be enabled by default when game builds mismatch. Each task records the recording/playback builds, the TrueView configuration, and its actually verifiable status; anything unknown is recorded as unknown. A replay must not be claimed to be a pixel-for-pixel record of the original screen.

When old maps or builds cannot be played normally, the task enters `blocked_assets` or `blocked_build`. Resources of unknown origin are never downloaded automatically, and raw demos are never modified.

### 5.3 Frame Times and Resumable Tasks

Each frame stores: task ID, player ID, round ID, frame index, demo time anchor, time-anchor source, frame path, frame perspective state, preprocessing version, and capture quality.

Actual game time and wall-clock time must be kept separate. Automatic capture requires a validated means of mapping frames to demo time; a tick must not be inferred merely from "N seconds elapsed since playback started". If the client cannot provide a reliable anchor, a manually assisted alignment task is produced first, and seemingly precise training labels are not generated.

Pauses, seeks, player switches, playback-speed changes, and the settling period after loading do not enter valid clips. Playback speed and client interpolation settings are fixed to avoid mixing timelines. Duplicate frames, dropped frames, and UI occlusions are marked separately.

Task states are planned / running / captured / aligned / rejected / blocked / failed. Each task commits and retries independently; a capture failure never overwrites a clip that has already passed validation.

## 6. Stage Three: Time Alignment and Resampling

Native ticks, command numbers, execution times, client times, and subticks are preserved first. Only when policy samples are constructed is everything mapped onto a unified game timeline; not all demos are hardcoded to a fixed tick rate.

The default control grid is 20 Hz: observation time `t_i` predicts the actions in the interval `[t_i, t_i+0.05s)`, and the previous action in the model input comes from the previous interval. Boundaries strictly follow the same rule, and actions from the target interval are prohibited from leaking into the history.

Command times and server execution times are explicitly distinguished by source. Measurable client/server/rendering offsets are recorded, and only monotonic, explainable alignment transforms are used. Labels must not be dynamically shifted per sample using future events in order to raise training scores.

Manual review covers at least 30 event clips, spanning shooting, counter-strafing, jumping, reloading, scoping, death, and player switches; labeled previews and error distributions are saved. 50 ms is the initial time resolution, not a pre-guaranteed accuracy. Clips whose alignment residual exceeds one control interval, or whose direction is ambiguous, are barred from the precise action-imitation data.

Direction reversals, multiple presses/releases, and rapid switches within an interval keep their raw records, and resampling ambiguity is reported. Conversions are never silently dropped by "taking the first one". The first version uses a fixed interval aggregation rule and acknowledges its coarseness; high-frequency critical clips are listed for separate evaluation, and the control frequency is raised if necessary, rather than discarding all counter-strafe transition clips.

There is no interpolation across death, rounds, maps, or discontinuous command segments. Identical frames produced by standing still normally are not deleted wholesale just because there is "no change". Missing frames may keep their true intervals, but segments beyond the alignment tolerance are cut; the previous frame is never repeated to fake a new observation.

## 7. Unified Action Format: CS2 Controls v2

### 7.1 Why an Adjustment Is Needed Before Training

The current model uses mouse counts and a single overall history-validity bit. Professional players may have different sensitivities, and demo actions may be partially missing, so new training data uniformly uses angular actions and per-branch validity. This is a new CS2 training contract; no backward-compatible loading of the old format is added.

### 7.2 Output Branches

| Branch | Target and semantics |
|---|---|
| forward | 3 classes: none, forward, back |
| strafe | 3 classes: none, left, right |
| jump / crouch / walk / use / attack / attack2 | Expected held state of the 6 buttons |
| reload / drop | Press events within an interval (2 of them), producing a pulse at execution time |
| weapon_slot | 6 classes: keep, slots 1–5 |
| delta_yaw / delta_pitch | Expected view-angle change over the next control interval, in degrees |

Movement direction is determined from the decoded local movement intent, with a fixed dead zone for analog settings; the first version aggregates by the direction that dominates in duration, and ties are decided by the last valid direction before the end of the interval. Button held states take the valid command at the interval boundary; short pulses are handled separately under the event semantics above. Intervals with multiple transitions, or that exceed the first version's expressive capacity, carry an ambiguity mark and are not conflated with high-precision labels.

Weapon-switch supervision preferentially comes from actual selection commands and their item-to-slot mapping. Observing only that the current weapon changed does not justify reconstructing the exact key press or wheel action by default; when provenance is insufficient, that branch is not supervised.

Angles use classification plus a residual:

- Yaw bucket centers: ±180, ±90, ±45, ±20, ±10, ±5, ±2, ±1, ±0.5, ±0.2, ±0.05, 0 — 23 classes in total, sorted numerically.
- Pitch bucket centers: ±89, ±45, ±20, ±10, ±5, ±2, ±1, ±0.5, ±0.2, ±0.05, 0 — 21 classes in total.
- Each axis adds a local residual output. The target is first assigned to the nearest bucket, then normalized using half the spacing to the adjacent bucket center on the side of the residual's sign as the scale; boundaries use one-sided spacing. Targets outside the defined range, or rotations that cannot be disambiguated, are marked invalid and are never silently clipped into reliable targets.

Ordered angle changes are preferentially obtained from reliable usercmd view-angle trajectories, with yaw wrap-around handled correctly. When the command stream is insufficient to distinguish rotations greater than half a turn, or the source mixes in replay camera/recoil, angle labels are marked unreliable. Mouse intent must not be claimed to be recovered merely from adjacent screenshots or entity orientation.

At execution time, angle commands are converted into relative mouse input from the fixed sensitivity, axis coefficients, scope state, and measured calibration. Raw mouse counts are kept for diagnostics and are not mixed in directly as cross-player targets. This conversion does not require reading real-time yaw/pitch through GSI, but the execution response needs separate validation.

### 7.3 Action History and Missing Labels

Action history values consist of 6 dimensions of direction one-hot, 8 dimensions of buttons, 6 dimensions of slot one-hot, and 2 dimensions of angles — 22 dimensions in total; plus 13 branch-validity bits, for 35 dimensions overall. The validity bits correspond to the two directions, the 8 buttons, the slot, and the two angle axes.

A zero input paired with a false validity bit means "missing" and cannot be interpreted as a real no-op. Training targets likewise store a mask, provenance, and confidence for each branch; the same action schema is used for demo data, actually executed actions at run time, and model export.

Label provenance levels:

- A: Direct commands verified by quality checks; participate in behavior cloning by default, with confidence 1.
- B: State-derived labels that can be independently verified; provenance is retained, and they do not participate in the first version's precise behavior cloning by default; later experiments may open them per branch, with a recommended starting weight of no more than 0.5.
- C: Inferred only from ambiguous evidence such as displacement or ammunition changes, or missing/stale data; the action supervision weight is 0.

When the entire dataset lacks trustworthy labels for a branch, it is reported as "this capability was not trained". The export package must carry a capability list; the deployment side must not treat a randomly initialized branch as a trained feature.

## 8. Information Scope of GSI State and Training Inputs

The field order, value meanings, and per-field masks of the existing 94-dimensional state are preserved. The offline conversion layer constructs features for the target player; it does not forge provider identity, and it does not pass spectator GSI directly into the existing local identity check function.

Only health, armor, weapons, ammunition, team, economy, and round state that can actually be obtained at run time are used. World coordinates, the states of all enemies, later damage, and final outcomes are not inputs to current decisions; they may be used for offline auditing, statistics, or explicit training labels.

When simulating the observation stream, the restricted local state is first formed on the native timeline, and GSI buffering, throttling, heartbeat, and missing data are then simulated according to the configuration. Each input sample may use only the latest state whose simulated reception time is no later than that sample, retaining gsi_age, gsi_fresh, and availability. Latency parameters are calibrated against actual GSI recordings; initial reproducible assumptions must be stated in the report and never treated as measured values.

When raw state is missing, the mask is false; backfilling from the future is never allowed. Simulated missing data and latency act only on the training inputs and do not modify the raw records used for quality verification. The validation set uses a fixed simulation seed and reports the ideal-state, calibrated-state, and stream-interruption stress tests together.

The current model receives neither professional team voice nor game audio. For actions that depend on such information, complete imitation is not promised; evaluation distinguishes information-incomplete situations from ordinary visible scenes.

## 9. Dataset Contract and Window Construction

### 9.1 Per-Step Records

Each step includes at least:

| Field group | Contents |
|---|---|
| Identifiers | dataset, match_group, demo_hash, map, round, player, episode, step |
| Time | game_time, raw tick/command mapping, actual interval length, alignment residual, source |
| Frames | image shard key, raw frame reference, rendering and preprocessing versions, frame availability |
| State | values[94], available[94], age, fresh |
| History | Action values from the previous actual interval, 13 validity bits |
| Target | Actions of the current target interval, per-branch masks, confidence, provenance, ambiguity marks |
| Boundaries | valid, episode_start, loss_mask, reason for death / round change / interruption |

Player names, SteamIDs, and match identities are used for traceability and grouping, not as identity features for predicting actions.

### 9.2 Trajectories and Windows

The first version takes only clips of living players during valid match phases. Pauses, warmup, loading, spectator UI, and the buy phase are outside the scope of action training; no further no-op labels are manufactured after death. Intervals at death boundaries that cannot be fully determined should be masked or truncated, with explicit statistics.

Windows never cross episodes. Lengths of 32, 64, and 128 steps are supported, with right-side padding; the model's valid and the targets' loss_mask are kept separate. The default 128-step window has 32 steps of context at the front and 96 steps counted in the loss; shorter windows use the same proportion of context. At the true first step of an episode, the history is invalid; when a window starts in the middle of a round, an existing previous action can still serve as input.

The Transformer applies causal attention over the entire window; loss_mask only determines where the loss is counted. Context does not imply unlimited memory, and a sequence must not be secretly spliced with another player's data at a window boundary.

### 9.3 Sampling

The initial training sampling mix is 50% natural uniform windows, 30% engagements, 10% utility throws, 10% target events, with a deterministic priority used for classification to prevent double counting. The proportions are an adjustable starting point; when events are scarce, sampling falls back to uniform and the true proportions are recorded.

Neither only kills nor only winning rounds are used, and actions before a loss are not all treated as mistakes. Repeated sampling of the same match, player, and event is capped. Validation and test always use the natural distribution, with event subsets reported separately; future events are used only for selecting training samples and for labels, never as input.

### 9.4 Storage and Space

Structured records use Parquet; images use randomly accessible JPEG shards with key indexes and are decoded to RGB at training time. One high-resolution rendered source with a manifest is kept first, and training shards are then generated per preprocessing version, without relying on millions of scattered files.

Estimated per hour of one player's trajectory: 20 Hz produces 72,000 frames; if the small images average 10–25 KB, this takes roughly 0.72–1.8 GB. This estimate excludes source images/video and indexes. Ten perspectives multiply the storage and rendering effort accordingly; the final dataset size cannot be inferred from the 5.54 GB size of the demos.

The program first measures the true frame size, valid player hours, and rendering speed on a small sample, and only then gives the full disk and time budget. Space is checked before writing shards; if space is insufficient, derivation tasks stop and raw assets are never deleted.

## 10. Training Objectives

The first version's main objective is masked behavior cloning. PPO is not used, and no pretense of online interaction is made on static demos.

Directions and slots use cross-entropy; buttons use per-branch BCEWithLogits; angle buckets use cross-entropy; angle residuals use Huber loss. Each branch is first normalized by its number of valid labels and confidence weights, and the supervised branches are then combined with weights, so that branches with more labels do not automatically dominate the loss.

Recommended initial group weights: movement 1, buttons 1, weapon slot 0.5, angle classification 1, angle residual 0.25. Branches inside the button group are normalized individually. Class-imbalance weights are computed from the training set only and are capped, for example at 5; validation still reports the raw metrics without artificial resampling.

A batch with no labels at all is skipped and counted; if this happens consecutively, training aborts and reports a data problem rather than marching through the whole run on zero loss. The number of valid labels, the gradients, and whether parameters actually update are checked for every branch.

The existing value head is given no loss in the first version's behavior cloning and is not treated as a reliable value output. If auxiliary heads for round outcomes or short-term events are added later, their targets must be named in a separate experiment and ablated against the main task. By default, no complex auxiliary objectives that might mask data errors are added.

Mild brightness/contrast and JPEG perturbations may be applied consistently within the same sequence. Arbitrary horizontal flips, random crops that move the crosshair, independent per-frame jitter, and geometric transformations that do not synchronously alter action semantics are prohibited.

## 11. Training Stages and Single-GPU Parameters

All values are starting points for the first round of experiments and need to be adjusted based on actual memory usage, speed, and validation results; neither training time nor playing strength is promised in advance.

| Item | smoke | short | sequence |
|---|---|---|---|
| Data | Small sample, single overfitting check | Training split | Training split |
| Sequence length | 32 | 32→64 | 128 |
| Model | Same as the production configuration | Production model | Production model |
| Micro-batch | 1 | 1 | 1 |
| Accumulation steps | 1 | 8 | 8 |
| CNN learning rate | 1e-4 | 1e-4 | 3e-5 |
| Transformer/action-head learning rate | 3e-4 | 3e-4 | 1e-4 |
| Termination | Check after roughly 200 updates | Validation convergence or budget reached | Validation convergence or budget reached |

Common settings: AdamW, weight_decay 0.01, gradient clipping norm 1.0, roughly 5% warmup followed by cosine decay, single GPU. Validation/save intervals are counted in optimizer updates: initially, evaluate every 500 updates, save `last` every 1000 updates, and save `best` separately. The smoke stage uses shorter intervals.

BF16 is used where the device check supports it; otherwise FP16 with GradScaler. micro-batch=1 and accumulation=8 mean that each optimizer update aggregates 8 windows; this does not mean 8 full windows are placed in GPU memory at once. Accumulate first, then unscale/clip/step; the learning-rate schedule likewise advances only on real optimizer steps.

In the 6 GB environment, the peak memory of a full training step is probed first. On OOM, the current configuration and error information are kept, uncommitted gradients are cleared, and the run exits with a suggestion to adjust via shorter sequences, chunked CNN encoding, or activation checkpointing; samples, resolution, and the action schema are never silently changed mid-training.

The DataLoader starts with worker=0, and tries 2 only after Windows spawn is confirmed to work; prefetching is limited, and pin_memory is enabled based on measurements. Parsing, game rendering, and training run separately. Logs include peak memory, read time, GPU utilization, and the number of valid loss-counted steps per second.

The first version jointly trains the CNN and the Transformer. Visual features may be cached only once the CNN is explicitly frozen and the preprocessing is fixed; the cache key includes the CNN weight hash, precision, and preprocessing version. Reusing old features after a CNN update is prohibited.

## 12. Behavioral Contract of the Training Loop

Each update follows: take a window → validate the batch and schema → preprocess RGB → encode history/state/time → causal forward pass → compute per-branch masked, confidence-weighted loss → accumulate the backward pass → clip and update → record statistics → validate/save on schedule.

Training uses correctly aligned expert action history, not actions from the target interval. During pretraining, model predictions are not substituted for label provenance; the discrepancy between expert history and true closed-loop behavior is exposed through dedicated evaluation and addressed later with reliable self-collected data or interactive learning.

A non-finite loss/gradient, the absence of valid supervision, read failures, and schema mismatches all produce explicit errors. Confirmed corrupted shards may be quarantined, but a new manifest must be generated; the dataset is never silently swapped mid-run.

## 13. Validation, Model Selection, and Success Criteria

### 13.1 Offline Metrics

Directions and slots report cross-entropy, macro-F1, confusion matrices, and label coverage; each button reports precision, recall, F1, PR-AUC, and the number of positives. Model quality must not be judged by accuracy inflated by the large number of no-ops.

Angles report classification loss and, after decoding, angle MAE/median/P90, stratified into subsets such as small tracking adjustments, turns, and scoped aiming; events report timing deviation within the control grid, without claiming precision beyond the sampling frequency. Metrics are computed only on reliable labels, and the amount of excluded data is reported at the same time.

Results are stratified by map, side, weapon, round phase, GSI latency/missingness, and data source. Validation inputs use the same restricted information scope as training.

### 13.2 Baselines That Must Be Compared

1. Simple baselines such as constant/no-op, repeating the previous action, and class frequencies.
2. Current-frame CNN with no history.
3. CNN + Transformer over frame history, with action history removed.
4. The full model.
5. State-only input, or frames masked out, to check whether the model is taking a shortcut through the HUD/GSI alone.

Comparisons use a fixed split, the same subset of reliable labels, and comparable training budgets, and record differences in model size. Shuffled history, altered time intervals, and state interruptions are stress tests and are not mixed into ordinary validation scores.

### 13.3 Model Selection

The primary metric is pre-specified as the normalized, trusted action-supervision loss on the natural validation distribution; button positive recall and angle error are monitored at the same time. When the main score improves but an important branch clearly worsens, the branch report is kept; a single score is not used to paper over regressions.

Early stopping starts with patience = 5 full validations; the maximum number of optimizer updates is an explicit configuration budget. Validation thresholds are tunable, but the test set is never used for them. `best` and `last` are saved separately; the final test serves only to report the final solution, and the best test score is never repeatedly cherry-picked.

### 13.4 Online Capability Boundaries

Demo playback does not change the world in response to model outputs, so metrics on demos are not closed-loop win rates. In the first stage after export, predictions and latency are recorded in a read-only shadow mode; afterwards, movement, aiming, counter-strafing, and target actions are evaluated in local, controllable scenarios.

Until a later capture program provides reliable raw relative mouse records, executor calibration, and the angular action contract, the existing desktop cursor delta is not used as an aiming label, and completion of real-machine action transfer is not declared.

## 14. Checkpoints, Resumption, and Export

A training checkpoint contains: all model parameters, the model/action/state schemas, the angle bucket and residual definitions, the optimizer, scheduler, and GradScaler, the current stage and optimizer step, accumulation state, RNG state, the data manifest/split hashes, preprocessing and parsing/rendering versions, the sampler position, and the best validation result.

Checkpoints are committed only at complete optimizer-step boundaries; temporary files are written and verified, then atomically replaced. Ctrl+C saves an `interrupt` checkpoint at a safe update boundary; if the interruption hits an incomplete accumulation, that group's gradients and data-cursor progress are discarded and execution returns to the last recoverable boundary — an exact restore to the middle of a batch is never claimed.

With worker=0 in the first version, the data cursor can be saved explicitly. Once asynchronous workers are added later, prefetch queues must be handled; before that is implemented, only restoration from explicit epoch/update boundaries is claimed, with no guarantee of bitwise per-sample reproduction. Enabling deterministic mode additionally requires recording the CUDA nondeterminism limitations.

Strict resumption requires the schema, dataset, and model shapes to match; layers are never silently ignored, old heads are never padded in, and old mouse counts are never converted. Switching short → sequence is an explicit stage migration: model parameters are kept, the optimizer and learning-rate schedule are rebuilt from the configuration, and the migration is recorded; it is not passed off as a strict resumption of the same run.

An export contains only inference parameters and the necessary configuration, not the optimizer. It is accompanied by a model card: the number of source matches, valid player hours, per-branch training coverage, validation scores, input limits, sampling rate, preprocessing masks, executor calibration requirements, and the capabilities not yet trained. The untrained status of the value head must be stated explicitly.

## 15. Command-Line and Task State Design

The following is a proposed subcommand interface, not commands that can currently be run:

| Command | Input | Output |
|---|---|---|
| `python -m pretraining scan` | Demo root directory | Catalog and grouping candidates |
| `python -m pretraining split` | Catalog, grouping list, seed | Fixed train/val/test manifest |
| `python -m pretraining audit` | Catalog, parsing configuration | Raw parsing, quality report, rejection list |
| `python -m pretraining render-plan` | Rounds that passed the audit | Player-perspective tasks and workload estimates |
| `python -m pretraining render` | Rendering tasks, explicit CS2 path | Images and frame-time manifests |
| `python -m pretraining align` | Frame manifests, commands, and events | Alignment results, manual spot-check previews |
| `python -m pretraining build` | Data that passed alignment | Labels, training shards, dataset manifest |
| `python -m pretraining prepare` | Configuration and asset list | Orchestrates the preparation stages above, resumable per stage |
| `python -m pretraining train` | Dataset manifest, stage configuration | Checkpoints and logs |
| `python -m pretraining evaluate` | Checkpoint, specified split | Offline report |
| `python -m pretraining export` | Best checkpoint | Inference model package |
| `python -m pretraining status` | Dataset/run path | Completed, failed, blocked, and remaining tasks |

`--dry-run` displays the plan, resources, and items to check without launching the game or training. `--resume` validates the task configuration and output hashes, reuses complete results, and never reuses half-written files. A first run must not by default expand all 17 assets into 10-perspective recordings.

Exit code conventions: 0 means the requested stage completed; 2 means a configuration/schema error; 3 means assets/labels failed the quality checks; 4 means rendering or resources are blocked; 5 means a training numerical/resource failure; 130 means user interruption. Detailed lists distinguish "some stages completed" from "the whole pipeline succeeded".

## 16. Tests and Acceptance

The necessary checks for the implementation stage include:

- Parsing fixtures: full, delta, mixed, missing baseline, command resets, interleaved players, and missing subtick; samples where stale content is returned in legacy columns must be detected.
- Data contracts: real zeros kept distinct from missing values, branch masks, no action inheritance across rounds, slot mapping, yaw wrap-around, and angle dimensions.
- Time causality: future frames/states/commands cannot influence past outputs, and modifying later-received data does not change earlier samples.
- Information isolation: all perspectives/maps are split by match group, hidden enemy fields never enter the inputs, and spectator images carry no omniscient overlays.
- Model: all supervised branches have gradients; a small batch can be overfitted; padding and context masks are correct; features are not cached before the CNN is frozen.
- Recovery: configuration and cursor are correct after an interruption, checkpoints are written atomically, and corrupted shards, configuration changes, and schema mismatches fail explicitly.
- Resources: a complete optimizer step, AMP, accumulation, and OOM reporting on 6 GB; the exit codes of CUDA processes and worker-thread backpropagation are 0.
- Capture: repeated rendering of the same small clip produces time mappings within the declared tolerance, and perspective switches and pauses are not packaged into continuous samples.

Data quality thresholds, alignment tolerances, valid-label thresholds, and acceptable online performance should all be measured and frozen in small-sample reports; this design draft does not fabricate claims that they have been passed.

## 17. Implementation Order and Definition of Done

| Milestone | Work content | Reviewable deliverables |
|---|---|---|
| M1 | scan, split, parsing, and audit | A field/timing/label coverage report for one demo, with trainable branches made explicit |
| M2 | Replay and alignment small sample | POV, time manifests, and annotated previews for 2 players, 3–5 rounds |
| M3 | Actions v2, GSI adapter, packing, and dataset | Versioned, reproducible training samples and contract tests |
| M4 | smoke and short training | Small sample that can be overfitted, per-branch gradients, complete exit checks |
| M5 | 128-step joint training, baselines, and recovery | Trusted validation report, best/last, resumption evidence |
| M6 | Full data preparation and inference export | Full quality report, final test, model card, and deployment contract |

Every stage is allowed to produce genuine failure/blocked reports. For example, if M1 proves that the current demos have no reliable mouse commands, training may continue on the branches that do have labels, or wait for supplementary data; mouse labels are never manufactured for the sake of "completeness". If M2 cannot render/align reliably, the pipeline does not proceed to visual behavior cloning.

The completion criteria for the full program are: a raw demo can be converted through explicit stages into an audited dataset; training is resumable; the model can be evaluated on independent matches and exported; and capabilities and limitations are traceable. Possessing these capabilities does not mean professional match level has been reached.

## 18. References and Known Facts

- [demoparser2 project and field list](https://github.com/LaihoE/demoparser): the candidate Python backend for parsing events, states, and usercmd fields.
- [demoparser #340](https://github.com/LaihoE/demoparser/issues/340): records the delta usercmd problem that caused missing/stale actions, and the 0.42.0 fix; it is the basis for mandatory quality audits, and it must not be used to assert that the local assets are affected or have passed validation.
- [demoparser current usercmd handling source](https://github.com/LaihoE/demoparser/blob/main/src/parser/src/second_pass/parser.rs): the handling of full/delta commands and history/subtick, usable for verifying versions at implementation time.
- [demoinfocs release notes](https://github.com/markus-wa/demoinfocs-golang/releases): the alternative parsing capability for complete UserCmd, subtick, and input history; its specific version and the local assets require separate evaluation.
- [Valve: Introducing TrueView](https://www.counter-strike.net/newsentry/578276333072678918?l=english): the official explanation of replay client prediction, version mismatches, and effect timing differences.

All other training parameters, architecture adjustments, directory layouts, and threshold strategies are this project's design recommendations, not results already validated by external sources.
