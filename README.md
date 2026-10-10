# HomeHand — Unitree G1 + two Inspire hands tidy a kitchen counter (MuJoCo)

A fixed-base **Unitree G1** humanoid with **two 6-DoF Inspire dexterous hands** looks at a kitchen counter with
its head camera, **detects and recognises** the household objects on it (YCB: tomato soup can, mustard bottle,
sugar box, potted meat can), and clears them into a bin: **the hand closest to an object takes it**, the two
hands alternate, each object gets **its own grasp type**, and the robot **drops** the objects into the bin.

Everything — simulation, perception, planning, three controllers (scripted expert, **ACT**, **Diffusion
Policy**), a **safety layer**, a sim-gap evaluation with Wilson confidence intervals and a failure taxonomy —
is served by one command as a web app. It runs on a laptop (i5-11400H, GTX 1650 4 GB, 16 GB RAM) on Ubuntu,
on **Windows 11** (native, Python 3.12: tests, viewer, web app and the whole training / evaluation pipeline),
or in Docker.

Demo video: [media/demo_seed8.mp4](media/demo_seed8.mp4) — one full episode in real time, YOLO perception, two of the four objects lying on their side (taken from above), each drop on a different spot.

```
head camera (RGB-D) ─► YOLOv8n-seg ─► masks + depth ─► 3-D footprint fit ─► object poses
                                                                              │
                                                     planner: nearest hand, outermost first, re-perceive
                                                                              │
                         pick & place skill: scripted expert │ ACT │ Diffusion Policy (13 joint targets)
                                                                              │
                     safety layer: joint & velocity limits, self-collision, force, workspace, protective stop
                                                                              │
                                    MuJoCo 3 (G1 + 2× Inspire, 26 position actuators, 500 Hz / 25 Hz control)
```

## Quick start

```bash
# Python 3.10+, git
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[learn]"                                 # core + torch/ultralytics (see GPU note below)
homehand fetch-assets                                     # G1 (Menagerie), Inspire (dex-urdf), YCB  ~90 MB
homehand build-model                                      # -> generated/scene_kitchen.xml
homehand serve                                            # opens http://127.0.0.1:8000
```

GPU (optional, for training): `pip install torch --index-url https://download.pytorch.org/whl/cu121` before the
line above. Without a GPU everything still runs; training uses the CPU.

**Windows notes.** Use PowerShell (`.venv\Scripts\activate`) and Git for Windows (needed by `fetch-assets`).
Rendering uses WGL/GLFW, no extra setup. If `pip install torch` fails with *"file is being used by another
process"* (antivirus scanning the 2.4 GB wheel), download it first and install from the file:
`pip download --no-deps -d wheels torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121` then
`pip install wheels\torch-*.whl`. On a flaky connection clone large repos with `git -c http.version=HTTP/1.1`.

The web UI is pre-built into `homehand/web_static`, so Node.js is only needed to change the frontend
(`cd web && npm install && npm run build`).

Train and evaluate everything (≈11 h on the laptop above; stages run one after another so RAM/VRAM stay bounded):

```bash
homehand pipeline     # demos -> YOLO data -> YOLO -> ACT -> Diffusion Policy -> evaluations
python scripts/run_all.py   # the pipeline + grasp benchmark + motion quality + sim-gap ablation -> RESULTS.md
```

Both are resumable: rerun the same command after an interruption and finished stages are skipped
(training restarts from its last checkpoint).

or step by step:

```bash
homehand collect --n 400                      # expert demonstrations (parallel, RAM-guarded)
homehand perception gen-data --n 1500         # synthetic head-camera images, auto-labelled masks
homehand perception train --epochs 40         # YOLOv8n-seg
homehand perception bench --detector yolo     # recall + 3-D localisation error vs ground truth
homehand train --policy act                   # ACT
homehand train --policy diffusion             # Diffusion Policy
homehand eval --controller act_expert_v1 --randomization medium -n 50
homehand demo --seed 3 --out demo.mp4         # one episode to a video
```

## Docker

The image contains the code and its dependencies; assets and results stay on the host (`third_party/`,
`generated/`, `data/`, `models/` are mounted), because the G1 + Inspire model is not redistributable.

```bash
scripts/docker_wheels.sh                               # once: pre-download the wheels (offline, retry-safe build)
docker compose build                                   # core + tests; add --build-arg EXTRAS=dev,learn for PyTorch/YOLO (CPU)
docker compose run --rm homehand build-model           # once: generated/ from third_party/
docker compose up web                                  # dashboard on http://localhost:8000
xhost +SI:localuser:$USER && docker compose run --rm view   # native MuJoCo viewer on the host display
docker compose run --rm test                           # tests
```

Headless rendering (head camera, web stream) uses OSMesa on the CPU; the viewer uses the host X server with
`/dev/dri` passed through. GPU training stays on the host
unless the NVIDIA container toolkit is installed. The container is capped at 6 GB of RAM.

## The web app

| tab | what it does |
|---|---|
| **Live** | run an episode in (scaled) real time: controller, perception, sim-gap level, seed, camera; video stream, head-camera detections, current hand / object / grasp type, grip forces, safety panel, planner log |
| **Evaluation** | launch batch evaluations; runs table with task success and objects-cleared rate (95 % Wilson CI), comparison chart, failure taxonomy, sim-gap curves, per-object/grasp-type success, safety statistics |
| **Episodes** | every evaluated episode, filter by outcome, replay any of them |
| **Models** | ACT vs Diffusion Policy training curves, dataset statistics, detector mAP and localisation benchmark, grasp library |

## What is inside

**Robot model** (`homehand/model`). Two interchangeable builders produce the same `generated/g1_inspire.xml`
(fixed-base G1, both 7-DoF arms actuated, two Inspire hands in the "handshake" orientation: fingers forward,
thumb up, palm inward):

* `build_repo.py` (default when `third_party/unitree_g1_inspire/` exists): adapts a ready-made G1 + Inspire MJCF
  (hands attached, detailed skin-coloured visual meshes). Floating base, legs and waist are welded, torque motors
  become position actuators, joints are renamed, collision geometry is the URDF primitives + simplified meshes.
  **That model's mimic constraints are inverted** (the actuated joint was made to follow the follower, so the
  fingers curled 25-45 % too little); the builder swaps them to match the URDF `<mimic>` tags.
* `build.py` (fallback, fully reproducible from `homehand fetch-assets`): Unitree G1 from MuJoCo Menagerie plus
  Inspire hands converted from dex-urdf by a small URDF→MJCF converter.

That G1 + Inspire MJCF ships without a licence, so it is not part of this repository: put your copy in
`third_party/unitree_g1_inspire` (or set `HOMEHAND_G1_INSPIRE_DIR`); without it the Menagerie builder is used.
Arm joints get passive damping; gravity is compensated on the arms. The build also slims the model (unused meshes removed, YCB textures 512², convex hulls
precomputed): one simulation process uses ~0.8 GB instead of 1.7 GB.

**Perception** (`homehand/perception`). RGB-D head camera. YOLOv8n-seg is trained on synthetic images rendered
from that camera, auto-labelled from MuJoCo's segmentation buffer, with lighting, camera pose, object and arm
pose randomisation. Masks + depth are back-projected to 3-D and a footprint of the class' known size is
fitted (tightest rectangle for boxes; camera-facing edge + radius for round objects) → position, height, yaw.
An oracle detector (segmentation buffer) is kept as an upper-bound baseline.

**Planning** (`homehand/control/planner.py`). Perceive → assign each object to the hand whose ready pose is
closest → alternate hands, outermost object first (the back of the hand faces outwards) → run the skill →
perceive again (so dropped or pushed objects are handled) → up to two attempts per object.

**Rest poses.** Objects stand upright or lie tipped over on their side (in the outer slots, ~half of them).
The head camera tells the two apart from the measured height of the object (98 % correct with YOLO), and for
a lying object fits its long-axis heading from the depth points.

**Grasp library** (`homehand/control/grasps.py`) — one grasp per object *and rest pose*:

| object | upright: side grasp | lying: grasp from above |
|---|---|---|
| tomato soup can | cylindrical power wrap | top wrap, thumb along the can, fingers tilted 55° down |
| mustard bottle | low power wrap below the shoulder | top wrap around the body, fingers tilted 55° down |
| sugar box | palmar box grasp | steep claw on the flat box, fingers tilted 65° down, thumb deep in opposition |
| potted meat can | tripod pinch | top pinch with the fingers *along* the can, thumb across it |

Upright objects are approached from behind in a "handshake" pose with the thumb pre-rotated into
opposition. Lying objects are taken from above, palm down. The geometry of each top grasp (tilt, which way the
thumb runs, where the object sits in the hand) was found with `scripts/top_grasp_search.py`, a fast
grasp-only search over ~300-700 candidates per object, ranked by how robust a candidate's neighbourhood is.
The decisive detail: the thumb has to swing into opposition *before* the hand comes down — closed last, it is
blocked by the counter and the fingers drag the object out of the hand. Where the object ends up in the closed
hand (it moves 3-5 cm) is measured and used to place the drop. The Inspire hand's power grasp wraps around
its thumb axis and the palm-down hand only reaches with the fingers pointing forward (±40°), which decides
the alignment of each grasp. `homehand grasp-bench` checks all of it: 188/192 single-object trials, 63/64 lying.

**Pick-and-place skill** (`homehand/control/skills.py`). Each phase (reach, approach, grasp, lift, transport,
drop, return) is one motion primitive:

* free-space moves (reach, transport, return) swing the arm in **joint space** between IK key poses solved
  from a working posture (the way a person moves the arm, and on the well-conditioned branch of the 7-DoF
  redundancy); moves near objects (descend, approach, lift) follow straight **Cartesian** paths, sampled every
  2 cm and solved with converged, warm-started **mink** IK (joint limits + arm/torso collision avoidance);
  all IK runs *before* a motion starts, so nothing twitches while the arm moves;
* a **minimum-jerk** time scaling runs over each path; its duration is set by the slowest of palm speed,
  joint speed (peak 90 % of the 2 rad/s limit) and wrist rotation, so motions start and stop softly and are
  never clipped by the safety layer;
* **natural posture**: at rest the upper arms hang along the torso, elbows at the hips, forearms forward
  and ~45° up, wrists straight, fingers relaxed; the waiting hand stays above every object. Free-space
  moves pass high (1.12-1.22 m) so the forearm never sweeps a tall neighbour;
* the hand yaw for an upright box is chosen among its two face pairs **within the yaw window the arm can
  reach at that position** (measured with the IK);
* **drop spots vary**: three spots per hand across the bin, 3.5 cm inside its near wall (the strip the G1
  can reach), each drop taking the spot furthest from what is already in the bin; side grasps release 4.5 cm
  above the rim, top grasps 8 cm (palm nearly flat by then: the steep grasp pose cannot reach over the bin);
* a proprioceptive empty-grasp check for side grasps; objects lying against the bin are left alone rather
  than pushing the fingers into its wall (checked on the forward kinematics of the planned grasp).

Contact geometry: the palm collides with the convex hull of its own visual mesh (the URDF primitives sat up
to 5 mm inside the rendered palm), so what you see never sinks into an object or the bin (measured worst case
over 24 episodes: 1 mm, fingers included).

**Human-like motion** (`homehand/control/human_motion.py`, `clearance.py`; expert v2, `HOMEHAND_EXPERT_STYLE=v1`
restores the previous one). Principles from human motor control, measured with `scripts/human_likeness.py`:

* the primitives of one reach (swing over + come down behind the object) and of one carry (lift + transport) are
  blended into **one movement** with a single speed profile, instead of stopping at every via pose; the slow
  final approach onto the object stays a separate movement (blended into it, the PD-tracked hand cut the corner
  and pushed tall objects);
* reach-to-grasp uses a **skewed bell-shaped speed profile** (v ~ t²(1−t)³: peak at 40 %, long deceleration
  towards the object, Marteniuk 1987) with a **Fitts' law** lower bound on its duration;
* the hand **pre-shapes during the reach** (thumb first, fully open just before the final approach) instead of
  opening in one go (Jeannerod 1984);
* **clearance to the other objects**: a failure analysis of 50 episodes showed that 8 of 11 lost objects were
  *not* the one being picked (the forearm, sloping down behind the hand over the bin, touched the top of a tall box
  in the inner slot; fingertips brushed neighbours). The arm is checked as probe spheres against the other
  objects' boxes: the release point goes up while the carry would touch one, and an object whose grasp would
  brush a neighbour waits until the neighbour is gone (the easy ones first).

**Vision-language-action policy** (`homehand/data/record_vla.py`, `export_lerobot.py`, `policy/smolvla_skill.py`,
`kaggle/`). Eye-in-hand cameras on both hands + the head camera (256×256), the 26 joint positions and an instruction
("pick up the mustard bottle with the left hand and put it in the bin", four phrasings + one held out for testing)
are recorded for every successful pick of the expert and exported as a LeRobotDataset; **SmolVLA** is fine-tuned on
Kaggle (free T4, see `kaggle/README.md`) and runs as the skill inside the same planner and safety layer
(`scripts/eval_smolvla.py`). The planner stays the high level (which object, which hand); the VLA does the skill.

**Learning** (`homehand/policy`). Every successful skill execution of the expert inside full tidy episodes is
a demonstration (features: active arm + hand joints, palm position, perceived object pose, grasp type, side,
progress → 13 joint targets). **ACT** (CVAE + transformer, chunk 25, temporal ensembling) and **Diffusion
Policy** (1-D conditional U-Net, DDPM training / 30-step DDIM, chunk 16, receding horizon 8) are trained on the
same data and plugged into the same planner, so the evaluation compares the skills under identical
perception and planning.

**Safety layer** (`homehand/control/safety.py`) between *every* controller and the robot: joint position limits
with a 3 % margin, velocity limits (arm 2 rad/s, fingers 3 rad/s), monitoring of arm–arm / arm–torso contact,
palm distance, contact force against the environment (> 80 N) and palm workspace; a violation lasting 0.2 s
triggers a protective stop (the robot holds still, the episode is labelled `safety_stop`). The IK plans inside
the same limits, so the expert's commands are never clipped. Actuator torque limits come from the model (G1
arm 25 N·m, wrist 5 N·m, Inspire fingers 2 N·m).

**Towards the real hand** (`homehand/control/inspire_real.py`). The simulated hand commands are converted
to the RH56DFX interface of Unitree's `dfx_inspire_service` (six normalised targets, 1 = open, order pinky →
thumb rotation, angle registers 0-1000, per-finger force limit ≤ 9.8 N), rate-limited like the safety layer,
with a hold on non-finite commands and a per-channel calibration hook. Not validated on hardware.

**Sim-gap evaluation** (`homehand/eval`). No real robot, so the gap is estimated by perturbing what the
simulator assumes: friction, object mass, perception noise, action latency and finger stiffness at four levels
(`nominal`, `low`, `medium`, `high`). Each run reports task success (all objects in the bin) and the
objects-cleared rate with 95 % Wilson intervals, the failure taxonomy (`not_attempted`, `grasp_fail`,
`dropped`, `misplaced`, `knocked_over`, `timeout`, `safety_stop`, `sim_error`) and safety statistics.
Physics blow-ups are detected and reported as `sim_error`, never hidden.

**Resources.** Worker processes are capped by free RAM (`homehand/resources.py`, ~0.9 GB per worker + 2.5 GB
reserve) and the pipeline runs GPU stages one at a time.

## Results

Full tables in [RESULTS.md](RESULTS.md) (all regenerated on Windows 11 with `python scripts/run_all.py`).
Headline numbers, 4 objects per episode (about half of the outer ones lying on their side), YOLO perception
unless stated, 95 % Wilson intervals:

| | clean episodes | objects in the bin | protective stops |
|---|---|---|---|
| scripted expert, nominal (50 ep.) | 66 % [52–78] | 87 % [82–91] | 1 |
| scripted expert, oracle masks (50 ep.) | 76 % [63–86] | 92 % [88–95] | 1 |
| ACT, 25 k steps, nominal (50 ep.) | 64 % [50–76] | 86 % [80–90] | 0 |
| Diffusion Policy, 30 k steps, 30 DDIM steps, nominal (50 ep.) | 30 % [19–44] | 70 % [63–76] | 5 |
| expert, single-object grasp benchmark (144 trials, upright + lying) | – | 99 % (143/144) | – |

* **Lying objects make the task harder**: the earlier upright-only setup reached 90 % clean / 98 % of the
  objects; with top grasps of lying objects added, the expert's losses are mostly knocked-over and dropped
  objects in the full scene, while every grasp works in isolation (143/144).
* **ACT matches the expert** on the same 870 demonstrations (86 % vs 87 % of the objects at nominal, and within
  the confidence intervals at every sim-gap level: 90–93 % vs 86–88 %). **Diffusion Policy** stays well below
  (62–70 %) and is the only controller with a noticeable number of protective stops (force > 80 N against the
  counter or bin, stopped correctly by the safety layer).
* **Motion**: minimum-jerk, joint-space planned motions: RMS arm joint acceleration 2.0 rad/s², no protective
  stop in 24 ground-truth episodes, deepest robot–object/bin overlap 2 mm (the wrist against a carried box).
* **Sim gap**: one factor at a time at the `high` extreme (16 episodes each), the expert depends most on
  **finger–object friction** (×0.5: 52/64 objects vs 61/64 nominal, mostly knocked over), then on **pose noise**
  (12 mm: 56/64) and **finger stiffness** (×0.6: 58/64); 120 ms of latency changes nothing.
* **Human-like motion (expert v2)**: same success as v1 (YOLO, 50 ep.: 174/200 objects for both; ground truth:
  186 vs 185/200) with 4 of 5 kinematic markers now in the human range — no stop while carrying (v1: 1), carry
  SPARC −1.67 (v1 −2.27), reach speed peak at 34 % of the movement, hand fully open at 62 % of the reach. Still
  unlike a person: ~4× slower (joint speed limits) and a high, curved path around the other objects.
* **SmolVLA** (images + language + joints, 3 k fine-tuning steps on a Kaggle T4 = ~0.4 epoch of 320 demos):
  2/40 objects, 0/10 clean episodes, 3 protective stops (pressing on the counter, stopped by the safety layer);
  with an instruction phrasing never seen in training 3/20 objects — the language conditioning does not break.
  The pipeline runs end to end; the policy is far from trained (SmolVLA fine-tunes usually take ≥ 20 k steps).
  On the T4 the VLM had to be loaded in float32 + AMP instead of bfloat16 (no bf16 hardware: > 7 s/step → 1.4 s/step).
* **Found on the way**: the mass randomisation changed `body_mass` without `mj_setConst`, so a +20 % mass made
  grasped objects slip and the simulation diverge — earlier sim-gap numbers measured that bug. Fixed and covered
  by a regression test.

## Licences

Code: MIT. Assets are downloaded at install time and keep their licences: Unitree G1 model (MuJoCo Menagerie,
BSD-3-Clause), Inspire hand (dex-urdf, **CC BY-NC-SA 4.0 — non-commercial**), YCB objects (CC BY 4.0).
