# Results

Measured on the development laptop (i5-11400H, GTX 1650 4 GB, 16 GB RAM) with `homehand eval` (seeds 0..n-1,
4 objects per episode). Intervals are 95 % Wilson score intervals. *Task success* = every object of the
episode ended in the bin; *objects cleared* = fraction of all objects that ended in the bin.
Randomised runs use only 10 episodes each, so their intervals overlap heavily (e.g. `high` scoring above
`medium` is sampling noise); the one-factor table further down is the clearer sim-gap measurement.
Runs made before the mass-randomisation fix (stale `mj_setConst`) are excluded.

## Controllers × sim-gap level

| run | controller | perception | randomisation | episodes | task success | objects cleared | protective stops |
|---|---|---|---|---|---|---|---|
| 3 | expert | yolo | nominal | 20 | 90% [70–97] | 98% [91–99] | 0 |
| 4 | act_expert_v1 | yolo | nominal | 20 | 70% [48–85] | 90% [81–95] | 0 |
| 5 | diffusion_expert_v1 (10 DDIM steps) | yolo | nominal | 20 | 0% [0–16] | 1% [0–7] | 11 |
| 13 | expert | oracle | nominal | 20 | 80% [58–92] | 95% [88–98] | 0 |
| 14 | diffusion_expert_v1 (30 DDIM steps) | yolo | nominal | 20 | 0% [0–16] | 30% [21–41] | 4 |
| 15 | expert | yolo | low | 10 | 80% [49–94] | 95% [84–99] | 0 |
| 16 | act_expert_v1 | yolo | low | 10 | 40% [17–69] | 80% [65–90] | 0 |
| 17 | expert | yolo | medium | 10 | 40% [17–69] | 82% [68–91] | 0 |
| 18 | act_expert_v1 | yolo | medium | 10 | 20% [6–51] | 75% [60–86] | 0 |
| 19 | expert | yolo | high | 10 | 70% [40–89] | 92% [80–97] | 0 |
| 20 | act_expert_v1 | yolo | high | 10 | 70% [40–89] | 90% [77–96] | 0 |
| 21 | diffusion_expert_v2 (30k steps, 30 DDIM) | yolo | nominal | 20 | 45% [26–66] | 78% [67–85] | 0 |

## Failure taxonomy (objects)

| run | success | not_attempted | grasp_fail | dropped | misplaced | knocked_over | timeout | safety_stop | sim_error |
|---|---|---|---|---|---|---|---|---|---|
| 3 expert / nominal | 78 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 |
| 4 act_expert_v1 / nominal | 72 | 1 | 0 | 0 | 5 | 2 | 0 | 0 | 0 |
| 5 diffusion_expert_v1 (10 DDIM steps) / nominal | 1 | 19 | 6 | 8 | 0 | 46 | 0 | 0 | 0 |
| 13 expert / nominal | 76 | 0 | 0 | 1 | 2 | 1 | 0 | 0 | 0 |
| 14 diffusion_expert_v1 (30 DDIM steps) / nominal | 24 | 11 | 6 | 8 | 0 | 31 | 0 | 0 | 0 |
| 15 expert / low | 38 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 |
| 16 act_expert_v1 / low | 32 | 1 | 0 | 1 | 5 | 1 | 0 | 0 | 0 |
| 17 expert / medium | 33 | 0 | 0 | 0 | 6 | 1 | 0 | 0 | 0 |
| 18 act_expert_v1 / medium | 30 | 2 | 1 | 0 | 6 | 1 | 0 | 0 | 0 |
| 19 expert / high | 37 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 0 |
| 20 act_expert_v1 / high | 36 | 1 | 0 | 0 | 2 | 1 | 0 | 0 | 0 |
| 21 diffusion_expert_v2 (30k steps, 30 DDIM) / nominal | 62 | 3 | 5 | 2 | 1 | 7 | 0 | 0 | 0 |

## Per object / grasp type (nominal runs)

| run | object | grasp | success |
|---|---|---|---|
| 3 expert | tomato soup can | cylindrical power wrap | 100% [84–100] |
| 3 expert | mustard bottle | low power wrap | 100% [84–100] |
| 3 expert | sugar box | palmar box grasp | 90% [70–97] |
| 3 expert | potted meat can | tripod pinch | 100% [84–100] |
| 4 act_expert_v1 | tomato soup can | cylindrical power wrap | 90% [70–97] |
| 4 act_expert_v1 | mustard bottle | low power wrap | 90% [70–97] |
| 4 act_expert_v1 | sugar box | palmar box grasp | 85% [64–95] |
| 4 act_expert_v1 | potted meat can | tripod pinch | 95% [76–99] |
| 5 diffusion_expert_v1 (10 DDIM steps) | tomato soup can | cylindrical power wrap | 0% [0–16] |
| 5 diffusion_expert_v1 (10 DDIM steps) | mustard bottle | low power wrap | 0% [0–16] |
| 5 diffusion_expert_v1 (10 DDIM steps) | sugar box | palmar box grasp | 5% [1–24] |
| 5 diffusion_expert_v1 (10 DDIM steps) | potted meat can | tripod pinch | 0% [0–16] |
| 13 expert | tomato soup can | cylindrical power wrap | 100% [84–100] |
| 13 expert | mustard bottle | low power wrap | 95% [76–99] |
| 13 expert | sugar box | palmar box grasp | 90% [70–97] |
| 13 expert | potted meat can | tripod pinch | 95% [76–99] |
| 14 diffusion_expert_v1 (30 DDIM steps) | tomato soup can | cylindrical power wrap | 50% [30–70] |
| 14 diffusion_expert_v1 (30 DDIM steps) | mustard bottle | low power wrap | 15% [5–36] |
| 14 diffusion_expert_v1 (30 DDIM steps) | sugar box | palmar box grasp | 40% [22–61] |
| 14 diffusion_expert_v1 (30 DDIM steps) | potted meat can | tripod pinch | 15% [5–36] |
| 21 diffusion_expert_v2 (30k steps, 30 DDIM) | tomato soup can | cylindrical power wrap | 90% [70–97] |
| 21 diffusion_expert_v2 (30k steps, 30 DDIM) | mustard bottle | low power wrap | 65% [43–82] |
| 21 diffusion_expert_v2 (30k steps, 30 DDIM) | sugar box | palmar box grasp | 70% [48–85] |
| 21 diffusion_expert_v2 (30k steps, 30 DDIM) | potted meat can | tripod pinch | 85% [64–95] |

## Grasp benchmark (single object, ground-truth pose)

`homehand grasp-bench --out data/grasp_bench.json`: every object with both hands, inner and outer
slot, yaws spread over the full circle. Failure modes: *missed* (fingers closed on nothing), *pushed*
(object swept before the grasp), *slipped* (fell out while carried), *bounced* (left the bin).

| object | hand | slot | success | failures |
|---|---|---|---|---|
| meat_can | left | inner | 11/12 | slipped 1 |
| meat_can | left | outer | 12/12 | - |
| meat_can | right | inner | 12/12 | - |
| meat_can | right | outer | 12/12 | - |
| mustard | left | inner | 12/12 | - |
| mustard | left | outer | 12/12 | - |
| mustard | right | inner | 12/12 | - |
| mustard | right | outer | 12/12 | - |
| soup_can | left | inner | 12/12 | - |
| soup_can | left | outer | 12/12 | - |
| soup_can | right | inner | 12/12 | - |
| soup_can | right | outer | 12/12 | - |
| sugar_box | left | inner | 11/12 | missed 1 |
| sugar_box | left | outer | 11/12 | missed 1 |
| sugar_box | right | inner | 12/12 | - |
| sugar_box | right | outer | 11/12 | missed 1 |
| **all** | | | **188/192** | |

## Motion quality (expert, full tidy episodes, ground-truth poses)

`python scripts/motion_quality.py --out data/motion_quality.json`. Joint acceleration / jerk from the
measured arm joint positions at 25 Hz; interpenetration = deepest overlap between any robot mesh (visual
or collision) and the objects, bin or counter.

| episodes | clean | objects cleared | protective stops | mean duration | RMS joint acc. | RMS joint jerk | max interpenetration |
|---|---|---|---|---|---|---|---|
| 24 | 21 | 93/96 | 0 | 49.5 s | 2.68 rad/s² | 75.7 rad/s³ | 0.92 mm |

## Sim-gap factors one at a time (expert, ground-truth poses)

`python scripts/simgap_ablation.py --out data/simgap_ablation.json`: each factor fixed at the extreme
of the `high` preset, everything else nominal. The random presets above mix all factors and use few
episodes, so this table is the better guide to what the controller depends on.

| factor | clean episodes | objects cleared | protective stops | failures |
|---|---|---|---|---|
| nominal | 7/8 | 31/32 | 0 | knocked_over 1 |
| friction x0.5 | 4/8 | 27/32 | 0 | dropped 1, knocked_over 2, misplaced 2 |
| friction x1.5 | 5/8 | 29/32 | 0 | dropped 1, knocked_over 1, misplaced 1 |
| mass x0.5 | 6/8 | 30/32 | 0 | knocked_over 1, misplaced 1 |
| mass x2.0 | 7/8 | 31/32 | 0 | knocked_over 1 |
| latency 3 steps (120 ms) | 6/8 | 30/32 | 0 | knocked_over 1, misplaced 1 |
| finger stiffness x0.6 | 5/8 | 29/32 | 0 | dropped 1, knocked_over 1, misplaced 1 |
| pose noise 12 mm | 5/8 | 29/32 | 0 | knocked_over 1, misplaced 2 |

## Learning

- **act_expert_v1**: 9.014 M params, 10000 steps, batch 128, cuda, 23.6 min, final validation action-MSE 0.00352
- **diffusion_expert_v1**: 18.238 M params, 10000 steps, batch 128, cuda, 39.0 min, final validation action-MSE 0.14
- **diffusion_expert_v2**: 18.238 M params, 30000 steps, batch 128, cuda, 67.1 min, final validation action-MSE 0.03183
- dataset **expert_v1**: 846 successful skill executions / 267773 frames from 300 tidy episodes; per object {'meat_can': 225, 'mustard': 209, 'soup_can': 229, 'sugar_box': 183}; per hand {'left': 532, 'right': 314}

## Perception

YOLOv8n-seg on held-out synthetic images: box mAP50 0.995 (mAP50-95 0.991), mask mAP50 0.995 (mAP50-95 0.960).

| detector | recall | position error mean | p90 | wrist-yaw error | latency |
|---|---|---|---|---|---|
| oracle | 100.0% | 2.9 mm | 4.9 mm | 11.6° | 3.5 ms |
| yolo | 100.0% | 3.3 mm | 4.9 mm | 12.0° | 12.5 ms |
