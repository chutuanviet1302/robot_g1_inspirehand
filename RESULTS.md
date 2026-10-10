# Results

Measured on the development laptop (i5-11400H, GTX 1650 4 GB, 16 GB RAM) with `homehand eval` (seeds 0..n-1,
4 objects per episode). Intervals are 95 % Wilson score intervals. *Task success* = every object of the
episode ended in the bin; *objects cleared* = fraction of all objects that ended in the bin.
Randomised runs use 25 episodes each, so neighbouring levels have overlapping
intervals; the one-factor table further down is the clearer sim-gap measurement.
Reproduce everything with `python scripts/run_all.py`.

## Controllers × sim-gap level

| run | controller | perception | randomisation | episodes | task success | objects cleared | protective stops |
|---|---|---|---|---|---|---|---|
| 1 | expert | yolo | nominal | 50 | 66% [52–78] | 87% [82–91] | 1 |
| 2 | act_expert_v1 | yolo | nominal | 50 | 64% [50–76] | 86% [80–90] | 0 |
| 3 | diffusion_expert_v1 | yolo | nominal | 50 | 30% [19–44] | 70% [63–76] | 5 |
| 4 | expert | yolo | low | 25 | 60% [41–77] | 87% [79–92] | 0 |
| 5 | act_expert_v1 | yolo | low | 25 | 60% [41–77] | 90% [83–94] | 1 |
| 6 | diffusion_expert_v1 | yolo | low | 25 | 24% [12–43] | 69% [59–77] | 2 |
| 7 | expert | yolo | medium | 25 | 60% [41–77] | 88% [80–93] | 1 |
| 8 | act_expert_v1 | yolo | medium | 25 | 76% [57–88] | 93% [86–97] | 0 |
| 9 | diffusion_expert_v1 | yolo | medium | 25 | 36% [20–55] | 68% [58–76] | 6 |
| 10 | expert | yolo | high | 25 | 64% [45–80] | 86% [78–91] | 1 |
| 11 | act_expert_v1 | yolo | high | 25 | 72% [52–86] | 91% [84–95] | 0 |
| 12 | diffusion_expert_v1 | yolo | high | 25 | 16% [6–35] | 62% [52–71] | 5 |
| 13 | expert | oracle | nominal | 50 | 76% [63–86] | 92% [88–95] | 1 |

## Failure taxonomy (objects)

| run | success | not_attempted | grasp_fail | dropped | misplaced | knocked_over | timeout | safety_stop | sim_error |
|---|---|---|---|---|---|---|---|---|---|
| 1 expert / nominal | 174 | 2 | 1 | 8 | 4 | 10 | 1 | 0 | 0 |
| 2 act_expert_v1 / nominal | 171 | 2 | 3 | 10 | 1 | 12 | 1 | 0 | 0 |
| 3 diffusion_expert_v1 / nominal | 140 | 15 | 5 | 18 | 8 | 12 | 2 | 0 | 0 |
| 4 expert / low | 87 | 0 | 4 | 4 | 2 | 2 | 1 | 0 | 0 |
| 5 act_expert_v1 / low | 90 | 1 | 2 | 3 | 1 | 2 | 1 | 0 | 0 |
| 6 diffusion_expert_v1 / low | 69 | 5 | 9 | 4 | 3 | 10 | 0 | 0 | 0 |
| 7 expert / medium | 88 | 1 | 3 | 2 | 2 | 4 | 0 | 0 | 0 |
| 8 act_expert_v1 / medium | 93 | 0 | 1 | 3 | 1 | 2 | 0 | 0 | 0 |
| 9 diffusion_expert_v1 / medium | 68 | 10 | 9 | 4 | 2 | 6 | 1 | 0 | 0 |
| 10 expert / high | 86 | 1 | 1 | 5 | 0 | 7 | 0 | 0 | 0 |
| 11 act_expert_v1 / high | 91 | 2 | 1 | 1 | 1 | 3 | 1 | 0 | 0 |
| 12 diffusion_expert_v1 / high | 62 | 9 | 9 | 10 | 0 | 6 | 4 | 0 | 0 |
| 13 expert / nominal | 185 | 1 | 0 | 1 | 4 | 8 | 1 | 0 | 0 |

## Per object / grasp type (nominal runs)

| run | object | grasp | success |
|---|---|---|---|
| 1 expert | tomato soup can | cylindrical power wrap | 86% [74–93] |
| 1 expert | mustard bottle | low power wrap | 82% [69–90] |
| 1 expert | sugar box | palmar box grasp | 84% [71–92] |
| 1 expert | potted meat can | tripod pinch | 96% [87–99] |
| 2 act_expert_v1 | tomato soup can | cylindrical power wrap | 86% [74–93] |
| 2 act_expert_v1 | mustard bottle | low power wrap | 86% [74–93] |
| 2 act_expert_v1 | sugar box | palmar box grasp | 76% [63–86] |
| 2 act_expert_v1 | potted meat can | tripod pinch | 94% [84–98] |
| 3 diffusion_expert_v1 | tomato soup can | cylindrical power wrap | 70% [56–81] |
| 3 diffusion_expert_v1 | mustard bottle | low power wrap | 64% [50–76] |
| 3 diffusion_expert_v1 | sugar box | palmar box grasp | 66% [52–78] |
| 3 diffusion_expert_v1 | potted meat can | tripod pinch | 80% [67–89] |
| 13 expert | tomato soup can | cylindrical power wrap | 96% [87–99] |
| 13 expert | mustard bottle | low power wrap | 90% [79–96] |
| 13 expert | sugar box | palmar box grasp | 84% [71–92] |
| 13 expert | potted meat can | tripod pinch | 100% [93–100] |

## Grasp benchmark (single object, ground-truth pose)

`homehand grasp-bench --out data/grasp_bench.json`: every object with both hands, inner and outer
slot, yaws spread over the full circle. Failure modes: *missed* (fingers closed on nothing), *pushed*
(object swept before the grasp), *slipped* (fell out while carried), *bounced* (left the bin).

| object | hand | slot | success | failures |
|---|---|---|---|---|
| meat_can | left | inner/upright | 6/6 | - |
| meat_can | left | outer/lying | 6/6 | - |
| meat_can | left | outer/upright | 6/6 | - |
| meat_can | right | inner/upright | 6/6 | - |
| meat_can | right | outer/lying | 6/6 | - |
| meat_can | right | outer/upright | 6/6 | - |
| mustard | left | inner/upright | 6/6 | - |
| mustard | left | outer/lying | 6/6 | - |
| mustard | left | outer/upright | 6/6 | - |
| mustard | right | inner/upright | 6/6 | - |
| mustard | right | outer/lying | 5/6 | slipped 1 |
| mustard | right | outer/upright | 6/6 | - |
| soup_can | left | inner/upright | 6/6 | - |
| soup_can | left | outer/lying | 6/6 | - |
| soup_can | left | outer/upright | 6/6 | - |
| soup_can | right | inner/upright | 6/6 | - |
| soup_can | right | outer/lying | 6/6 | - |
| soup_can | right | outer/upright | 6/6 | - |
| sugar_box | left | inner/upright | 6/6 | - |
| sugar_box | left | outer/lying | 6/6 | - |
| sugar_box | left | outer/upright | 6/6 | - |
| sugar_box | right | inner/upright | 6/6 | - |
| sugar_box | right | outer/lying | 6/6 | - |
| sugar_box | right | outer/upright | 6/6 | - |
| **all** | | | **143/144** | |

## Motion quality (expert, full tidy episodes, ground-truth poses)

`python scripts/motion_quality.py --out data/motion_quality.json`. Joint acceleration / jerk from the
measured arm joint positions at 25 Hz; interpenetration = deepest overlap between any robot mesh (visual
or collision) and the objects, bin or counter.

| episodes | clean | objects cleared | protective stops | mean duration | RMS joint acc. | RMS joint jerk | max interpenetration |
|---|---|---|---|---|---|---|---|
| 24 | 18 | 88/96 | 0 | 54.2 s | 2.0 rad/s² | 56.3 rad/s³ | 2.0 mm |

## Sim-gap factors one at a time (expert, ground-truth poses)

`python scripts/simgap_ablation.py --out data/simgap_ablation.json`: each factor fixed at the extreme
of the `high` preset, everything else nominal. The random presets above mix all factors and use few
episodes, so this table is the better guide to what the controller depends on.

| factor | clean episodes | objects cleared | protective stops | failures |
|---|---|---|---|---|
| nominal | 13/16 | 61/64 | 0 | dropped 1, knocked_over 2 |
| friction x0.5 | 8/16 | 52/64 | 0 | dropped 2, knocked_over 8, misplaced 2 |
| friction x1.5 | 11/16 | 59/64 | 0 | knocked_over 3, misplaced 2 |
| mass x0.5 | 12/16 | 60/64 | 0 | knocked_over 2, misplaced 1, not_attempted 1 |
| mass x2.0 | 12/16 | 59/64 | 0 | knocked_over 2, misplaced 2, timeout 1 |
| latency 3 steps (120 ms) | 13/16 | 61/64 | 0 | knocked_over 2, misplaced 1 |
| finger stiffness x0.6 | 10/16 | 58/64 | 0 | knocked_over 2, misplaced 4 |
| pose noise 12 mm | 9/16 | 56/64 | 0 | dropped 1, knocked_over 3, misplaced 1, not_attempted 1, timeout 2 |

## Learning

- **act_expert_v1**: 9.019 M params, 25000 steps, batch 128, cuda, 167.2 min, final validation action-MSE 0.01056
- **diffusion_expert_v1**: 18.24 M params, 30000 steps, batch 128, cuda, 107.0 min, final validation action-MSE 0.04144
- dataset **expert_v1**: 870 successful skill executions / 324197 frames from 300 tidy episodes; per object {'meat_can': 236, 'mustard': 213, 'soup_can': 225, 'sugar_box': 196}; per hand {'left': 547, 'right': 323}

## Perception

YOLOv8n-seg on held-out synthetic images: box mAP50 0.995 (mAP50-95 0.993), mask mAP50 0.995 (mAP50-95 0.970).

| detector | recall | position error mean | p90 | yaw error | upright / lying correct | latency |
|---|---|---|---|---|---|---|
| oracle | 100.0% | 2.4 mm | 4.5 mm | 9.1° | 100% | 8.4 ms |
| yolo | 100.0% | 3.5 mm | 5.4 mm | 9.7° | 98% | 50.8 ms |
