"""Navigation with the robot's own sensor: a 2-D lidar, an occupancy grid, A* and a path follower.

- Lidar: `N_RAYS` horizontal rays from the pelvis at `LIDAR_Z` above the floor (MuJoCo ray casts against the scene's
  collision geoms; the robot's own geoms are in other groups and are not seen), range `RANGE`.
- Occupancy grid: `CELL` m cells; every return marks its cell occupied (the room is static while the robot walks).
  Unseen cells count as free: the path is planned optimistically and replanned as soon as a new obstacle appears
  on it.
- A*: 8-connected, on the grid inflated by the robot's radius (`INFLATE`), with a cost bonus for staying away
  from obstacles.
- Follower: pure pursuit for forward speed and turn rate; turns on the spot first when the path points far off
  the heading (the walking policy turns poorly while walking fast).
"""

from __future__ import annotations

import heapq

import mujoco
import numpy as np

from homehand.locomotion.walker import wrap

N_RAYS = 90
RANGE = 4.0
LIDAR_Z = 0.30           # m above the floor: chairs' seats and legs, boxes, the counter
CELL = 0.05
INFLATE = 0.32           # m: robot half-width (0.25) + margin
LOOKAHEAD = 0.45
GEOM_GROUPS = np.array([1, 0, 0, 0, 1, 0], dtype=np.uint8)   # scene fixtures (0) and obstacles (4)


class OccupancyGrid:
    def __init__(self, x_range=(-4.0, 1.5), y_range=(-2.5, 2.5)):
        self.x0, self.y0 = x_range[0], y_range[0]
        self.nx = int(round((x_range[1] - x_range[0]) / CELL))
        self.ny = int(round((y_range[1] - y_range[0]) / CELL))
        self.occ = np.zeros((self.nx, self.ny), dtype=bool)
        self._inflated = None

    def cell(self, x: float, y: float) -> tuple[int, int]:
        return int((x - self.x0) / CELL), int((y - self.y0) / CELL)

    def center(self, i: int, j: int) -> np.ndarray:
        return np.array([self.x0 + (i + 0.5) * CELL, self.y0 + (j + 0.5) * CELL])

    def inside(self, i: int, j: int) -> bool:
        return 0 <= i < self.nx and 0 <= j < self.ny

    def mark(self, pts: np.ndarray) -> int:
        new = 0
        for x, y in pts:
            i, j = self.cell(x, y)
            if self.inside(i, j) and not self.occ[i, j]:
                self.occ[i, j] = True
                new += 1
        if new:
            self._inflated = None
        return new

    def inflated(self) -> tuple[np.ndarray, np.ndarray]:
        """(blocked cells, distance to the nearest obstacle in m)."""
        if self._inflated is None:
            from scipy.ndimage import distance_transform_edt
            dist = distance_transform_edt(~self.occ) * CELL
            self._inflated = (dist < INFLATE, dist)
        return self._inflated


def scan(model: mujoco.MjModel, data: mujoco.MjData, origin_body: int, base_xy, yaw: float) -> np.ndarray:
    """World (x, y) of the lidar returns."""
    o = np.array([base_xy[0], base_xy[1], LIDAR_Z])
    gid = np.zeros(1, dtype=np.int32)
    pts = []
    for a in np.linspace(-np.pi, np.pi, N_RAYS, endpoint=False):
        v = np.array([np.cos(yaw + a), np.sin(yaw + a), 0.0])
        dist = mujoco.mj_ray(model, data, o, v, GEOM_GROUPS, 1, origin_body, gid)
        if 0 <= dist < RANGE:
            pts.append(o[:2] + dist * v[:2])
    return np.array(pts).reshape(-1, 2)


def astar(grid: OccupancyGrid, start, goal) -> list[np.ndarray] | None:
    blocked, dist = grid.inflated()
    s, g = grid.cell(*start), grid.cell(*goal)
    blocked = blocked.copy()
    # the robot stands where it stands: free its own footprint and the goal's
    for c in (s, g):
        i0, j0 = c
        r = int(INFLATE / CELL) + 1
        blocked[max(0, i0 - r):i0 + r + 1, max(0, j0 - r):j0 + r + 1] &= grid.occ[max(0, i0 - r):i0 + r + 1,
                                                                                  max(0, j0 - r):j0 + r + 1]
    if not (grid.inside(*s) and grid.inside(*g)):
        return None
    moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
    openq = [(0.0, s)]
    cost = {s: 0.0}
    parent = {}
    while openq:
        _, c = heapq.heappop(openq)
        if c == g:
            path = [c]
            while c in parent:
                c = parent[c]
                path.append(c)
            return [grid.center(*p) for p in reversed(path)]
        for di, dj, w in moves:
            n = (c[0] + di, c[1] + dj)
            if not grid.inside(*n) or blocked[n]:
                continue
            clearance = 0.6 / max(float(dist[n]), 0.05)        # prefer the middle of free space
            nc = cost[c] + w * (1.0 + 0.15 * clearance)
            if nc < cost.get(n, 1e18):
                cost[n] = nc
                parent[n] = c
                h = np.hypot(n[0] - g[0], n[1] - g[1])
                heapq.heappush(openq, (nc + h, n))
    return None


def path_blocked(grid: OccupancyGrid, path: list[np.ndarray], from_index: int = 0) -> bool:
    blocked, _ = grid.inflated()
    for p in path[from_index:]:
        i, j = grid.cell(*p)
        if grid.inside(i, j) and blocked[i, j]:
            return True
    return False


def follow(path: list[np.ndarray], pos, yaw: float, v_max: float = 0.4) -> tuple[np.ndarray, int]:
    """Velocity command (forward, sideways, yaw rate) and the index of the path point nearest the robot."""
    d = [np.linalg.norm(p - pos[:2]) for p in path]
    k = int(np.argmin(d))
    target = path[-1]
    for p in path[k:]:
        if np.linalg.norm(p - pos[:2]) >= LOOKAHEAD:
            target = p
            break
    heading = np.arctan2(target[1] - pos[1], target[0] - pos[0])
    err = wrap(heading - yaw)
    turn = float(np.clip(1.8 * err, -0.8, 0.8))
    if abs(err) > 0.6:                                    # turn on the spot first
        return np.array([0.0, 0.0, turn]), k
    remaining = sum(np.linalg.norm(path[i + 1] - path[i]) for i in range(k, len(path) - 1))
    v = float(np.clip(0.8 * remaining, 0.12, v_max)) * (1.0 - abs(err) / 0.9)
    return np.array([v, 0.0, turn]), k
