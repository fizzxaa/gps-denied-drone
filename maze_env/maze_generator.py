"""
Maze generator matching the brief: arena <= 15m x 15m, 2m x 2m rooms, 1m-wide corridors between
them, single entry/exit, 2.44m (8ft) ceiling clearance.

Earlier version's bug: ROOM_SIZE and CORRIDOR_WIDTH were defined but never used -- every "cell" was
actually drawn as one uniform 3m x 3m open square with a 3m-wide gap wherever two cells connected.
That is not a maze of rooms and corridors; it is a much easier, wide-open grid.

This version draws the real geometry: each grid cell is a genuine 2m x 2m room (walls on all four
sides). Between two CONNECTED rooms, a narrow 1m x 1m corridor tunnel bridges the 1m gap between
them, itself walled on both sides so it is a real, physical constriction the drone must pass
through -- not just an absence of wall. Between two rooms that are NOT connected, the whole 1m gap
is solid.
"""

import random
import numpy as np
import pybullet as p

WALL_HEIGHT = 2.44          # 8 ft ceiling clearance
WALL_THICKNESS = 0.08
ROOM_SIZE = 2.0              # 2m x 2m rooms, per brief
CORRIDOR_WIDTH = 1.0         # >= 1m clear width, per brief -- also the corridor's LENGTH here,
                              # since it is the gap between two room edges
ARENA_MAX = 15.0             # 15m x 15m max footprint


class MazeGenerator:
    """
    n x n grid of 2m x 2m ROOM cells, each pair of orthogonally-adjacent rooms either sealed off or
    joined by a straight 1m-wide, 1m-long corridor. Grid spacing (room-center to room-center) is
    ROOM_SIZE + CORRIDOR_WIDTH = 3.0m, so a given grid_n occupies exactly the same footprint as the
    earlier version -- only what happens BETWEEN room centers changed.
    """

    def __init__(self, grid_n=5, seed=None, entry_cell=(0, 0), exit_cell=None):
        self.rng = random.Random(seed)
        self.grid_n = grid_n
        self.entry_cell = entry_cell
        self.exit_cell = exit_cell if exit_cell else (grid_n - 1, grid_n - 1)

        self.room_half = ROOM_SIZE / 2.0
        self.corr_half = CORRIDOR_WIDTH / 2.0
        self.cell_pitch = ROOM_SIZE + CORRIDOR_WIDTH
        footprint = self.cell_pitch * grid_n
        if footprint > ARENA_MAX:
            raise ValueError(f"grid_n={grid_n} gives footprint {footprint:.1f}m > {ARENA_MAX}m arena max.")

        self.origin = np.array([-(footprint - self.cell_pitch) / 2.0] * 2)

        # walls_x[i, j] / walls_y[i, j]: True = SEALED boundary, False = OPEN (a corridor is built).
        # Same indexing as before: walls_x[i,j] is the boundary between room (i-1,j) and (i,j);
        # walls_y[i,j] is the boundary between room (i,j-1) and (i,j).
        self.walls_x = np.zeros((grid_n + 1, grid_n), dtype=bool)
        self.walls_y = np.zeros((grid_n, grid_n + 1), dtype=bool)
        self._generate_topology()

    # ------------------------------------------------------------------
    # Topology (recursive backtracker maze on the room grid) -- UNCHANGED logic, only geometry below differs
    # ------------------------------------------------------------------
    def _generate_topology(self):
        n = self.grid_n
        self.walls_x[:, :] = True
        self.walls_y[:, :] = True
        visited = np.zeros((n, n), dtype=bool)
        stack = [self.entry_cell]
        visited[self.entry_cell] = True
        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < n and 0 <= ny < n and not visited[nx, ny]:
                    neighbors.append((nx, ny, dx, dy))
            if not neighbors:
                stack.pop()
                continue
            nx, ny, dx, dy = self.rng.choice(neighbors)
            if dx == 1:
                self.walls_x[cx + 1, cy] = False
            elif dx == -1:
                self.walls_x[cx, cy] = False
            elif dy == 1:
                self.walls_y[cx, cy + 1] = False
            elif dy == -1:
                self.walls_y[cx, cy] = False
            visited[nx, ny] = True
            stack.append((nx, ny))

        extra_links = max(1, n)
        for _ in range(extra_links):
            cx, cy = self.rng.randrange(n), self.rng.randrange(n)
            direction = self.rng.choice(["x", "y"])
            if direction == "x" and cx + 1 < n:
                self.walls_x[cx + 1, cy] = False
            elif direction == "y" and cy + 1 < n:
                self.walls_y[cx, cy + 1] = False

    def cell_center(self, cell):
        cx, cy = cell
        return self.origin + np.array([cx, cy]) * self.cell_pitch

    # ------------------------------------------------------------------
    # Geometry: every wall as (x0, y0, x1, y1, along_x) box specs
    # ------------------------------------------------------------------
    def _room_boundary_walls(self, cx, cy):
        """4 room-perimeter segments for a room with NO neighbours at all (used only as a fallback;
        normally every side is produced by the boundary logic below, once per shared edge)."""
        rh = self.room_half
        return [(cx - rh, cy + rh, cx + rh, cy + rh, True), (cx - rh, cy - rh, cx + rh, cy - rh, True),
                (cx - rh, cy - rh, cx - rh, cy + rh, False), (cx + rh, cy - rh, cx + rh, cy + rh, False)]

    def _boundary_segments(self, open_, cA, cB, axis):
        """One shared boundary between adjacent room centers cA (lower/left) and cB (upper/right)
        along `axis` ('x' or 'y'). Returns wall segments as (x0,y0,x1,y1)."""
        rh, ch = self.room_half, self.corr_half
        segs = []
        if axis == "x":                                             # boundary is vertical, rooms differ in x
            xA, xB = cA[0] + rh, cB[0] - rh                          # xA < xB, gap = CORRIDOR_WIDTH
            y = cA[1]
            if not open_:
                segs.append((xA, y - rh, xB, y - rh))                # seal: fill the gap with one solid block
                segs.append((xA, y - rh, xA, y + rh))
                segs.append((xB, y - rh, xB, y + rh))
                segs.append((xA, y + rh, xB, y + rh))
            else:
                segs.append((xA, y - rh, xA, y - ch))                # room A's face: two stubs flank the doorway
                segs.append((xA, y + ch, xA, y + rh))
                segs.append((xB, y - rh, xB, y - ch))                # room B's face
                segs.append((xB, y + ch, xB, y + rh))
                segs.append((xA, y - ch, xB, y - ch))                # corridor side walls
                segs.append((xA, y + ch, xB, y + ch))
        else:                                                        # boundary is horizontal, rooms differ in y
            yA, yB = cA[1] + rh, cB[1] - rh
            x = cA[0]
            if not open_:
                segs.append((x - rh, yA, x - rh, yB))
                segs.append((x - rh, yA, x + rh, yA))
                segs.append((x - rh, yB, x + rh, yB))
                segs.append((x + rh, yA, x + rh, yB))
            else:
                segs.append((x - rh, yA, x - ch, yA))
                segs.append((x + ch, yA, x + rh, yA))
                segs.append((x - rh, yB, x - ch, yB))
                segs.append((x + ch, yB, x + rh, yB))
                segs.append((x - ch, yA, x - ch, yB))
                segs.append((x + ch, yA, x + ch, yB))
        return segs

    def _all_boundary_segments(self):
        segs = []
        n, pitch = self.grid_n, self.cell_pitch
        for i in range(n + 1):
            for j in range(n):
                cB = self.origin + np.array([i, j]) * pitch
                cA = self.origin + np.array([i - 1, j]) * pitch
                if i == 0 or i == n:
                    segs.append((cB if i == 0 else cA)[0] + (self.room_half if i == 0 else -self.room_half))
                    segs.pop()                                       # outer edge handled by outer wall below
                    continue
                segs += self._boundary_segments(not self.walls_x[i, j], cA, cB, axis="x")
        for i in range(n):
            for j in range(n + 1):
                if j == 0 or j == n:
                    continue
                cB = self.origin + np.array([i, j]) * pitch
                cA = self.origin + np.array([i, j - 1]) * pitch
                segs += self._boundary_segments(not self.walls_y[i, j], cA, cB, axis="y")
        # outer perimeter: every room edge on the arena boundary gets a solid wall (no doors to nowhere)
        rh = self.room_half
        for i in range(n):
            for j in range(n):
                cx, cy = self.origin + np.array([i, j]) * pitch
                if i == 0:
                    segs.append((cx - rh, cy - rh, cx - rh, cy + rh))
                if i == n - 1:
                    segs.append((cx + rh, cy - rh, cx + rh, cy + rh))
                if j == 0:
                    segs.append((cx - rh, cy - rh, cx + rh, cy - rh))
                if j == n - 1:
                    segs.append((cx - rh, cy + rh, cx + rh, cy + rh))
        return segs

    def wall_segments(self):
        """World-space wall line segments [(x0,y0,x1,y1), ...] -- exactly what build_in_pybullet() draws."""
        return np.array(self._all_boundary_segments())

    # ------------------------------------------------------------------
    def build_in_pybullet(self, client=0):
        wall_ids = []
        half_h = WALL_HEIGHT / 2.0

        def add_wall(x0, y0, x1, y1):
            length = float(np.hypot(x1 - x0, y1 - y0))
            if length < 1e-6:
                return
            along_x = abs(x1 - x0) > abs(y1 - y0)
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            sx = length / 2.0 if along_x else WALL_THICKNESS / 2.0
            sy = WALL_THICKNESS / 2.0 if along_x else length / 2.0
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[sx, sy, half_h])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[sx, sy, half_h], rgbaColor=[0.75, 0.72, 0.68, 1.0])
            wall_ids.append(p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                              baseVisualShapeIndex=vis, basePosition=[cx, cy, half_h]))

        for x0, y0, x1, y1 in self._all_boundary_segments():
            add_wall(x0, y0, x1, y1)

        n, pitch = self.grid_n, self.cell_pitch
        floor_half = (n * pitch) / 2.0 + 1.0
        fcol = p.createCollisionShape(p.GEOM_BOX, halfExtents=[floor_half, floor_half, 0.05])
        fvis = p.createVisualShape(p.GEOM_BOX, halfExtents=[floor_half, floor_half, 0.05], rgbaColor=[0.5, 0.5, 0.55, 1.0])
        self.floor_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=fcol, baseVisualShapeIndex=fvis,
                                          basePosition=[0, 0, -0.05])
        ccol = p.createCollisionShape(p.GEOM_BOX, halfExtents=[floor_half, floor_half, 0.02])
        cvis = p.createVisualShape(p.GEOM_BOX, halfExtents=[floor_half, floor_half, 0.02], rgbaColor=[0.2, 0.2, 0.2, 0.35])
        self.ceiling_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=ccol, baseVisualShapeIndex=cvis,
                                            basePosition=[0, 0, WALL_HEIGHT])
        self.wall_ids = wall_ids
        self._add_clutter()
        return wall_ids

    def _add_clutter(self):
        """Scattered crate-like obstacles in a subset of rooms. Rooms are now only 2m x 2m, so clutter
        is smaller and pulled in from the room edge more, or it would block the 1m doorway."""
        entry_xy, exit_xy = self.entry_world_xy(), self.exit_world_xy()
        for cx, cy in self.all_room_centers():
            if np.linalg.norm([cx - entry_xy[0], cy - entry_xy[1]]) < 0.1:
                continue
            if np.linalg.norm([cx - exit_xy[0], cy - exit_xy[1]]) < 0.1:
                continue
            if self.rng.random() > 0.4:
                continue
            for _ in range(self.rng.randint(1, 2)):
                ox, oy = self.rng.uniform(-0.45, 0.45), self.rng.uniform(-0.45, 0.45)
                half_w = self.rng.uniform(0.08, 0.15)
                half_h = self.rng.uniform(0.12, 0.25)
                shade = self.rng.uniform(0.35, 0.55)
                color = [shade + self.rng.uniform(-0.05, 0.05), shade * 0.85, shade * 0.6, 1.0]
                col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half_w, half_w, half_h])
                vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[half_w, half_w, half_h], rgbaColor=color)
                p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                                  basePosition=[cx + ox, cy + oy, half_h])

    # ------------------------------------------------------------------
    # Layout identity + convenience
    # ------------------------------------------------------------------
    def layout_signature(self):
        import hashlib
        h = hashlib.md5()
        h.update(bytes([self.grid_n]))
        h.update(self.walls_x.tobytes())
        h.update(self.walls_y.tobytes())
        return h.hexdigest()

    def layout_split(self, val_fraction=0.2):
        bucket = int(self.layout_signature()[:8], 16) % 100
        return "val" if bucket < int(val_fraction * 100) else "train"

    def entry_world_xy(self):
        return self.cell_center(self.entry_cell)

    def exit_world_xy(self):
        return self.cell_center(self.exit_cell)

    def all_room_centers(self):
        return [self.cell_center((i, j)) for i in range(self.grid_n) for j in range(self.grid_n)]

    def bounds(self):
        """True outer wall extent -- NOT n*cell_pitch/2 (that formula matched the old geometry, where
        an 'open' boundary spanned the full cell pitch; here a room's own wall sits only room_half
        past its outermost centre)."""
        half = self.cell_pitch * (self.grid_n - 1) / 2.0 + self.room_half
        return -half, half
