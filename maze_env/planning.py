"""
Path planning on the occupancy grid. Replaces the old pathfinding.py + exploration.py.

What was wrong before (measured, not guessed):
  * A* ran over raw free cells with NO clearance, so paths hugged walls (median closest approach
    0.08 m, 36% of the path within 0.3 m of a wall). The safety filter then blocked motion toward
    those walls and the drone sat at corners.
  * Frontiers were chosen by straight-line distance, so the "nearest" one was often behind a wall.
  * Exploration stopped when no frontier was >0.6 m away and dropped free cells at random.

Now:
  * Walls are inflated by ROBOT_RADIUS; a soft cost keeps paths near the middle of passages.
  * ONE Dijkstra sweep from the drone gives the true travel cost to every reachable cell; frontiers
    and unseen floor are ranked by that real cost, so unreachable targets are never chosen.
  * Paths are string-pulled with a clearance-aware line-of-sight test (no corner cutting).
"""

import heapq
import numpy as np

from maze_env import config as C

_NB8 = [(-1, -1, 1.4142), (-1, 0, 1.0), (-1, 1, 1.4142), (0, -1, 1.0),
        (0, 1, 1.0), (1, -1, 1.4142), (1, 0, 1.0), (1, 1, 1.4142)]
_INF = 1e18


class Planner:
    def __init__(self, occ_map, robot_radius=C.ROBOT_RADIUS, soft_margin=0.85):
        self.map = occ_map
        self.r = robot_radius
        self.soft = soft_margin
        self.R = int(np.ceil(soft_margin / occ_map.res)) + 1
        self.blacklist = []          # [(xy, expiry_time)]
        self.last_trav = None

    # ------------------------------------------------------------------
    def refresh(self):
        m = self.map
        L = m.log_odds
        self.occ = L > 0.4
        self.free = L < -0.3
        self.unk = np.abs(L) < 0.05
        obst = self.occ | (~self.free & ~self.unk)          # ambiguous cells count as obstacles
        self.dist = self._distance_field(obst)
        self.trav = self.free & (self.dist >= self.r)
        pen = np.clip((self.soft - self.dist) / max(self.soft - self.r, 1e-6), 0.0, 1.0)
        self.cost = 1.0 + 3.0 * pen

    def _distance_field(self, obst):
        n, m = obst.shape
        R = self.R
        pad = np.pad(obst, R, constant_values=False)
        dist = np.full((n, m), R + 1.0)
        for dx in range(-R, R + 1):
            for dy in range(-R, R + 1):
                d = np.hypot(dx, dy)
                if d > R:
                    continue
                sh = pad[R + dx:R + dx + n, R + dy:R + dy + m]
                dist = np.where(sh & (d < dist), d, dist)
        return dist * self.map.res

    # ------------------------------------------------------------------
    def dijkstra(self, start_xy, max_cost=250.0):
        n = self.map.n
        sx, sy = self.map.world_to_grid(start_xy)
        trav = self.trav.copy()
        # the drone may currently be closer to a wall than the planning clearance: let it leave
        # through free cells that still have some clearance
        r = 5
        x0, x1, y0, y1 = max(sx - r, 0), min(sx + r + 1, n), max(sy - r, 0), min(sy + r + 1, n)
        trav[x0:x1, y0:y1] |= self.free[x0:x1, y0:y1] & (self.dist[x0:x1, y0:y1] >= 0.10)
        trav[sx, sy] = True
        self.last_trav = trav
        ft = trav.ravel().tolist()
        cost = self.cost.ravel().tolist()
        dist = [_INF] * (n * n)
        parent = [-1] * (n * n)
        s = sx * n + sy
        dist[s] = 0.0
        heap = [(0.0, s)]
        pop, push = heapq.heappop, heapq.heappush
        while heap:
            d, u = pop(heap)
            if d > dist[u]:
                continue
            if d > max_cost:
                break
            ux, uy = divmod(u, n)
            for dx, dy, step in _NB8:
                vx, vy = ux + dx, uy + dy
                if vx < 0 or vx >= n or vy < 0 or vy >= n:
                    continue
                v = vx * n + vy
                if not ft[v]:
                    continue
                if dx and dy and not (ft[(ux + dx) * n + uy] and ft[ux * n + vy]):
                    continue                                   # no diagonal corner cutting
                nd = d + step * cost[v]
                if nd < dist[v]:
                    dist[v] = nd
                    parent[v] = u
                    push(heap, (nd, v))
        self.D = np.array(dist)
        self.parent = parent
        self.start_idx = s
        return self.D

    # ------------------------------------------------------------------
    def frontier_mask(self):
        # Unknown cells hugging a wall are just the un-observed sliver beside it (free space is
        # deliberately not marked right up to a wall), not unexplored territory.
        unk = self.unk & (self.dist >= 0.45)
        nb = np.zeros_like(unk)
        p = np.pad(unk, 1, constant_values=False)
        n, m = unk.shape
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    nb |= p[1 + dx:1 + dx + n, 1 + dy:1 + dy + m]
        fr = self.free & nb & self.trav
        cnt = np.zeros(fr.shape, dtype=np.int8)                # drop isolated speckles
        pf = np.pad(fr, 1, constant_values=False)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    cnt += pf[1 + dx:1 + dx + n, 1 + dy:1 + dy + m]
        return fr & (cnt >= 1)

    def unseen_mask(self):
        """Reachable-looking floor the camera has never had in view (clustered, not speckle)."""
        u = self.trav & ~self.map.seen
        cnt = np.zeros(u.shape, dtype=np.int8)
        pu = np.pad(u, 1, constant_values=False)
        n, m = u.shape
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    cnt += pu[1 + dx:1 + dx + n, 1 + dy:1 + dy + m]
        return u & (cnt >= 4)

    def _blacklisted_cells(self, now):
        self.blacklist = [(xy, t) for xy, t in self.blacklist if t > now]
        if not self.blacklist:
            return None
        n, res, half = self.map.n, self.map.res, self.map.half_size
        gx = (np.arange(n) * res - half + res / 2)
        X, Y = np.meshgrid(gx, gx, indexing="ij")
        bad = np.zeros((n, n), dtype=bool)
        for xy, _ in self.blacklist:
            bad |= (X - xy[0]) ** 2 + (Y - xy[1]) ** 2 < 0.7 ** 2
        return bad

    def select_goal(self, now, frontier_bonus=1.5):
        """Cheapest reachable frontier or unseen-floor cell. Returns (cell_idx, kind) or (None, None)."""
        bad = self._blacklisted_cells(now)
        fr = self.frontier_mask()
        un = self.unseen_mask()
        if bad is not None:
            fr &= ~bad
            un &= ~bad
        D = self.D
        best, best_score, kind = None, _INF, None
        fi = np.flatnonzero(fr.ravel() & (D < _INF))
        if len(fi):
            j = fi[np.argmin(D[fi])]
            best, best_score, kind = int(j), D[j] - frontier_bonus, "frontier"
        ui = np.flatnonzero(un.ravel() & (D < _INF))
        if len(ui):
            j = ui[np.argmin(D[ui])]
            if D[j] < best_score:
                best, best_score, kind = int(j), D[j], "unseen"
        return best, kind

    def goal_still_valid(self, goal_idx, kind, now):
        if goal_idx is None or self.D[goal_idx] >= _INF:
            return False
        bad = self._blacklisted_cells(now)
        n = self.map.n
        gx, gy = divmod(goal_idx, n)
        if bad is not None and bad[gx, gy]:
            return False
        if kind == "frontier":
            return bool(self.frontier_mask()[gx, gy])
        if kind == "unseen":
            return bool(self.unseen_mask()[gx, gy])
        return True

    def nearest_reachable_to(self, xy):
        """Reachable traversable cell closest (straight-line) to a world point, e.g. the exit."""
        n, res, half = self.map.n, self.map.res, self.map.half_size
        idx = np.flatnonzero((self.D < _INF).ravel())
        if len(idx) == 0:
            return None
        gx, gy = np.divmod(idx, n)
        wx = gx * res - half + res / 2
        wy = gy * res - half + res / 2
        return int(idx[np.argmin((wx - xy[0]) ** 2 + (wy - xy[1]) ** 2)])

    def clearest_cell_nearby(self, start_xy, radius=1.8, min_clear=0.55):
        n, res, half = self.map.n, self.map.res, self.map.half_size
        idx = np.flatnonzero(((self.D < _INF) & (self.dist.ravel() >= min_clear)))
        if len(idx) == 0:
            return None
        gx, gy = np.divmod(idx, n)
        wx = gx * res - half + res / 2
        wy = gy * res - half + res / 2
        d = np.hypot(wx - start_xy[0], wy - start_xy[1])
        ok = d < radius
        if not ok.any():
            return None
        return int(idx[ok][np.argmin(self.D[idx[ok]])])

    # ------------------------------------------------------------------
    def path_to(self, goal_idx):
        """World-space waypoints (N,2) from the drone to goal_idx, string-pulled."""
        n = self.map.n
        cells = []
        u = goal_idx
        while u != -1:
            cells.append(u)
            u = self.parent[u]
        cells.reverse()
        cxy = np.array([divmod(c, n) for c in cells])
        if len(cxy) <= 2:
            keep = list(range(len(cxy)))
        else:
            keep = [0]
            a = 0
            i = 2
            while i < len(cxy):
                if not self._line_free(cxy[a], cxy[i]):
                    keep.append(i - 1)
                    a = i - 1
                i += 1
            keep.append(len(cxy) - 1)
        res, half = self.map.res, self.map.half_size
        return cxy[keep] * res - half + res / 2

    def _line_free(self, a, b):
        k = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) * 2 + 2
        t = np.linspace(0.0, 1.0, k)
        xs = np.rint(a[0] + (b[0] - a[0]) * t).astype(int)
        ys = np.rint(a[1] + (b[1] - a[1]) * t).astype(int)
        return bool(self.last_trav[xs, ys].all())
