import numpy as _np
"""
Navigator: decides WHERE to fly (explore / return to exit / escape), follows the path smoothly,
and a safety governor that is the last thing between a command and the motors.

Everything here works in the ESTIMATOR's frame, or in the drone's body frame using lidar. It never
reads the simulator's true pose.
"""

import numpy as np

from maze_env import config as C
from maze_env.planning import Planner


# ----------------------------------------------------------------------------
def safety_govern(v_body, angles, ranges, r_stop=0.30, gain=2.2, push_dist=0.32, push_gain=1.2):
    """
    Body-frame velocity filter using the live lidar scan.
      1. Speed toward any obstacle is capped at gain*(range - r_stop): the closer the wall the
         slower we may approach it, zero at r_stop. Sliding ALONG a wall is untouched.
         (The old filter summed over 180 rays and throttled sideways motion too.)
      2. Within push_dist of a wall, add a gentle push away, so the drone never rests against one.
    """
    v = np.asarray(v_body, dtype=float).copy()
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    allowed = gain * np.maximum(ranges - r_stop, 0.0)
    for _ in range(4):
        viol = dirs @ v - allowed
        i = int(np.argmax(viol))
        if viol[i] <= 1e-4:
            break
        v = v - viol[i] * dirs[i]
    close = ranges < push_dist
    if close.any():
        w = (push_dist - ranges[close]) / push_dist
        push = -(dirs[close] * w[:, None]).sum(axis=0) * (push_gain / max(close.sum() ** 0.5, 1.0))
        mag = np.linalg.norm(push)
        if mag > 0.5:
            push *= 0.5 / mag
        v = v + push
    return v, float(ranges.min())


def slew_limit(v_prev, v_des, dt, max_accel=C.MAX_ACCEL):
    delta = v_des - v_prev
    m = np.linalg.norm(delta)
    lim = max_accel * dt
    return v_des if m <= lim else v_prev + delta * (lim / m)


# ----------------------------------------------------------------------------
class Navigator:
    def __init__(self, occ_map, exit_xy, replan_period=0.5, room_centers=None):
        self.map = occ_map
        self.planner = Planner(occ_map)
        self.exit_xy = np.asarray(exit_xy, dtype=float)
        # Explicit backstop on top of the frontier/unseen-cell logic: a corridor mouth can let lidar
        # glimpse a few cells of the room beyond without the drone ever actually flying in to look at
        # it (measured: this stranded ~2 rooms per maze, unmapped, while exploration declared itself
        # done). Every room center gets checked directly against occ_map.seen; if a room the drone has
        # never actually looked at is reachable, it always outranks the generic frontier/unseen goal.
        self.room_centers = [np.asarray(c, dtype=float) for c in (room_centers or [])]
        self._room_goal_since = {}
        self._room_visited = set()  # indices into room_centers confirmed genuinely scanned
        self.replan_period = replan_period
        self.mode = "explore"              # explore | exit | escape
        self.resume_mode = "explore"       # what to go back to after an escape maneuver
        self.goal_idx = None
        self.goal_kind = None
        self.path = None
        self.next_replan = 0.0
        self.escape_until = 0.0
        self.explored_out = False          # nothing left to explore or look at
        self._ref_pos, self._ref_t = None, 0.0
        self.stuck_events = 0
        self.log_goals = 0
        self._retries = 0
        self.exit_cost = float('inf')

    # ---- planning ------------------------------------------------------
    def _plan(self, now, pos):
        pl = self.planner
        pl.refresh()
        pl.dijkstra(pos)
        g_exit = pl.nearest_reachable_to(self.exit_xy)
        self.exit_cost = float(pl.D[g_exit]) if g_exit is not None else float("inf")
        if self.mode == "escape":
            g = pl.clearest_cell_nearby(pos)
            if g is None or now > self.escape_until:
                self.mode = self.resume_mode
            else:
                self.goal_idx, self.goal_kind = g, "escape"
        if self.mode == "exit":
            g = pl.nearest_reachable_to(self.exit_xy)
            self.goal_idx, self.goal_kind = g, "exit"
        elif self.mode == "explore":
            room_goal = self._unvisited_room_goal(now)
            if room_goal is not None and (self.goal_kind != "room" or not pl.goal_still_valid(self.goal_idx, "room", now)):
                self.goal_idx, self.goal_kind = room_goal, "room"
                self.log_goals += 1
            elif room_goal is None and (not pl.goal_still_valid(self.goal_idx, self.goal_kind, now) or self.goal_kind in ("escape", "room")):
                self.goal_idx, self.goal_kind = pl.select_goal(now)
                self.log_goals += 1
            if self.goal_idx is None and self.planner.blacklist and self._retries < 2:
                # targets we gave up on may be reachable now that more is mapped: retry twice
                self._retries += 1
                self.planner.blacklist = []
                self.goal_idx, self.goal_kind = pl.select_goal(now)
            if self.goal_idx is None:
                self.explored_out = True
        if self.goal_idx is None:
            self.path = None
            return
        self.path = pl.path_to(self.goal_idx)

    def _room_is_seen(self, c, margin=0.35, grid=3):
        """A room only counts as done once most of its floor has actually been in the camera's view --
        NOT just its single centre cell. A single grazing lidar ray could mark that one cell 'seen'
        while the drone never actually entered or scanned the room (measured: this let 4 of 9 rooms in
        a maze get silently skipped while still passing a centre-only check)."""
        m = self.map
        offsets = np.linspace(-margin, margin, grid)
        hits = 0
        for dx in offsets:
            for dy in offsets:
                gx, gy = m.world_to_grid((c[0] + dx, c[1] + dy))
                hits += int(m.seen[gx, gy])
        return hits / (grid * grid) >= 0.75

    def _unvisited_room_goal(self, now):
        """Cheapest reachable, not-blacklisted cell near a room whose floor the camera has not actually
        scanned. Returns a grid index, or None if every room is either seen or unreachable."""
        if not self.room_centers:
            return None
        pl = self.planner
        m = self.map
        bad = pl._blacklisted_cells(now)
        best, best_cost, best_key = None, float("inf"), None
        for i, c in enumerate(self.room_centers):
            if i in self._room_visited:
                continue
            if self._room_is_seen(c):
                self._room_visited.add(i)
                continue
            gx, gy = m.world_to_grid(c)
            first_seen = self._room_goal_since.get((gx, gy))
            if first_seen is not None and now - first_seen > 25.0:
                # scanned from right next to it for 25s and it STILL never registered -- a geometry
                # edge case, not worth spending the rest of the mission on
                self.planner.blacklist.append((c.copy(), now + 200.0))
                self._room_visited.add(i)
                continue
            idx = pl.nearest_reachable_to(c)
            if idx is None:
                continue
            ix, iy = divmod(idx, m.n)
            if bad is not None and bad[ix, iy]:
                continue
            cost = float(pl.D[idx])
            if cost < best_cost:
                best, best_cost, best_key = idx, cost, (gx, gy)
        if best is not None:
            self._room_goal_since.setdefault(best_key, now)
        return best

    def goal_xy(self):
        if self.goal_idx is None:
            return None
        gx, gy = divmod(self.goal_idx, self.map.n)
        return self.map.grid_to_world(gx, gy)

    # ---- path following ------------------------------------------------
    def _pursuit_target(self, pos, lookahead=0.75):
        P = self.path
        if P is None or len(P) == 0:
            return None, 0.0
        if len(P) == 1:
            return P[0], float(np.linalg.norm(P[0] - pos))
        seg = P[1:] - P[:-1]
        seg_len = np.linalg.norm(seg, axis=1)
        seg_len = np.maximum(seg_len, 1e-6)
        t = np.clip(((pos - P[:-1]) * seg).sum(1) / seg_len ** 2, 0.0, 1.0)
        proj = P[:-1] + seg * t[:, None]
        i = int(np.argmin(np.linalg.norm(proj - pos, axis=1)))
        remaining = float((1 - t[i]) * seg_len[i] + seg_len[i + 1:].sum())
        # walk `lookahead` along the polyline from the projection
        need = lookahead
        p = proj[i]
        for j in range(i, len(seg)):
            end = P[j + 1]
            d = np.linalg.norm(end - p)
            if d >= need:
                return p + (end - p) * (need / max(d, 1e-9)), remaining
            need -= d
            p = end
        return P[-1], remaining

    # ---- main entry ----------------------------------------------------
    def command(self, now, pos, min_clear):
        """Returns the desired velocity (2,) in the estimator frame plus a status string."""
        if now >= self.next_replan:
            self._plan(now, pos)
            self.next_replan = now + self.replan_period

        # stall detector: real progress, not "the estimate moved". Sitting still because the drone is
        # deliberately parked at an unvisited room's centre, scanning, is not stuck -- give that its
        # own longer patience window before treating no-movement as a real problem.
        if self._ref_pos is None:
            self._ref_pos, self._ref_t = pos.copy(), now
        else:
            g = self.goal_xy()
            parked_scanning = self.goal_kind == "room" and g is not None and np.linalg.norm(pos - g) < 0.6
            window = 15.0 if parked_scanning else 6.0
            if now - self._ref_t >= window:
                moved = np.linalg.norm(pos - self._ref_pos)
                if moved < 0.5 and self.mode in ("explore", "exit"):
                    self._on_stuck(now, pos)
                self._ref_pos, self._ref_t = pos.copy(), now

        if self.path is None:
            return np.zeros(2), "no-path"
        target, remaining = self._pursuit_target(pos)
        if target is None:
            return np.zeros(2), "no-target"
        to_t = target - pos
        dist = np.linalg.norm(to_t)
        direction = to_t / dist if dist > 1e-3 else np.zeros(2)

        speed = C.MAX_SPEED
        speed *= float(np.clip((min_clear - 0.30) / 0.30, 0.6, 1.0))      # gentler near walls
        if self.goal_kind in ("exit", "escape"):
            speed = min(speed, max(0.25, 0.7 * remaining))                    # ease into the exit
        if self.mode == "escape" and remaining < 0.3:
            self.mode = self.resume_mode
            self.goal_idx, self.goal_kind = None, None
            self.next_replan = 0.0
        # goal reached -> ask for a new one immediately (no stopping at intermediate goals)
        if remaining < 0.45 and self.goal_kind in ("frontier", "unseen"):
            self.next_replan = 0.0
            if self.goal_kind == "frontier":
                self._mark_visited_frontier(now)
        return direction * speed, self.mode

    def _mark_visited_frontier(self, now):
        g = self.goal_xy()
        if g is not None:
            # if it is STILL a frontier after we stood on it, unknown cells behind it are not
            # observable (thin wall gaps etc.) -- stop chasing it
            self.planner.blacklist.append((g.copy(), now + 30.0))

    def _on_stuck(self, now, pos):
        self.stuck_events += 1
        g = self.goal_xy()
        if g is not None:
            self.planner.blacklist.append((g.copy(), now + 45.0))
        self.goal_idx, self.goal_kind = None, None
        if self.mode != "escape":
            self.resume_mode = self.mode
        self.mode = "escape"
        self.escape_until = now + 4.0
        self.next_replan = 0.0

    def start_exit(self):
        self.mode = "exit"
        self.resume_mode = "exit"
        self.goal_idx, self.goal_kind, self.path = None, None, None
        self.next_replan = 0.0
