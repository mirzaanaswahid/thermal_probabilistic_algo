#!/usr/bin/env python3
# ============================================================
#  dummy_swarm_stream.py
#  ------------------------------------------------------------
#  Continuously prints ROS-style status messages for 78 fixed-
#  wing UAVs.  Each line mimics what you would see from
#
#      ros2 topic echo /uav_<id>/status
#
#  – Battery starts at 100 % and—IF the aircraft stayed in
#    “NOMINAL_FLIGHT” the whole time—would linearly fall to
#    20 % in exactly 2 h (7200 s).
#  – Five flight modes are modelled; each scales the drain
#    rate up or down:
#        * NOMINAL_FLIGHT        (×1.00)
#        * BUFFER_ZONE           (×1.10)
#        * EVENT_INVESTIGATION   (×1.25)
#        * THERMAL_SOAR          (×0.25)
#        * GLIDING               (×0.10)
#  – Every 30–180 s a UAV randomly flips to a new mode.
#  – Output frequency is 1 Hz.  Use Ctrl-C to stop.
#
#  You can pipe the output into `ros2 topic pub` or just watch
#  the console.  No ROS 2 Python dependencies are required.
#
#  Author: <your name>, 2025-05
# ============================================================

import time
import json
import random
import itertools
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List

# --------------------  configuration knobs  -----------------
N_UAVS          = 78          # swarm size
LOOP_HZ         = 1.0         # status publish frequency
MISSION_LENGTH  = 120 * 60    # seconds (2 hours)
BAT_FULL        = 100.0       # %
BAT_CRITICAL    = 20.0        # %
SEED            = 42

# Fixed *nominal* drain so that 100 → 20 in 7200 s
BASE_DRAIN = (BAT_FULL - BAT_CRITICAL) / MISSION_LENGTH  # % per second

# Flight-mode multipliers
MODE_DRAIN: Dict[str, float] = {
    "NOMINAL_FLIGHT":       1.00,
    "BUFFER_ZONE":          1.10,
    "EVENT_INVESTIGATION":  1.25,
    "THERMAL_SOAR":         0.25,
    "GLIDING":              0.10,
}
MODES: List[str] = list(MODE_DRAIN.keys())

# --------------------  data structures  ---------------------
@dataclass
class UAV:
    uid: int
    battery: float = BAT_FULL
    mode: str = "NOMINAL_FLIGHT"
    _time_to_next_flip: float = field(default_factory=lambda: random.randint(30, 180))

    def step(self, dt: float):
        """Advance simulation by dt seconds."""
        # battery drain
        drain = BASE_DRAIN * MODE_DRAIN[self.mode] * dt
        self.battery = max(BAT_CRITICAL, self.battery - drain)

        # mode change countdown
        self._time_to_next_flip -= dt
        if self._time_to_next_flip <= 0:
            self._change_mode()
            self._time_to_next_flip = random.randint(30, 180)

    # ------------------  private helpers  -------------------
    def _change_mode(self):
        self.mode = random.choice(MODES)

    # ----------------  message serialisation  ---------------
    def ros_status(self, sim_time: float) -> str:
        """Return a JSON string similar to a ROS 2 std_msgs/String."""
        msg = {
            "sim_time":     round(sim_time, 1),
            "id":           self.uid,
            "mode":         self.mode,
            "battery":      round(self.battery, 2),
        }
        stamp = f"{datetime.utcnow().timestamp():.3f}"
        return (f"[INFO] [{stamp}] [/uav_{self.uid:02d}/status] "
                f"{json.dumps(msg, separators=(',', ':'))}")

# --------------------  main loop  ---------------------------
def main() -> None:
    random.seed(SEED)
    uavs = [UAV(i + 1) for i in range(N_UAVS)]
    dt = 1.0 / LOOP_HZ
    sim_time = 0.0

    try:
        for _ in itertools.count():
            tic = time.perf_counter()

            # update and print
            for u in uavs:
                u.step(dt)
                print(u.ros_status(sim_time))

            # advance clock
            sim_time += dt
            if sim_time >= MISSION_LENGTH:
                break

            # simple timing to maintain LOOP_HZ
            toc = time.perf_counter()
            time.sleep(max(0.0, dt - (toc - tic)))
    except KeyboardInterrupt:
        print("\n[INFO] Simulation interrupted by user.")

# --------------------  entry point  -------------------------
if __name__ == "__main__":
    main()

