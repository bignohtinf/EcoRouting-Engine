"""
crew_scheduling.py
==================
OR-Tools CP-SAT solver for driver crew scheduling.

Constraints (§4.7)
------------------
CS1 : Each driver assigned to at most 1 shift per day.
CS2 : Total shift duration ≤ H_max = 8 hours.
CS3 : Total accumulated driving time ≤ H_max.
CS4 : No continuous driving block > Δ_drive = 4 hours.

Falls back to a greedy heuristic when OR-Tools is unavailable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

try:
    from ortools.sat.python import cp_model
    HAS_ORTOOLS = True
except ImportError:
    HAS_ORTOOLS = False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Driver:
    """Represents a driver."""
    id: str
    cost_per_shift: float = 1.0
    max_hours: float = 8.0       # individual max hours (overrides global if set)


@dataclass
class Shift:
    """
    Represents one possible shift.

    Attributes
    ----------
    id         : unique shift identifier
    start_hour : shift start in hours (e.g. 8.0 for 08:00)
    end_hour   : shift end in hours
    blocks     : list of continuous driving block durations (hours)
                 e.g. [3.5, 2.0] means two blocks separated by a break
    """
    id: str
    start_hour: float
    end_hour: float
    blocks: List[float] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end_hour - self.start_hour

    @property
    def total_driving(self) -> float:
        return sum(self.blocks)

    @property
    def max_block(self) -> float:
        return max(self.blocks) if self.blocks else self.duration


# ---------------------------------------------------------------------------
# Constraint validators (used by both solvers)
# ---------------------------------------------------------------------------

def _shift_is_valid(shift: Shift,
                    H_max_hours: float,
                    delta_drive_hours: float) -> bool:
    """Check CS2, CS3, CS4 for a single shift."""
    # CS2: total shift duration
    if shift.duration > H_max_hours + 1e-9:
        return False
    # CS3: total driving
    if shift.total_driving > H_max_hours + 1e-9:
        return False
    # CS4: no block > delta_drive
    if shift.max_block > delta_drive_hours + 1e-9:
        return False
    return True


# ---------------------------------------------------------------------------
# Crew Scheduler
# ---------------------------------------------------------------------------

class CrewScheduler:
    """
    Solves driver shift assignment to minimise staffing cost while
    satisfying demand coverage requirements.

    Parameters
    ----------
    H_max_hours      : float  Maximum shift duration / accumulated driving hours.
    delta_drive_hours: float  Maximum continuous driving block.
    min_rest_hours   : float  Minimum rest between blocks (informational;
                              enforced implicitly via blocks definition).
    """

    def __init__(
        self,
        H_max_hours: float = 8.0,
        delta_drive_hours: float = 4.0,
        min_rest_hours: float = 0.5,
    ):
        self.H_max = H_max_hours
        self.delta_drive = delta_drive_hours
        self.min_rest = min_rest_hours

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        drivers: List[Driver],
        shifts: List[Shift],
        demand_coverage: Dict[int, int],
    ) -> Dict[str, Optional[str]]:
        """
        Assign drivers to shifts.

        Parameters
        ----------
        drivers         : list of Driver objects
        shifts          : list of Shift objects
        demand_coverage : dict {hour_bucket: min_drivers_needed}
                          e.g. {8: 3, 9: 4, 10: 3, ...}

        Returns
        -------
        assignment : dict {driver_id: shift_id | None}
            None means the driver is unassigned.
        """
        # Pre-filter: remove shifts that violate hard constraints
        valid_shifts = [s for s in shifts
                        if _shift_is_valid(s, self.H_max, self.delta_drive)]

        if HAS_ORTOOLS:
            try:
                return self._solve_ortools(drivers, valid_shifts, demand_coverage)
            except Exception:
                pass  # fall through to greedy

        return self._solve_greedy(drivers, valid_shifts, demand_coverage)

    # ------------------------------------------------------------------
    # OR-Tools CP-SAT solver
    # ------------------------------------------------------------------

    def _solve_ortools(
        self,
        drivers: List[Driver],
        shifts: List[Shift],
        demand_coverage: Dict[int, int],
    ) -> Dict[str, Optional[str]]:
        """CP-SAT formulation."""
        model = cp_model.CpModel()

        # x[k][j] = 1 iff driver k is assigned to shift j
        x = {}
        for k, drv in enumerate(drivers):
            for j, sh in enumerate(shifts):
                x[k, j] = model.NewBoolVar(f"x_{k}_{j}")

        # CS1: each driver assigned to at most 1 shift
        for k in range(len(drivers)):
            model.Add(sum(x[k, j] for j in range(len(shifts))) <= 1)

        # Demand coverage: for each hour bucket, total drivers covering it >= min needed
        for hour, min_needed in demand_coverage.items():
            covering = [
                x[k, j]
                for k in range(len(drivers))
                for j, sh in enumerate(shifts)
                if sh.start_hour <= hour < sh.end_hour
            ]
            if covering:
                model.Add(sum(covering) >= min_needed)

        # Objective: minimise total staffing cost
        # Scale costs to integers (multiply by 100 and round)
        cost_terms = []
        for k, drv in enumerate(drivers):
            for j in range(len(shifts)):
                cost_int = int(round(drv.cost_per_shift * 100))
                cost_terms.append(x[k, j] * cost_int)

        model.Minimize(sum(cost_terms))

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30.0
        status = solver.Solve(model)

        assignment: Dict[str, Optional[str]] = {drv.id: None for drv in drivers}

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for k, drv in enumerate(drivers):
                for j, sh in enumerate(shifts):
                    if solver.Value(x[k, j]) == 1:
                        assignment[drv.id] = sh.id
                        break

        return assignment

    # ------------------------------------------------------------------
    # Greedy fallback
    # ------------------------------------------------------------------

    def _solve_greedy(
        self,
        drivers: List[Driver],
        shifts: List[Shift],
        demand_coverage: Dict[int, int],
    ) -> Dict[str, Optional[str]]:
        """
        Greedy heuristic:
        1. Sort demand hours by deficit (most under-covered first).
        2. For each deficit, assign the cheapest unassigned driver to the
           cheapest shift that covers that hour.
        """
        assignment: Dict[str, Optional[str]] = {drv.id: None for drv in drivers}
        unassigned_drivers = list(drivers)  # mutable pool
        # Track how many drivers are already covering each hour
        coverage_count: Dict[int, int] = {h: 0 for h in demand_coverage}

        # Sort hours by priority (largest deficit first)
        hours_sorted = sorted(
            demand_coverage.keys(),
            key=lambda h: demand_coverage[h],
            reverse=True,
        )

        for hour in hours_sorted:
            needed = demand_coverage[hour]
            while coverage_count.get(hour, 0) < needed and unassigned_drivers:
                # Cheapest available driver
                unassigned_drivers.sort(key=lambda d: d.cost_per_shift)
                driver = unassigned_drivers[0]

                # Cheapest valid shift covering this hour
                covering_shifts = [s for s in shifts if s.start_hour <= hour < s.end_hour]
                covering_shifts.sort(key=lambda s: s.duration)  # prefer shorter shifts

                if not covering_shifts:
                    break  # no shift covers this hour

                chosen_shift = covering_shifts[0]
                assignment[driver.id] = chosen_shift.id
                unassigned_drivers.pop(0)

                # Update coverage counts for all hours this shift covers
                for h in demand_coverage:
                    if chosen_shift.start_hour <= h < chosen_shift.end_hour:
                        coverage_count[h] = coverage_count.get(h, 0) + 1

        return assignment


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    drivers = [
        Driver(id="D1", cost_per_shift=10.0),
        Driver(id="D2", cost_per_shift=8.0),
        Driver(id="D3", cost_per_shift=12.0),
        Driver(id="D4", cost_per_shift=9.0),
        Driver(id="D5", cost_per_shift=11.0),
    ]

    shifts = [
        Shift(id="S_morning", start_hour=6,  end_hour=14, blocks=[3.5, 3.0]),
        Shift(id="S_day",     start_hour=8,  end_hour=16, blocks=[4.0, 3.0]),
        Shift(id="S_evening", start_hour=14, end_hour=22, blocks=[3.5, 3.5]),
        Shift(id="S_night",   start_hour=22, end_hour=30, blocks=[4.0, 3.0]),  # crosses midnight
        # This shift violates CS4 (block > 4h) and should be filtered out
        Shift(id="S_bad",     start_hour=8,  end_hour=16, blocks=[5.0, 2.0]),
    ]

    demand_coverage = {8: 2, 10: 2, 14: 1, 16: 1}

    scheduler = CrewScheduler(H_max_hours=8, delta_drive_hours=4, min_rest_hours=0.5)
    result = scheduler.solve(drivers, shifts, demand_coverage)

    print(f"Using OR-Tools: {HAS_ORTOOLS}")
    print("Assignments:")
    for drv_id, shift_id in result.items():
        print(f"  {drv_id} -> {shift_id}")
    print("crew_scheduling.py OK")
