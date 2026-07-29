from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any, Optional, Tuple, Type

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.callback.abstract_callback import AbstractCallback
from nuplan.planning.simulation.callback.multi_callback import MultiCallback
from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory

logger = logging.getLogger(__name__)

# Lookahead [m] for the "end the simulation at the longitudinal end of the mapped road" check
END_AT_MAP_EDGE_LOOKAHEAD_M = 10.0


def _normalize_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class Simulation:
    """
    This class queries data for initialization of a planner, and propagates simulation a step forward based on the
        planned trajectory of a planner.
    """

    def __init__(
        self,
        simulation_setup: SimulationSetup,
        callback: Optional[AbstractCallback] = None,
        simulation_history_buffer_duration: float = 2,
        end_at_map_edge: bool = False,
    ):
        """
        Create Simulation.
        :param simulation_setup: Configuration that describes the simulation.
        :param callback: A callback to be executed for this simulation setup
        :param simulation_history_buffer_duration: [s] Duration to pre-load scenario into the buffer.
        :param end_at_map_edge: If True, end the simulation once the ego reaches the longitudinal end
            of the mapped road (the forward lane graph dead-ends within a short lookahead), while it
            is still on-road. Used under nucontrol alternative routing, where routes can run past the
            cropped map edge; stopping early keeps drivable_area_compliance from being zeroed by an
            off-map frame. Default False keeps every other simulation identical.
        """
        if simulation_history_buffer_duration < simulation_setup.scenario.database_interval:
            raise ValueError(
                f"simulation_history_buffer_duration {simulation_history_buffer_duration} has to be larger than the scenario database_interval {simulation_setup.scenario.database_interval}"
            )

        # Store all engines
        self._setup = simulation_setup

        # Proxy
        self._time_controller = simulation_setup.time_controller
        self._ego_controller = simulation_setup.ego_controller
        self._observations = simulation_setup.observations
        self._scenario = simulation_setup.scenario
        self._callback = MultiCallback([]) if callback is None else callback

        # History where the steps of a simulation are stored
        self._history = SimulationHistory(self._scenario.map_api, self._scenario.get_mission_goal())

        # Rolling window of past states
        # We add self._scenario.database_interval to the buffer duration here to ensure that the minimum
        # simulation_history_buffer_duration is satisfied
        self._simulation_history_buffer_duration = simulation_history_buffer_duration + self._scenario.database_interval

        # The + 1 here is to account for duration. For example, 20 steps at 0.1s starting at 0s will have a duration
        # of 1.9s. At 21 steps the duration will achieve the target 2s duration.
        self._history_buffer_size = int(self._simulation_history_buffer_duration / self._scenario.database_interval) + 1
        self._history_buffer: Optional[SimulationHistoryBuffer] = None

        # Flag that keeps track whether simulation is still running
        self._is_simulation_running = True

        # End-at-map-edge early termination (see end_at_map_edge above)
        self._end_at_map_edge = end_at_map_edge
        self._end_lookahead_m = END_AT_MAP_EDGE_LOOKAHEAD_M

    def __reduce__(self) -> Tuple[Type[Simulation], Tuple[Any, ...]]:
        """
        Hints on how to reconstruct the object when pickling.
        :return: Object type and constructor arguments to be used.
        """
        # NOTE: keep every constructor arg here — an omitted one silently reverts to its default when
        # the object is unpickled on a Ray worker (this is what carries _end_at_map_edge to workers).
        return self.__class__, (
            self._setup,
            self._callback,
            self._simulation_history_buffer_duration,
            self._end_at_map_edge,
        )

    def is_simulation_running(self) -> bool:
        """
        Check whether a simulation reached the end
        :return True if simulation hasn't reached the end, otherwise false.
        """
        return not self._time_controller.reached_end() and self._is_simulation_running

    def reset(self) -> None:
        """
        Reset all internal states of simulation.
        """
        # Clear created log
        self._history.reset()

        # Reset all simulation internal members
        self._setup.reset()

        # Clear history buffer
        self._history_buffer = None

        # Restart simulation
        self._is_simulation_running = True

    def initialize(self) -> PlannerInitialization:
        """
        Initialize the simulation
         - Initialize Planner with goals and maps
        :return data needed for planner initialization.
        """
        self.reset()

        # Initialize history from scenario
        self._history_buffer = SimulationHistoryBuffer.initialize_from_scenario(
            self._history_buffer_size, self._scenario, self._observations.observation_type()
        )

        # Initialize observations
        self._observations.initialize()

        # Add the current state into the history buffer
        self._history_buffer.append(self._ego_controller.get_state(), self._observations.get_observation())

        # Return the planner initialization structure for this simulation
        return PlannerInitialization(
            route_roadblock_ids=self._scenario.get_route_roadblock_ids(),
            mission_goal=self._scenario.get_mission_goal(),
            map_api=self._scenario.map_api,
        )

    def get_planner_input(self) -> PlannerInput:
        """
        Construct inputs to the planner for the current iteration step
        :return Inputs to the planner.
        """
        if self._history_buffer is None:
            raise RuntimeError("Simulation was not initialized!")

        if not self.is_simulation_running():
            raise RuntimeError("Simulation is not running, stepping can not be performed!")

        # Extract current state
        iteration = self._time_controller.get_iteration()

        # Extract traffic light status data
        traffic_light_data = list(self._scenario.get_traffic_light_status_at_iteration(iteration.index))
        logger.debug(f"Executing {iteration.index}!")
        return PlannerInput(iteration=iteration, history=self._history_buffer, traffic_light_data=traffic_light_data)

    def propagate(self, trajectory: AbstractTrajectory) -> None:
        """
        Propagate the simulation based on planner's trajectory and the inputs to the planner
        This function also decides whether simulation should still continue. This flag can be queried through
        reached_end() function
        :param trajectory: computed trajectory from planner.
        """
        if self._history_buffer is None:
            raise RuntimeError("Simulation was not initialized!")

        if not self.is_simulation_running():
            raise RuntimeError("Simulation is not running, simulation can not be propagated!")

        # Measurements
        iteration = self._time_controller.get_iteration()
        ego_state, observation = self._history_buffer.current_state
        traffic_light_status = list(self._scenario.get_traffic_light_status_at_iteration(iteration.index))

        # Add new sample to history
        logger.debug(f"Adding to history: {iteration.index}")
        self._history.add_sample(
            SimulationHistorySample(iteration, ego_state, trajectory, observation, traffic_light_status)
        )

        # Propagate state to next iteration
        next_iteration = self._time_controller.next_iteration()

        # Propagate state
        if next_iteration:
            self._ego_controller.update_state(iteration, next_iteration, ego_state, trajectory)
            self._observations.update_observation(iteration, next_iteration, self._history_buffer)
        else:
            self._is_simulation_running = False

        # Append new state into history buffer
        self._history_buffer.append(self._ego_controller.get_state(), self._observations.get_observation())

        # End the simulation once the ego reaches the longitudinal end of the mapped road. The check
        # is on the just-appended NEXT state (not yet added to self._history via add_sample): when it
        # is at/over the map edge we stop before it is recorded, so the recorded history's last frame
        # is the prior on-road state and drivable_area_compliance stays 1.
        if self._end_at_map_edge and self._is_simulation_running:
            next_ego_state, _ = self._history_buffer.current_state
            if self._reached_drivable_end(next_ego_state):
                logger.info(
                    "Ending simulation early at iteration %d: ego reached the longitudinal end of the "
                    "mapped road (scenario token %s).",
                    iteration.index,
                    self._scenario.token,
                )
                self._is_simulation_running = False

    def _reached_drivable_end(self, ego_state: EgoState) -> bool:
        """
        Whether the drivable road ends longitudinally ahead of the ego within self._end_lookahead_m.

        This detects the ego running out of mapped road along its direction of travel (e.g. an
        alternative route continuing past the cropped nuPlan map edge) — NOT lateral off-road drift.
        If the ego is not contained in any lane/lane-connector it is treated as lateral off-road and
        this returns False (out of scope). Otherwise the forward lane graph is walked in travel
        direction, accumulating baseline lengths capped at the lookahead: if any path reaches the
        lookahead the road continues (False); if every path terminates (an edge with no outgoing
        edges) below the lookahead the road ends ahead (True). Internal junctions expose outgoing
        lane-connectors, so only the true map-crop boundary yields empty outgoing edges.

        :param ego_state: current ego state.
        :return: True if the drivable road dead-ends ahead within the lookahead, else False.
        """
        map_api = self._scenario.map_api
        ego_point = ego_state.rear_axle.point
        ego_heading = ego_state.rear_axle.heading

        # Find the containing lane/lane-connector whose baseline direction best matches the ego
        # heading, so the forward walk follows the ego's travel direction (not the oncoming lane).
        candidates = map_api.get_all_map_objects(ego_point, SemanticMapLayer.LANE) + map_api.get_all_map_objects(
            ego_point, SemanticMapLayer.LANE_CONNECTOR
        )
        current_edge = None
        best_heading_error = None
        for edge in candidates:
            if not edge.contains_point(ego_point):
                continue
            discrete_path = edge.baseline_path.discrete_path
            nearest = min(discrete_path, key=lambda s: (s.x - ego_point.x) ** 2 + (s.y - ego_point.y) ** 2)
            heading_error = abs(_normalize_angle(nearest.heading - ego_heading))
            if best_heading_error is None or heading_error < best_heading_error:
                best_heading_error = heading_error
                current_edge = edge

        # Lateral off-road (no containing lane): explicitly out of scope, keep the simulation running.
        if current_edge is None:
            return False

        # Remaining drivable length ahead on the current edge from the ego's along-track station.
        baseline = current_edge.baseline_path
        station = baseline.get_nearest_arc_length_from_position(ego_point)
        remaining_ahead = baseline.length - station

        # BFS forward over the lane graph, capped at the lookahead. Reaching the lookahead on any
        # path means the road continues; if the frontier empties first, every path dead-ended.
        frontier = deque([(current_edge, remaining_ahead)])
        visited = {current_edge.id}
        reached_dead_end = False
        while frontier:
            edge, dist_ahead = frontier.popleft()
            if dist_ahead >= self._end_lookahead_m:
                return False
            outgoing_edges = edge.outgoing_edges
            if not outgoing_edges:
                reached_dead_end = True
                continue
            for next_edge in outgoing_edges:
                if next_edge.id in visited:
                    continue
                visited.add(next_edge.id)
                frontier.append((next_edge, dist_ahead + next_edge.baseline_path.length))
        return reached_dead_end

    @property
    def scenario(self) -> AbstractScenario:
        """
        :return: used scenario in this simulation.
        """
        return self._scenario

    @property
    def setup(self) -> SimulationSetup:
        """
        :return: Setup for this simulation.
        """
        return self._setup

    @property
    def callback(self) -> AbstractCallback:
        """
        :return: Callback for this simulation.
        """
        return self._callback

    @property
    def history(self) -> SimulationHistory:
        """
        :return History from the simulation.
        """
        return self._history

    @property
    def history_buffer(self) -> SimulationHistoryBuffer:
        """
        :return SimulationHistoryBuffer from the simulation.
        """
        if self._history_buffer is None:
            raise RuntimeError(
                "_history_buffer is None. Please initialize the buffer by calling Simulation.initialize()"
            )
        return self._history_buffer
