from typing import List, Optional, Set

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase
from nuplan.planning.metrics.metric_result import (
    MetricStatistics,
    MetricStatisticsType,
    Statistic,
    TimeSeries,
)
from nuplan.planning.metrics.utils.state_extractors import (
    extract_ego_center,
    extract_ego_time_point,
)
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory


class NavigationComplianceStatistics(MetricBase):
    """
    Navigation compliance metric.
    Checks whether the ego vehicle's center is within a lane or lane connector
    belonging to a route roadblock at each timestep.
    The final metric score is binary (0 or 1) based on the last timestep only.
    """

    def __init__(
        self,
        name: str,
        category: str,
        metric_score_unit: Optional[str] = None,
    ) -> None:
        """
        Initialize the NavigationComplianceStatistics class.
        :param name: Metric name.
        :param category: Metric category.
        :param metric_score_unit: Metric final score unit.
        """
        super().__init__(
            name=name, category=category, metric_score_unit=metric_score_unit
        )

    @staticmethod
    def _is_on_route(
        pose: Point2D, map_api: AbstractMap, route_roadblock_ids: Set[str]
    ) -> bool:
        """
        Check if a pose is within a lane or lane connector belonging to a route roadblock.
        :param pose: ego center position.
        :param map_api: map API.
        :param route_roadblock_ids: set of route roadblock/roadblock-connector IDs.
        :return: True if the pose is on-route.
        """
        # Check lane first (most common case outside intersections)
        lane = map_api.get_one_map_object(pose, SemanticMapLayer.LANE)
        if lane is not None and isinstance(lane, LaneGraphEdgeMapObject):
            return lane.get_roadblock_id() in route_roadblock_ids

        # At intersections, ego may be in a lane connector instead
        lane_connectors = map_api.get_all_map_objects(
            pose, SemanticMapLayer.LANE_CONNECTOR
        )
        for connector in lane_connectors:
            if (
                isinstance(connector, LaneGraphEdgeMapObject)
                and connector.get_roadblock_id() in route_roadblock_ids
            ):
                return True

        return False

    def compute_score(
        self,
        scenario: AbstractScenario,
        metric_statistics: List[Statistic],
        time_series: Optional[TimeSeries] = None,
    ) -> float:
        """Inherited, see superclass."""
        return float(metric_statistics[0].value)

    def compute(
        self, history: SimulationHistory, scenario: AbstractScenario
    ) -> List[MetricStatistics]:
        """
        Return the navigation compliance metric.
        :param history: History from a simulation engine.
        :param scenario: Scenario running this metric.
        :return: navigation compliance statistics.
        """
        ego_states = history.extract_ego_state
        ego_poses = extract_ego_center(ego_states)
        ego_timestamps = extract_ego_time_point(ego_states)
        map_api = history.map_api
        route_roadblock_ids = set(scenario.get_route_roadblock_ids())

        # Compute on-route status at each timestep
        on_route_per_timestep = [
            float(self._is_on_route(pose, map_api, route_roadblock_ids))
            for pose in ego_poses
        ]

        # Final score is based on last timestep only (binary, PDM-style)
        on_route_at_end = on_route_per_timestep[-1] if on_route_per_timestep else 0.0

        statistics = [
            Statistic(
                name=f"{self.name}",
                unit=MetricStatisticsType.BOOLEAN.unit,
                value=on_route_at_end,
                type=MetricStatisticsType.BOOLEAN,
            ),
        ]

        time_series = TimeSeries(
            unit="boolean",
            time_stamps=list(ego_timestamps),
            values=on_route_per_timestep,
        )

        results = self._construct_metric_results(
            metric_statistics=statistics,
            time_series=time_series,
            scenario=scenario,
            metric_score_unit=self._metric_score_unit,
        )

        return results  # type: ignore
