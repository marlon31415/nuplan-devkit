import math
import pickle
import types
import unittest

from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario
from nuplan.planning.simulation.callback.multi_callback import MultiCallback
from nuplan.planning.simulation.controller.perfect_tracking import PerfectTrackingController
from nuplan.planning.simulation.observation.tracks_observation import TracksObservation
from nuplan.planning.simulation.simulation import (
    END_AT_MAP_EDGE_LOOKAHEAD_M,
    Simulation,
    _normalize_angle,
)
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.simulation.simulation_time_controller.step_simulation_time_controller import (
    StepSimulationTimeController,
)


class _FakePoint:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


class _FakeStateSE2:
    def __init__(self, x: float, y: float, heading: float) -> None:
        self.x = x
        self.y = y
        self.heading = heading


class _FakeBaseline:
    """Minimal PolylineMapObject stand-in for the drivable-end detector."""

    def __init__(self, length: float, heading: float) -> None:
        self.length = length
        # Two-vertex straight baseline pointing along `heading`.
        self.discrete_path = [_FakeStateSE2(0.0, 0.0, heading), _FakeStateSE2(length, 0.0, heading)]

    def get_nearest_arc_length_from_position(self, point: _FakePoint) -> float:
        return 0.0  # ego sits at the start of its current edge -> whole edge is ahead


class _FakeEdge:
    def __init__(self, edge_id: str, length: float, heading: float = 0.0, contains: bool = False) -> None:
        self.id = edge_id
        self.baseline_path = _FakeBaseline(length, heading)
        self.outgoing_edges = []
        self._contains = contains

    def contains_point(self, point: _FakePoint) -> bool:
        return self._contains


class _FakeMapApi:
    def __init__(self, lanes=None, lane_connectors=None) -> None:
        self._by_layer = {
            SemanticMapLayer.LANE: lanes or [],
            SemanticMapLayer.LANE_CONNECTOR: lane_connectors or [],
        }

    def get_all_map_objects(self, point, layer):
        return list(self._by_layer.get(layer, []))


def _ego(heading: float = 0.0):
    rear_axle = types.SimpleNamespace(point=_FakePoint(0.0, 0.0), heading=heading)
    return types.SimpleNamespace(rear_axle=rear_axle)


def _detector_self(map_api, lookahead: float = 10.0):
    """A minimal stand-in exposing only what Simulation._reached_drivable_end reads."""
    return types.SimpleNamespace(
        _scenario=types.SimpleNamespace(map_api=map_api),
        _end_lookahead_m=lookahead,
    )


class TestReachedDrivableEnd(unittest.TestCase):
    """Unit tests for Simulation._reached_drivable_end (pure map/graph predicate)."""

    def _call(self, map_api, ego, lookahead: float = 10.0) -> bool:
        return Simulation._reached_drivable_end(_detector_self(map_api, lookahead), ego)

    def test_dead_end_within_lookahead_is_true(self) -> None:
        """current(5) -> outgoing(3, no further) totals 8 < 10 -> road ends ahead."""
        cur = _FakeEdge("cur", length=5.0, contains=True)
        tail = _FakeEdge("tail", length=3.0)
        cur.outgoing_edges = [tail]
        self.assertTrue(self._call(_FakeMapApi(lanes=[cur]), _ego()))

    def test_road_continues_is_false(self) -> None:
        """current(5) -> outgoing(8) reaches 13 >= 10 -> road continues."""
        cur = _FakeEdge("cur", length=5.0, contains=True)
        nxt = _FakeEdge("nxt", length=8.0)
        cur.outgoing_edges = [nxt]
        self.assertFalse(self._call(_FakeMapApi(lanes=[cur]), _ego()))

    def test_branch_with_one_long_path_is_false(self) -> None:
        """A junction: one branch dead-ends short, another reaches the lookahead -> continues."""
        cur = _FakeEdge("cur", length=2.0, contains=True)
        short = _FakeEdge("short", length=3.0)  # dead-end at 5
        long = _FakeEdge("long", length=9.0)  # reaches 11 >= 10
        cur.outgoing_edges = [short, long]
        self.assertFalse(self._call(_FakeMapApi(lanes=[cur]), _ego()))

    def test_lateral_off_road_is_false(self) -> None:
        """No containing lane -> lateral off-road, explicitly out of scope."""
        self.assertFalse(self._call(_FakeMapApi(lanes=[]), _ego()))

    def test_picks_heading_aligned_lane(self) -> None:
        """Two containing lanes; the oncoming one dead-ends, the aligned one continues.

        Correct (heading-aligned) selection must yield False (road continues).
        """
        aligned = _FakeEdge("aligned", length=6.0, heading=0.0, contains=True)
        aligned.outgoing_edges = [_FakeEdge("aligned_next", length=6.0)]  # 12 >= 10
        oncoming = _FakeEdge("oncoming", length=3.0, heading=math.pi, contains=True)  # dead-end at 3
        self.assertFalse(self._call(_FakeMapApi(lanes=[aligned, oncoming]), _ego(heading=0.0)))

    def test_lane_connector_used_when_no_lane(self) -> None:
        """The detector also considers LANE_CONNECTOR objects (junctions)."""
        conn = _FakeEdge("conn", length=4.0, contains=True)
        conn.outgoing_edges = [_FakeEdge("conn_tail", length=2.0)]  # 6 < 10 -> dead-end
        self.assertTrue(self._call(_FakeMapApi(lane_connectors=[conn]), _ego()))


class TestEndAtMapEdgePickling(unittest.TestCase):
    """The end_at_map_edge flag must survive pickling to Ray workers (via __reduce__)."""

    def _build(self, end_at_map_edge: bool) -> Simulation:
        scenario = MockAbstractScenario(number_of_past_iterations=10)
        setup = SimulationSetup(
            time_controller=StepSimulationTimeController(scenario),
            observations=TracksObservation(scenario),
            ego_controller=PerfectTrackingController(scenario),
            scenario=scenario,
        )
        return Simulation(
            simulation_setup=setup,
            callback=MultiCallback([]),
            simulation_history_buffer_duration=2,
            end_at_map_edge=end_at_map_edge,
        )

    def test_default_is_off(self) -> None:
        """Default keeps the flag off so non-alt runs are unaffected."""
        self.assertFalse(self._build(False)._end_at_map_edge)

    def test_reduce_carries_flag(self) -> None:
        """__reduce__ must include the flag as the 4th constructor arg."""
        sim = self._build(True)
        _, args = sim.__reduce__()
        self.assertEqual(len(args), 4)
        self.assertIs(args[3], True)

    def test_pickle_roundtrip_preserves_flag(self) -> None:
        """A pickled/unpickled Simulation keeps end_at_map_edge (Ray-worker propagation)."""
        restored = pickle.loads(pickle.dumps(self._build(True)))
        self.assertTrue(restored._end_at_map_edge)
        self.assertEqual(restored._end_lookahead_m, END_AT_MAP_EDGE_LOOKAHEAD_M)


class TestNormalizeAngle(unittest.TestCase):
    def test_wraps_to_pi_interval(self) -> None:
        self.assertAlmostEqual(_normalize_angle(1.5 * math.pi), -0.5 * math.pi)
        self.assertAlmostEqual(_normalize_angle(-1.5 * math.pi), 0.5 * math.pi)
        self.assertAlmostEqual(_normalize_angle(0.25 * math.pi), 0.25 * math.pi)


if __name__ == "__main__":
    unittest.main()
