"""Pure unit tests for the alternative-route ego-progress helpers.

Fakes only (no map data), mirroring the style of
``nuplan-devkit/nuplan/planning/simulation/test/test_end_at_map_edge.py``. Covers:
- ``get_route_baseline_roadblock_linkedlist_from_ids`` (route_extractor): build the progress
  linked-list straight from an ordered roadblock/connector id chain.
- ``alt_reference_progress_m`` (ego_progress_along_expert_route): the synthetic reference formula.
"""

import unittest

from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.metrics.evaluation_metrics.common.ego_progress_along_expert_route import (
    ALT_GOAL_PROGRESS_FRACTION,
    ALT_REFERENCE_WINDOW_S,
    alt_reference_progress_m,
)
from nuplan.planning.metrics.utils.route_extractor import (
    get_route_baseline_roadblock_linkedlist_from_ids,
)


class _FakeBaseline:
    """Identity stand-in for a lane baseline (PolylineMapObject)."""


class _FakeEdge:
    def __init__(self) -> None:
        self.baseline_path = _FakeBaseline()


class _FakeRoadblock:
    def __init__(self, has_edges: bool = True) -> None:
        self.interior_edges = [_FakeEdge()] if has_edges else []


class _FakeMapApi:
    """Resolves given roadblock ids on ROADBLOCK and connector ids on ROADBLOCK_CONNECTOR."""

    def __init__(self, roadblocks=None, connectors=None) -> None:
        self._by_layer = {
            SemanticMapLayer.ROADBLOCK: roadblocks or {},
            SemanticMapLayer.ROADBLOCK_CONNECTOR: connectors or {},
        }

    def get_map_object(self, rid, layer):
        return self._by_layer.get(layer, {}).get(rid)


def _ids(linked_list):
    """Walk the linked list, returning (road_block, base_line) identity pairs in order."""
    out = []
    node = linked_list.head
    while node is not None:
        out.append((node.road_block, node.base_line))
        node = node.next
    return out


class TestLinkedListFromIds(unittest.TestCase):
    def test_builds_ordered_pairs_including_connectors(self) -> None:
        rb0, rb1 = _FakeRoadblock(), _FakeRoadblock()
        conn0 = _FakeRoadblock()
        map_api = _FakeMapApi(roadblocks={"rb0": rb0, "rb1": rb1}, connectors={"conn0": conn0})
        ll = get_route_baseline_roadblock_linkedlist_from_ids(map_api, ["rb0", "conn0", "rb1"])
        pairs = _ids(ll)
        self.assertEqual([p[0] for p in pairs], [rb0, conn0, rb1])  # order + connectors included
        # base_line is the roadblock's first interior edge baseline
        self.assertEqual(pairs[0][1], rb0.interior_edges[0].baseline_path)

    def test_unresolved_ids_are_skipped(self) -> None:
        rb0 = _FakeRoadblock()
        map_api = _FakeMapApi(roadblocks={"rb0": rb0})
        ll = get_route_baseline_roadblock_linkedlist_from_ids(map_api, ["missing", "rb0", "also_missing"])
        self.assertEqual([p[0] for p in _ids(ll)], [rb0])

    def test_roadblock_without_interior_edges_skipped(self) -> None:
        rb0 = _FakeRoadblock(has_edges=False)
        map_api = _FakeMapApi(roadblocks={"rb0": rb0})
        ll = get_route_baseline_roadblock_linkedlist_from_ids(map_api, ["rb0"])
        self.assertIsNone(ll.head)

    def test_empty_chain_is_empty_list(self) -> None:
        self.assertIsNone(get_route_baseline_roadblock_linkedlist_from_ids(_FakeMapApi(), []).head)


class TestAltReferenceProgress(unittest.TestCase):
    def test_full_window_is_half_the_goal(self) -> None:
        self.assertAlmostEqual(alt_reference_progress_m(670.54, 60.0), 335.27)

    def test_short_window_scales_linearly(self) -> None:
        # 15 s -> 0.5 * 15/60 = 0.125 of the goal distance
        self.assertAlmostEqual(alt_reference_progress_m(670.54, 15.0), 83.8175)

    def test_linear_in_window(self) -> None:
        self.assertAlmostEqual(alt_reference_progress_m(100.0, 30.0), 25.0)
        self.assertAlmostEqual(alt_reference_progress_m(100.0, 60.0), 50.0)

    def test_constants(self) -> None:
        self.assertEqual(ALT_GOAL_PROGRESS_FRACTION, 0.5)
        self.assertEqual(ALT_REFERENCE_WINDOW_S, 60.0)


if __name__ == "__main__":
    unittest.main()
