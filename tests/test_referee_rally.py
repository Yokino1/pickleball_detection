import json
import unittest

from src.court import CourtLayout, ProjectionResult
from src.referee import (
    REFEREE_CONTRACT_VERSION,
    REFEREE_SCHEMA_VERSION,
    RallyRefereeEngine,
    RefereeFrameResult,
    ScoreRecorder,
    build_referee_engine,
)


def projection(
    y_ft=None,
    *,
    x_ft=10.0,
    observed=True,
    valid=True,
    inside=True,
    event=None,
    active_side="left",
):
    track_status = "observed" if observed else "predicted"
    court_xy = (
        [float(x_ft), float(y_ft)]
        if valid and y_ft is not None
        else None
    )
    return ProjectionResult(
        coordinate_system="pickleball_full_court_ft",
        coordinate_system_version=1,
        active_side=active_side,
        calibration_id="test_calibration",
        calibration_source="manual_test",
        image_xy=[100.0, 100.0] if valid else None,
        ball_court_xy=court_xy,
        projection_status="homography" if valid else "unavailable",
        projection_valid=valid,
        homography_available=True,
        reprojection_error_px=0.0 if valid else None,
        track_status=track_status,
        observed=observed,
        predicted=not observed,
        inside_court=inside if valid else None,
        event=event,
    )


def bounce_event(
    y_ft,
    *,
    frame_index,
    inside=True,
    extra_events=(),
    contact_timestamp_s=None,
):
    result = {
        "events": ["bounce_candidate", *extra_events],
        "contact_frame_index": int(frame_index),
        "contact_court_xy": [10.0, float(y_ft)],
        "contact_inside_court": inside,
        "evidence": ["test_bounce"],
    }
    if contact_timestamp_s is not None:
        result["contact_timestamp_s"] = float(contact_timestamp_s)
    return result


def referee(**overrides):
    values = {
        "net_deadband_ft": 0.75,
        "route_confirm_observed_frames": 3,
        "route_min_netward_displacement_ft": 1.5,
        "net_cross_confirm_observed_frames": 2,
        "net_cross_min_displacement_ft": 0.15,
        "max_observation_gap_ms": 120.0,
        "unavailable_timeout_ms": 200.0,
        "post_rally_guard_ms": 100.0,
        "require_serve_arming": False,
    }
    values.update(overrides)
    return RallyRefereeEngine(CourtLayout(), **values)


class Sequence:
    def __init__(self, engine):
        self.engine = engine
        self.frame_index = -1
        self.timestamp_s = -0.02

    def step(
        self,
        y_ft=None,
        *,
        x_ft=10.0,
        observed=True,
        valid=True,
        inside=True,
        event=None,
        active_side="left",
        timestamp_s=None,
        discontinuity_reason=None,
        eligible_player_centers_court_xy=(),
    ):
        self.frame_index += 1
        if timestamp_s is None:
            self.timestamp_s += 0.02
        else:
            self.timestamp_s = float(timestamp_s)
        return self.engine.update(
            projection(
                y_ft,
                x_ft=x_ft,
                observed=observed,
                valid=valid,
                inside=inside,
                event=event,
                active_side=active_side,
            ),
            timestamp_s=self.timestamp_s,
            frame_index=self.frame_index,
            discontinuity_reason=discontinuity_reason,
            eligible_player_centers_court_xy=(
                eligible_player_centers_court_xy
            ),
        )

    def start_low_y_serve(self):
        results = [
            self.step(8.0, active_side="right"),
            self.step(10.0, active_side="right"),
            self.step(12.0, active_side="right"),
        ]
        return results[-1]

    def cross_to_high_y(self):
        self.step(24.0)
        return self.step(25.0)


class RallyRefereeEngineTest(unittest.TestCase):
    def test_contract_and_schema_versions_are_independent_fields(self):
        record = RefereeFrameResult(
            phase="WAIT_SERVE",
            frame_index=0,
            timestamp_s=0.0,
            rally_id=None,
            physical_half=None,
            physical_side=None,
            server_half=None,
            server_side=None,
            route_origin_half=None,
            route_origin_side=None,
            route_destination_half=None,
            route_destination_side=None,
            leg_index=0,
            net_crossed=False,
            first_bounce_half=None,
            score={"left": 0, "right": 0},
            contract_version=7,
            schema_version=9,
        ).to_dict()

        self.assertEqual(REFEREE_CONTRACT_VERSION, 1)
        self.assertEqual(REFEREE_SCHEMA_VERSION, 1)
        self.assertEqual(record["contract_version"], 7)
        self.assertEqual(record["schema_version"], 9)

    def test_production_default_ignores_clip_starting_mid_motion(self):
        sequence = Sequence(
            RallyRefereeEngine(
                CourtLayout(),
                route_min_netward_displacement_ft=1.5,
            )
        )

        for y_ft in (6.0, 8.0, 10.0, 12.0, 14.0, 16.0):
            result = sequence.step(y_ft)

        self.assertEqual(result.phase, "WAIT_SERVE")
        self.assertIsNone(result.rally_id)
        self.assertFalse(sequence.engine.diagnostics()["serve_armed"])
        self.assertEqual(sequence.engine.diagnostics()["rallies_started"], 0)

    def test_production_default_arms_after_stable_ball_then_serves(self):
        sequence = Sequence(
            RallyRefereeEngine(
                CourtLayout(),
                route_min_netward_displacement_ft=1.5,
                post_rally_rearm_stable_observed_frames=3,
                post_rally_rearm_stable_displacement_ft=0.25,
            )
        )

        sequence.step(8.0, x_ft=10.0)
        sequence.step(8.05, x_ft=10.05)
        armed = sequence.step(8.0, x_ft=10.0)

        self.assertEqual(armed.phase, "WAIT_SERVE")
        self.assertIn("serve_armed", armed.events)
        self.assertTrue(sequence.engine.diagnostics()["serve_armed"])

        sequence.step(8.0)
        sequence.step(10.0)
        started = sequence.step(12.0)
        self.assertIn("serve_confirmed", started.events)
        self.assertEqual(started.phase, "IN_RALLY")

    def test_production_default_arms_after_sustained_missing(self):
        sequence = Sequence(
            RallyRefereeEngine(
                CourtLayout(),
                route_min_netward_displacement_ft=1.5,
                post_rally_rearm_missing_ms=200.0,
            )
        )

        sequence.step(None, valid=False, timestamp_s=0.0)
        sequence.step(None, valid=False, timestamp_s=0.1)
        armed = sequence.step(None, valid=False, timestamp_s=0.21)

        self.assertIn("serve_armed", armed.events)
        self.assertEqual(
            sequence.engine.diagnostics()["serve_arming_missing_count"],
            1,
        )

        sequence.step(8.0)
        sequence.step(10.0)
        started = sequence.step(12.0)
        self.assertIn("serve_confirmed", started.events)

    def test_post_rally_guard_rearm_directly_arms_next_serve(self):
        sequence = Sequence(
            RallyRefereeEngine(
                CourtLayout(),
                route_min_netward_displacement_ft=1.5,
                post_rally_rearm_stable_observed_frames=3,
                post_rally_rearm_stable_displacement_ft=0.25,
                post_rally_guard_ms=100.0,
                post_rally_rearm_missing_ms=50.0,
            )
        )
        for y_ft in (8.0, 8.05, 8.0):
            sequence.step(y_ft)
        sequence.step(8.0)
        sequence.step(10.0)
        sequence.step(12.0)
        event = bounce_event(19.0, frame_index=sequence.frame_index + 1)
        terminal = sequence.step(19.0, event=event)

        ready = sequence.step(
            None,
            valid=False,
            timestamp_s=terminal.timestamp_s + 0.2,
        )

        self.assertEqual(ready.phase, "WAIT_SERVE")
        self.assertIn("ready_for_next_serve", ready.events)
        self.assertTrue(sequence.engine.diagnostics()["serve_armed"])

    def test_observed_netward_motion_confirms_serve(self):
        sequence = Sequence(referee())

        result = sequence.start_low_y_serve()

        self.assertEqual(result.phase, "IN_RALLY")
        self.assertEqual(result.events, ["serve_confirmed"])
        self.assertEqual(result.rally_id, 1)
        self.assertEqual(result.server_half, "low_y")
        self.assertEqual(result.server_side, "left")
        self.assertEqual(result.route_origin_half, "low_y")
        self.assertEqual(result.leg_index, 1)
        self.assertEqual(result.rally_over, "N")
        self.assertEqual(result.target_ball_state_0811, "target_ball_tracked")
        self.assertEqual(len(result.hit_records), 1)
        self.assertEqual(result.hit_records[0].hit_count, 1)
        self.assertEqual(result.hit_records[0].hit_side, "left")

    def test_predicted_positions_do_not_confirm_serve(self):
        engine = referee()
        sequence = Sequence(engine)

        for y_ft in (8.0, 10.0, 12.0, 18.0):
            result = sequence.step(y_ft, observed=False)

        self.assertEqual(result.phase, "WAIT_SERVE")
        self.assertIsNone(result.rally_id)
        self.assertEqual(
            result.target_ball_state_0811,
            "target_ball_predicted",
        )
        self.assertEqual(engine.diagnostics()["rallies_started"], 0)

    def test_net_crossing_requires_hysteresis_and_direction(self):
        sequence = Sequence(referee())
        sequence.start_low_y_serve()

        deadband = sequence.step(22.1)
        first_safe_sample = sequence.step(22.9)
        wrong_direction = sequence.step(22.8)
        confirmed = sequence.step(23.2)

        self.assertFalse(deadband.net_crossed)
        self.assertFalse(first_safe_sample.net_crossed)
        self.assertFalse(wrong_direction.net_crossed)
        self.assertTrue(confirmed.net_crossed)
        self.assertIn("net_crossing_confirmed", confirmed.events)

    def test_same_side_landing_without_crossing_is_serve_net(self):
        sequence = Sequence(referee())
        sequence.start_low_y_serve()
        event = bounce_event(
            19.0,
            frame_index=sequence.frame_index + 1,
        )

        result = sequence.step(19.0, event=event)
        rally_result = result.rally_result

        self.assertIsNotNone(rally_result)
        self.assertEqual(rally_result.terminal_event, "serve_net")
        self.assertEqual(rally_result.fault_half, "low_y")
        self.assertEqual(rally_result.fault_side, "left")
        self.assertEqual(rally_result.rally_winner, "right")
        self.assertEqual(rally_result.point_awarded_to, "right")
        self.assertEqual(result.score, {"left": 0, "right": 1})
        self.assertEqual(result.event_type_0811, "off_net")
        self.assertEqual(result.rally_over, "Y")
        self.assertFalse(result.manual_confirmation_required)
        self.assertEqual(rally_result.event_type_0811, "off_net")
        self.assertEqual(rally_result.rally_over, "Y")
        self.assertEqual(rally_result.hit_records[0].first_landing_inout, "IN")

    def test_expected_server_side_is_diagnostic_only(self):
        sequence = Sequence(referee(expected_server_side="right"))

        started = sequence.start_low_y_serve()

        self.assertEqual(started.phase, "IN_RALLY")
        self.assertEqual(started.server_side, "left")
        self.assertEqual(started.expected_server_side, "right")
        self.assertFalse(started.server_side_match)
        self.assertIn("expected_server_side_mismatch", started.warnings)
        self.assertEqual(
            sequence.engine.score_recorder.score,
            {"left": 0, "right": 0},
        )

        sequence.engine.set_expected_server_side("left")
        self.assertTrue(sequence.engine.diagnostics()["server_side_match"])

    def test_first_confirmed_serve_initializes_service_owner(self):
        sequence = Sequence(
            referee(service_ownership_enabled=True)
        )

        started = sequence.start_low_y_serve()

        self.assertEqual(started.phase, "IN_RALLY")
        self.assertEqual(started.service_owner_half, "low_y")
        self.assertEqual(started.service_owner_side, "left")
        self.assertIn("service_owner_initialized", started.events)
        self.assertIn("service_owner_side_match", started.evidence)
        diagnostics = sequence.engine.diagnostics()
        self.assertEqual(diagnostics["service_owner_side"], "left")
        self.assertEqual(
            diagnostics["service_owner_source"],
            "first_serve_detected",
        )

    def test_serve_rejects_player_box_center_inside_court(self):
        sequence = Sequence(
            referee(require_server_player_center_outside_court=True)
        )

        sequence.step(
            8.0,
            eligible_player_centers_court_xy=((10.0, 5.0),),
        )
        sequence.step(
            10.0,
            eligible_player_centers_court_xy=((10.0, 5.0),),
        )
        rejected = sequence.step(
            12.0,
            eligible_player_centers_court_xy=((10.0, 5.0),),
        )

        self.assertEqual(rejected.phase, "WAIT_SERVE")
        self.assertIsNone(rejected.rally_id)
        self.assertFalse(rejected.eligible_player_center_outside_court)
        self.assertIn(
            "serve_rejected_player_center_inside_court",
            rejected.events,
        )
        self.assertIn(
            "eligible_server_player_center_not_outside_court",
            rejected.warnings,
        )
        self.assertEqual(
            sequence.engine.diagnostics()["server_position_rejections"],
            1,
        )

    def test_serve_accepts_and_records_player_box_center_outside_court(self):
        sequence = Sequence(
            referee(require_server_player_center_outside_court=True)
        )
        outside_center = (-0.5, 5.0)

        sequence.step(
            8.0,
            eligible_player_centers_court_xy=(outside_center,),
        )
        sequence.step(
            10.0,
            eligible_player_centers_court_xy=(outside_center,),
        )
        accepted = sequence.step(
            12.0,
            eligible_player_centers_court_xy=(outside_center,),
        )

        self.assertEqual(accepted.phase, "IN_RALLY")
        self.assertEqual(accepted.rally_id, 1)
        self.assertTrue(accepted.eligible_player_center_outside_court)
        self.assertIn("serve_confirmed", accepted.events)
        self.assertIn(
            "eligible_server_player_center_outside_court",
            accepted.evidence,
        )

        terminal = sequence.step(
            19.0,
            event=bounce_event(
                19.0,
                frame_index=sequence.frame_index + 1,
            ),
        )
        self.assertEqual(
            terminal.rally_result.server_player_centers_court_xy,
            (outside_center,),
        )
        self.assertTrue(
            terminal.rally_result.server_player_center_outside_court
        )

    def test_rally_winner_owns_and_gates_next_serve(self):
        sequence = Sequence(
            referee(
                service_ownership_enabled=True,
                post_rally_rearm_missing_ms=200.0,
            )
        )
        sequence.start_low_y_serve()
        terminal = sequence.step(
            19.0,
            event=bounce_event(
                19.0,
                frame_index=sequence.frame_index + 1,
            ),
        )

        self.assertEqual(terminal.rally_result.rally_winner, "right")
        self.assertEqual(terminal.rally_result.next_server_half, "high_y")
        self.assertEqual(terminal.rally_result.next_server_side, "right")
        self.assertEqual(terminal.service_owner_side, "right")

        ready = sequence.step(
            None,
            valid=False,
            timestamp_s=terminal.timestamp_s + 0.25,
        )
        self.assertEqual(ready.phase, "WAIT_SERVE")
        self.assertEqual(ready.service_owner_side, "right")

        rejected = sequence.start_low_y_serve()
        self.assertEqual(rejected.phase, "WAIT_SERVE")
        self.assertIsNone(rejected.rally_id)
        self.assertIn(
            "serve_rejected_wrong_service_owner",
            rejected.events,
        )

        sequence.step(36.0)
        sequence.step(34.0)
        accepted = sequence.step(32.0)
        self.assertEqual(accepted.phase, "IN_RALLY")
        self.assertEqual(accepted.rally_id, 2)
        self.assertEqual(accepted.server_half, "high_y")
        self.assertEqual(accepted.server_side, "right")
        self.assertEqual(accepted.service_owner_side, "right")

    def test_unknown_result_retains_service_owner(self):
        sequence = Sequence(
            referee(service_ownership_enabled=True)
        )
        started = sequence.start_low_y_serve()
        terminal = sequence.step(
            None,
            valid=False,
            timestamp_s=started.timestamp_s + 0.25,
        )

        self.assertEqual(terminal.rally_result.status, "unresolved")
        self.assertIsNone(terminal.rally_result.rally_winner)
        self.assertEqual(terminal.rally_result.next_server_side, "left")
        self.assertEqual(terminal.service_owner_side, "left")

    def test_terminal_decision_latency_uses_contact_timestamp(self):
        sequence = Sequence(referee())
        sequence.start_low_y_serve()
        decision_time_s = sequence.timestamp_s + 0.02
        contact_time_s = decision_time_s - 0.015
        event = bounce_event(
            19.0,
            frame_index=sequence.frame_index + 1,
            contact_timestamp_s=contact_time_s,
        )

        result = sequence.step(19.0, event=event, timestamp_s=decision_time_s)

        self.assertAlmostEqual(result.decision_latency_ms["serve_net"], 15.0)
        self.assertAlmostEqual(result.rally_result.decision_latency_ms, 15.0)
        stats = sequence.engine.diagnostics()["decision_latency_ms"]["serve_net"]
        self.assertEqual(stats["count"], 1)
        self.assertAlmostEqual(stats["average"], 15.0)
        self.assertAlmostEqual(stats["max"], 15.0)

    def test_direction_reversal_infers_return_without_hit_candidate(self):
        sequence = Sequence(referee())
        sequence.start_low_y_serve()
        crossed = sequence.cross_to_high_y()
        self.assertTrue(crossed.net_crossed)

        sequence.step(28.0)
        sequence.step(31.0)
        sequence.step(30.0)
        return_route = sequence.step(27.0)
        event = bounce_event(
            24.0,
            frame_index=sequence.frame_index + 1,
        )
        terminal = sequence.step(24.0, event=event)

        self.assertIn("return_route_confirmed", return_route.events)
        self.assertIn("volleyed_return", return_route.events)
        self.assertEqual(return_route.event_type_0811, "volleyed")
        self.assertEqual(return_route.hit_records[0].event_type_0811, "volleyed")
        self.assertNotIn(
            "recent_paddle_hit_candidate_support",
            return_route.evidence,
        )
        self.assertEqual(return_route.route_origin_half, "high_y")
        self.assertEqual(return_route.leg_index, 2)
        self.assertFalse(return_route.net_crossed)
        self.assertEqual(
            terminal.rally_result.terminal_event,
            "return_net",
        )
        self.assertEqual(terminal.rally_result.fault_side, "right")
        self.assertEqual(terminal.rally_result.rally_winner, "left")

    def test_out_of_bounds_precedes_simultaneous_second_bounce(self):
        sequence = Sequence(referee())
        sequence.start_low_y_serve()
        sequence.cross_to_high_y()
        first = bounce_event(
            30.0,
            frame_index=sequence.frame_index + 1,
        )
        sequence.step(30.0, event=first)
        second_out = bounce_event(
            45.0,
            frame_index=sequence.frame_index + 1,
            inside=False,
            extra_events=(
                "out_of_bounds_bounce_candidate",
                "second_bounce_candidate",
            ),
        )

        result = sequence.step(
            40.0,
            inside=False,
            event=second_out,
        )

        self.assertEqual(
            result.rally_result.terminal_event,
            "out_of_bounds",
        )
        self.assertEqual(result.rally_result.fault_side, "left")
        self.assertEqual(result.rally_result.rally_winner, "right")

    def test_second_inside_bounce_ends_rally(self):
        sequence = Sequence(referee())
        sequence.start_low_y_serve()
        sequence.cross_to_high_y()
        first = bounce_event(
            30.0,
            frame_index=sequence.frame_index + 1,
        )
        first_result = sequence.step(30.0, event=first)
        second = bounce_event(
            34.0,
            frame_index=sequence.frame_index + 1,
            extra_events=("second_bounce_candidate",),
        )

        result = sequence.step(34.0, event=second)

        self.assertIn("first_bounce_recorded", first_result.events)
        self.assertEqual(result.rally_result.terminal_event, "second_bounce")
        self.assertEqual(result.rally_result.fault_half, "high_y")
        self.assertEqual(result.rally_result.rally_winner, "left")
        self.assertEqual(result.score, {"left": 1, "right": 0})

    def test_confirmed_routes_produce_per_hit_records(self):
        sequence = Sequence(referee())
        sequence.start_low_y_serve()
        sequence.cross_to_high_y()
        first = bounce_event(
            30.0,
            frame_index=sequence.frame_index + 1,
            contact_timestamp_s=sequence.timestamp_s + 0.02,
        )
        first_bounce = sequence.step(30.0, event=first)
        sequence.step(31.0)
        sequence.step(30.0)
        returned = sequence.step(27.0)

        self.assertEqual(first_bounce.event_type_0811, "firstlanding_in")
        self.assertAlmostEqual(
            first_bounce.decision_latency_ms["first_bounce_recorded"],
            0.0,
        )
        self.assertEqual(len(returned.hit_records), 2)
        first_hit, second_hit = returned.hit_records
        self.assertEqual(first_hit.event_type_0811, "firstlanding_in")
        self.assertEqual(first_hit.first_landing_position, (10.0, 30.0))
        self.assertEqual(first_hit.first_landing_inout, "IN")
        self.assertIsNotNone(first_hit.hit_end_timestamp_s)
        self.assertEqual(second_hit.hit_count, 2)
        self.assertEqual(second_hit.hit_half, "high_y")
        self.assertEqual(second_hit.hit_side, "right")
        self.assertIsNone(second_hit.hit_end_timestamp_s)

        terminal = sequence.step(
            24.0,
            event=bounce_event(
                24.0,
                frame_index=sequence.frame_index + 1,
            ),
        )
        self.assertEqual(len(terminal.rally_result.hit_records), 2)
        self.assertEqual(
            terminal.rally_result.hit_records[-1].event_type_0811,
            "off_net",
        )

    def test_contact_coordinate_takes_precedence_over_current_ball(self):
        sequence = Sequence(referee())
        sequence.start_low_y_serve()
        event = bounce_event(
            19.0,
            frame_index=sequence.frame_index + 1,
        )

        result = sequence.step(24.0, event=event)

        self.assertEqual(result.physical_half, "high_y")
        self.assertEqual(result.rally_result.terminal_event, "serve_net")
        self.assertEqual(result.rally_result.fault_half, "low_y")
        self.assertIn(
            "contact_court_xy_used",
            result.rally_result.evidence,
        )

    def test_terminal_event_and_score_are_emitted_only_once(self):
        engine = referee(post_rally_guard_ms=100.0)
        sequence = Sequence(engine)
        sequence.start_low_y_serve()
        event_frame = sequence.frame_index + 1
        event = bounce_event(19.0, frame_index=event_frame)

        first = sequence.step(19.0, event=event)
        duplicate = sequence.step(19.0, event=event)
        after_guard = sequence.step(
            None,
            valid=False,
            timestamp_s=first.timestamp_s + 0.2,
        )

        self.assertIsNotNone(first.rally_result)
        self.assertIsNone(duplicate.rally_result)
        self.assertIsNone(after_guard.rally_result)
        self.assertEqual(first.score, {"left": 0, "right": 1})
        self.assertEqual(duplicate.score, first.score)
        self.assertEqual(after_guard.score, first.score)
        self.assertEqual(engine.diagnostics()["rallies_completed"], 1)

    def test_score_recorder_is_idempotent_by_rally_id(self):
        recorder = ScoreRecorder()

        first = recorder.award(
            7,
            rally_winner="right",
            server_side="left",
        )
        duplicate = recorder.award(
            7,
            rally_winner="left",
            server_side="left",
        )

        self.assertFalse(first.duplicate)
        self.assertTrue(duplicate.duplicate)
        self.assertEqual(duplicate.point_awarded_to, "right")
        self.assertEqual(recorder.score, {"left": 0, "right": 1})

    def test_score_recorder_accumulates_without_two_point_cap(self):
        recorder = ScoreRecorder()

        for rally_id in range(1, 8):
            recorder.award(
                rally_id,
                rally_winner="left",
                server_side="left" if rally_id % 2 else "right",
                status="demo_inferred",
            )

        self.assertEqual(recorder.score, {"left": 7, "right": 0})

    def test_guard_expiry_clears_previous_rally_context(self):
        sequence = Sequence(referee(post_rally_guard_ms=100.0))
        sequence.start_low_y_serve()
        event = bounce_event(
            19.0,
            frame_index=sequence.frame_index + 1,
        )
        terminal = sequence.step(19.0, event=event)

        ready = sequence.step(
            None,
            valid=False,
            timestamp_s=terminal.timestamp_s + 0.2,
        )

        self.assertEqual(ready.phase, "WAIT_SERVE")
        self.assertIn("ready_for_next_serve", ready.events)
        self.assertIsNone(ready.rally_id)
        self.assertIsNone(ready.server_half)
        self.assertIsNone(ready.server_side)
        self.assertIsNone(ready.route_origin_half)
        self.assertIsNone(ready.route_destination_half)
        self.assertEqual(ready.leg_index, 0)

    def test_guard_does_not_rearm_from_continuing_rebound_motion(self):
        sequence = Sequence(
            referee(
                post_rally_guard_ms=100.0,
                post_rally_rearm_missing_ms=50.0,
                post_rally_rearm_stable_observed_frames=3,
                post_rally_rearm_stable_displacement_ft=0.25,
            )
        )
        sequence.start_low_y_serve()
        event = bounce_event(
            19.0,
            frame_index=sequence.frame_index + 1,
        )
        terminal = sequence.step(19.0, event=event)

        for offset, y_ft in ((0.12, 18.0), (0.14, 16.0), (0.16, 14.0)):
            still_guarded = sequence.step(
                y_ft,
                timestamp_s=terminal.timestamp_s + offset,
            )

        self.assertEqual(still_guarded.phase, "POST_RALLY_GUARD")
        self.assertEqual(
            still_guarded.score,
            {"left": 0, "right": 1},
        )
        self.assertNotIn("ready_for_next_serve", still_guarded.events)

        rearmed = sequence.step(
            None,
            valid=False,
            timestamp_s=terminal.timestamp_s + 0.4,
        )
        self.assertEqual(rearmed.phase, "WAIT_SERVE")
        self.assertIn("ready_for_next_serve", rearmed.events)

    def test_ambiguous_bounce_stays_pending_then_times_out_unknown(self):
        sequence = Sequence(
            referee(
                unavailable_timeout_ms=200.0,
                pending_bounce_timeout_ms=200.0,
            )
        )
        sequence.start_low_y_serve()
        event = bounce_event(
            22.1,
            frame_index=sequence.frame_index + 1,
        )
        event["contact_court_xy"] = None

        pending = sequence.step(20.0, event=event)

        self.assertIsNone(pending.rally_result)
        self.assertEqual(pending.phase, "IN_RALLY")
        self.assertIn("bounce_pending_uncertain", pending.events)
        self.assertEqual(pending.score, {"left": 0, "right": 0})

        result = sequence.step(
            20.0,
            timestamp_s=pending.timestamp_s + 0.21,
        )

        self.assertEqual(result.rally_result.status, "unresolved")
        self.assertEqual(result.rally_result.terminal_event, "unknown")
        self.assertIsNone(result.rally_result.rally_winner)
        self.assertIsNone(result.rally_result.point_awarded_to)
        self.assertEqual(result.score, {"left": 0, "right": 0})
        self.assertEqual(
            result.rally_result.score_before,
            result.rally_result.score_after,
        )
        payload = result.to_dict()
        self.assertEqual(payload["contract_version"], 1)
        self.assertEqual(payload["rally_result"]["contract_version"], 1)
        self.assertEqual(payload["event_type_0811"], "UNKNOWN")
        self.assertEqual(payload["rally_over"], "UNKNOWN")
        self.assertTrue(payload["manual_confirmation_required"])
        self.assertEqual(payload["rally_result"]["event_type_0811"], "UNKNOWN")
        self.assertEqual(payload["rally_result"]["rally_over"], "UNKNOWN")
        self.assertTrue(
            payload["rally_result"]["manual_confirmation_required"]
        )
        json.dumps(payload, allow_nan=False)

    def test_net_deadband_landing_without_crossing_is_net_fault(self):
        sequence = Sequence(referee())
        sequence.start_low_y_serve()
        event = bounce_event(
            22.1,
            frame_index=sequence.frame_index + 1,
        )

        result = sequence.step(22.1, event=event)

        self.assertEqual(result.rally_result.terminal_event, "serve_net")
        self.assertEqual(result.rally_result.fault_side, "left")
        self.assertEqual(result.rally_result.rally_winner, "right")
        self.assertEqual(result.score, {"left": 0, "right": 1})
        self.assertIn(
            "net_deadband_landing_without_confirmed_crossing",
            result.rally_result.evidence,
        )

    def test_waiting_state_recovers_inbound_hit_then_scores_return_net(self):
        sequence = Sequence(
            referee(
                service_ownership_enabled=True,
                initial_server_side="left",
                require_server_player_center_outside_court=True,
            )
        )
        inside_center = ((10.0, 10.0),)

        for offset, y_ft in enumerate(
            (20.3, 19.9, 19.5, 19.1, 18.7, 18.3, 17.9, 17.5, 17.1, 16.7, 16.3)
        ):
            sequence.step(
                y_ft,
                timestamp_s=49.3 + offset * 0.02,
                eligible_player_centers_court_xy=inside_center,
            )
        sequence.step(
            17.0,
            event={"events": ["paddle_hit_candidate"]},
            discontinuity_reason="local_track_change",
            timestamp_s=49.54,
            eligible_player_centers_court_xy=inside_center,
        )
        sequence.step(
            18.0,
            timestamp_s=49.56,
            eligible_player_centers_court_xy=inside_center,
        )
        recovered = sequence.step(
            19.0,
            timestamp_s=49.58,
            eligible_player_centers_court_xy=inside_center,
        )

        self.assertEqual(recovered.phase, "IN_RALLY")
        self.assertEqual(recovered.leg_index, 2)
        self.assertEqual(recovered.route_origin_half, "low_y")
        self.assertIn(
            "rally_recovered_from_observed_return",
            recovered.events,
        )
        self.assertIn("return_route_confirmed", recovered.events)
        self.assertEqual(sequence.engine.diagnostics()["serves_confirmed"], 0)
        self.assertEqual(sequence.engine.diagnostics()["rallies_started"], 1)

        terminal = sequence.step(
            22.1,
            event=bounce_event(
                22.1,
                frame_index=sequence.frame_index + 1,
            ),
            eligible_player_centers_court_xy=inside_center,
        )

        self.assertEqual(terminal.rally_result.terminal_event, "return_net")
        self.assertEqual(terminal.rally_result.fault_side, "left")
        self.assertEqual(terminal.rally_result.rally_winner, "right")
        self.assertEqual(terminal.score, {"left": 0, "right": 1})

    def test_waiting_hit_without_inbound_route_does_not_bypass_serve_gate(self):
        sequence = Sequence(
            referee(require_server_player_center_outside_court=True)
        )
        inside_center = ((10.0, 10.0),)

        sequence.step(
            16.0,
            event={"events": ["paddle_hit_candidate"]},
            eligible_player_centers_court_xy=inside_center,
        )
        sequence.step(
            18.0,
            eligible_player_centers_court_xy=inside_center,
        )
        rejected = sequence.step(
            20.0,
            eligible_player_centers_court_xy=inside_center,
        )

        self.assertEqual(rejected.phase, "WAIT_SERVE")
        self.assertIsNone(rejected.rally_id)
        self.assertNotIn(
            "rally_recovered_from_observed_return",
            rejected.events,
        )
        self.assertIn(
            "serve_rejected_player_center_inside_court",
            rejected.events,
        )

    def test_route_confirmation_accumulates_within_time_window(self):
        sequence = Sequence(referee())
        sequence.start_low_y_serve()
        sequence.cross_to_high_y()
        sequence.step(30.0)

        for y_ft in (29.6, 29.2, 28.8):
            result = sequence.step(y_ft)
            self.assertNotIn("return_route_confirmed", result.events)
        result = sequence.step(28.4)

        self.assertIn("return_route_confirmed", result.events)
        self.assertEqual(result.route_origin_half, "high_y")
        self.assertEqual(result.leg_index, 2)

    def test_late_return_recovers_only_from_fresh_safe_observations(self):
        sequence = Sequence(referee(route_min_netward_displacement_ft=4.0))
        sequence.start_low_y_serve()
        sequence.cross_to_high_y()
        sequence.step(30.0)
        sequence.step(29.0)
        sequence.step(28.0)
        sequence.step(22.0)

        skipped = sequence.step(
            20.9,
            discontinuity_reason="camera_side_switch",
        )
        first_fresh = sequence.step(20.6)
        recovered = sequence.step(20.2)

        self.assertNotIn("return_route_confirmed_late", skipped.events)
        self.assertNotIn("return_route_confirmed_late", first_fresh.events)
        self.assertIn("return_route_confirmed_late", recovered.events)
        self.assertIn("net_crossing_confirmed_late", recovered.events)
        self.assertEqual(recovered.route_origin_half, "high_y")
        self.assertEqual(recovered.route_destination_half, "low_y")
        self.assertEqual(recovered.leg_index, 2)
        self.assertTrue(recovered.net_crossed)

    def test_repeated_conflict_bounces_settle_pending_second_bounce(self):
        sequence = Sequence(referee())
        sequence.start_low_y_serve()
        sequence.cross_to_high_y()
        first = bounce_event(19.0, frame_index=sequence.frame_index + 1)

        pending = sequence.step(19.0, event=first)
        second = bounce_event(
            18.0,
            frame_index=sequence.frame_index + 1,
            extra_events=("second_bounce_candidate",),
        )
        result = sequence.step(18.0, event=second)

        self.assertIsNone(pending.rally_result)
        self.assertIn("bounce_pending_uncertain", pending.events)
        self.assertEqual(result.rally_result.terminal_event, "second_bounce")
        self.assertEqual(result.rally_result.fault_half, "low_y")
        self.assertEqual(result.rally_result.rally_winner, "right")
        self.assertEqual(result.score, {"left": 0, "right": 1})

    def test_upstream_event_quality_warning_reaches_terminal_result(self):
        sequence = Sequence(referee())
        sequence.start_low_y_serve()
        event = bounce_event(19.0, frame_index=sequence.frame_index + 1)
        event["warnings"] = ["tracker_diagnostic_not_track_scoped"]

        result = sequence.step(19.0, event=event)

        self.assertIn(
            "tracker_diagnostic_not_track_scoped",
            result.rally_result.warnings,
        )

    def test_guard_stability_uses_two_dimensional_displacement(self):
        sequence = Sequence(
            referee(
                post_rally_guard_ms=100.0,
                post_rally_rearm_stable_observed_frames=3,
                post_rally_rearm_stable_displacement_ft=0.25,
            )
        )
        sequence.start_low_y_serve()
        event = bounce_event(19.0, frame_index=sequence.frame_index + 1)
        terminal = sequence.step(19.0, event=event)

        for offset, x_ft in ((0.12, 8.0), (0.14, 9.0), (0.16, 10.0)):
            guarded = sequence.step(
                18.0,
                x_ft=x_ft,
                timestamp_s=terminal.timestamp_s + offset,
            )

        self.assertEqual(guarded.phase, "POST_RALLY_GUARD")
        self.assertNotIn("ready_for_next_serve", guarded.events)

    def test_factory_applies_physical_half_side_mapping(self):
        engine = build_referee_engine(
            {
                "runtime": {
                    "court_projection": {
                        "net_deadband_ft": 1.25,
                    },
                    "referee": {
                        "enabled": True,
                        "contract_version": 1,
                        "scoring_mode": "rally_point",
                        "expected_server_side": "team_b",
                        "service_ownership_enabled": True,
                        "initial_server_side": "team_b",
                        "require_serve_arming": False,
                        "side_mapping": {
                            "low_y": "team_b",
                            "high_y": "team_a",
                        },
                        "route_confirm_observed_frames": 3,
                        "route_min_netward_displacement_ft": 1.5,
                        "post_rally_guard_ms": 100.0,
                        "rally_lost_timeout_ms": 3210.0,
                    }
                }
            },
            CourtLayout(),
        )
        self.assertIsNotNone(engine)
        self.assertAlmostEqual(engine.net_deadband_ft, 1.25)
        self.assertEqual(
            engine.half_to_side,
            {"low_y": "team_b", "high_y": "team_a"},
        )
        self.assertAlmostEqual(engine.unavailable_timeout_s, 3.21)
        self.assertEqual(engine.expected_server_side, "team_b")
        self.assertTrue(engine.service_ownership_enabled)
        self.assertEqual(engine.service_owner_side, "team_b")
        sequence = Sequence(engine)
        sequence.start_low_y_serve()
        event = bounce_event(
            19.0,
            frame_index=sequence.frame_index + 1,
        )

        result = sequence.step(19.0, event=event)

        self.assertEqual(result.rally_result.fault_side, "team_b")
        self.assertEqual(result.rally_result.rally_winner, "team_a")
        self.assertEqual(result.rally_result.next_server_side, "team_a")
        self.assertEqual(result.service_owner_side, "team_a")
        self.assertEqual(result.score, {"team_b": 0, "team_a": 1})


if __name__ == "__main__":
    unittest.main()
