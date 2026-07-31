import unittest
from types import SimpleNamespace

from src.court import (
    CourtEventInterpreter,
    CourtLayout,
    CourtPanelRenderer,
    ProjectionResult,
)


def projection(
    *,
    inside=True,
    image_xy=(100.0, 100.0),
    track_status="observed",
):
    return ProjectionResult(
        coordinate_system="pickleball_full_court_ft",
        coordinate_system_version=1,
        active_side="left",
        calibration_id="left_test",
        calibration_source="manual_test",
        image_xy=[float(image_xy[0]), float(image_xy[1])],
        ball_court_xy=[10.0, 12.0] if inside else [-1.0, 12.0],
        projection_status="homography",
        projection_valid=True,
        homography_available=True,
        reprojection_error_px=0.0,
        track_status=track_status,
        observed=track_status == "observed",
        predicted=track_status == "predicted",
        inside_court=inside,
    )


def track(
    *,
    horizontal_velocity=300.0,
    vertical_velocity=-200.0,
    status="observed",
):
    return SimpleNamespace(
        status=status,
        velocity=[horizontal_velocity, vertical_velocity],
    )


def diagnostics(*, bounce=0, impact=0):
    return {
        "left": {
            "bounce_recoveries": bounce,
            "impact_recoveries": impact,
        },
        "right": {
            "bounce_recoveries": 0,
            "impact_recoveries": 0,
        },
    }


class CourtEventInterpreterTest(unittest.TestCase):
    def update(
        self,
        interpreter,
        projected,
        *,
        bounce=0,
        impact=0,
        vertical_velocity=-200.0,
        timestamp_s=None,
        frame_index=None,
        eligible_players=None,
        track_status="observed",
        horizontal_velocity=300.0,
        active_side="left",
        local_track_id=7,
        discontinuity_reason=None,
    ):
        return interpreter.update(
            projected,
            track=track(
                horizontal_velocity=horizontal_velocity,
                vertical_velocity=vertical_velocity,
                status=track_status,
            ),
            active_side=active_side,
            local_track_id=local_track_id,
            tracker_diagnostics=diagnostics(
                bounce=bounce,
                impact=impact,
            ),
            frame_scale_overrides={"left": 1.0, "right": 1.0},
            timestamp_s=timestamp_s,
            frame_index=frame_index,
            eligible_players=eligible_players or [],
            discontinuity_reason=discontinuity_reason,
        )

    def test_default_trail_is_half_of_previous_thirty_frames(self):
        renderer = CourtPanelRenderer(CourtLayout())

        self.assertEqual(renderer.trail.maxlen, 15)

    def test_airborne_projection_outside_court_stays_green(self):
        interpreter = CourtEventInterpreter()

        result = self.update(interpreter, projection(inside=False))

        self.assertEqual(result.phase, "airborne")
        self.assertEqual(result.display_color, "fluorescent_green")
        self.assertEqual(result.display_text_zh, "飞行")
        self.assertEqual(result.events, [])

    def test_inside_bounce_is_yellow_until_rising_is_confirmed(self):
        interpreter = CourtEventInterpreter(
            ground_min_hold_frames=3,
            rise_confirm_frames=2,
        )
        projected = projection(inside=True)

        contact = self.update(interpreter, projected, bounce=1)
        held_1 = self.update(interpreter, projected, bounce=1)
        held_2 = self.update(interpreter, projected, bounce=1)
        airborne = self.update(interpreter, projected, bounce=1)

        self.assertIn("bounce_candidate", contact.events)
        self.assertEqual(contact.display_color, "yellow")
        self.assertEqual(contact.display_text_zh, "落地")
        self.assertFalse(contact.fault_candidate)
        self.assertEqual(held_1.display_color, "yellow")
        self.assertEqual(held_2.display_color, "yellow")
        self.assertEqual(airborne.display_color, "fluorescent_green")

    def test_outside_bounce_is_red_but_airborne_outside_is_not(self):
        interpreter = CourtEventInterpreter()
        projected = projection(inside=False)

        contact = self.update(interpreter, projected, bounce=1)

        self.assertEqual(contact.display_color, "red")
        self.assertEqual(contact.display_text_zh, "出界")
        self.assertTrue(contact.fault_candidate)
        self.assertEqual(contact.fault_reasons, ["out_of_bounds"])
        self.assertIn("bounce_candidate", contact.events)
        self.assertIn("out_of_bounds_bounce_candidate", contact.events)

    def test_second_bounce_candidate_is_counted_since_last_hit(self):
        interpreter = CourtEventInterpreter()
        projected = projection(inside=True)

        first = self.update(
            interpreter,
            projected,
            bounce=1,
            timestamp_s=0.0,
        )
        second = self.update(
            interpreter,
            projected,
            bounce=2,
            timestamp_s=0.2,
        )

        self.assertEqual(first.bounce_index_since_last_hit, 1)
        self.assertEqual(second.bounce_index_since_last_hit, 2)
        self.assertIn("second_bounce_candidate", second.events)
        self.assertEqual(second.display_text_zh, "二弹")
        self.assertTrue(second.fault_candidate)
        self.assertEqual(second.fault_reasons, ["second_bounce"])

    def test_second_bounce_has_no_short_rule_time_window(self):
        interpreter = CourtEventInterpreter(
            rally_state_timeout_ms=5000.0,
            event_cooldown_ms=0.0,
        )
        projected = projection(inside=True)

        self.update(
            interpreter,
            projected,
            bounce=1,
            timestamp_s=0.0,
        )
        second = self.update(
            interpreter,
            projected,
            bounce=2,
            timestamp_s=3.5,
        )

        self.assertIn("second_bounce_candidate", second.events)
        self.assertEqual(second.display_text_zh, "二弹")

    def test_stale_rally_timeout_prevents_cross_rally_second_bounce(self):
        interpreter = CourtEventInterpreter(
            rally_state_timeout_ms=5000.0,
            event_cooldown_ms=0.0,
        )
        projected = projection(inside=True)

        self.update(
            interpreter,
            projected,
            bounce=1,
            timestamp_s=0.0,
        )
        later = self.update(
            interpreter,
            projected,
            bounce=2,
            timestamp_s=6.0,
        )

        self.assertEqual(later.bounce_index_since_last_hit, 1)
        self.assertNotIn("second_bounce_candidate", later.events)
        self.assertIn("rally_state_reset_after_timeout", later.warnings)

    def test_out_of_bounds_takes_precedence_over_second_bounce(self):
        interpreter = CourtEventInterpreter(event_cooldown_ms=0.0)

        self.update(
            interpreter,
            projection(inside=True),
            bounce=1,
            timestamp_s=0.0,
        )
        outside = self.update(
            interpreter,
            projection(inside=False),
            bounce=2,
            timestamp_s=0.5,
        )

        self.assertIn("out_of_bounds_bounce_candidate", outside.events)
        self.assertNotIn("second_bounce_candidate", outside.events)
        self.assertEqual(outside.display_text_zh, "出界")
        self.assertEqual(outside.fault_reasons, ["out_of_bounds"])

    def test_missing_ball_and_predicted_ball_have_distinct_chinese_states(self):
        interpreter = CourtEventInterpreter()
        projected = projection()
        missing = interpreter.update(
            projected,
            track=None,
            active_side=None,
            local_track_id=None,
            tracker_diagnostics=diagnostics(),
            frame_scale_overrides={"left": 1.0, "right": 1.0},
        )
        predicted_projection = projection(track_status="predicted")
        predicted = self.update(
            interpreter,
            predicted_projection,
            track_status="predicted",
        )

        self.assertEqual(missing.display_state, "not_visible")
        self.assertEqual(missing.display_text_zh, "消失在屏幕")
        self.assertEqual(predicted.display_state, "predicted")
        self.assertEqual(predicted.display_text_zh, "短时预测")

    def test_r9_bounce_near_player_body_is_reclassified_as_hit(self):
        interpreter = CourtEventInterpreter()
        player = {
            "bbox": [80.0, 60.0, 140.0, 160.0],
            "eligible_player": True,
        }

        result = self.update(
            interpreter,
            projection(image_xy=(100.0, 100.0)),
            bounce=1,
            eligible_players=[player],
        )

        self.assertIn("paddle_hit_candidate", result.events)
        self.assertNotIn("bounce_candidate", result.events)
        self.assertEqual(result.display_color, "purple")
        self.assertIn("r9_bounce_reclassified_as_hit", result.evidence)

    def test_multiframe_vertical_v_detects_bounce_without_r9_recovery(self):
        interpreter = CourtEventInterpreter()
        points = [
            (90.0, 90.0),
            (100.0, 120.0),
            (110.0, 105.0),
            (120.0, 90.0),
        ]

        results = [
            self.update(
                interpreter,
                projection(image_xy=point),
                timestamp_s=index * 0.02,
                frame_index=index,
            )
            for index, point in enumerate(points)
        ]

        self.assertEqual(results[-1].events, ["bounce_candidate"])
        self.assertEqual(results[-1].display_color, "yellow")
        self.assertEqual(results[-1].contact_frame_index, 1)
        self.assertEqual(results[-1].contact_image_xy, [100.0, 120.0])
        self.assertIn(
            "vertical_down_to_up_reversal",
            results[-1].evidence,
        )

    def test_horizontal_reversal_near_player_detects_hit_without_new_model(self):
        interpreter = CourtEventInterpreter()
        player = {
            "bbox": [80.0, 60.0, 140.0, 160.0],
            "eligible_player": True,
        }
        points = [
            (80.0, 100.0),
            (110.0, 100.0),
            (90.0, 102.0),
            (70.0, 104.0),
        ]

        results = [
            self.update(
                interpreter,
                projection(image_xy=point),
                timestamp_s=index * 0.02,
                frame_index=index,
                eligible_players=[player],
            )
            for index, point in enumerate(points)
        ]

        self.assertEqual(results[-1].events, ["paddle_hit_candidate"])
        self.assertEqual(results[-1].display_color, "purple")
        self.assertGreater(
            results[-1].metrics["direction_change_deg"],
            90.0,
        )

    def test_same_side_track_change_reversal_near_player_detects_hit(self):
        interpreter = CourtEventInterpreter()
        player = {
            "bbox": [80.0, 60.0, 140.0, 160.0],
            "eligible_player": True,
        }
        self.update(
            interpreter,
            projection(image_xy=(180.0, 100.0)),
            timestamp_s=0.0,
            frame_index=0,
            local_track_id=89,
            eligible_players=[player],
        )
        self.update(
            interpreter,
            projection(image_xy=(140.0, 100.0)),
            timestamp_s=0.02,
            frame_index=1,
            local_track_id=89,
            eligible_players=[player],
        )

        result = self.update(
            interpreter,
            projection(image_xy=(170.0, 100.0)),
            timestamp_s=0.04,
            frame_index=2,
            local_track_id=90,
            discontinuity_reason="local_track_change",
            eligible_players=[player],
        )

        self.assertIn("paddle_hit_candidate", result.events)
        self.assertEqual(result.display_color, "purple")
        self.assertIn(
            "same_side_track_change_reversal",
            result.evidence,
        )
        self.assertEqual(
            result.metrics["player_relation"],
            "extended_player_reach",
        )
        self.assertGreater(result.metrics["direction_change_deg"], 90.0)

    def test_fast_side_switch_inside_extended_reach_detects_hit(self):
        interpreter = CourtEventInterpreter()
        player = {
            "bbox": [80.0, 60.0, 140.0, 160.0],
            "eligible_player": True,
        }
        self.update(
            interpreter,
            projection(image_xy=(100.0, 100.0)),
            timestamp_s=0.0,
            frame_index=0,
            active_side="left",
            local_track_id=2,
        )

        result = self.update(
            interpreter,
            projection(image_xy=(50.0, 100.0)),
            timestamp_s=0.02,
            frame_index=1,
            horizontal_velocity=400.0,
            vertical_velocity=0.0,
            active_side="right",
            local_track_id=3,
            discontinuity_reason="camera_side_switch",
            eligible_players=[player],
        )

        self.assertIn("paddle_hit_candidate", result.events)
        self.assertEqual(result.display_color, "purple")
        self.assertIn(
            "camera_side_switch_extended_reach",
            result.evidence,
        )
        self.assertEqual(
            result.metrics["player_relation"],
            "extended_player_reach",
        )

    def test_slow_side_switch_inside_extended_reach_is_not_hit(self):
        interpreter = CourtEventInterpreter()
        player = {
            "bbox": [80.0, 60.0, 140.0, 160.0],
            "eligible_player": True,
        }
        self.update(
            interpreter,
            projection(image_xy=(100.0, 100.0)),
            timestamp_s=0.0,
            active_side="left",
            local_track_id=2,
        )

        result = self.update(
            interpreter,
            projection(image_xy=(50.0, 100.0)),
            timestamp_s=0.02,
            horizontal_velocity=100.0,
            vertical_velocity=0.0,
            active_side="right",
            local_track_id=3,
            discontinuity_reason="camera_side_switch",
            eligible_players=[player],
        )

        self.assertNotIn("paddle_hit_candidate", result.events)

    def test_impossible_cross_id_jump_near_player_is_not_hit(self):
        interpreter = CourtEventInterpreter()
        player = {
            "bbox": [950.0, 60.0, 1050.0, 160.0],
            "eligible_player": True,
        }
        self.update(
            interpreter,
            projection(image_xy=(0.0, 100.0)),
            timestamp_s=0.0,
            local_track_id=1,
        )
        self.update(
            interpreter,
            projection(image_xy=(20.0, 100.0)),
            timestamp_s=0.02,
            local_track_id=1,
        )

        result = self.update(
            interpreter,
            projection(image_xy=(1000.0, 100.0)),
            timestamp_s=0.04,
            local_track_id=2,
            discontinuity_reason="local_track_change",
            eligible_players=[player],
        )

        self.assertNotIn("paddle_hit_candidate", result.events)

    def test_bounce_is_suppressed_immediately_after_hit(self):
        interpreter = CourtEventInterpreter(
            event_cooldown_ms=0.0,
            post_hit_bounce_suppression_ms=180.0,
        )
        projected = projection()
        hit = self.update(
            interpreter,
            projected,
            impact=1,
            timestamp_s=0.0,
        )
        self.update(
            interpreter,
            projected,
            timestamp_s=0.02,
        )
        suppressed = self.update(
            interpreter,
            projected,
            bounce=1,
            timestamp_s=0.08,
        )

        self.assertIn("paddle_hit_candidate", hit.events)
        self.assertNotIn("bounce_candidate", suppressed.events)
        self.assertIn(
            "bounce_candidate_suppressed_post_hit",
            suppressed.warnings,
        )

    def test_bounce_count_resets_when_ball_changes_court_half(self):
        interpreter = CourtEventInterpreter()
        projected = projection(inside=True)
        first = self.update(
            interpreter,
            projected,
            bounce=1,
            timestamp_s=0.0,
        )
        other_half = interpreter.update(
            projected,
            track=track(),
            active_side="right",
            local_track_id=9,
            tracker_diagnostics={
                "left": {
                    "bounce_recoveries": 1,
                    "impact_recoveries": 0,
                },
                "right": {
                    "bounce_recoveries": 1,
                    "impact_recoveries": 0,
                },
            },
            frame_scale_overrides={"left": 1.0, "right": 1.0},
            discontinuity_reason="camera_side_switch",
            timestamp_s=0.2,
        )

        self.assertEqual(first.bounce_index_since_last_hit, 1)
        self.assertEqual(other_half.bounce_index_since_last_hit, 1)
        self.assertNotIn("second_bounce_candidate", other_half.events)

    def test_hit_candidate_is_purple_then_returns_to_green(self):
        interpreter = CourtEventInterpreter(hit_flash_frames=3)
        projected = projection(inside=True)

        flash_1 = self.update(interpreter, projected, impact=1)
        flash_2 = self.update(interpreter, projected, impact=1)
        flash_3 = self.update(interpreter, projected, impact=1)
        airborne = self.update(interpreter, projected, impact=1)

        self.assertIn("paddle_hit_candidate", flash_1.events)
        self.assertIn("paddle_not_directly_detected", flash_1.warnings)
        self.assertEqual(flash_1.display_color, "purple")
        self.assertEqual(flash_1.display_text_zh, "击球")
        self.assertEqual(flash_2.display_color, "purple")
        self.assertEqual(flash_3.display_color, "purple")
        self.assertEqual(airborne.display_color, "fluorescent_green")
        self.assertEqual(airborne.bounce_index_since_last_hit, 0)


if __name__ == "__main__":
    unittest.main()
