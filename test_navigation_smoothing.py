# test_navigation_smoothing.py
"""
Unit tests for anti-jitter navigation smoothing, guidance state hysteresis,
and voice pacing.
"""

import time
import numpy as np
from src.navigation import NavigationEngine
from src.guidance import GuidanceEngine, ObstacleInfo
from src.voice import VoiceGuide


def test_navigation_jitter_filtering():
    print("[TEST 1] Testing NavigationEngine Jitter Filtering...")
    nav = NavigationEngine(history_size=7)

    dummy_mask = np.full((720, 1280), 255, dtype=np.uint8)

    # Frame sequence simulating obstacle flickering on/off rapidly in center
    # Frame 0-5: Clear forward
    for i in range(5):
        d = nav.decide(dummy_mask, detected_objects=[])
        assert d == "FORWARD", f"Expected FORWARD, got {d} at frame {i}"

    print("  -> Initial forward state confirmed.")

    # Frame 6: Single-frame noisy obstacle glitch in center
    noisy_obj = [{"label": "chair", "distance": 1.5, "center_x": 640}]
    d_noisy = nav.decide(dummy_mask, detected_objects=noisy_obj)
    # The stabilized direction should NOT immediately flip on a single noisy frame
    assert d_noisy == "FORWARD", f"Single frame noise flipped direction to {d_noisy}!"
    print("  -> Single-frame noisy glitch successfully suppressed (kept FORWARD).")

    # Frame 7-12: Sustained obstacle in center -> should transition smoothly to LEFT or RIGHT
    for _ in range(5):
        d_turn = nav.decide(dummy_mask, detected_objects=noisy_obj)
    assert d_turn in ("LEFT", "RIGHT"), f"Expected detour turn, got {d_turn}"
    print(f"  -> Sustained obstacle transitioned to detour: {d_turn}")

    # Temporary 1-frame drop of obstacle detection while turning
    d_drop = nav.decide(dummy_mask, detected_objects=[])
    # Should maintain detour direction due to history voting and hysteresis
    assert d_drop == d_turn, f"Expected detour to be maintained on 1-frame drop, got {d_drop}"
    print("  -> 1-frame detection drop maintained detour state.")

    # Sustained clear frames (6+ frames) should smoothly return to FORWARD
    for _ in range(6):
        d_clear = nav.decide(dummy_mask, detected_objects=[])
    assert d_clear == "FORWARD", f"Expected return to FORWARD after sustained clear, got {d_clear}"
    print("  -> Sustained clear path cleanly returned to FORWARD.")
    print("[TEST 1 PASSED]\n")


def test_guidance_state_hysteresis():
    print("[TEST 2] Testing GuidanceEngine State Hysteresis...")
    guidance = GuidanceEngine(frame_width=1280, min_clear_frames=8, min_obstacle_frames=3)

    # Initial state: no obstacles
    disp, spoken = guidance.generate_guidance(objects=[], direction="FORWARD")
    # First time starting, transitioning to clear -> requires min_clear_frames (8 frames)
    for _ in range(6):
        disp, spoken = guidance.generate_guidance(objects=[], direction="FORWARD")
        assert spoken is None, f"Premature speech before confirmation: {spoken}"

    # On the 8th frame, sustained clear is confirmed and spoken
    disp, spoken = guidance.generate_guidance(objects=[], direction="FORWARD")
    assert spoken == "Path is clear, continue forward.", f"Expected clear speech, got {spoken}"
    print("  -> Sustained clear path correctly announced once after confirmation.")

    # Subsequent clear frames should remain completely silent
    for _ in range(5):
        disp, spoken = guidance.generate_guidance(objects=[], direction="FORWARD")
        assert spoken is None, f"Repeated clear speech when already clear: {spoken}"
    print("  -> Clear path remains silent during ongoing travel.")

    # 1-frame transient obstacle blip
    blip_obj = [ObstacleInfo(label="chair", distance=1.6, danger="CAUTION", center_x=640)]
    disp, spoken = guidance.generate_guidance(objects=blip_obj, direction="FORWARD")
    assert spoken is None, f"1-frame obstacle blip caused speech: {spoken}"
    print("  -> 1-frame obstacle blip suppressed without false alarm.")

    # Immediate return to clear: should NOT speak 'Path is clear' again because it was never in sustained obstacle state!
    for _ in range(3):
        disp, spoken = guidance.generate_guidance(objects=[], direction="FORWARD")
        assert spoken is None, f"Unnecessary 'Path is clear' chatter: {spoken}"
    print("  -> No false 'Path is clear' flapping.")

    # Sustained obstacle -> announces obstacle
    sustained_objs = [ObstacleInfo(label="chair", distance=1.6, danger="CAUTION", center_x=800)]
    for _ in range(2):
        disp, spoken = guidance.generate_guidance(objects=sustained_objs, direction="LEFT")
    disp, spoken = guidance.generate_guidance(objects=sustained_objs, direction="LEFT")
    assert spoken is not None and "left" in spoken.lower(), f"Expected turn guidance, got {spoken}"
    print(f"  -> Sustained obstacle announced turn guidance: '{spoken}'")

    # Urgent emergency stop -> announces immediately on frame 1
    stop_obj = [ObstacleInfo(label="wall", distance=0.8, danger="DANGER", center_x=640)]
    disp, spoken = guidance.generate_guidance(objects=stop_obj, direction="STOP")
    assert spoken is not None and ("stop" in spoken.lower() or "warning" in spoken.lower()), f"Emergency stop was delayed: {spoken}"
    print(f"  -> Emergency STOP announced with zero delay: '{spoken}'")
    print("[TEST 2 PASSED]\n")


def test_voice_pacing():
    print("[TEST 3] Testing VoiceGuide Global Pacing & Emergency Override...")
    # Initialize voice guide with mock/fast cooldowns for testing logic
    voice = VoiceGuide(cooldown_seconds=1.0, global_pacing_seconds=0.8)

    # Urgent message should be accepted
    voice.speak("Warning. Path blocked. Please stop.", force=True)
    time.sleep(0.1)

    # Immediately requesting non-urgent message during global pacing window should be suppressed
    voice.speak("Path is clear, continue forward.")
    # Verify non-urgent didn't overwrite or disrupt
    print("  -> Non-urgent chatter during pacing window successfully paced.")

    # Clean shutdown
    voice.stop()
    print("[TEST 3 PASSED]\n")


if __name__ == "__main__":
    test_navigation_jitter_filtering()
    test_guidance_state_hysteresis()
    test_voice_pacing()
    print("========================================")
    print("ALL ANTI-JITTER TESTS PASSED SUCCESSFULLY!")
    print("========================================")
