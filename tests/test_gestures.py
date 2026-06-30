"""Gesture classifier tests — synthetic COCO-17 skeletons, pure geometry."""
from kovio.adapters.gestures import (
    GestureClassifier,
    NOSE, L_EYE, R_EYE, L_EAR, R_EAR,
    L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_HIP, R_HIP,
)


def skeleton(**overrides):
    """A neutral standing person (arms down). Override joints by name.

    Image coords: y grows downward. Shoulders y=100, hips y=250 -> torso=150,
    nose y=60. Each value is (x, y, conf).
    """
    kp = [None] * 17
    kp[NOSE] = (200, 60, 1.0)
    kp[L_EYE], kp[R_EYE] = (190, 55, 1.0), (210, 55, 1.0)
    kp[L_EAR], kp[R_EAR] = (180, 58, 1.0), (220, 58, 1.0)
    kp[L_SHOULDER], kp[R_SHOULDER] = (160, 100, 1.0), (240, 100, 1.0)
    kp[L_ELBOW], kp[R_ELBOW] = (150, 175, 1.0), (250, 175, 1.0)
    kp[L_WRIST], kp[R_WRIST] = (150, 250, 1.0), (250, 250, 1.0)
    kp[L_HIP], kp[R_HIP] = (170, 250, 1.0), (230, 250, 1.0)
    for name, val in overrides.items():
        kp[globals()[name]] = val
    return kp


def kinds(hits):
    return sorted(h.kind for h in hits)


def test_arms_down_is_nothing():
    gc = GestureClassifier()
    assert gc.classify(1, skeleton(), now=0.0) == []


def test_overhead_is_high_five():
    gc = GestureClassifier()
    hits = gc.classify(1, skeleton(R_WRIST=(250, 30, 1.0)), now=0.0)
    assert "high_five" in kinds(hits)
    assert next(h for h in hits if h.kind == "high_five").side == "right"


def test_static_raised_hand_is_not_a_wave():
    gc = GestureClassifier()
    # Raised (y=70 < shoulder-margin) but not overhead and not oscillating.
    out = []
    for t in (0.0, 0.3, 0.6, 0.9):
        out = gc.classify(1, skeleton(R_WRIST=(250, 70, 1.0)), now=t)
    assert "wave" not in kinds(out)


def test_oscillating_raised_hand_is_a_wave():
    gc = GestureClassifier()
    xs = [320, 260, 320, 260, 320]  # side-to-side, amplitude >> jitter
    out = []
    for i, x in enumerate(xs):
        out = gc.classify(1, skeleton(R_WRIST=(x, 70, 1.0)), now=i * 0.3)
    assert "wave" in kinds(out)


def test_handshake_needs_forward_depth_hint():
    gc = GestureClassifier()
    arm = skeleton(R_WRIST=(300, 200, 1.0))  # mid-torso height, extended
    # No depth hint -> no handshake.
    assert "handshake" not in kinds(gc.classify(1, arm, now=0.0))
    # With the forward hint from depth -> handshake.
    hits = gc.classify(2, arm, now=0.0, forward_right=True)
    assert "handshake" in kinds(hits)


def test_chest_height_forward_is_fist_bump():
    gc = GestureClassifier()
    hits = gc.classify(1, skeleton(R_WRIST=(300, 130, 1.0)), now=0.0, forward_right=True)
    assert "fist_bump" in kinds(hits)


def test_no_shoulders_bails_out():
    gc = GestureClassifier()
    kp = skeleton(R_WRIST=(250, 30, 1.0))
    kp[L_SHOULDER] = (160, 100, 0.0)  # zero confidence
    kp[R_SHOULDER] = (240, 100, 0.0)
    assert gc.classify(1, kp, now=0.0) == []


def test_left_arm_high_five_reports_left_side():
    gc = GestureClassifier()
    hits = gc.classify(1, skeleton(L_WRIST=(150, 30, 1.0)), now=0.0)
    hf = [h for h in hits if h.kind == "high_five"]
    assert hf and hf[0].side == "left"
