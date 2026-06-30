"""Keypoint gaze-proxy tests (facing_camera)."""
from kovio.adapters.detectors import facing_camera


def _head(nose_x, le_x, re_x, conf=1.0):
    kp = [None] * 17
    kp[0] = (nose_x, 50, conf)   # nose
    kp[1] = (le_x, 48, conf)     # left eye
    kp[2] = (re_x, 48, conf)     # right eye
    return kp


def test_frontal_face_is_looking():
    assert facing_camera(_head(nose_x=200, le_x=190, re_x=210)) is True


def test_profile_nose_outside_eyes_is_not_looking():
    # Head turned: nose has slid past both eyes.
    assert facing_camera(_head(nose_x=230, le_x=190, re_x=210)) is False


def test_low_confidence_eye_is_not_looking():
    kp = _head(nose_x=200, le_x=190, re_x=210)
    kp[2] = (210, 48, 0.1)  # right eye barely seen -> likely profile
    assert facing_camera(kp, conf_min=0.3) is False


def test_missing_keypoints_safe():
    assert facing_camera(None) is False
    assert facing_camera([None] * 17) is False
