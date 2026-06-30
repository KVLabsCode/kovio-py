"""Pure-helper tests for detectors: YOLO decode, NMS, phone association."""
import numpy as np

from kovio.adapters.detectors import (
    Box, parse_yolo_output, nms, associate_phones, iou,
)


def test_parse_yolo_output_decodes_and_scales():
    # 80-class detection head, 2 anchors. Anchor 0 is a confident person.
    preds = np.zeros((84, 2), dtype="float32")
    preds[0:4, 0] = [320, 320, 64, 128]              # box for anchor 0
    preds[4 + 0, 0] = 0.9                              # person class score
    preds[4 + 67, 1] = 0.05                            # anchor 1: weak phone
    boxes = parse_yolo_output(preds, conf_threshold=0.4, img_w=640, img_h=480)
    assert len(boxes) == 1
    b = boxes[0]
    assert b.cls == 0 and abs(b.conf - 0.9) < 1e-6
    # sx=1.0, sy=0.75 -> y scaled by 0.75
    assert abs(b.x1 - 288) < 1e-4 and abs(b.x2 - 352) < 1e-4
    assert abs(b.y1 - 192) < 1e-4 and abs(b.y2 - 288) < 1e-4


def test_nms_keeps_highest_per_class():
    a = Box(0, 0.9, 0, 0, 100, 100)
    b = Box(0, 0.6, 10, 10, 105, 105)   # heavy overlap, same class -> dropped
    c = Box(67, 0.8, 0, 0, 100, 100)    # different class -> kept
    out = nms([a, b, c], iou_threshold=0.5)
    assert a in out and c in out and b not in out
    assert iou(a, b) > 0.5


def test_associate_phone_inside_person():
    person = Box(0, 0.9, 0, 0, 100, 200)
    phone = Box(67, 0.8, 40, 80, 60, 110)  # centre (50,95) inside person
    count, holders = associate_phones([person], [phone])
    assert count == 1 and holders == {0}


def test_associate_phone_nearest_when_outside():
    p0 = Box(0, 0.9, 0, 0, 50, 100)        # centre (25,50)
    p1 = Box(0, 0.9, 200, 0, 250, 100)     # centre (225,50)
    phone = Box(67, 0.8, 52, 40, 62, 60)   # centre (57,50) just outside p0, near it
    count, holders = associate_phones([p0, p1], [phone])
    assert count == 1 and holders == {0}


def test_far_phone_is_not_counted():
    person = Box(0, 0.9, 0, 0, 50, 100)
    phone = Box(67, 0.8, 600, 400, 610, 420)  # nowhere near anyone
    count, holders = associate_phones([person], [phone])
    assert count == 0 and holders == set()


def test_two_people_one_phone_each():
    p0 = Box(0, 0.9, 0, 0, 100, 200)
    p1 = Box(0, 0.9, 300, 0, 400, 200)
    ph0 = Box(67, 0.8, 40, 80, 60, 110)     # in p0
    ph1 = Box(67, 0.8, 340, 80, 360, 110)   # in p1
    count, holders = associate_phones([p0, p1], [ph0, ph1])
    assert count == 2 and holders == {0, 1}
