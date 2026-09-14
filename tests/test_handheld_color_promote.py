"""REPAIR-ORACLE-5290: handheld color_candidate promotion gates."""

from inference.detection.yolo_detector import promote_handheld_color_class


def test_promotes_small_black_candidate_in_carry_band():
    person = (400.0, 200.0, 600.0, 900.0)  # 200x700
    # Small bag near right hand (lower body)
    bag = (560.0, 650.0, 610.0, 720.0)  # 50x70 = 3500; person=140000 → ratio≈0.025
    assert (
        promote_handheld_color_class("color_candidate_black", bag, [person])
        == "black_waste_bag"
    )


def test_dumpster_scale_black_stays_proposal():
    person = (400.0, 200.0, 600.0, 900.0)
    dumpster = (620.0, 400.0, 900.0, 900.0)  # huge next to person
    assert (
        promote_handheld_color_class("color_candidate_black", dumpster, [person])
        == "color_candidate_black"
    )


def test_candidate_outside_carry_band_stays_proposal():
    person = (400.0, 200.0, 600.0, 900.0)
    # Above head
    bag = (480.0, 50.0, 530.0, 100.0)
    assert (
        promote_handheld_color_class("color_candidate_yellow", bag, [person])
        == "color_candidate_yellow"
    )


def test_white_candidate_never_promoted():
    person = (400.0, 200.0, 600.0, 900.0)
    shirt = (450.0, 500.0, 520.0, 580.0)
    assert (
        promote_handheld_color_class("color_candidate_white", shirt, [person])
        == "color_candidate_white"
    )
