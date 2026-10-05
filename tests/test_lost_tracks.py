"""Never silently lose a person who may have fallen: tracks lost while lying are kept and flagged."""

from core.lost_tracks import LostTracks

W, H = 1280, 720


def step(lt, ts, people, labels=None, falls=None, furniture=()):
    return lt.update(ts, people, labels or {}, falls or {}, list(furniture), W, H)


def test_a_person_lost_while_lying_is_kept_and_alerted_after_3_s():
    lt = LostTracks()
    box = (400.0, 500.0, 800.0, 650.0)
    step(lt, 0.0, {1: box}, labels={1: "Lying down"})
    assert step(lt, 0.1, {}) == [] and lt.snapshot(0.1)[0]["reason"] == "lying down"
    assert step(lt, 2.0, {}) == []
    alerts = step(lt, 3.2, {})
    assert len(alerts) == 1 and alerts[0].alert_type == "person_lost_lying"
    assert "check on them" in alerts[0].message
    assert step(lt, 10.0, {}) == []  # one alert per loss
    assert lt.snapshot(10.0)[0]["seconds"] == 10.0  # since last seen
    step(lt, 31.0, {})
    assert lt.snapshot(31.0) == []  # kept up to 30 s


def test_seen_again_where_lost_clears_it_even_with_a_new_track_id():
    lt = LostTracks()
    step(lt, 0.0, {1: (400.0, 500.0, 800.0, 650.0)}, falls={1: "fallen"})
    step(lt, 1.0, {})
    assert lt.snapshot(1.0)[0]["reason"] == "fall detector: fallen"
    step(lt, 2.0, {7: (420.0, 480.0, 790.0, 660.0)})
    assert lt.snapshot(2.0) == []


def test_walking_out_of_frame_is_a_normal_exit():
    lt = LostTracks()
    step(lt, 0.0, {1: (1100.0, 100.0, 1230.0, 600.0)}, labels={1: "Walking"})
    step(lt, 0.1, {1: (1150.0, 100.0, 1279.0, 600.0)}, labels={1: "Walking"})
    step(lt, 0.2, {})
    assert lt.snapshot(0.2) == []


def test_lost_on_a_bed_or_couch_counts_even_without_a_lying_label():
    lt = LostTracks()
    bed = ("bed", (300.0, 350.0, 1000.0, 700.0))
    step(lt, 0.0, {1: (500.0, 300.0, 700.0, 600.0)}, labels={1: "Sitting"}, furniture=[bed])
    step(lt, 0.1, {}, furniture=[bed])
    assert lt.snapshot(0.1)[0]["reason"] == "on/near bed"


def test_an_upright_person_vanishing_mid_room_is_not_flagged():
    lt = LostTracks()
    step(lt, 0.0, {1: (500.0, 100.0, 650.0, 600.0)}, labels={1: "Standing"})
    step(lt, 0.1, {})
    assert lt.snapshot(0.1) == []
