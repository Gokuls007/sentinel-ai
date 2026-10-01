# Your search questions (8–10 of the 30)

Write questions the way you'd actually type them: casual, short, with typos if that's how you
type. Claude then works out the expected answer for each against the seeded test database
and adds it to `questions.json` with `"source": "user"`. Leave the **Expected** lines alone.

## What's in the test database

It's a fixed log of 45 events (`tests/search/seed.py`). Pretend **now is Wednesday 30 Sep
2026, 15:00**. "This week" starts Monday 28 Sep, and "last week" is Mon 21 to Sun 27 Sep.
The log covers 15–30 Sep.

| Camera | What it sees | Zones |
|---|---|---|
| `cam-0` | Warehouse floor | Loading Dock (time-limited), Forklift Lane (one-way), Chemical Storage (restricted) |
| `cam-1` | Yard | Yard Gate (restricted); some events have no zone |
| `laptop` | Assembly workbench | Workbench (posture / REBA alerts) |

- **Event types:** fall, zone_intrusion, time_exceeded (stayed too long), wrong_direction,
  loitering, ergo_risk (bad posture), near_miss.
- **Severities:** low, medium, high, critical.
- **Other fields:** each event has a track id (one tracked person on one camera). Falls
  can be marked as confirmed or as a false alarm.
- Questions about something that isn't there are fine, for example a zone that doesn't
  exist or a day with nothing. The expected answer is then "none".

## Questions

### U1
Question:
Expected: _(Claude fills in)_

### U2
Question:
Expected: _(Claude fills in)_

### U3
Question:
Expected: _(Claude fills in)_

### U4
Question:
Expected: _(Claude fills in)_

### U5
Question:
Expected: _(Claude fills in)_

### U6
Question:
Expected: _(Claude fills in)_

### U7
Question:
Expected: _(Claude fills in)_

### U8
Question:
Expected: _(Claude fills in)_

### U9 (optional)
Question:
Expected: _(Claude fills in)_

### U10 (optional)
Question:
Expected: _(Claude fills in)_
