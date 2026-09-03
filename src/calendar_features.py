"""Chinese calendar features known before the forecast date."""

from calendar import monthrange

import pandas as pd


HOLIDAYS = {
    # 2013
    "2013-01-01", "2013-01-02", "2013-01-03",
    "2013-02-09", "2013-02-10", "2013-02-11", "2013-02-12",
    "2013-02-13", "2013-02-14", "2013-02-15",
    "2013-04-04", "2013-04-05", "2013-04-06",
    "2013-04-29", "2013-04-30", "2013-05-01",
    "2013-06-10", "2013-06-11", "2013-06-12",
    "2013-09-19", "2013-09-20", "2013-09-21",
    "2013-10-01", "2013-10-02", "2013-10-03", "2013-10-04",
    "2013-10-05", "2013-10-06", "2013-10-07",
    # 2014
    "2014-01-01",
    "2014-01-31", "2014-02-01", "2014-02-02", "2014-02-03",
    "2014-02-04", "2014-02-05", "2014-02-06",
    "2014-04-05", "2014-04-06", "2014-04-07",
    "2014-05-01", "2014-05-02", "2014-05-03",
    "2014-05-31", "2014-06-01", "2014-06-02",
    "2014-09-06", "2014-09-07", "2014-09-08",
    "2014-10-01", "2014-10-02", "2014-10-03", "2014-10-04",
    "2014-10-05", "2014-10-06", "2014-10-07",
}

SPECIAL_WORKDAYS = {
    "2013-01-05", "2013-01-06", "2013-02-16", "2013-02-17",
    "2013-04-07", "2013-04-27", "2013-04-28", "2013-06-08",
    "2013-06-09", "2013-09-22", "2013-09-29", "2013-10-12",
    "2014-01-26", "2014-02-08", "2014-05-04", "2014-09-28",
    "2014-10-11",
}

HOLIDAY_DATES = {pd.Timestamp(value) for value in HOLIDAYS}
SPECIAL_WORKDAY_DATES = {pd.Timestamp(value) for value in SPECIAL_WORKDAYS}


def is_workday(date):
    date = pd.Timestamp(date).normalize()
    return date in SPECIAL_WORKDAY_DATES or (
        date.dayofweek < 5 and date not in HOLIDAY_DATES
    )


def _holiday_blocks():
    blocks = []
    for date in sorted(HOLIDAY_DATES):
        if not blocks or date != blocks[-1][-1] + pd.Timedelta(days=1):
            blocks.append([date])
        else:
            blocks[-1].append(date)
    return blocks


HOLIDAY_BLOCKS = _holiday_blocks()
HOLIDAY_POSITION = {
    date: (index + 1, len(block))
    for block in HOLIDAY_BLOCKS
    for index, date in enumerate(block)
}
HOLIDAY_STARTS = [block[0] for block in HOLIDAY_BLOCKS]
HOLIDAY_ENDS = [block[-1] for block in HOLIDAY_BLOCKS]


def _distance_to_next(date, candidates, default=30):
    future = [candidate for candidate in candidates if candidate >= date]
    return min((future[0] - date).days, default) if future else default


def _distance_from_last(date, candidates, default=30):
    past = [candidate for candidate in candidates if candidate <= date]
    return min((date - past[-1]).days, default) if past else default


def _adjacent_workday_flag(date, direction):
    probe = date + pd.Timedelta(days=direction)
    for _ in range(7):
        if probe in HOLIDAY_DATES:
            return 1
        if is_workday(probe):
            return 0
        probe += pd.Timedelta(days=direction)
    return 0


def calendar_feature_dict(date):
    date = pd.Timestamp(date).normalize()
    holiday_index, holiday_length = HOLIDAY_POSITION.get(date, (0, 0))
    days_to_start = _distance_to_next(date, HOLIDAY_STARTS)
    days_from_end = _distance_from_last(date, HOLIDAY_ENDS)
    _, days_in_month = monthrange(date.year, date.month)
    month_dates = pd.date_range(
        pd.Timestamp(date.year, date.month, 1),
        pd.Timestamp(date.year, date.month, days_in_month),
    )
    workdays = [value for value in month_dates if is_workday(value)]
    workday_position = workdays.index(date) + 1 if date in workdays else 0
    workdays_remaining = sum(value > date for value in workdays)

    return {
        "is_holiday": int(date in HOLIDAY_DATES),
        "is_special_workday": int(date in SPECIAL_WORKDAY_DATES),
        "is_workday": int(is_workday(date)),
        "is_non_workday": int(not is_workday(date)),
        "holiday_day_index": holiday_index,
        "holiday_length": holiday_length,
        "days_to_holiday_start": days_to_start,
        "days_from_holiday_end": days_from_end,
        "is_pre_holiday_1d": int(days_to_start == 1),
        "is_pre_holiday_3d": int(0 < days_to_start <= 3),
        "is_post_holiday_1d": int(days_from_end == 1),
        "is_post_holiday_3d": int(0 < days_from_end <= 3),
        "is_last_workday_before_holiday": int(
            is_workday(date) and _adjacent_workday_flag(date, 1)
        ),
        "is_first_workday_after_holiday": int(
            is_workday(date) and _adjacent_workday_flag(date, -1)
        ),
        "workday_of_month": workday_position,
        "workdays_to_month_end": workdays_remaining,
        "is_first_workday_of_month": int(bool(workdays) and date == workdays[0]),
        "is_last_workday_of_month": int(bool(workdays) and date == workdays[-1]),
    }


CALENDAR_FEATURE_COLUMNS = set(calendar_feature_dict("2014-01-01")) | {
    "days_to_next_holiday",
    "days_from_last_holiday",
    "is_day_before_holiday",
    "is_day_after_holiday",
}
