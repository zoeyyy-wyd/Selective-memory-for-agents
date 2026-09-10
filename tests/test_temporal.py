from datetime import datetime

from smem.temporal import Interval, parse_session_date, parse_temporal

NOW = datetime(2023, 6, 10, 12, 0)


def test_now_words():
    c = parse_temporal("Which city does the user live in now?", NOW)
    assert c.mode == "now" and c.interval.start == NOW


def test_month_without_year_picks_most_recent_past_occurrence():
    c = parse_temporal("Where did I live in March?", NOW)
    assert c.mode == "interval"
    assert c.interval.start == datetime(2023, 3, 1) and c.interval.end.month == 3 and c.interval.end.day == 31


def test_month_later_than_now_rolls_back_a_year():
    c = parse_temporal("What did I do in September?", NOW)
    assert c.interval.start.year == 2022


def test_change_questions_return_whole_chain():
    assert parse_temporal("How many times did I move?", NOW).mode == "whole_chain"
    assert parse_temporal("Has my job changed over time?", NOW).mode == "whole_chain"


def test_ago_and_last_unit():
    c = parse_temporal("What was I doing two weeks ago?", NOW)
    assert c.mode == "interval" and c.interval.start < NOW
    c = parse_temporal("What did I buy last month?", NOW)
    assert c.interval.start == datetime(2023, 5, 1)


def test_before_latest_and_none():
    assert parse_temporal("Where did I live previously?", NOW).mode == "before_latest"
    assert parse_temporal("What is my allergy?", NOW).mode == "none"


def test_interval_intersects_half_open_validity():
    march = Interval(datetime(2023, 3, 1), datetime(2023, 3, 31, 23, 59, 59))
    assert march.intersects(datetime(2023, 2, 4), datetime(2023, 5, 19))
    assert not march.intersects(datetime(2023, 5, 19), None)
    assert march.intersects(datetime(2023, 3, 31), None)
    assert not march.intersects(datetime(2023, 1, 1), datetime(2023, 3, 1))


def test_parse_session_date_formats():
    assert parse_session_date("2023/05/20 (Sat) 02:21") == datetime(2023, 5, 20, 2, 21)
    assert parse_session_date("2023-05-20") == datetime(2023, 5, 20)
