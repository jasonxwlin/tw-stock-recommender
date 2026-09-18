#!/usr/bin/env python3
"""Self-check for the FOMC/PCE HTML parsers in tech_analysis.py (ponytail: non-trivial regex parsing)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import date
from tech_analysis import _parse_fomc_events, _parse_bea_events, _parse_month_day

FOMC_FIXTURE = """
<div class="panel panel-default"><div class="panel-heading"><h4><a id="1">2026 FOMC Meetings</a></h4></div>
<div class="row fomc-meeting" ">
    <div class="fomc-meeting__month col-xs-5"><strong>January</strong></div>
    <div class="fomc-meeting__date col-xs-4">27-28</div>
</div>
<div class="fomc-meeting--shaded row fomc-meeting" ">
    <div class="fomc-meeting--shaded fomc-meeting__month col-xs-5"><strong>March</strong></div>
    <div class="fomc-meeting__date col-xs-4">17-18*</div>
</div>
<div class="panel panel-default"><div class="panel-heading"><h4><a id="2">2027 FOMC Meetings</a></h4></div>
<div class="row fomc-meeting" ">
    <div class="fomc-meeting__month col-xs-5"><strong>January</strong></div>
    <div class="fomc-meeting__date col-xs-4">26-27</div>
</div>
"""

BEA_FIXTURE = """
<th id="x" scope="col">Year 2026</th>
<tr class="scheduled-releases-type-press">
    <td class="scheduled-date no-wrap"><div class="release-date">September 30</div></td>
    <td class="release-title views-field">Personal Income and Outlays, August 2026</td>
</tr>
<tr class="scheduled-releases-type-press">
    <td class="scheduled-date no-wrap"><div class="release-date">September 24</div></td>
    <td class="release-title views-field">GDP (Advance Estimate), 3rd Quarter 2026</td>
</tr>
<tr class="scheduled-releases-type-press">
    <td class="scheduled-date no-wrap"><div class="release-date">September 25</div></td>
    <td class="release-title views-field">GDP by County and Personal Income by County, 2025</td>
</tr>
<tr class="scheduled-releases-type-press">
    <td class="scheduled-date no-wrap"><div class="release-date">September 3</div></td>
    <td class="release-title views-field">U.S. International Trade in Goods and Services, July 2026</td>
</tr>
"""


def test_parse_month_day():
    assert _parse_month_day("January 28", 2026) == date(2026, 1, 28)
    assert _parse_month_day("garbage", 2026) is None


def test_parse_fomc_events_window():
    events = _parse_fomc_events(FOMC_FIXTURE, date(2026, 1, 20), date(2026, 1, 30))
    assert [e['date'] for e in events] == [date(2026, 1, 28)]

    events_wide = _parse_fomc_events(FOMC_FIXTURE, date(2026, 1, 1), date(2027, 1, 27))
    assert [e['date'] for e in events_wide] == [
        date(2026, 1, 28), date(2026, 3, 18), date(2027, 1, 27)
    ]


def test_parse_bea_events_filters_and_labels_rows():
    events = _parse_bea_events(BEA_FIXTURE, date(2026, 9, 1), date(2026, 10, 1))
    events.sort(key=lambda e: e['date'])
    assert [e['date'] for e in events] == [date(2026, 9, 24), date(2026, 9, 30)]
    assert [e['name'] for e in events] == ['美國GDP公布', '美國PCE物價指數公布']
