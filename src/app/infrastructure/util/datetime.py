""" Date(time) related utils """
from datetime import date, datetime, time, timedelta


def daterange(start_date: date, end_date: date, include_end_date: bool=True):
    if include_end_date:
        days = int((end_date + timedelta(days=1) - start_date).days)
    else:
        days = int((end_date - start_date).days)
    for n in range(days):
        yield start_date + timedelta(n)


def to_seconds(dt):
    """ Convert datetime to seconds since epoch """
    if isinstance(dt, datetime):
        return int(dt.timestamp())
    elif isinstance(dt, date):
        return int(datetime.combine(dt, time(hour=12)).timestamp())  # Assume noon for wildfires
    return 0

