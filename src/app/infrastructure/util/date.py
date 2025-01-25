""" Date(time) related utils """
from datetime import date, timedelta


def daterange(start_date: date, end_date: date, include_end_date: bool=True):
    if include_end_date:
        days = int((end_date + timedelta(days=1) - start_date).days)
    else:
        days = int((end_date - start_date).days)
    for n in range(days):
        yield start_date + timedelta(n)
