import cftime
import xarray as xr


NO_DAYS_IN_YEAR = 365
NO_DAYS_IN_MONTH = 30


def timeInterpolateStart(input_tensor, dt):
    """
    Interpolate yearly climate data to daily resolution by linearly interpolating from the start of the year.

    Note: E = Jan 1st; ==> (364/365)*E_2009 + (1/365)*E_2010 for July 3rd 2009 formula
        For module climatebench_daily
    """
    emmission_type = ...
    E = input_tensor[emmission_type]
    E_curr = E.sel(time=dt.year)

    # calculate the number of days from the start of the year
    days_from_start = dt.timetuple().tm_yday - 1

    # For dates after January 1st
    if dt.month > 1 or (dt.month == 1 and dt.day > 1):
        E_prev = E.sel(time=dt.year - 1)
        output = (
            NO_DAYS_IN_YEAR - days_from_start
        ) / NO_DAYS_IN_YEAR * E_curr + days_from_start / NO_DAYS_IN_YEAR * E_prev
    else:
        # For January 1st or earlier
        E_next = E.sel(time=dt.year + 1)
        output = (
            NO_DAYS_IN_YEAR - days_from_start
        ) / NO_DAYS_IN_YEAR * E_curr + days_from_start / NO_DAYS_IN_YEAR * E_next

    return output


def timeInterpolateMiddle(input_tensor, dt, yearly_reso_idx):
    """
    Interpolate yearly climate data to daily resolution by linearly interpolating from the middle of the year.

    Note: E = July 2nd; ==> (364/365)*E_2009 + (1/365)*E_2010 for July 3rd 2009 formula
        For module climatebench_daily
    """
    E_curr = input_tensor[yearly_reso_idx]  # Getting the tensor for the current year using the index

    # Passed July 2nd
    if dt.month > 7 or (dt.month == 7 and dt.day > 2):
        # calculate the number of days from the middle of the year
        days_from_middle = 183 - dt.timetuple().tm_yday
        E_next = input_tensor[yearly_reso_idx + 1]

        inputs = (
            NO_DAYS_IN_YEAR - days_from_middle
        ) / NO_DAYS_IN_YEAR * E_curr + days_from_middle / NO_DAYS_IN_YEAR * E_next
    else:
        # calculate the number of days from the middle of the year
        days_from_middle = dt.timetuple().tm_yday - 183
        E_prev = input_tensor[yearly_reso_idx - 1]

        inputs = (
            NO_DAYS_IN_YEAR - days_from_middle
        ) / NO_DAYS_IN_YEAR * E_prev + days_from_middle / NO_DAYS_IN_YEAR * E_curr

    return inputs


def timeInterpolateMiddleModified(
    input_xr: xr.Dataset, ssp_index_datetime: cftime.datetime, no_prev_year: bool = False, no_next_year: bool = False
) -> xr.Dataset:
    """
    Interpolate yearly climate data to daily resolution by linearly interpolating from the middle of the year.

    Note: E = July 2nd; ==> (364/365)*E_2009 + (1/365)*E_2010 for July 3rd 2009 formula
        For module climatebench_daily_modified

    Args:
     - input_xr: xarray.Dataset
     - ssp_index_datetime: cftime.DatetimeNoLeap
     # Flags to indicate if there is no previous or next year
        - no_prev_year: bool
        - no_next_year: bool

    Returns:
        - interpolated_values: xarray.Dataset
    """
    DAY = ssp_index_datetime.day
    MONTH = ssp_index_datetime.month
    YEAR = ssp_index_datetime.year
    JULY_2ND = cftime.DatetimeNoLeap(YEAR, 7, 2)
    DAYS_FROM_MIDDLE = (ssp_index_datetime - JULY_2ND).days

    try:
        # Get the values for the current year
        E_curr = input_xr.sel(time=YEAR)
    except Exception:
        # Use the nearest year if the current year is not available
        # Due to the lone 2101 in output, we can just use the 2100 input data
        E_curr = input_xr.sel(time=YEAR, method="nearest")

    # Passed July 2nd
    if MONTH > 7 or (MONTH == 7 and DAY > 2):
        try:
            # Get the values for the next year
            # NOTE: Interpolates with same year if there is no previous year
            E_next = input_xr.sel(time=YEAR + 1) if not no_next_year else input_xr.sel(time=YEAR)
        except Exception:
            # Get the values using the same year
            E_next = input_xr.sel(time=YEAR, method="nearest")  # Need to figure out why YEAR became 2101
        # Calculate the interpolated values
        interpolated_values = (
            NO_DAYS_IN_YEAR - DAYS_FROM_MIDDLE
        ) / NO_DAYS_IN_YEAR * E_curr + DAYS_FROM_MIDDLE / NO_DAYS_IN_YEAR * E_next
    else:
        try:
            # Get the values for the previous year
            # NOTE: Interpolates with same year if there is no next year
            E_prev = input_xr.sel(time=YEAR - 1) if not no_prev_year else input_xr.sel(time=YEAR)
        except Exception:
            # Use the same year for interpolation
            E_prev = input_xr.sel(time=YEAR, method="nearest")

        # Calculate the interpolated values
        interpolated_values = (
            NO_DAYS_IN_YEAR - DAYS_FROM_MIDDLE
        ) / NO_DAYS_IN_YEAR * E_prev + DAYS_FROM_MIDDLE / NO_DAYS_IN_YEAR * E_curr

    return interpolated_values


def timeInterpolateMonthly(input_xr: xr.Dataset, index_datetime: cftime.datetime) -> xr.Dataset:
    """
    Interpolate yearly climate data to daily resolution by linearly interpolating from the middle of the month.

    Note: For module climatebench_daily_modified/Middle of month set set 15th as the middle of the month

    Args:
        - input_xr: xarray.Dataset
        - index_datetime: cftime.DatetimeNoLeap
        # Flags to indicate if there is no previous or next month
        - no_prev_month: bool
        - no_next_month: bool
    """
    DAY = index_datetime.day
    MONTH = index_datetime.month
    YEAR = index_datetime.year
    MIDDLE_OF_MONTH = cftime.DatetimeNoLeap(YEAR, MONTH, 15)
    DAYS_FROM_MIDDLE = (index_datetime - MIDDLE_OF_MONTH).days

    try:
        # Get the values for the current month
        curr = input_xr.sel(time=f"{YEAR}-{MONTH:02d}")
    except Exception:
        # Use the nearest month if the current month is not available
        # Due to the lone 2101 in output, we can just use the 2100 input data
        try:
            if YEAR == 2101:
                curr = input_xr.sel(time="2100-12")
            else:
                curr = input_xr.sel(time=f"{YEAR-1}-{MONTH:02d}")
        except Exception:
            curr = input_xr.sel(time="2100-12")  # Need to figure out why YEAR became 2101
            print("Error occured during interpolation")
            print(f"Year: {YEAR}, Month: {MONTH}")

    # Interpolate the values
    if DAY > 15:
        try:
            # Calculate next month handling the edge case of December
            next_month_string = f"{YEAR}-{MONTH + 1:02d}" if MONTH < 12 else f"{YEAR + 1}-{1:02d}"
            # Get the values for the next month
            next_month = input_xr.sel(time=next_month_string)
        except Exception:
            # Use the same month for interpolation if the next month is not available (i.e. December 2100)
            next_month = input_xr.sel(time=f"{YEAR}-{MONTH:02d}")

        interpolated_values = ((NO_DAYS_IN_MONTH - DAYS_FROM_MIDDLE) / NO_DAYS_IN_MONTH * curr).squeeze() + (
            DAYS_FROM_MIDDLE / NO_DAYS_IN_MONTH * next_month
        ).squeeze()
    else:
        try:
            # Calculate previous month handling the edge case of January
            prev_month_string = f"{YEAR}-{MONTH - 1:02d}" if MONTH > 1 else f"{YEAR - 1}-{12:02d}"
            # Get the values for the previous month
            prev_month = input_xr.sel(time=prev_month_string)
        except Exception:
            # Use the same month for interpolation if the previous month is not available (i.e. January 2015)
            prev_month = input_xr.sel(time=f"{YEAR}-{MONTH:02d}")

        interpolated_values = ((NO_DAYS_IN_MONTH - DAYS_FROM_MIDDLE) / NO_DAYS_IN_MONTH * prev_month).squeeze() + (
            DAYS_FROM_MIDDLE / NO_DAYS_IN_MONTH * curr
        ).squeeze()

    return interpolated_values
