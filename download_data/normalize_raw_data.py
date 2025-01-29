import xarray as xr

# Load the raw and piControl data (Historical)
raw = xr.open_dataset("/data/temp/outputs_historical_daily_raw.nc")
piControl = xr.open_dataset("/data/CESM2_piControl_r1i1p1f1_climatology_daily.nc")


# Normalize the raw data by subtracting the piControl data
raw = raw.assign_coords(dayofyear=raw['time'].dt.dayofyear)
piControl = piControl.squeeze()
piControl_expanded = piControl.sel(dayofyear=raw['dayofyear'])
normalized = raw - piControl_expanded

normalized.to_netcdf("/data/outputs_historical_daily.nc")