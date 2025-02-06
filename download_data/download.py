import os

os.environ["S3_ENDPOINT_URL"] = "https://rosedata.ucsd.edu"
os.environ["S3_BUCKET_NAME"] = "climatebench"
from src.utilities.s3utils import download_s3_object

# -------------- Edit this to the directory where you'd like the data to be in --------------
local_data_dir = "/data"
# ---------------------------------------
files = [
    # Outputs daily
    # "outputs_ssp126_daily.nc",
    # "outputs_ssp245_daily.nc",
    # "outputs_ssp370_daily.nc",
    # "outputs_ssp585_daily.nc",
    # # PI Control
    # "CESM2_piControl_r1i1p1f1_climatology_daily.nc",
    # # Inputs
    # "inputs_ssp126.nc",
    # "inputs_ssp245.nc",
    # "inputs_ssp370.nc",
    # "inputs_ssp585.nc",
    # "inputs_historical.nc",
    # # RSDT
    # "CESM2-rsdt-Amon-gn-ssp126.nc",
    # "CESM2-rsdt-Amon-gn-historical.nc",
    # "CESM2-rsdt-Amon-gn-piControl.nc",
    # # Raw data daily
    # "outputs_ssp126_daily_raw.nc",
    # "outputs_ssp245_daily_raw.nc",
    # "outputs_ssp370_daily_raw.nc",
    # "outputs_ssp585_daily_raw.nc",
    "outputs_historical_daily_raw.nc",
]


s3_data_dir = "data/"

# Ensure local data directory exists
if not os.path.exists(local_data_dir):
    os.makedirs(local_data_dir, exist_ok=True)

for file in files:
    print(f"Downloading {file}")
    s3_file_path = os.path.join(s3_data_dir, file)
    local_file_path = os.path.join(local_data_dir, file)
    download_s3_object(local_file_path=local_file_path, s3_file_path=s3_file_path)