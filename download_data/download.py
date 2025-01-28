import os

os.environ["S3_ENDPOINT_URL"] = "https://rosedata.ucsd.edu"
os.environ["S3_BUCKET_NAME"] = "climatebench"
from src.utilities.s3utils import download_s3_object

# -------------- Edit this to the directory where you'd like the data to be in --------------
local_data_dir = "./data"
# ---------------------------------------
files = [
    "outputs_ssp126_daily_raw.nc",
    # "outputs_ssp245_daily_raw.nc",
    # "outputs_ssp370_daily_raw.nc",
    # "outputs_ssp585_daily_raw.nc",
    # "outputs_historical_daily_raw.nc",
]
s3_data_dir = "data/"

for file in files:
    s3_file_path = os.path.join(s3_data_dir, file)
    local_file_path = os.path.join(local_data_dir, file)
    download_s3_object(local_file_path=local_file_path, s3_file_path=s3_file_path)