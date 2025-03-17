from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch
import xarray as xr

from src.evaluation.torchmetrics import Metric
from src.utilities.spectra import ZonalEnergySpectrum
from src.utilities.utils import extract_xarray_metadata, reconstruct_xarray, torch_to_numpy


class SpectraAggregator(Metric):
    """
    Aggregator for spectra metrics.
    """

    def __init__(
        self,
        is_ensemble: bool,
        spectra_type: str = "zonal_60_90",  # "zonal" or "meridional" or "basic" (for non-earth data)
        var_names: Optional[List[str]] = None,
        coords: Optional[Dict[str, np.ndarray]] = None,
        data_to_log: str = "preds",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name = "spectra"
        self.is_ensemble = is_ensemble
        self.var_names = var_names
        self.spectra_type = spectra_type
        assert "zonal" in spectra_type  # , "meridional", "basic"]
        if "zonal" in spectra_type:
            _, subsel_spectra_l, subsel_spectra_r = spectra_type.split("_")
            self._subsel_spectra = {"latitude": slice(int(subsel_spectra_l), int(subsel_spectra_r))}

        assert data_to_log in ["preds", "targets"]
        self.data_to_log = data_to_log
        self._data_coords = coords
        self.dims = None
        self._spectra_vars_to_xr_metadata = dict()
        self.add_state("_n_batches", default=torch.tensor(0.0), dist_reduce_fx="sum")
        if coords is not None:
            assert "latitude" in coords, "data_coords must contain 'latitude'"
            assert "longitude" in coords, "data_coords must contain 'longitude'"

    def update_running_spectra(self, spectra: xr.DataArray, var_name: str):
        to_add_spectrum = spectra.sum(dim="batch")
        to_add_spectrum_tensor = torch.tensor(to_add_spectrum.values)
        if var_name not in self._spectra_vars_to_xr_metadata.keys():
            xr_metadata = extract_xarray_metadata(to_add_spectrum)
            self._spectra_vars_to_xr_metadata[var_name] = xr_metadata
            # Add state
            self.add_state(
                f"_running_spectra_{var_name}", default=torch.zeros_like(to_add_spectrum_tensor), dist_reduce_fx="sum"
            )
        # Update state
        self.__dict__[f"_running_spectra_{var_name}"] += to_add_spectrum_tensor

    def get_aggregated_spectrum(self, var_name: str) -> xr.DataArray:
        # Compute mean spectrum, should be called from compute() to get a proper DDP reduction
        running_spectrum = self.__dict__[f"_running_spectra_{var_name}"]
        running_spectrum = running_spectrum / self._n_batches
        running_spectrum = reconstruct_xarray(running_spectrum, self._spectra_vars_to_xr_metadata[var_name])
        return running_spectrum

    @torch.inference_mode()
    def update(
        self,
        target_data: Mapping[str, torch.Tensor],
        gen_data: Mapping[str, torch.Tensor],
        target_data_norm: Mapping[str, torch.Tensor] = None,
        gen_data_norm: Mapping[str, torch.Tensor] = None,
        metadata: Mapping[str, Any] = None,
    ):
        if self.data_to_log == "preds":
            data = gen_data
        elif self.data_to_log == "targets":
            print("Logging targets.")
            data = target_data
        else:
            raise ValueError(f"Unknown data_to_log: {self.data_to_log}")

        if torch.is_tensor(data):  # add dummy key
            data = {"": data}
        data = torch_to_numpy(data)

        names = self.var_names if self.var_names is not None else data.keys()
        for i, name in enumerate(names):
            spectra_compute_class = ZonalEnergySpectrum(variable_name=name)
            # Map gen_data to xarray
            self.dims = ("batch", "longitude", "latitude")
            if self.is_ensemble:
                self.dims = ("ensemble",) + self.dims

            data_xr = xr.Dataset({name: xr.DataArray(data[name], dims=self.dims, coords=self._data_coords)})

            # Compute spectra
            spectra = spectra_compute_class.compute(data_xr.load()).sel(**self._subsel_spectra)
            self.update_running_spectra(spectra, name)

        self._n_batches += torch.tensor(spectra.sizes["batch"])

    @torch.inference_mode()
    def compute(self, prefix: str = "", epoch: Optional[int] = None):
        """
        Returns logs as can be reported to WandB.

        Args:
            prefix: Label to prepend to all log keys.
            epoch: Current epoch number.
        """
        prefix = prefix + "/" if prefix else ""
        mean_dims = ["latitude"] if not self.is_ensemble else ["ensemble", "latitude"]
        logs = dict(x_axes=["wavelength", "wavenumber"])
        for name in self._spectra_vars_to_xr_metadata.keys():
            log_key = f"{prefix}{name}/spectrum".rstrip("/")
            spectra_v = self.get_aggregated_spectrum(name)
            mean_spectra = spectra_v.mean(dim=mean_dims)
            # Log the mean spectra for each wavelength and wavenumber separately (so that we can plot them as x-axis)
            for i, wavenumber in enumerate(spectra_v.zonal_wavenumber):
                wavenumber_float = float(wavenumber)
                if wavenumber_float not in logs.keys():
                    spectra_wave_length = float(spectra_v.wavelength.mean(dim="latitude")[i].values)
                    logs[wavenumber_float] = dict()
                    logs[wavenumber_float]["wavelength"] = spectra_wave_length
                    logs[wavenumber_float]["wavenumber"] = wavenumber_float
                logs[wavenumber_float][log_key] = float(mean_spectra.sel(zonal_wavenumber=wavenumber).values)
        return {}, {}, {"wavenumber": logs}
