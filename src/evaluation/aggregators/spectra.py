from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch
import xarray as xr

from src.utilities.spectra import ZonalEnergySpectrum
from src.utilities.utils import torch_to_numpy


class SpectraAggregator:
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
    ):
        self._n_batches = 0
        self.is_ensemble = is_ensemble
        self.var_names = var_names
        self.spectra_type = spectra_type
        assert "zonal" in spectra_type  # , "meridional", "basic"]
        if "zonal" in spectra_type:
            _, subsel_spectra_l, subsel_spectra_r = spectra_type.split("_")
            self._subsel_spectra = {"latitude": slice(int(subsel_spectra_l), int(subsel_spectra_r))}

        assert data_to_log in ["preds", "targets"]
        self.data_to_log = data_to_log
        self._running_spectra = dict()
        self._data_coords = coords
        self.dims = None
        if coords is not None:
            assert "latitude" in coords, "data_coords must contain 'latitude'"
            assert "longitude" in coords, "data_coords must contain 'longitude'"

    @torch.inference_mode()
    def record_batch(
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

        names = self.var_names if self.var_names is not None else gen_data.keys()
        for i, name in enumerate(names):
            spectra_compute_class = ZonalEnergySpectrum(variable_name=name)
            # Map gen_data to xarray
            self.dims = ("batch", "longitude", "latitude")
            if self.is_ensemble:
                self.dims = ("ensemble",) + self.dims

            data_xr = xr.Dataset({name: xr.DataArray(data[name], dims=self.dims, coords=self._data_coords)})

            # Compute spectra
            spectra = spectra_compute_class.compute(data_xr.load()).sel(**self._subsel_spectra)
            if name not in self._running_spectra.keys():
                self._running_spectra[name] = spectra.sum(dim="batch")
            else:
                self._running_spectra[name] += spectra.sum(dim="batch")

        self._n_batches += spectra.sizes["batch"]

    @torch.inference_mode()
    def get_logs(self, label: str = "", epoch: Optional[int] = None) -> Dict[str, float]:
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
            epoch: Current epoch number.
        """
        label = label + "/" if label else ""
        mean_dims = ["latitude"] if not self.is_ensemble else ["ensemble", "latitude"]
        logs = dict(x_axes=["wavelength", "wavenumber"])
        for name, spectra_v in self._running_spectra.items():
            log_key = f"{label}{name}/spectrum".rstrip("/")
            spectra_v /= self._n_batches
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
