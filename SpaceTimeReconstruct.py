"""
st_filter.py
============

Space–time (k–ω) band-pass filtering using equatorial wave dispersion relations.

This module implements a **Wheeler–Kiladis–style space–time filter**
with dispersion-curve constraints from the Matsuno (1966) equatorial
shallow-water wave theory.

The implementation follows **exactly** the FFT conventions used in the
original user script:

Forward transform
-----------------
1. FFT over time axis
2. IFFT over longitude axis (scaled by nlon)
3. Sum over latitude

Inverse transform
-----------------
1. IFFT over time
2. FFT over longitude (scaled by 1/nlon)

This design allows direct reproduction of common Kelvin / Rossby / IG
wave reconstructions used in tropical dynamics studies.

Dependencies
------------
- numpy
- DispersionRelation.EquatorialWaveDispersion
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Sequence, Tuple, List, Optional, Dict, Any, Literal

from DispersionRelation import EquatorialWaveDispersion


WaveType = Literal["Kelvin", "MRG", "IG0", "IG1", "IG2", "ER1"]


@dataclass(frozen=True)
class DispersionParams:
    """
    Parameters controlling the equatorial wave dispersion relation.

    Attributes
    ----------
    n_planetary_wave : int
        Total number of zonal wavenumbers used in the dispersion solver.
        Typically equals the longitude FFT length.
    rlat : float
        Reference latitude (radians). Use 0.0 for equatorial waves.
    equivalent_depth : tuple of float
        List of equivalent depths (meters). Multiple values define an
        envelope of dispersion curves.
    s_min, s_max : int
        Zonal wavenumber index range used internally by the solver.
    """
    n_planetary_wave: int = 576
    rlat: float = 0.0
    equivalent_depth: Tuple[float, ...] = (8.0, 90.0)
    s_min: int = -288
    s_max: int = 287


@dataclass(frozen=True)
class BandpassParams:
    """
    Parameters defining the k–ω band-pass filter.

    Attributes
    ----------
    k_range : (int, int)
        Zonal wavenumber range (|k| limits).
    f_range : (float, float)
        Frequency range (cycles per unit time).
    nan_to_inf : bool
        If True, NaNs in dispersion curves are treated as unbounded
        (±∞), preventing artificial masking.
    """
    k_range: Tuple[int, int] = (1, 14)
    f_range: Tuple[float, float] = (1 / 30, 1 / 2.5)
    nan_to_inf: bool = True


class SpaceTimeFilter:
    """
    Space–time band-pass filter constrained by equatorial wave dispersion.

    This class wraps a **fully working Kelvin / equatorial wave filtering
    script** into a reusable, object-oriented interface without changing
    numerical behavior.

    Input data format
    -----------------
    data : ndarray, shape (time, lat, lon)

    Output format
    -------------
    reconstructed field : ndarray, shape (time, lon)

    Notes
    -----
    - Latitude is **summed**, not averaged.
    - Filtering is applied in (frequency, zonal-wavenumber) space.
    - Eastward and westward branches are handled symmetrically.
    """

    SUPPORTED_WAVES = ("Kelvin", "MRG", "IG0", "IG1", "IG2", "ER1")

    def __init__(
        self,
        *,
        dispersion: DispersionParams = DispersionParams(),
        bandpass: BandpassParams = BandpassParams(),
    ):
        """
        Initialize the space–time filter.

        Parameters
        ----------
        dispersion : DispersionParams
            Parameters controlling the dispersion relation solver.
        bandpass : BandpassParams
            Parameters defining the k–ω filtering window.
        """
        self.dispersion = dispersion
        self.bandpass = bandpass

        self._ewd: Optional[EquatorialWaveDispersion] = None
        self._disp_cache: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Dispersion handling
    # ------------------------------------------------------------------
    def _acquire_ewd(self) -> EquatorialWaveDispersion:
        """
        Lazily create and cache the EquatorialWaveDispersion object.

        Returns
        -------
        EquatorialWaveDispersion
            Configured dispersion solver.
        """
        if self._ewd is None:
            p = self.dispersion
            self._ewd = EquatorialWaveDispersion(
                nPlanetaryWave=p.n_planetary_wave,
                rlat=p.rlat,
                Ahe=list(p.equivalent_depth),
                s_min=p.s_min,
                s_max=p.s_max,
            )
        return self._ewd

    def list_waves(self) -> List[str]:
        """
        List available equatorial wave types.

        Returns
        -------
        list of str
            Names of supported wave modes.
        """
        return list(self._acquire_ewd().list_waves())

    def get_dispersion(self, wave_list: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract dispersion curves for selected waves.

        Parameters
        ----------
        wave_list : sequence of str
            Names of waves to extract (e.g. ["Kelvin"]).

        Returns
        -------
        freq : ndarray
            Dispersion frequencies. Shape depends on number of equivalent
            depths and wave types.
        wnum : ndarray
            Corresponding zonal wavenumbers.
        """
        if not isinstance(wave_list, (list, tuple)):
            raise ValueError("wave_list must be a list or tuple")

        ewd = self._acquire_ewd()
        available = ewd.list_waves()

        for w in wave_list:
            if w not in available:
                raise ValueError(f"Wave '{w}' not available")

        Afreq, Awnum, wave_name = ewd.compute(waves="all")

        freq = []
        wnum = []
        for w in wave_list:
            idx = wave_name.index(w)
            freq.append(np.asarray(Afreq[idx]))
            wnum.append(np.asarray(Awnum[idx]))

        return np.array(freq).squeeze(), np.array(wnum).squeeze()

    # ------------------------------------------------------------------
    # FFT / IFFT
    # ------------------------------------------------------------------
    @staticmethod
    def fft2_lat_sum(data: np.ndarray) -> np.ndarray:
        """
        Forward space–time transform.

        Parameters
        ----------
        data : ndarray, shape (time, lat, lon)

        Returns
        -------
        spectrum : ndarray, shape (time, lon)
            Complex space–time spectrum summed over latitude.
        """
        if data.ndim != 3:
            raise ValueError("data must have shape (time, lat, lon)")

        nlon = data.shape[2]
        spec = np.fft.fft(data, axis=0)
        spec = np.fft.ifft(spec, axis=2) * nlon
        return spec.sum(axis=1)

    @staticmethod
    def ifft2_time_lon(spec: np.ndarray) -> np.ndarray:
        """
        Inverse space–time transform.

        Parameters
        ----------
        spec : ndarray, shape (time, lon)

        Returns
        -------
        recon : ndarray, shape (time, lon)
            Real-valued reconstructed signal.
        """
        if spec.ndim != 2:
            raise ValueError("spec must have shape (time, lon)")

        nlon = spec.shape[-1]
        out = np.fft.ifft(spec, axis=0)
        out = np.fft.fft(out, axis=-1) / nlon
        return out.real

    # ------------------------------------------------------------------
    # Mask
    # ------------------------------------------------------------------
    def make_mask(
        self,
        *,
        fr_low: np.ndarray,
        fr_high: np.ndarray,
        fr_grid: np.ndarray,
        wn_grid: np.ndarray,
        k_range: Optional[Tuple[int, int]] = None,
        f_range: Optional[Tuple[float, float]] = None,
        nan_to_inf: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Construct a binary k–ω mask constrained by dispersion curves.

        Parameters
        ----------
        fr_low, fr_high : ndarray
            Lower and upper frequency bounds from dispersion curves.
        fr_grid, wn_grid : ndarray
            2-D frequency and wavenumber grids.
        k_range, f_range : tuple, optional
            Override default band-pass limits.
        nan_to_inf : bool, optional
            How to treat NaNs in dispersion curves.

        Returns
        -------
        mask : ndarray of int
            Binary mask (1 = keep, 0 = discard).
        """
        if fr_grid.shape != wn_grid.shape:
            raise ValueError("fr_grid and wn_grid must have same shape")

        if k_range is None:
            k_range = self.bandpass.k_range
        if f_range is None:
            f_range = self.bandpass.f_range
        if nan_to_inf is None:
            nan_to_inf = self.bandpass.nan_to_inf

        kmin, kmax = k_range
        fmin, fmax = f_range

        if nan_to_inf:
            fr_low = np.where(np.isnan(fr_low), -np.inf, fr_low)
            fr_high = np.where(np.isnan(fr_high), +np.inf, fr_high)

        mask = np.where(
            (
                (wn_grid >= kmin) & (wn_grid <= kmax) &
                (fr_grid >= fmin) & (fr_grid <= fmax) &
                (fr_grid >= fr_low[None, :]) & (fr_grid <= fr_high[None, :])
            ) |
            (
                (wn_grid <= -kmin) & (wn_grid >= -kmax) &
                (fr_grid <= -fmin) & (fr_grid >= -fmax) &
                (fr_grid <= -fr_low[::-1][None, :]) & (fr_grid >= -fr_high[::-1][None, :])
            ),
            1, 0
        )

        return mask.astype(np.int8)

    # ------------------------------------------------------------------
    # End-to-end API
    # ------------------------------------------------------------------
    def compute(
        self,
        *,
        data: np.ndarray,
        fr_grid: np.ndarray,
        wn_grid: np.ndarray,
        wave_type: WaveType = "Kelvin",
        return_mask: bool = False,
        return_spectrum: bool = False,
    ):
        """
        Perform space–time filtering and reconstruction.

        Parameters
        ----------
        data : ndarray, shape (time, lat, lon)
            Input field.
        fr_grid, wn_grid : ndarray
            Frequency and wavenumber grids.
        wave_type : str
            Equatorial wave type (default: Kelvin).
        return_mask : bool
            If True, return the binary mask.
        return_spectrum : bool
            If True, also return original and masked spectra.

        Returns
        -------
        recon : ndarray
            Reconstructed physical-space signal.
        """
        freq, _ = self.get_dispersion([wave_type])
        fr_low, fr_high = freq[0], freq[1]

        spec = self.fft2_lat_sum(data)
        mask = self.make_mask(
            fr_low=fr_low,
            fr_high=fr_high,
            fr_grid=fr_grid,
            wn_grid=wn_grid,
        )
        masked = spec * mask
        recon = self.ifft2_time_lon(masked)

        outputs = [recon]
        if return_mask:
            outputs.append(mask)
        if return_spectrum:
            outputs.extend([spec, masked])

        return outputs[0] if len(outputs) == 1 else tuple(outputs)
