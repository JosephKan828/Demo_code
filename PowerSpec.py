import numpy as np
from scipy.signal import detrend
from scipy.ndimage import convolve1d


class EquatorialSpectrumAnalyzer:
    """
    Compute symmetric/asymmetric equatorial space–time spectra
    and remove a smooth background.

    Workflow (equivalent to your original `output` function):
    1. Decompose data into symmetric and asymmetric parts about the equator.
    2. Apply windowing with overlap and Hanning taper in time.
    3. Compute FFT-based power spectra for each window and average.
    4. Sum over latitude, shift wavenumbers/frequencies, and keep positive frequencies.
    5. Estimate a smooth background spectrum via iterative 1D convolutions.
    6. Return raw spectra and spectra with background removed.

    Parameters
    ----------
    n_window : int
        Window length in time samples (default: 96).
    n_overlap : int
        Number of overlapping samples between consecutive windows (default: 48).
    kernel : np.ndarray, optional
        1D kernel used for background smoothing (default: [1, 2, 1]/4).
    f_running : int
        Number of smoothing passes along frequency axis (axis=0) (default: 10).
    low_k_running : int
        Number of smoothing passes along low wavenumbers (default: 10).
    high_k_running : int
        Number of smoothing passes along high wavenumbers (default: 40).
    """

    def __init__(
        self,
        n_window: int = 96,
        n_overlap: int = 48,
        kernel: np.ndarray | None = None,
        f_running: int = 10,
        low_k_running: int = 10,
        high_k_running: int = 40,
    ):
        self.n_window = n_window
        self.n_overlap = n_overlap

        if kernel is None:
            kernel = np.array([1, 2, 1], dtype=float) / 4.0
        self.kernel = kernel

        self.f_running = f_running
        self.low_k_running = low_k_running
        self.high_k_running = high_k_running

    # -----------------------------
    # Public API
    # -----------------------------
    def compute(self, data: np.ndarray):
        """
        Compute symmetric and asymmetric power spectra and
        their background-removed versions.

        This is the class equivalent of the original `output` function.

        Parameters
        ----------
        data : np.ndarray
            Input data with shape (time, lat, lon).

        Returns
        -------
        symm_ps : np.ndarray
            Symmetric power spectrum, shape (n_freq_pos, n_wavenumber).
        asym_ps : np.ndarray
            Asymmetric power spectrum, shape (n_freq_pos, n_wavenumber).
        symm_ps_rm_bg : np.ndarray
            Symmetric power spectrum divided by background, same shape as symm_ps.
        asym_ps_rm_bg : np.ndarray
            Asymmetric power spectrum divided by background, same shape as asym_ps.
        wn : np.ndarray
            Zonal wavenumbers, length n_wavenumber.
        fr_pos : np.ndarray
            Positive frequencies corresponding to the returned spectra.
        """
        if data.ndim != 3:
            raise ValueError("data must have shape (time, lat, lon)")

        n_time, n_lat, n_lon = data.shape

        # 1) Symmetric / asymmetric decomposition
        symm, asym = self._symm_asym(data)

        # 2) Windowing in time
        symm_chunk = self._windowing(symm, self.n_window, self.n_overlap)
        asym_chunk = self._windowing(asym, self.n_window, self.n_overlap)

        # 3) Compute spectra for each chunk
        symm_ps_chunks = []
        asym_ps_chunks = []

        for i in range(symm_chunk.shape[0]):
            symm_ps_chunks.append(self._compute_spectrum(symm_chunk[i]))
            asym_ps_chunks.append(self._compute_spectrum(asym_chunk[i]))

        symm_ps_chunks = np.stack(symm_ps_chunks, axis=0)  # (n_chunk, f, lat, lon)
        asym_ps_chunks = np.stack(asym_ps_chunks, axis=0)

        # 4) Average over chunks, sum over latitude, shift, and keep positive frequencies
        # Shape after mean over chunks: (f, lat, lon)
        symm_ps_mean = symm_ps_chunks.mean(axis=0).real  # (f, lat, lon)
        asym_ps_mean = asym_ps_chunks.mean(axis=0).real

        # Sum over latitude -> (f, lon)
        symm_ps_lat_sum = symm_ps_mean.sum(axis=1)
        asym_ps_lat_sum = asym_ps_mean.sum(axis=1)

        # Frequency / wavenumber arrays
        wn = np.fft.fftshift(np.fft.fftfreq(n_lon, d=1.0 / n_lon))
        fr = np.fft.fftshift(np.fft.fftfreq(self.n_window, d=1.0))

        # Shift spectra in wavenumber and frequency
        symm_ps_shift = np.fft.fftshift(symm_ps_lat_sum)
        asym_ps_shift = np.fft.fftshift(asym_ps_lat_sum)

        # Keep positive frequencies only and apply *2.0 like original code
        positive = fr > 0
        fr_pos = fr[positive]
        symm_ps = symm_ps_shift[positive] * 2.0  # (n_freq_pos, n_wavenumber)
        asym_ps = asym_ps_shift[positive] * 2.0

        # 5) Background spectrum from average of symm/asym
        bg_input = (symm_ps + asym_ps) / 2.0  # shape (n_freq_pos, n_wavenumber)
        bg_ps = self._background(bg_input)

        # 6) Background-removed spectra
        symm_ps_rm_bg = symm_ps / bg_ps
        asym_ps_rm_bg = asym_ps / bg_ps

        return symm_ps, asym_ps, symm_ps_rm_bg, asym_ps_rm_bg, wn, fr_pos

    # -----------------------------
    # Internal helpers
    # -----------------------------
    @staticmethod
    def _symm_asym(data: np.ndarray):
        """
        Symmetric/asymmetric decomposition about the equator (lat dimension).
        """
        if data.ndim != 3:
            raise ValueError("data must have shape (time, lat, lon)")

        symm = (data + np.flip(data, axis=1)) / 2.0
        asym = (data - np.flip(data, axis=1)) / 2.0
        return symm, asym

    @staticmethod
    def _windowing(data: np.ndarray, n_window: int, n_overlap: int):
        """
        Apply temporal windowing with overlap and Hanning taper.

        Parameters
        ----------
        data : np.ndarray
            Shape (time, lat, lon).
        n_window : int
            Window length in time.
        n_overlap : int
            Overlap between windows.

        Returns
        -------
        chunked_data : np.ndarray
            Shape (n_chunk, n_window, lat, lon).
        """
        if data.ndim != 3:
            raise ValueError("data must have shape (time, lat, lon)")

        t_len = data.shape[0]
        if n_window > t_len:
            raise ValueError("n_window cannot be larger than the time dimension")

        hanning = np.hanning(n_window)[:, None, None]
        chunks = []

        # Keep your original logic: step = n_overlap, number of chunks = t_len // n_window
        for i in range(t_len // n_window):
            start = i * n_overlap
            end = start + n_window
            if end > t_len:
                break

            chunk = detrend(data[start:end], axis=0) * hanning
            chunks.append(chunk)

        if not chunks:
            raise ValueError("No chunks were generated; check n_window and n_overlap.")

        return np.stack(chunks, axis=0)

    @staticmethod
    def _compute_spectrum(data: np.ndarray):
        """
        Compute power spectrum for a single (windowed) segment.

        Parameters
        ----------
        data : np.ndarray
            Shape (time, lat, lon) for a single chunk.

        Returns
        -------
        ps : np.ndarray
            Power spectrum, shape (freq, lat, lon).
        """
        if data.ndim != 3:
            raise ValueError("data must have shape (time, lat, lon)")

        n_time, _, n_lon = data.shape

        data_fft = np.fft.fft(data, axis=0)           # FFT over time
        data_fft = np.fft.ifft(data_fft, axis=2) * n_lon  # IFFT over lon, scaled

        ps = (data_fft * data_fft.conj()) / (n_time * n_lon) ** 2.0
        return ps

    def _background(self, data: np.ndarray):
        """
        Compute smoothed background spectrum via iterative 1D convolutions.

        Parameters
        ----------
        data : np.ndarray
            Typically 2D spectrum (freq, wavenumber) but can be 3D as well.

        Returns
        -------
        bg : np.ndarray
            Smoothed background, same shape as `data`.
        """
        data = data.copy()
        kernel = self.kernel
        half_freq = data.shape[0] // 2

        # Smooth along frequency axis (axis=0)
        for _ in range(self.f_running):
            data = convolve1d(data, kernel, axis=0, mode="reflect")

        # Smooth low wavenumbers (axis=1) for lower half of freq
        for _ in range(self.low_k_running):
            data[:half_freq] = convolve1d(
                data[:half_freq], kernel, axis=1, mode="reflect"
            )

        # Smooth high wavenumbers (axis=1) for upper half of freq
        for _ in range(self.high_k_running):
            data[half_freq:] = convolve1d(
                data[half_freq:], kernel, axis=1, mode="reflect"
            )

        return data
