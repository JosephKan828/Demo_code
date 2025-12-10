import numpy as np


class EquatorialWaveDispersion:
    """
    Compute dispersion curves of equatorial shallow-water waves
    following Matsuno (1966).

    Supported wave names
    --------------------
    "MRG"    : Mixed Rossby-Gravity (n=0, antisymmetric)
    "IG0"    : Inertial Gravity (n=0)
    "IG1"    : Inertial Gravity (n=1)
    "IG2"    : Inertial Gravity (n=2)
    "ER1"    : Equatorial Rossby (n=1)
    "Kelvin" : Equatorial Kelvin wave

    Parameters
    ----------
    nPlanetaryWave : int, optional
        Number of zonal wavenumber samples (s points).
    rlat : float, optional
        Latitude (radians). Use 0.0 for equator.
    Ahe : sequence of float, optional
        Equivalent depths [m] for the vertical modes.
    s_min, s_max : float, optional
        Minimum and maximum zonal wavenumber index s
        (dimensionless, used to generate the s-grid).

    Attributes
    ----------
    s : ndarray, shape (nPlanetaryWave,)
        Zonal wavenumber index (dimensionless).
    k : ndarray, shape (nPlanetaryWave,)
        Zonal wavenumber in rad/m.
    Ahe : ndarray, shape (nEquivDepth,)
        Equivalent depths [m].
    Beta : float
        Meridional gradient of Coriolis parameter at latitude rlat.
    """

    # ------------------------------------------------------------------
    #   Initialization
    # ------------------------------------------------------------------
    def __init__(
        self,
        nPlanetaryWave: int = 50,
        rlat: float = 0.0,
        Ahe=(50.0, 25.0, 12.0),
        s_min: float = -20.0,
        s_max: float = 20.0,
    ):
        # --- constants ---
        """
        Initialize the EquatorialWaveDispersion class.

        Parameters
        ----------
        nPlanetaryWave : int, optional
            Number of zonal wavenumber samples (s points).
        rlat : float, optional
            Latitude (radians). Use 0.0 for equator.
        Ahe : sequence of float, optional
            Equivalent depths [m] for the vertical modes.
        s_min, s_max : float, optional
            Minimum and maximum zonal wavenumber index s
            (dimensionless, used to generate the s-grid).

        Attributes
        ----------
        s : ndarray, shape (nPlanetaryWave,)
            Zonal wavenumber index (dimensionless).
        k : ndarray, shape (nPlanetaryWave,)
            Zonal wavenumber in rad/m.
        Ahe : ndarray, shape (nEquivDepth,)
            Equivalent depths [m].
        Beta : float
            Meridional gradient of Coriolis parameter at latitude rlat.
        """
        self.pi = np.pi
        self.radius = 6.37122e6        # [m]
        self.g = 9.80665               # [m s^-2]
        self.omega = 7.292e-5          # [s^-1]
        self.fillval = np.nan

        # --- parameters ---
        self.nPlanetaryWave = int(nPlanetaryWave)
        self.Ahe = np.atleast_1d(Ahe).astype(float)
        self.rlat = float(rlat)

        # --- planetary beta and Earth circumference at rlat ---
        self.Beta = 2.0 * self.omega * np.cos(abs(self.rlat)) / self.radius
        self.ll = 2.0 * self.pi * self.radius * np.cos(abs(self.rlat))

        # --- wavenumber grids ---
        self.s = np.linspace(s_min, s_max, self.nPlanetaryWave)
        self.k = 2.0 * self.pi * self.s / self.ll  # [rad m^-1]

        # --- registry of available wave types ---
        # Functions take (k, c, he) and return frequency in rad/s
        self._wave_registry = {
            "MRG":    self._disp_mrg,
            "IG0":    self._disp_ig0,
            "IG1":    lambda k, c, he: self._disp_ig_n(k, c, he, n=1),
            "IG2":    lambda k, c, he: self._disp_ig_n(k, c, he, n=2),
            "ER1":    lambda k, c, he: self._disp_er_n(k, c, n=1),
            "Kelvin": self._disp_kelvin,
        }

    # ------------------------------------------------------------------
    #   Public API
    # ------------------------------------------------------------------
    def list_waves(self):
        """Return a sorted list of available wave names."""
        return sorted(self._wave_registry.keys())

    def compute(self, waves="all"):
        """
        Compute dispersion curves for selected equatorial wave types.

        Parameters
        ----------
        waves : "all" or str or list of str, optional
            Which wave types to compute. Options are:
            "all" or any subset of:
                "MRG", "IG0", "IG1", "IG2", "ER1", "Kelvin".

        Returns
        -------
        Afreq : ndarray
            Frequency [cycles per day],
            shape (nWaveSel, nEquivDepth, nPlanetaryWave).
        Apzwn : ndarray
            Zonal wavenumber index s (dimensionless),
            shape (nWaveSel, nEquivDepth, nPlanetaryWave).
        wave_names : list of str
            Names of the waves in the same order as Afreq/Apzwn.
        """
        # --- normalize wave selection ---
        if waves == "all":
            wave_names = self.list_waves()
        elif isinstance(waves, str):
            wave_names = [waves]
        else:
            wave_names = list(waves)

        # validate
        for w in wave_names:
            if w not in self._wave_registry:
                raise ValueError(
                    f"Unknown wave name '{w}'. "
                    f"Available: {self.list_waves()}"
                )

        n_wave = len(wave_names)
        n_he = len(self.Ahe)
        n_s = self.nPlanetaryWave

        # allocate outputs
        Afreq = np.full((n_wave, n_he, n_s), self.fillval, dtype=float)
        Apzwn = np.broadcast_to(self.s, (n_wave, n_he, n_s))

        # constants for unit conversion
        two_pi = 2.0 * np.pi
        sec_per_day = 86400.0

        # --- main loop over wave types and equivalent depths ---
        for iw, wname in enumerate(wave_names):
            disp_fun = self._wave_registry[wname]

            for ie, he in enumerate(self.Ahe):
                c = np.sqrt(self.g * he)  # shallow-water gravity wave speed

                # dispersion in rad/s
                freq = disp_fun(self.k, c, he)

                # rad/s → cycles/day
                Afreq[iw, ie, :] = freq / two_pi * sec_per_day

        return Afreq, Apzwn, wave_names

    # ------------------------------------------------------------------
    #   Dispersion relations (internal, vectorized)
    # ------------------------------------------------------------------
    def _disp_mrg(self, k, c, he=None):
        """Mixed Rossby-Gravity wave (n=0 antisymmetric)."""
        freq = np.full_like(k, self.fillval, dtype=float)

        # westward branch: k < 0, ω > 0
        mask_w = k < 0.0
        if np.any(mask_w):
            kw = k[mask_w]
            dell = np.sqrt(1.0 + 4.0 * self.Beta / (kw**2 * c))
            freq[mask_w] = kw * c * (0.5 - 0.5 * dell)

        # k = 0 limit: IG frequency
        mask_eq = k == 0.0
        if np.any(mask_eq):
            freq[mask_eq] = np.sqrt(c * self.Beta)

        return freq

    def _disp_ig0(self, k, c, he=None):
        """Inertial Gravity wave with n=0."""
        freq = np.full_like(k, self.fillval, dtype=float)

        # k = 0: IG frequency
        mask_eq = k == 0.0
        if np.any(mask_eq):
            freq[mask_eq] = np.sqrt(c * self.Beta)

        # eastward branch (you can generalize to both signs if desired)
        mask_e = k > 0.0
        if np.any(mask_e):
            ke = k[mask_e]
            dell = np.sqrt(1.0 + 4.0 * self.Beta / (ke**2 * c))
            freq[mask_e] = ke * c * (0.5 + 0.5 * dell)

        return freq

    def _disp_ig_n(self, k, c, he, n):
        """
        Inertial Gravity wave with meridional mode index n >= 1.

        Solves the cubic dispersion relation via fixed-point iteration:
            ω^3 = [(2n+1)βc + c^2 k^2] ω + c^2 β k
        """
        beta_c = self.Beta * c
        c2 = self.g * he       # == c**2

        # initial guess: ignore the β k / ω term
        freq = np.sqrt((2 * n + 1) * beta_c + c2 * k**2)

        # fixed-point iteration
        for _ in range(5):
            freq = np.sqrt(
                (2 * n + 1) * beta_c
                + c2 * k**2
                + c2 * self.Beta * k / freq
            )

        return freq

    def _disp_kelvin(self, k, c, he=None):
        """Equatorial Kelvin wave: eastward only (k > 0)."""
        freq = np.full_like(k, self.fillval, dtype=float)
        mask_e = k > 0.0
        freq[mask_e] = k[mask_e] * c
        return freq

    def _disp_er_n(self, k, c, n):
        """Equatorial Rossby wave with meridional mode index n >= 1."""
        freq = np.full_like(k, self.fillval, dtype=float)

        # only westward k < 0 with ω > 0
        mask_w = k < 0.0
        if np.any(mask_w):
            kw = k[mask_w]
            dell = (self.Beta / c) * (2 * n + 1)  # (2n+1)/L^2
            freq[mask_w] = -self.Beta * kw / (kw**2 + dell)

        return freq
