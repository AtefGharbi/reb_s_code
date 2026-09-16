"""
Synthetic network, EV fleet, and BESS parameter generator for the REB-S case study.

IMPORTANT: These parameters are SYNTHETIC / representative values, not real
Saudi Electricity Company (SEC) data (no such proprietary dataset is
accessible from this environment). They are chosen to be qualitatively
consistent with the ranges described in the proposal (medium-voltage feeder,
mixed residential/commercial EV charging, small-scale BESS) so the pipeline
below is directly reusable: swap the constants / generators here for real
SEC feeder data and real fleet telemetry when available, and every
downstream script (scheduler, detector, evaluation) works unchanged.
"""
import numpy as np

DT_HOURS = 0.5          # 30-minute control interval
STEPS_PER_DAY = 48       # 24h / 0.5h


def time_of_use_price(steps_per_day=STEPS_PER_DAY):
    """Synthetic SEC-like time-of-use tariff (SAR/kWh), two peak windows."""
    t = np.arange(steps_per_day) * DT_HOURS
    base = 0.18
    evening_peak = 0.22 * np.exp(-0.5 * ((t - 19) / 1.6) ** 2)   # ~19:00
    midday_peak = 0.10 * np.exp(-0.5 * ((t - 13) / 2.0) ** 2)    # ~13:00 (AC load)
    price = base + evening_peak + midday_peak
    return price


def v2g_price(steps_per_day=STEPS_PER_DAY):
    """V2G compensation rate, set at a discount to retail price to reflect utility margin."""
    return 0.85 * time_of_use_price(steps_per_day)


def make_ev_fleet(n_ev, rng, steps_per_day=STEPS_PER_DAY):
    """Generate a synthetic EV fleet with mobility-derived arrival/departure/energy need."""
    evs = []
    for i in range(n_ev):
        # Home-charging commuter pattern: arrive evening, depart next morning
        arrival = int(np.clip(rng.normal(17.5, 1.5), 15, 22) / DT_HOURS)
        departure = int(np.clip(rng.normal(7.5, 1.0), 5, 9) / DT_HOURS) + steps_per_day
        cap_kwh = rng.choice([50.0, 62.0, 75.0])
        daily_km = np.clip(rng.normal(45, 15), 10, 120)
        consumption_kwh_per_km = 0.18
        energy_needed = daily_km * consumption_kwh_per_km
        soc_arrival = float(np.clip(1.0 - energy_needed / cap_kwh, 0.15, 0.6))
        soc_dep_min_true = float(np.clip(soc_arrival + energy_needed / cap_kwh * 0.9, 0.55, 0.95))
        p_max_kw = rng.choice([7.4, 11.0])
        evs.append(dict(
            id=i, arrival=arrival, departure=departure, cap_kwh=cap_kwh,
            soc_arrival=soc_arrival, soc_dep_min_true=soc_dep_min_true,
            p_max_kw=p_max_kw, soc_min=0.10, soc_max=1.0
        ))
    return evs


def make_bess_fleet(n_bess, rng):
    bess = []
    for b in range(n_bess):
        cap_kwh = rng.choice([100.0, 150.0, 200.0])
        bess.append(dict(
            id=b, cap_kwh=cap_kwh, p_max_kw=cap_kwh * 0.5,
            soc_init=0.5, soc_min=0.10, soc_max=0.95,
            degr_coeff=0.04 / cap_kwh,   # SAR per kWh-throughput, scaled by size
        ))
    return bess


def base_load_profile(steps_per_day=STEPS_PER_DAY, scale=40.0, rng=None):
    t = np.arange(2 * steps_per_day) * DT_HOURS
    tod = t % 24
    load = scale * (0.55 + 0.30 * np.exp(-0.5 * ((tod - 20) / 2.5) ** 2)
                     + 0.20 * np.exp(-0.5 * ((tod - 13) / 2.5) ** 2))
    if rng is not None:
        load = load * (1 + rng.normal(0, 0.03, size=load.shape))
    return load


ETA_CH = 0.95
ETA_DIS = 0.95
FEEDER_MAX_KW = 500.0
STATION_MAX_KW = 250.0
PEAK_WEIGHT = 0.05          # SAR per kW of peak, small relative to energy cost
GAMMA_RISK = 1.0             # risk-adaptation gain, eq. (11)
BUFFER_SOC = 0.08            # extra SoC margin at max risk, eq. (11)
