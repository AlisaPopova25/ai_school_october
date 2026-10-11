"""Python port of runCircadianModel.m / circadianModel.m / steps2light.m / lux2alpha.m.

Builds the "clock proxy" feature from the paper (Walch et al., SLEEP 2019) without MATLAB:
steps from the ambulatory week -> average day -> estimated light -> Forger (1999) circadian
oscillator integrated with RK4 -> x(t) during the PSG night.

Output: data/circadian_predictions/<subject>_clock_proxy.txt with columns  timestamp, x, xc
(same format the MATLAB script wrote, so TimeBasedFeatureService can read it unchanged).
"""
import math
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from source import utils

SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 24 * SECONDS_PER_HOUR
NUM_HOURS_TO_SAVE = 9
OUTPUT_STEP_SECONDS = 30
HOURS_OVERHANG_TO_CROP = 10
MAX_DAYS_TO_AVERAGE = 5
BIN_MINUTES = 10
NUM_DAYS_TO_SIMULATE = 60
TAU = 24.2
TIME_ZONE = ZoneInfo('America/New_York')


def clean_steps(data):
    data = data[data[:, 0] != 0]
    if data[:, 0].max() > 1.5e10:  # milliseconds -> seconds
        data[:, 0] = data[:, 0] / 1000
    return data


def lux_to_alpha(lux):
    return 0.16 * (lux ** 0.6 / 9500 ** 0.6)


def steps_to_light(steps, start_hour, dt_seconds, step_threshold=20):
    """steps2light.m: 50/500/1000 lux depending on clock time whenever steps >= threshold."""
    light = np.zeros(len(steps))
    for i in range(len(steps)):
        hour = (start_hour + (i + 1) * dt_seconds / 3600) % 24  # MATLAB uses 1-based i
        value = 500
        if 10 < hour < 16:
            value = 1000
        if hour < 7 or hour > 22:
            value = 50
        light[i] = value if steps[i] >= step_threshold else 0
    return light


def forger_rhs(y, alpha, tau=TAU):
    """Forger, Jewett, Kronauer 1999 'simple' model (the one used in the paper)."""
    x, xc, n = y
    G, k, mu, b = 19.875, 0.55, 0.23, 0.013
    bh = G * (1 - n) * alpha
    B = bh * (1 - 0.4 * x) * (1 - 0.4 * xc)
    return np.array([
        math.pi / 12 * (xc + B),
        math.pi / 12 * (mu * (xc - 4 * xc ** 3 / 3) - x * ((24 / (0.99669 * tau)) ** 2 + k * B)),
        60 * (alpha * (1 - n) - b * n),
    ])


def simulate(light_time_hours, light_alpha, duration_hours, dt=0.1, seed=0):
    """ode4 (fixed-step RK4) with zero-order-hold light: alpha(t) = light[last index with time < t]."""
    rng = np.random.default_rng(seed)
    y = rng.random(3)
    y[:2] = y[:2] * 2 - 1

    def alpha_at(t):
        idx = np.searchsorted(light_time_hours, t, side='left') - 1
        return light_alpha[idx] if idx >= 0 else 0.0

    tc = np.arange(0, duration_hours + 1e-9, dt)
    out = np.zeros((len(tc), 3))
    out[0] = y
    for i in range(1, len(tc)):
        t, h = tc[i - 1], tc[i] - tc[i - 1]
        k1 = forger_rhs(y, alpha_at(t))
        k2 = forger_rhs(y + 0.5 * h * k1, alpha_at(t + 0.5 * h))
        k3 = forger_rhs(y + 0.5 * h * k2, alpha_at(t + 0.5 * h))
        k4 = forger_rhs(y + h * k3, alpha_at(tc[i]))
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[i] = y
    return tc, out


# The public PhysioNet release stores time as seconds relative to PSG start (t=0), so the clock time
# needed by steps2light is not available. Per protocol, lights-out was at habitual bedtime; we assume
# PSG start ~ 23:00 local time. (The original MATLAB code used true Unix time.)
PSG_START_CLOCK_HOUR = 23.0


def local_hour(timestamp):
    if abs(timestamp) < 1e9:  # relative time (PhysioNet release)
        return (PSG_START_CLOCK_HOUR + timestamp / SECONDS_PER_HOUR) % 24
    local = datetime.fromtimestamp(timestamp, tz=timezone.utc).astimezone(TIME_ZONE)
    return local.hour + local.minute / 60


def build_for_subject(subject_id, seed=0):
    root = utils.get_project_root()
    steps = clean_steps(pd.read_csv(root.joinpath(f'data/steps/{subject_id}_steps.txt'), header=None).values.astype(float))
    steps = steps[np.argsort(steps[:, 0])]
    score = np.loadtxt(root.joinpath(f'outputs/cropped/{subject_id}_cleaned_psg.out'))

    dt = BIN_MINUTES * SECONDS_PER_MINUTE
    duration_days = (steps[:, 0].max() - steps[:, 0].min()) / SECONDS_PER_DAY

    if duration_days > MAX_DAYS_TO_AVERAGE:
        duration_days = MAX_DAYS_TO_AVERAGE
    else:
        # Pad the start by repeating the last recorded day (as in the MATLAB code)
        base = steps[:, 0].max() - MAX_DAYS_TO_AVERAGE * SECONDS_PER_DAY
        last_day = base + (MAX_DAYS_TO_AVERAGE - 1) * SECONDS_PER_DAY
        padding = [[base, np.interp(last_day, steps[:, 0], steps[:, 1])]]
        cumulative, start = 0, base
        while start < steps[:, 0].min():
            cumulative += dt
            start = base + cumulative
            padding.append([start, np.interp(last_day + cumulative % SECONDS_PER_DAY, steps[:, 0], steps[:, 1])])
        steps = np.vstack([np.array(padding), steps])
        duration_days = (steps[:, 0].max() - steps[:, 0].min()) / SECONDS_PER_DAY

    bins_per_day = SECONDS_PER_DAY // dt
    cropped_start = steps[:, 0].max() - SECONDS_PER_DAY * math.floor(duration_days)
    end_light_window = steps[:, 0].max() - dt / 2
    end_timestamp = steps[:, 0].max() + 3 * SECONDS_PER_DAY
    if score[:, 0].max() < steps[:, 0].max():
        end_timestamp = score[:, 0].max() + 3 * SECONDS_PER_DAY
        end_light_window = score[:, 0].max() - dt / 2
        cropped_start = score[:, 0].max() - SECONDS_PER_DAY * math.floor(duration_days)

    # Average day of steps in 10-minute bins
    total = np.zeros(bins_per_day)
    count = np.zeros(bins_per_day)
    for bin_index, t in enumerate(np.arange(cropped_start + dt / 2, end_light_window + 1e-9, dt)):
        in_bin = (steps[:, 0] >= t - dt / 2) & (steps[:, 0] < t + dt / 2)
        total[bin_index % bins_per_day] += steps[in_bin, 1].sum()
        count[bin_index % bins_per_day] += 1
    average_steps = total / count

    light = steps_to_light(average_steps, local_hour(cropped_start), dt)
    light = lux_to_alpha(np.tile(light, NUM_DAYS_TO_SIMULATE))
    timestamps = end_timestamp - dt * np.arange(NUM_DAYS_TO_SIMULATE * bins_per_day)[::-1]

    psg_start = score[:, 0].min()
    light[timestamps > psg_start] = 0  # lights off from the start of the PSG night
    keep = timestamps <= psg_start + HOURS_OVERHANG_TO_CROP * SECONDS_PER_HOUR
    timestamps, light = timestamps[keep], light[keep]

    hours = (timestamps - timestamps.min()) / SECONDS_PER_HOUR
    tc, y = simulate(hours, light, hours.max(), seed=seed)
    sim_seconds = tc * SECONDS_PER_HOUR + timestamps.min()

    out_t = np.arange(psg_start, psg_start + NUM_HOURS_TO_SAVE * SECONDS_PER_HOUR + 1e-9, OUTPUT_STEP_SECONDS)
    output = np.column_stack([out_t, np.interp(out_t, sim_seconds, y[:, 0]), np.interp(out_t, sim_seconds, y[:, 1])])
    out_dir = root.joinpath('data/circadian_predictions')
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir.joinpath(f'{subject_id}_clock_proxy.txt'), output, delimiter=',', fmt='%.12g')
    return output


def build_all(subject_ids):
    for subject_id in subject_ids:
        print(f'Circadian model for subject {subject_id}...')
        build_for_subject(subject_id)
