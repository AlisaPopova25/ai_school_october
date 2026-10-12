"""Standalone calculation of three additional sleep/wake MLP features.

Requires only NumPy. Input times must use seconds and a common time origin.
Process one participant/recording at a time. Acceleration must be in g,
heart rate in bpm, epochs are 30 seconds. No labels or model weights are used.

The numerical feature calculation matches source/preprocessing/fast_sleep_features.py.
"""
import numpy as np

FAST_NAMES = ['log_peak_5sec_movement', 'active_5sec_fraction_1min', 'relative_mean_hr_1min']


def interval_moments(samples, starts, ends, columns):
    samples = np.asarray(samples, dtype=float)
    if not np.isfinite(samples).all() or np.any(np.diff(samples[:, 0]) < 0):
        raise ValueError('Samples must be finite and time ordered.')
    left = np.searchsorted(samples[:, 0], starts, side='left')
    right = np.searchsorted(samples[:, 0], ends, side='left')
    count = right - left
    values = samples[:, columns]
    sums = np.vstack([np.zeros((1, len(columns))), np.cumsum(values, axis=0)])
    squares = np.vstack([np.zeros((1, len(columns))), np.cumsum(values ** 2, axis=0)])
    mean = (sums[right] - sums[left]) / np.maximum(count[:, None], 1)
    variance = np.maximum(0, (squares[right] - squares[left]) / np.maximum(count[:, None], 1) - mean ** 2)
    return mean, variance, count


def make_fast_features(motion, heart_rate, epoch_times, movement_threshold_g=.02):
    """Use contiguous aligned 30-second epochs; centered minutes include future data."""
    motion = np.asarray(motion, dtype=float)
    heart_rate = np.asarray(heart_rate, dtype=float)
    for samples, columns, name in [(motion, 4, 'motion'), (heart_rate, 2, 'heart_rate')]:
        if samples.ndim != 2 or samples.shape[1] != columns or len(samples) == 0:
            raise ValueError(f'{name} must be a nonempty array with {columns} columns.')
    if np.any(heart_rate[:, 1] <= 0):
        raise ValueError('Heart rate measurements must be positive bpm.')
    times = np.asarray(epoch_times, dtype=float).ravel()
    if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Epoch times must be finite and increase.')
    if not np.isfinite(movement_threshold_g) or movement_threshold_g <= 0:
        raise ValueError('Movement threshold must be positive.')
    segment_starts = np.r_[0, np.flatnonzero(np.abs(np.diff(times) - 30) > .02) + 1]
    segment_ends = np.r_[segment_starts[1:], len(times)]
    lengths = segment_ends - segment_starts
    lower = np.repeat(times[segment_starts], lengths)
    upper = np.repeat(times[segment_ends - 1] + 30, lengths)
    starts = times[:, None] + np.arange(-15, 45, 5)[None, :]
    ends = starts + 5
    starts = np.maximum(starts, lower[:, None])
    ends = np.maximum(starts, np.minimum(ends, upper[:, None]))
    _, variance, count = interval_moments(motion, starts.ravel(), ends.ravel(), [1, 2, 3])
    movement = np.sqrt(variance.sum(axis=1)).reshape(len(times), 12)
    available = count.reshape(len(times), 12) >= 2
    central_available = available[:, 3:9]
    if np.any(~central_available.any(axis=1)):
        raise ValueError('An epoch has no usable five-second motion interval.')
    peak = np.where(central_available, movement[:, 3:9], 0).max(axis=1)
    active = ((movement >= movement_threshold_g) & available).sum(axis=1) / available.sum(axis=1)
    hr, _, hr_count = interval_moments(heart_rate, np.maximum(times - 15, lower),
                                      np.minimum(times + 45, upper), [1])
    if np.any(hr_count == 0):
        raise ValueError('A minute has no observed heart rate.')
    median_hr = float(np.median(heart_rate[:, 1]))
    if median_hr <= 0:
        raise ValueError('Median heart rate must be positive.')
    # Fixed physical scaling, no fitted scaler. Median uses the unlabeled full recording.
    features = np.column_stack([np.log1p(peak / movement_threshold_g), active,
                                (hr[:, 0] - median_hr) / median_hr])
    return features, {'recording_median_hr': median_hr,
                      'movement_threshold_g': movement_threshold_g,
                      'feature_names': FAST_NAMES}


def load_numeric_table(path):
    """Read CSV or whitespace-separated numbers, with an optional header."""
    from pathlib import Path
    with Path(path).open(encoding='utf-8-sig') as stream:
        first = next((line.strip() for line in stream if line.strip()), '')
    if not first:
        raise ValueError(f'Empty input file: {path}')
    delimiter = ',' if ',' in first else None
    fields = first.split(',') if delimiter else first.split()
    try:
        [float(value) for value in fields]
        skiprows = 0
    except ValueError:
        skiprows = 1
    return np.loadtxt(path, delimiter=delimiter, skiprows=skiprows, ndmin=2, encoding='utf-8-sig')


def main():
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion', required=True, type=Path, help='Columns: timestamp,x,y,z; acceleration in g.')
    parser.add_argument('--heart-rate', required=True, type=Path, help='Columns: timestamp,heart_rate; bpm.')
    parser.add_argument('--epochs', required=True, type=Path, help='One column: start timestamps of valid 30-second epochs.')
    parser.add_argument('--output', default=Path('additional_features.csv'), type=Path)
    parser.add_argument('--movement-threshold-g', type=float, default=.02)
    args = parser.parse_args()
    motion = load_numeric_table(args.motion)
    hr = load_numeric_table(args.heart_rate)
    epochs = load_numeric_table(args.epochs)
    if epochs.shape[1] != 1:
        parser.error('--epochs must contain exactly one timestamp column.')
    times = epochs[:, 0]
    features, metadata = make_fast_features(motion, hr, times, args.movement_threshold_g)
    metadata.update({'epoch_duration_seconds': 30, 'window_seconds_relative_to_epoch_start': [-15, 45],
                     'input_acceleration_units': 'g', 'input_heart_rate_units': 'bpm',
                     'epochs': len(times), 'uses_future_observations': True})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output, np.column_stack([times, features]), delimiter=',',
               fmt=['%.6f', '%.12g', '%.12g', '%.12g'],
               header=','.join(['timestamp'] + FAST_NAMES), comments='')
    metadata_path = args.output.with_suffix('.metadata.json')
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved {len(times)} epochs to {args.output}')


if __name__ == '__main__':
    main()

