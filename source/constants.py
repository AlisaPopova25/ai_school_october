from source import utils


class Constants(object):
    # WAKE_THRESHOLD = 0.3  # These values were used for scikit-learn 0.20.3, See:
    # REM_THRESHOLD = 0.35  # https://scikit-learn.org/stable/whats_new.html#version-0-21-0
    WAKE_THRESHOLD = 0.5  #
    REM_THRESHOLD = 0.35

    INCLUDE_CIRCADIAN = True  # uses the Python port in preprocessing/time/clock_proxy/circadian_model.py
    EPOCH_DURATION_IN_SECONDS = 30
    SECONDS_PER_MINUTE = 60
    SECONDS_PER_DAY = 3600 * 24
    SECONDS_PER_HOUR = 3600
    VERBOSE = True
    CROPPED_FILE_PATH = utils.get_project_root().joinpath('outputs/cropped/')
    FEATURE_FILE_PATH = utils.get_project_root().joinpath('outputs/features/')
    FIGURE_FILE_PATH = utils.get_project_root().joinpath('outputs/figures/')
    LOWER_BOUND = -0.2
    MATLAB_PATH = '/Applications/MATLAB_R2019a.app/bin/matlab'  # Replace with your MATLAB path


# Output folders are not kept in git (no empty placeholder folders): create them on first use
for _path in (Constants.CROPPED_FILE_PATH, Constants.FEATURE_FILE_PATH, Constants.FIGURE_FILE_PATH):
    _path.mkdir(parents=True, exist_ok=True)
