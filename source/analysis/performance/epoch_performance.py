class SleepWakePerformance(object):
    def __init__(self, accuracy, wake_correct, sleep_correct, kappa, auc, sleep_predictive_value,
                 wake_predictive_value, f1_wake=None, f1_sleep=None, f1_macro=None):
        self.accuracy = accuracy
        self.wake_correct = wake_correct
        self.sleep_correct = sleep_correct
        self.kappa = kappa
        self.auc = auc
        self.wake_predictive_value = wake_predictive_value
        self.sleep_predictive_value = sleep_predictive_value
        self.f1_wake = f1_wake  # F1 for the rare class (wake as positive)
        self.f1_sleep = f1_sleep
        self.f1_macro = f1_macro  # mean of the two, each class counts equally


class ThreeClassPerformance(object):
    def __init__(self, accuracy, wake_correct, rem_correct, nrem_correct, kappa, f1_macro=None, f1_wake=None,
                 f1_nrem=None, f1_rem=None):
        self.accuracy = accuracy
        self.wake_correct = wake_correct
        self.rem_correct = rem_correct
        self.nrem_correct = nrem_correct
        self.kappa = kappa
        self.f1_macro = f1_macro  # macro F1 over wake / NREM / REM (at the same point as the best-accuracy kappa)
        self.f1_wake = f1_wake
        self.f1_nrem = f1_nrem
        self.f1_rem = f1_rem
