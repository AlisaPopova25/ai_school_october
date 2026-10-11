from source.analysis.performance.performance_summarizer import PerformanceSummarizer
from source.analysis.performance.raw_performance import RawPerformance
from source.analysis.setup.feature_set_service import FeatureSetService
from source.analysis.setup.feature_type import FeatureType


class TableBuilder(object):
    pass

    @staticmethod
    def print_table_sw(classifier_summary):
        performance_dictionary = classifier_summary.performance_dictionary
        attributed_classifier = classifier_summary.attributed_classifier

        frontmatter = '\\begin{table}  \\caption{Sleep/wake differentiation performance by ' \
                      + attributed_classifier.name \
                      + ' across different feature inputs in the Apple Watch (PPG, MEMS) dataset} ' \
                        '\\begin{tabular}{l*{5}{c}} & Accuracy & Wake correct (specificity) ' \
                        '& Sleep correct (sensitivity) & $\\kappa$ & AUC \\\\ '

        print(frontmatter)
        for feature_set in performance_dictionary:
            raw_performances: [RawPerformance] = performance_dictionary[feature_set]

            print('\\hline ' + FeatureSetService.get_label(list(feature_set)) + ' &')

            true_positive_thresholds, performance_summaries = PerformanceSummarizer.summarize_thresholds(
                raw_performances)

            for true_positive_threshold, performance in zip(true_positive_thresholds, performance_summaries):
                if true_positive_threshold == 0.8:
                    print(
                        str(round(performance.accuracy, 3)) + ' & ' +
                        str(round(performance.wake_correct, 3)) + ' & ' +
                        str(round(performance.sleep_correct, 3)) + ' & ' +
                        str(round(performance.kappa, 3)) + ' & ' +
                        str(round(performance.auc, 3)) + '  \\\\')
                else:
                    print('& ' +
                          str(round(performance.accuracy, 3)) + ' & ' +
                          str(round(performance.wake_correct, 3)) + ' & ' +
                          str(round(performance.sleep_correct, 3))
                          + ' & ' + str(round(performance.kappa, 3)) + ' &   \\\\')

        backmatter = '\\hline \\end{tabular}  \\label{tab:' + \
                     attributed_classifier.name[0:4] + \
                     'params} \\small \\vspace{.2cm} ' \
                     '\\caption*{Fraction of wake correct, fraction of sleep correct, accuracy, ' \
                     '$\\kappa$, and AUC for sleep-wake predictions of ' + attributed_classifier.name + \
                     ' with use of motion, HR, clock proxy, or combination of features. PPG, ' \
                     'photoplethysmography; MEMS, microelectromechanical systems; HR, heart rate; ' \
                     'AUC, area under the curve.} \\end{table}'

        print(backmatter)

    @staticmethod
    def print_table_three_class(classifier_summaries):

        frontmatter = '\\begin{tabular}{l | l | c | c | c | c | c } & & Wake Correct & NREM' \
                      ' Correct & REM Correct & Best accuracy & $\\kappa$ \\'

        print(frontmatter)

        for classifier_summary in classifier_summaries:
            performance_dictionary = classifier_summary.performance_dictionary
            attributed_classifier = classifier_summary.attributed_classifier

            for feature_set in performance_dictionary:

                if list(feature_set) == [FeatureType.count]:
                    print('\\hline ' + attributed_classifier.name + ' & ' + FeatureSetService.get_label(
                        list(feature_set)) + ' & ')

                else:
                    print(' & ' + FeatureSetService.get_label(list(feature_set)) + ' & ')

                three_class_performance_summary = performance_dictionary[feature_set]
                wake_correct = three_class_performance_summary.wake_correct
                nrem_correct = three_class_performance_summary.nrem_correct
                rem_correct = three_class_performance_summary.rem_correct
                accuracy = three_class_performance_summary.accuracy
                kappa = three_class_performance_summary.kappa

                print(str(round(wake_correct, 3)) + ' & ' +
                      str(round(nrem_correct, 3)) + ' & ' +
                      str(round(rem_correct, 3)) + ' & ' +
                      str(round(accuracy, 3)) + ' & ' +
                      str(round(kappa, 3)) + '\\\\')

        backmatter = '\\end{tabular}  \\label{tab:REM_params} \\small \\vspace{.2cm}'

        print(backmatter)

    @staticmethod
    def print_table_sw_f1(classifier_summary):
        """Added table: F1 at the same sensitivity thresholds as print_table_sw (that table is unchanged)."""
        attributed_classifier = classifier_summary.attributed_classifier
        print('\\begin{table}  \\caption{F1 for sleep/wake by ' + attributed_classifier.name +
              ' (wake = positive class for F1 wake)} \\begin{tabular}{l*{5}{c}} '
              '& Sleep correct (sensitivity) & Wake correct & F1 wake & F1 sleep & Macro F1 \\\\ ')
        for feature_set, raw_performances in classifier_summary.performance_dictionary.items():
            print('\\hline ' + FeatureSetService.get_label(list(feature_set)) + ' &')
            thresholds, summaries = PerformanceSummarizer.summarize_thresholds(raw_performances)
            for index, (threshold, performance) in enumerate(zip(thresholds, summaries)):
                prefix = '' if index == 0 else '& '
                print(prefix + str(round(threshold, 3)) + ' & ' + str(round(performance.wake_correct, 3)) + ' & ' +
                      str(round(performance.f1_wake, 3)) + ' & ' + str(round(performance.f1_sleep, 3)) + ' & ' +
                      str(round(performance.f1_macro, 3)) + '  \\\\')
        print('\\hline \\end{tabular} \\end{table}')

    @staticmethod
    def print_table_three_class_f1(classifier_summaries):
        """Added table: macro and per-class F1 at the best-accuracy point used for kappa in Table 6."""
        print('\\begin{tabular}{l | l | c | c | c | c } & & Macro F1 & F1 wake & F1 NREM & F1 REM \\\\')
        for classifier_summary in classifier_summaries:
            name = classifier_summary.attributed_classifier.name
            for index, (feature_set, summary) in enumerate(classifier_summary.performance_dictionary.items()):
                label = FeatureSetService.get_label(list(feature_set))
                print(('\\hline ' + name if index == 0 else '') + ' & ' + label + ' & ' +
                      str(round(summary.f1_macro, 3)) + ' & ' + str(round(summary.f1_wake, 3)) + ' & ' +
                      str(round(summary.f1_nrem, 3)) + ' & ' + str(round(summary.f1_rem, 3)) + ' \\\\')
        print('\\end{tabular}')
