import time

from source.analysis.figures.data_plot_builder import DataPlotBuilder
from source.analysis.setup.subject_builder import SubjectBuilder
from source.constants import Constants
from source.preprocessing.activity_count.activity_count_service import ActivityCountService
from source.preprocessing.feature_builder import FeatureBuilder
from source.preprocessing.raw_data_processor import RawDataProcessor
from source.preprocessing.time.clock_proxy import circadian_model


def run_preprocessing(subject_set):
    start_time = time.time()

    for subject in subject_set:
        print("Cropping data from subject " + str(subject) + "...")
        RawDataProcessor.crop_all(str(subject))

    if Constants.INCLUDE_CIRCADIAN:
        # Python port of runCircadianModel.m (no MATLAB needed); MESA variant is not ported
        circadian_model.build_all(subject_set)

    for subject in subject_set:
        FeatureBuilder.build(str(subject))

    end_time = time.time()
    print("Execution took " + str((end_time - start_time) / 60) + " minutes")


if __name__ == '__main__':
    subject_ids = SubjectBuilder.get_all_subject_ids()
    run_preprocessing(subject_ids)

    for subject_id in subject_ids:
        DataPlotBuilder.make_data_demo(subject_id, False)
