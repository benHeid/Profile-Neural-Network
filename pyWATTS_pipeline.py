import pandas as pd
from pywatts.callbacks import CSVCallback, LinePlotCallback
from pywatts.core.pipeline import Pipeline
from pywatts.modules import RollingGroupBy
from pywatts.modules.calendar_extraction import CalendarExtraction, CalendarFeature
from pywatts.modules.clock_shift import ClockShift
from pywatts.modules.profile_neural_network import ProfileNeuralNetwork
from pywatts.modules.rolling_mean import RollingMean
from pywatts.modules.root_mean_squared_error import RmseCalculator
from pywatts.modules.sample_module import Sampler
from pywatts.modules.trend_extraction import TrendExtraction
from pywatts.utils._xarray_time_series_utils import numpy_to_xarray
from pywatts.wrapper.function_module import FunctionModule

# NOTE If you choose a horizon greater than 24 you have to shift the profile -> Else future values may be considered for calculating the profile.
HORIZON = 24


def get_diff(x, profile):
    return numpy_to_xarray(x.values - profile.values, x, "difference")


drift_occured = False

if __name__ == "__main__":
    pipeline = Pipeline("pnn_pipeline")

    profile_moving = RollingMean(window_size=28, group_by=RollingGroupBy.WorkdayWeekend)(
        x=(pipeline["BldgX"]))
    difference = FunctionModule(get_diff)(x=pipeline["BldgX"], profile=profile_moving)
    trend = TrendExtraction(168, 5)(x=difference)
    calendar = CalendarExtraction(country="BadenWurttemberg",
                                  features=[CalendarFeature.hour_sine, CalendarFeature.month_sine,
                                            CalendarFeature.day_sine, CalendarFeature.monday, CalendarFeature.tuesday,
                                            CalendarFeature.wednesday, CalendarFeature.thursday, CalendarFeature.friday,
                                            CalendarFeature.hour_cos, CalendarFeature.day_cos,
                                            CalendarFeature.month_cos, CalendarFeature.saturday, CalendarFeature.sunday,
                                            CalendarFeature.workday])(
        x=pipeline["BldgX"])

    shifted_difference = ClockShift(36)(x=difference)
    sampled_difference = Sampler(36)(x=shifted_difference)
    sampled_calendar = Sampler(HORIZON)(x=calendar)
    sampled_humidity = Sampler(HORIZON)(x=pipeline["Humidity"])
    sampled_temperature = Sampler(HORIZON)(x=pipeline["Temperature"])
    sampled_profile_moving = Sampler(HORIZON)(x=profile_moving)
    sampled_trend = Sampler(HORIZON)(x=trend)

    target = Sampler(HORIZON)(x=pipeline["BldgX"])

    prediction_moving = ProfileNeuralNetwork(offset=24 * 7 * 11,
                                             epochs=1000)(historical_input=sampled_difference,
                                                        calendar=sampled_calendar,
                                                        temperature=sampled_temperature,
                                                        humidity=sampled_humidity,
                                                        profile=sampled_profile_moving,
                                                        trend=sampled_trend,
                                                        target=target,
                                                        callbacks=[LinePlotCallback("PNN")])

    rmse = RmseCalculator(offset=11 * 168)(pnn_moving=prediction_moving,  moving_pred=sampled_profile_moving, y=target,
                                           callbacks=[CSVCallback('RMSE')])

    rmse_cleaned = RmseCalculator(name="RMSE_cleaned", offset=11 * 168)(
        pnn_moving=prediction_moving, moving_pred=sampled_profile_moving, y=target,
        callbacks=[CSVCallback('RMSE')])

    data = pd.read_csv("data/data.csv", index_col="time", parse_dates=["time"],
                       infer_datetime_format=True)

    result_train = pipeline.train(data[:"05.18.2015"])
    result_test = pipeline.test(data["05.18.2015":])

    print("Finished")
