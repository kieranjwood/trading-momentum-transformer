"""Model Inputs"""
# import mom_trans.utils as utils
import numpy as np
import sklearn.preprocessing
import pandas as pd
import datetime as dt
import enum

from sklearn.preprocessing import MinMaxScaler

# Type defintions
class DataTypes(enum.IntEnum):
    """Defines numerical types of each column."""

    REAL_VALUED = 0
    CATEGORICAL = 1
    DATE = 2


class InputTypes(enum.IntEnum):
    """Defines input types of each column."""

    TARGET = 0
    OBSERVED_INPUT = 1
    KNOWN_INPUT = 2
    STATIC_INPUT = 3
    ID = 4  # Single column used as an entity identifier
    TIME = 5  # Single column used as a time index


def get_single_col_by_input_type(input_type, column_definition):
    """Returns name of single column.
    Args:
      input_type: Input type of column to extract
      column_definition: Column definition list for experiment
    """

    l = [tup[0] for tup in column_definition if tup[2] == input_type]

    if len(l) != 1:
        raise ValueError("Invalid number of columns for {}".format(input_type))

    return l[0]


def extract_cols_from_data_type(data_type, column_definition, excluded_input_types):
    """Extracts the names of columns that correspond to a define data_type.
    Args:
      data_type: DataType of columns to extract.
      column_definition: Column definition to use.
      excluded_input_types: Set of input types to exclude
    Returns:
      List of names for columns with data type specified.
    """
    return [
        tup[0]
        for tup in column_definition
        if tup[1] == data_type and tup[2] not in excluded_input_types
    ]


class ModelFeatures:
    """Defines and formats data for the MomentumCp dataset.
    Attributes:
      column_definition: Defines input and data type of column used in the
        experiment.
      identifiers: Entity identifiers used in experiments.
    """

    def __init__(
        self,
        df,
        total_time_steps,
        start_boundary=1990,
        test_boundary=2020,
        test_end=2021,
        changepoint_lbws=None,
        train_valid_sliding=False,
        # add_buffer_years_to_test=1,  # TODO FIX THIS!!!!
        transform_real_inputs=False,  # TODO remove this
        train_valid_ratio=0.9,
        split_tickers_individually=True,
        add_ticker_as_static=False,
        time_features=False,
        lags=None,
        asset_class_dictionary=None,
        static_ticker_type_feature = False,
    ):
        """Initialises formatter. Splits data frame into training-validation-test data frames.
        This also calibrates scaling object, and transforms data for each split."""
        self._column_definition = [
            ("ticker", DataTypes.CATEGORICAL, InputTypes.ID),
            ("date", DataTypes.DATE, InputTypes.TIME),
            ("target_returns", DataTypes.REAL_VALUED, InputTypes.TARGET),
            ("norm_daily_return", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            ("norm_monthly_return", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            ("norm_quarterly_return", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            ("norm_biannual_return", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            ("norm_annual_return", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            ("macd_8_24", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            ("macd_16_48", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            ("macd_32_96", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ]
        df = df.dropna()
        df = df[df["year"] >= start_boundary].copy()
        years = df["year"]

        self.identifiers = None
        self._real_scalers = None
        self._cat_scalers = None
        self._target_scaler = None
        self._num_classes_per_cat_input = None
        self.total_time_steps = total_time_steps
        self.lags = lags

        if changepoint_lbws:
            for lbw in changepoint_lbws:
                self._column_definition.append(
                    (f"cp_score_{lbw}", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT)
                )
                self._column_definition.append(
                    (f"cp_rl_{lbw}", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT)
                )

        if time_features:
            self._column_definition.append(
                ("days_from_start", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT)
            )
            self._column_definition.append(
                ("day_of_week", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT)
            )
            self._column_definition.append(
                (f"day_of_month", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT)
            )
            self._column_definition.append(
                (f"week_of_year", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT)
            )
            # self._column_definition.append(
            #     (f"month_of_year", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT)
            # )

            # dataframe could have later years
            start_date = dt.datetime(start_boundary, 1, 1)
            days_from_start_max = (dt.datetime(test_end - 1, 12, 31) - start_date).days
            df["days_from_start"] = (df.index - start_date).days
            df["days_from_start"] = np.minimum(
                df["days_from_start"], days_from_start_max
            )

            df["days_from_start"] = (
                MinMaxScaler().fit_transform(df[["days_from_start"]].values).flatten()
            )
            df["day_of_week"] = (
                MinMaxScaler().fit_transform(df[["day_of_week"]].values).flatten()
            )
            df["day_of_month"] = (
                MinMaxScaler().fit_transform(df[["day_of_month"]].values).flatten()
            )
            df["week_of_year"] = (
                MinMaxScaler().fit_transform(df[["week_of_year"]].values).flatten()
            )
            # df["month_of_year"] = MinMaxScaler().fit_transform(df[["month_of_year"]].values).flatten()

        if add_ticker_as_static:
            self._column_definition.append(
                (f"static_ticker", DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT)
            )
            df["static_ticker"] = df["ticker"]
            if static_ticker_type_feature:
                df["static_ticker_type"] = df["ticker"].map(
                    lambda t: asset_class_dictionary[t]
                )
                self._column_definition.append(
                    (
                        f"static_ticker_type",
                        DataTypes.CATEGORICAL,
                        InputTypes.STATIC_INPUT,
                    )
                )

        self.transform_real_inputs = transform_real_inputs

        # for static_variables
        # self._column_definition.append(("ticker", DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT))

        test = df.loc[years >= test_boundary]

        if split_tickers_individually:
            trainvalid = df.loc[years < test_boundary]
            if lags:
                tickers = (
                    trainvalid.groupby("ticker")["ticker"].count()
                    * (1.0 - train_valid_ratio)
                ) >= total_time_steps
                tickers = tickers[tickers].index.tolist()
            else:
                tickers = list(trainvalid.ticker.unique())

            train, valid = [], []
            for ticker in tickers:
                calib_data = trainvalid[trainvalid.ticker == ticker]
                T = len(calib_data)
                train_valid_split = int(train_valid_ratio * T)
                train.append(calib_data.iloc[:train_valid_split, :].copy())
                valid.append(calib_data.iloc[train_valid_split:, :].copy())

            train = pd.concat(train)
            valid = pd.concat(valid)

            test = test[test.ticker.isin(tickers)]
        else:
            trainvalid = df.loc[years < test_boundary]
            dates = np.sort(trainvalid.index.unique())
            split_index = int(train_valid_ratio * len(dates))
            train_dates = pd.DataFrame({"date": dates[:split_index]})
            valid_dates = pd.DataFrame({"date": dates[split_index:]})

            train = (
                trainvalid.reset_index()
                .merge(train_dates, on="date")
                .set_index("date")
                .copy()
            )
            valid = (
                trainvalid.reset_index()
                .merge(valid_dates, on="date")
                .set_index("date")
                .copy()
            )
            if lags:
                tickers = (
                    valid.groupby("ticker")["ticker"].count() > self.total_time_steps
                )
                tickers = tickers[tickers].index.tolist()
                train = train[train.ticker.isin(tickers)]

            else:
                # at least one full training sequence
                # tickers = (
                #     train.groupby("ticker")["ticker"].count() > self.total_time_steps
                # )
                # tickers = tickers[tickers].index.tolist()
                tickers = list(train.ticker.unique())
            valid = valid[valid.ticker.isin(tickers)]
            test = test[test.ticker.isin(tickers)]

        # don't think this is needed...
        if test_end:
            # test = test[test["year"] < ((test_end + add_buffer_years_to_test))]
            test = test[test["year"] < test_end]

        test_with_buffer = pd.concat(
            [
                pd.concat(
                    [
                        trainvalid[trainvalid.ticker == t].iloc[
                            -(self.total_time_steps - 1) :
                        ],  # TODO this
                        test[test.ticker == t],
                    ]
                ).sort_index()
                for t in tickers
            ]
        )

        # to deal with case where fixed window did not have a full sequence
        if lags:
            for t in tickers:
                test_ticker = test[test["ticker"] == t]
                diff = len(test_ticker) - self.total_time_steps
                if diff < 0:
                    test = pd.concat(
                        [trainvalid[trainvalid["ticker"] == t][diff:], test]
                    )
                    # maybe should sort here but probably not needed

        self.tickers = tickers
        self.num_tickers = len(tickers)
        self.set_scalers(train)

        train, valid, test, test_with_buffer = [
            self.transform_inputs(data)
            for data in [train, valid, test, test_with_buffer]
        ]

        if lags:
            self.train = self._batch_data_smaller_output(
                train, train_valid_sliding, self.lags
            )
            self.valid = self._batch_data_smaller_output(
                valid, train_valid_sliding, self.lags
            )
            self.test_fixed = self._batch_data_smaller_output(test, False, self.lags)
            self.test_sliding = self._batch_data_smaller_output(
                test_with_buffer, True, self.lags
            )
        else:
            self.train = self._batch_data(train, train_valid_sliding)
            self.valid = self._batch_data(valid, train_valid_sliding)
            self.test_fixed = self._batch_data(test, False)
            self.test_sliding = self._batch_data(test_with_buffer, True)

    def set_scalers(self, df):
        """Calibrates scalers using the data supplied.
        Args:
          df: Data to use to calibrate scalers.
        """
        column_definitions = self.get_column_definition()
        id_column = get_single_col_by_input_type(InputTypes.ID, column_definitions)
        target_column = get_single_col_by_input_type(
            InputTypes.TARGET, column_definitions
        )

        # Extract identifiers in case required
        self.identifiers = list(df[id_column].unique())

        # Format real scalers
        real_inputs = extract_cols_from_data_type(
            DataTypes.REAL_VALUED,
            column_definitions,
            {InputTypes.ID, InputTypes.TIME, InputTypes.TARGET},
        )

        data = df[real_inputs].values
        self._real_scalers = sklearn.preprocessing.StandardScaler().fit(data)
        self._target_scaler = sklearn.preprocessing.StandardScaler().fit(
            df[[target_column]].values
        )  # used for predictions

        # Format categorical scalers
        categorical_inputs = extract_cols_from_data_type(
            DataTypes.CATEGORICAL,
            column_definitions,
            {InputTypes.ID, InputTypes.TIME, InputTypes.TARGET},
        )

        categorical_scalers = {}
        num_classes = []
        for col in categorical_inputs:
            # Set all to str so that we don't have mixed integer/string columns
            srs = df[col].apply(str)
            categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(
                srs.values
            )
            num_classes.append(srs.nunique())

        # Set categorical scaler outputs
        self._cat_scalers = categorical_scalers
        self._num_classes_per_cat_input = num_classes

    def transform_inputs(self, df):
        """Performs feature transformations.
        This includes both feature engineering, preprocessing and normalisation.
        Args:
          df: Data frame to transform.
        Returns:
          Transformed data frame.
        """
        output = df.copy()

        if self._real_scalers is None and self._cat_scalers is None:
            raise ValueError("Scalers have not been set!")

        column_definitions = self.get_column_definition()

        real_inputs = extract_cols_from_data_type(
            DataTypes.REAL_VALUED,
            column_definitions,
            {InputTypes.ID, InputTypes.TIME, InputTypes.TARGET},
        )
        categorical_inputs = extract_cols_from_data_type(
            DataTypes.CATEGORICAL,
            column_definitions,
            {InputTypes.ID, InputTypes.TIME, InputTypes.TARGET},
        )

        # Format real inputs
        if self.transform_real_inputs:
            output[real_inputs] = self._real_scalers.transform(df[real_inputs].values)

        # Format categorical inputs
        for col in categorical_inputs:
            string_df = df[col].apply(str)
            output[col] = self._cat_scalers[col].transform(string_df)

        return output

    def format_predictions(self, predictions):
        """Reverts any normalisation to give predictions in original scale.
        Args:
          predictions: Dataframe of model predictions.
        Returns:
          Data frame of unnormalised predictions.
        """
        output = predictions.copy()

        column_names = predictions.columns

        if self.transform_real_inputs:
            for col in column_names:
                if col not in {"forecast_time", "identifier"}:
                    output[col] = self._target_scaler.inverse_transform(
                        predictions[col]
                    )
        else:
            categorical_inputs = extract_cols_from_data_type(
                DataTypes.CATEGORICAL,
                self.get_column_definition(),
                {InputTypes.ID, InputTypes.TIME, InputTypes.TARGET},
            )
            for col in column_names:
                if col in categorical_inputs:
                    output[col] = self._target_scaler.inverse_transform(
                        predictions[col]
                    )

        return output

    def get_column_definition(self):
        """ "Returns formatted column definition in order expected."""

        column_definition = self._column_definition

        # Sanity checks first.
        # Ensure only one ID and time column exist
        def _check_single_column(input_type):

            length = len([tup for tup in column_definition if tup[2] == input_type])

            if length != 1:
                raise ValueError(
                    "Illegal number of inputs ({}) of type {}".format(
                        length, input_type
                    )
                )

        _check_single_column(InputTypes.ID)
        _check_single_column(InputTypes.TIME)

        identifier = [tup for tup in column_definition if tup[2] == InputTypes.ID]
        time = [tup for tup in column_definition if tup[2] == InputTypes.TIME]
        real_inputs = [
            tup
            for tup in column_definition
            if tup[1] == DataTypes.REAL_VALUED
            and tup[2] not in {InputTypes.ID, InputTypes.TIME, InputTypes.TARGET}
        ]
        categorical_inputs = [
            tup
            for tup in column_definition
            if tup[1] == DataTypes.CATEGORICAL
            and tup[2] not in {InputTypes.ID, InputTypes.TIME, InputTypes.TARGET}
        ]
        target = [tup for tup in column_definition if tup[2] == InputTypes.TARGET]

        return identifier + time + real_inputs + categorical_inputs + target

    @staticmethod
    def _unpack(data):
        return (
            data["inputs"],
            data["outputs"],
            data["active_entries"],
            data["identifier"],
            data["date"],
        )

    def _batch_data(self, data, sliding_window):
        """Batches data for training.

        Converts raw dataframe from a 2-D tabular format to a batched 3-D array
        to feed into Keras model.

        Args:
          data: DataFrame to batch

        Returns:
          Batched Numpy array with shape=(?, self.time_steps, self.input_size)
        """
        # TODO this works but is a bit of a mess
        data = data.copy()
        data["date"] = data.index.strftime("%Y-%m-%d")

        id_col = get_single_col_by_input_type(InputTypes.ID, self._column_definition)
        time_col = get_single_col_by_input_type(
            InputTypes.TIME, self._column_definition
        )
        target_col = get_single_col_by_input_type(
            InputTypes.TARGET, self._column_definition
        )

        input_cols = [
            tup[0]
            for tup in self._column_definition
            if tup[2] not in {InputTypes.ID, InputTypes.TIME, InputTypes.TARGET}
        ]

        data_map = {}

        if sliding_window:
            # Functions.
            def _batch_single_entity(input_data):
                time_steps = len(input_data)
                lags = self.total_time_steps  # + int(self.extra_lookahead_steps)
                x = input_data.values
                if time_steps >= lags:
                    return np.stack(
                        [x[i : time_steps - (lags - 1) + i, :] for i in range(lags)],
                        axis=1,
                    )
                else:
                    return None

            for _, sliced in data.groupby(id_col):

                col_mappings = {
                    "identifier": [id_col],
                    "date": [time_col],
                    "outputs": [target_col],
                    "inputs": input_cols,
                }

                for k in col_mappings:
                    cols = col_mappings[k]
                    arr = _batch_single_entity(sliced[cols].copy())

                    if k not in data_map:
                        data_map[k] = [arr]
                    else:
                        data_map[k].append(arr)

            # Combine all data
            for k in data_map:
                data_map[k] = np.concatenate(data_map[k], axis=0)

            active_entries = np.ones_like(data_map["outputs"])
            if "active_entries" not in data_map:
                data_map["active_entries"] = active_entries
            else:
                data_map["active_entries"].append(active_entries)

        else:
            for _, sliced in data.groupby(id_col):

                col_mappings = {
                    "identifier": [id_col],
                    "date": [time_col],
                    "inputs": input_cols,
                    "outputs": [target_col],
                }

                time_steps = len(sliced)
                lags = self.total_time_steps
                additional_time_steps_required = lags - (time_steps % lags)

                def _batch_single_entity(input_data):
                    x = input_data.values
                    if additional_time_steps_required > 0:
                        x = np.concatenate(
                            [x, np.zeros((additional_time_steps_required, x.shape[1]))]
                        )
                    return x.reshape(-1, lags, x.shape[1])

                # for k in col_mappings:
                k = "outputs"
                cols = col_mappings[k]
                arr = _batch_single_entity(sliced[cols].copy())

                batch_size = arr.shape[0]
                sequence_lengths = [
                    (
                        lags
                        if i != batch_size - 1
                        else lags - additional_time_steps_required
                    )
                    for i in range(batch_size)
                ]
                active_entries = np.ones((arr.shape[0], arr.shape[1], arr.shape[2]))
                for i in range(batch_size):
                    active_entries[i, sequence_lengths[i] :, :] = 0
                sequence_lengths = np.array(sequence_lengths, dtype=np.int)

                if "active_entries" not in data_map:
                    data_map["active_entries"] = [
                        active_entries[sequence_lengths > 0, :, :]
                    ]
                else:
                    data_map["active_entries"].append(
                        active_entries[sequence_lengths > 0, :, :]
                    )

                if k not in data_map:
                    data_map[k] = [arr[sequence_lengths > 0, :, :]]
                else:
                    data_map[k].append(arr[sequence_lengths > 0, :, :])

                for k in set(col_mappings) - {"outputs"}:
                    cols = col_mappings[k]
                    arr = _batch_single_entity(sliced[cols].copy())

                    if k not in data_map:
                        data_map[k] = [arr[sequence_lengths > 0, :, :]]
                    else:
                        data_map[k].append(arr[sequence_lengths > 0, :, :])

            # Combine all data
            for k in data_map:
                data_map[k] = np.concatenate(data_map[k], axis=0)

        active_flags = (np.sum(data_map["active_entries"], axis=-1) > 0.0) * 1.0
        data_map["inputs"] = data_map["inputs"][: len(active_flags)]
        data_map["outputs"] = data_map["outputs"][: len(active_flags)]
        data_map["active_entries"] = active_flags
        data_map["identifier"] = data_map["identifier"][: len(active_flags)]
        data_map["identifier"][data_map["identifier"] == 0] = ""
        data_map["date"] = data_map["date"][: len(active_flags)]
        data_map["date"][data_map["date"] == 0] = ""
        return data_map

    def _batch_data_smaller_output(self, data, sliding_window, output_length):
        """Batches data for training.

        Converts raw dataframe from a 2-D tabular format to a batched 3-D array
        to feed into Keras model.

        Args:
          data: DataFrame to batch

        Returns:
          Batched Numpy array with shape=(?, self.time_steps, self.input_size)
        """
        # TODO this works but is a bit of a mess
        data = data.copy()
        data["date"] = data.index.strftime("%Y-%m-%d")

        id_col = get_single_col_by_input_type(InputTypes.ID, self._column_definition)
        time_col = get_single_col_by_input_type(
            InputTypes.TIME, self._column_definition
        )
        target_col = get_single_col_by_input_type(
            InputTypes.TARGET, self._column_definition
        )

        input_cols = [
            tup[0]
            for tup in self._column_definition
            if tup[2] not in {InputTypes.ID, InputTypes.TIME, InputTypes.TARGET}
        ]

        data_map = {}

        col_mappings = {
            "identifier": [id_col],
            "date": [time_col],
            "inputs": input_cols,
            "outputs": [target_col],
        }

        if sliding_window:
            for _, sliced in data.groupby(id_col):

                time_steps = len(sliced)
                batch_size = time_steps - self.total_time_steps + 1
                seq_len = self.total_time_steps
                for k in col_mappings:
                    cols = col_mappings[k]
                    arr = sliced[cols].copy().values
                    arr = np.concatenate(
                        [arr[start : start + seq_len] for start in range(0, batch_size)]
                    ).reshape(-1, seq_len, arr.shape[1])

                    if k not in data_map:
                        data_map[k] = [arr]
                    else:
                        data_map[k].append(arr)

        else:
            for _, sliced in data.groupby(id_col):

                time_steps = len(sliced)
                batch_size = (
                    time_steps - self.total_time_steps + output_length
                ) // output_length
                active_time_steps = batch_size * output_length + (
                    self.total_time_steps - output_length
                )
                disregard_time_steps = time_steps % active_time_steps
                seq_len = self.total_time_steps
                for k in col_mappings:
                    cols = col_mappings[k]
                    arr = sliced[cols].copy().values[disregard_time_steps:]
                    arr = np.concatenate(
                        [
                            arr[start : start + seq_len]
                            for start in range(
                                0, output_length * batch_size, output_length
                            )
                        ]
                    ).reshape(-1, seq_len, arr.shape[1])

                    if k not in data_map:
                        data_map[k] = [arr]
                    else:
                        data_map[k].append(arr)

        # Combine all data
        for k in data_map:
            data_map[k] = np.concatenate(data_map[k], axis=0)

        data_map["active_entries"] = (np.sum(data_map["inputs"], axis=-1) > 0.0) * 1.0
        # TODO
        data_map["identifier"][data_map["identifier"] == 0] = ""
        data_map["date"][data_map["date"] == 0] = ""

        data_map["inputs_identifier"] = data_map["identifier"].copy()
        data_map["identifier"] = data_map["identifier"][:, -output_length:, :]

        data_map["inputs_date"] = data_map["date"].copy()
        data_map["date"] = data_map["date"][:, -output_length:, :]

        data_map["outputs"] = data_map["outputs"][:, -output_length:, :]

        return data_map

    def _get_input_columns(self):
        """Returns names of all input columns."""
        return [
            tup[0]
            for tup in self.get_column_definition()
            if tup[2] not in {InputTypes.ID, InputTypes.TIME, InputTypes.TARGET}
        ]

    @property
    def num_classes_per_cat_input(self):
        """Returns number of categories per relevant input.

        This is seqeuently required for keras embedding layers.
        """
        return self._num_classes_per_cat_input

    @property
    def input_params(self):
        """Returns the relevant indexes and input sizes."""

        # Functions
        def _extract_tuples_from_data_type(data_type, defn):
            return [
                tup
                for tup in defn
                if tup[1] == data_type
                and tup[2] not in {InputTypes.ID, InputTypes.TIME, InputTypes.TARGET}
            ]

        def _get_locations(input_types, defn):
            return [i for i, tup in enumerate(defn) if tup[2] in input_types]

        # Start extraction
        column_definition = [
            tup
            for tup in self.get_column_definition()
            if tup[2] not in {InputTypes.ID, InputTypes.TIME, InputTypes.TARGET}
        ]

        categorical_inputs = _extract_tuples_from_data_type(
            DataTypes.CATEGORICAL, column_definition
        )
        real_inputs = _extract_tuples_from_data_type(
            DataTypes.REAL_VALUED, column_definition
        )

        input_size = len(self._get_input_columns())

        # TODO artefact of previous code - need to clean up
        locations = {
            # "lstm_input_size": input_size,
            "input_size": input_size,
            "output_size": len(
                _get_locations({InputTypes.TARGET}, self._column_definition)
            ),
            "category_counts": self.num_classes_per_cat_input,
            # "input_obs_loc": _get_locations({InputTypes.TARGET}, column_definition),
            "static_input_loc": _get_locations(
                {InputTypes.STATIC_INPUT}, column_definition
            ),
            "known_regular_inputs": _get_locations(
                {InputTypes.STATIC_INPUT, InputTypes.KNOWN_INPUT}, real_inputs
            ),
            "known_categorical_inputs": _get_locations(
                {InputTypes.STATIC_INPUT, InputTypes.KNOWN_INPUT}, categorical_inputs
            ),
        }

        return locations
