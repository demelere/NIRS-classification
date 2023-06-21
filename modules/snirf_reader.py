from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd

COL_TIMESTAMP = "Timestamp"
COL_DURATION = "Duration"
COL_VALUE = "Value"
COL_EVENT = "Event"

COLS_DEFAULT = [COL_TIMESTAMP, COL_DURATION, COL_VALUE]


def read_snirf(
    filename: Union[str, Path]
) -> Tuple[np.ndarray, List[float], List[str], pd.DataFrame]:
    """Reads NIRS data and task events from a SNIRF file

    Parameters
    ----------
    filename : str | Path
        Path to the SNIRF file

    Returns
    -------
    data_nirs : np.ndarray
        An array of shape (num_timestamps, num_measurements)
    timestamps : List[float]
        The timestamps corresponding to the first dimension of `data_nirs`
    measurements : List[str]
        The names of the measurements corresponding to the second dimension of `data_nirs`
    df_events : pd.DataFrame
        DataFrame of task events, with columns (Timestamp, Duration, Value, Event)
    """
    data_nirs, timestamps, measurement_names = _get_data_from_snirf(filename)
    df_events = _get_events_from_snirf(filename)
    return data_nirs, timestamps, measurement_names, df_events


def _get_data_from_snirf(
    filename: Union[str, Path]
) -> Tuple[np.ndarray, np.ndarray, list]:
    with h5py.File(Path(filename).resolve(), "r") as file:
        data_nirs = np.array(file["nirs"]["data1"]["dataTimeSeries"])
        timestamps = np.array(file["nirs"]["data1"]["time"])
        unique_source_labels = np.array(file["nirs"]["probe"]["sourceLabels"]).astype(
            str
        )
        unique_detector_labels = np.array(
            file["nirs"]["probe"]["detectorLabels"]
        ).astype(str)
        measurement_names = []
        for idx_measurement in range(data_nirs.shape[1]):
            label_measurement = np.array(
                file["nirs"]["data1"][f"measurementList{idx_measurement+1}"][
                    "dataTypeLabel"
                ]
            ).astype(str)
            idx_detector = (
                np.array(
                    file["nirs"]["data1"][f"measurementList{idx_measurement+1}"][
                        "detectorIndex"
                    ]
                )
                - 1
            )
            idx_source = (
                np.array(
                    file["nirs"]["data1"][f"measurementList{idx_measurement+1}"][
                        "sourceIndex"
                    ]
                )
                - 1
            )
            measurement_names.append(
                f"{unique_source_labels[idx_source]}_{unique_detector_labels[idx_detector]} {label_measurement}"
            )
    return data_nirs, timestamps, measurement_names


def _get_events_from_snirf(filename: Union[str, Path]) -> Optional[pd.DataFrame]:

    df_events = pd.DataFrame()

    with h5py.File(Path(filename).resolve(), "r") as file:
        event_dfs = [
            pd.DataFrame(
                np.array(stim_info["data"]),
                columns=(
                    np.array(stim_info["dataLabels"]).astype(str)
                    if "dataLabels" in stim_info
                    else COLS_DEFAULT
                ),
            ).assign(
                Event=[np.array(stim_info["name"]).astype(str)]
                * len(np.array(stim_info["data"]))
            )
            for stim_name, stim_info in sorted(file["nirs"].items())
            if stim_name.startswith("stim")
        ]

    if not event_dfs:
        return
    df_events = pd.concat(event_dfs, ignore_index=True)

    df_events.sort_values(by=[COL_TIMESTAMP], inplace=True, ignore_index=True)
    df_events.reset_index(inplace=True, drop=True)
    df_events = df_events[
        [COL_TIMESTAMP, COL_EVENT, COL_DURATION]
        + [
            col
            for col in df_events.columns
            if col not in [COL_TIMESTAMP, COL_EVENT, COL_DURATION]
        ]
    ]

    # merge one-hot encoded columns
    col_prefixes = [col.split(".")[0] for col in df_events.columns]
    counter = Counter(col_prefixes)

    for col_prefix in set(col_prefixes):
        if counter[col_prefix] == 1 and col_prefix in df_events.columns:
            continue
        cols = [col for col in df_events.columns if col.startswith(col_prefix)]
        for col in cols:
            df_events.loc[df_events[col] == 1.0, col_prefix] = ".".join(
                col.split(".")[1:]
            )
        df_events.drop(columns=cols, inplace=True)

    return df_events
