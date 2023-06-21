from __future__ import annotations

from typing import TYPE_CHECKING

import nilearn.glm.contrasts
import nilearn.glm.first_level
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from typing import Dict, List, Optional, Tuple

    from nilearn.glm.regression import RegressionResults


_OUTPUT_STATS = ["stat", "p_value", "effect_size"]


def run_glm(
    data_nirs: np.ndarray,
    timestamps: np.ndarray,
    df_events: pd.DataFrame,
    conditions: Tuple[str, str],
    **glm_kwargs,
) -> np.ndarray:
    """Runs a GLM and returns statistics for the contrast `conditions[0] - conditions[1]`

    Parameters
    ----------
    data_nirs : np.ndarray
        2+D array, with time in the zeroth axis
    timestamps : np.ndarray
        Timestamps for each row in the zeroth dimension of `data_nirs`
    df_events : pd.DataFrame
        DataFrame describing the task events (e.g. blocks and block types)
    conditions : Tuple[str, str]
        2-tuple of block types defining the contrast to compute

    Returns
    -------
    data_glm_contrasts : np.ndarray
        GLM contrast results, with the last dimension of length 3 containing
        the values for the t-statistic, p value, and effect size
    """
    if len(conditions) != 2:
        raise ValueError("Specify two conditions for which to compute a contrast")

    df_onsets = _make_df_onsets(df_events, conditions)
    df_design_matrix = nilearn.glm.first_level.make_first_level_design_matrix(
        timestamps, df_onsets
    )

    labels, glm_estimates = _run_glm(data_nirs, df_design_matrix, **glm_kwargs)

    data_contrast = _compute_contrast(
        labels,
        glm_estimates,
        conditions,
        list(df_design_matrix.columns),
    )

    if data_contrast.ndim != data_nirs.ndim:
        data_contrast = data_contrast.reshape(
            *data_nirs.shape[1:], data_contrast.shape[-1]
        )

    return data_contrast


def _make_df_onsets(df_events: pd.DataFrame, conditions: List[str]):
    """helper function to make an onsets dataframe from an events dataframe

    Parameters
    ----------
    df_events: pd.DataFrame
        an events dataframe e.g. as returned by `flatten_taskv2_events()
    conditions: List[str]
        List of block types to extract

    Returns
    -------
    pd.DataFrame
        a 3-column dataframe ["trial_type", "onset", "duration"], ready for use in nilearn GLM

    """
    if not isinstance(conditions, (tuple, list)):
        raise TypeError("`conditions` must be a list of strings")

    onset_dfs: List[pd.DataFrame] = []
    for condition in conditions:
        is_row_selected = (df_events["Event"] == "StartBlock") & (
            df_events["BlockType"] == condition
        )
        if not is_row_selected.any():
            raise ValueError(f"No data for condition '{condition}'")

        df_condition = df_events.loc[is_row_selected, ["Timestamp", "Duration"]].rename(
            columns={"Timestamp": "onset", "Duration": "duration"}
        )
        df_condition.loc[:, "trial_type"] = condition
        onset_dfs.append(df_condition)

    df_onsets = pd.concat(onset_dfs, ignore_index=True)
    return df_onsets


def _run_glm(
    data_nirs: np.ndarray,
    df_design_matrix: pd.DataFrame,
    noise_model: str = "auto",
    bins: Optional[int] = None,
    n_jobs: int = -1,
    **kwargs,
) -> Tuple[np.ndarray, Dict[str, RegressionResults]]:
    """
    Run GLM on data using supplied design matrix.

    This is a wrapper function for nilearn.glm.first_level.run_glm

    Parameters
    ----------
    data_nirs : np.ndarray
        2+D array to be fit by the GLM, with time in the zeroth dimension
    df_design_matrix : pd.DataFrame
        The design matrix, as a pandas dataframe indexed by time (num_timepoints, num_regressors)
    noise_model : str, optional
        The temporal variance model, used to correct for autocorrelation in the
        timeseries. One of ['ols', 'arN', 'auto']. To specify the order of an
        autoregressive model place the order after the characters `ar`, for
        example to specify a third order model use `ar3`. If the string `auto`
        is provided a model with order 4 times the sample rate will be used.
    bins : Optional[int], optional
        Maximum number of discrete bins for the AR coef histogram/clustering.
        If an autoregressive model with order greater than one is specified
        then adaptive quantification is performed and the coefficients will be
        clustered via K-means with `bins` number of clusters. By default the
        value is None (default), which will set the number of bins to the
        number of channels, effectively estimating the AR model for each
        channel.
    n_jobs : int, optional
        The number of CPUs to use to do the computation. -1 (default) means 'all CPUs'
    **kwargs
        other keyword arguments to `nilearn.glm.first_level.run_glm`

    Returns
    -------
    labels : np.ndarray
        labels array of length num_features
    glm_estimates : dict
        Dictionary whose keys correspond to the different labels and values are RegressionResults instances.
    """
    if noise_model == "auto":
        sfreq = np.round(1.0 / np.median(np.diff(df_design_matrix.index.values)), 3)
        noise_model = f"ar{int(np.round(sfreq * 4))}"
    if bins is None:
        bins = min(1000, data_nirs.shape[1])

    Y = data_nirs.reshape(data_nirs.shape[0], -1)
    is_valid = ~np.any(np.isnan(Y), axis=0) & (np.std(Y, axis=0) >= np.finfo(float).eps)

    labels_valid, glm_estimates = nilearn.glm.first_level.run_glm(
        Y[:, is_valid],
        df_design_matrix.values,
        noise_model=noise_model,
        bins=bins,
        n_jobs=n_jobs,
        **kwargs,
    )

    labels = np.full(Y.shape[1], -9999, dtype=labels_valid.dtype)
    labels[is_valid] = labels_valid

    return labels, glm_estimates


def _compute_contrast(
    labels: np.ndarray,
    glm_estimates: Dict[str, RegressionResults],
    conditions: Tuple[str, str],
    design_columns: List[str],
) -> np.ndarray:
    """compute a contrast on the results of a glm regression

    Parameters
    ----------
    labels : np.ndarray
        an array indicating the key in glm_estimates that corresponds to each channel
    glm_estimates : dict
        the dictionary of RegressionResults instances
    conditions : Tuple[str, str]
        2-tuple of condition names defining the contrast
    design_columns : List[str]
        columns of the design matrix

    Returns
    -------
    data_contrast : np.ndarray
        Array of contrast statistics, with the last dimension of length 3
        containing the values for the t-statistic, p value, and effect size
    """
    con_val = nilearn.glm.contrasts.expression_to_contrast_vector(
        " - ".join(conditions), design_columns
    )

    # normalize contrast: all >0 sums to 1, all <0 sums to -1
    con_val[con_val > 0] = con_val[con_val > 0] / np.sum(con_val[con_val > 0])
    con_val[con_val < 0] = -con_val[con_val < 0] / np.sum(con_val[con_val < 0])

    if np.all(con_val == 0):
        raise ValueError("Contrast is null")

    con = nilearn.glm.contrasts.compute_contrast(
        labels,
        glm_estimates,
        con_val,
        contrast_type="t",
    )

    data_contrast = np.asarray([getattr(con, stat)() for stat in _OUTPUT_STATS])
    data_contrast = np.moveaxis(data_contrast, 0, -1)

    return data_contrast
