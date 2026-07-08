"""TAQ (Trade and Quote) data cleaning utilities.

Requires: WRDS access. The Windows path resolver (_WindowsDownloadPath) uses
ctypes.windll and is only available on Windows. On other platforms, supply
parent_path explicitly to DataCleaner.clean().

The DataCleaner class reads the output of the SAS scripts in sas_scripts/ and
produces a clean intraday return matrix ready for the funcgarch estimators.
"""

import sys
import ctypes
from typing import Any
from uuid import UUID
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from ctypes import windll, wintypes
    _WINDOWS_AVAILABLE = True
except (ImportError, OSError):
    _WINDOWS_AVAILABLE = False


def _time_grid(start_time: str, end_time: str, interval_minutes: int) -> list[str]:
    """Generate a list of 'HH:MM' timestamps at regular intervals.

    Args:
        start_time: Start time in 'HH:MM' format, e.g. '09:30'.
        end_time: End time in 'HH:MM' format, e.g. '16:30'.
        interval_minutes: Step size in minutes.

    Returns:
        List of timestamp strings.
    """
    start = datetime.strptime(start_time, '%H:%M')
    end   = datetime.strptime(end_time,   '%H:%M')
    times = []
    cur   = start
    while cur <= end:
        times.append(cur.strftime('%H:%M'))
        cur += timedelta(minutes=interval_minutes)
    return times


def pivot_intraday(
    data: pd.DataFrame,
    market_open: str = '09:30',
    market_close: str = '16:30',
    interval_minutes: int = 1,
) -> pd.DataFrame:
    """Pivot raw TAQ price series into a (time, date) DataFrame.

    Resamples `data` to `interval_minutes`-minute bars and fills a matrix
    whose columns are trading dates and rows are intraday timestamps.

    Args:
        data: Price series with a DatetimeIndex.
        market_open: Market open time, default '09:30'.
        market_close: Market close time, default '16:30'.
        interval_minutes: Bar size in minutes.

    Returns:
        DataFrame of shape (n_bars, n_days) with NaN columns dropped.
    """
    def _make_dt(date, time_str):
        return pd.to_datetime(str(date).split(' ')[0] + ' ' + time_str)

    time_list    = _time_grid(market_open, market_close, interval_minutes)
    daily_dates  = data.resample('1d').last().index.unique()
    result       = pd.DataFrame(columns=daily_dates, index=time_list)
    data_resampled = data.resample(str(60 * interval_minutes) + 'S').last()

    for col in tqdm(result.columns[:-1]):
        start_dt = _make_dt(col, market_open)
        end_dt   = _make_dt(col, market_close)
        mask = (data_resampled.index >= start_dt) & (data_resampled.index <= end_dt)
        result[col] = data_resampled[mask].values

    return result.iloc[:-1].dropna(axis=1)


if _WINDOWS_AVAILABLE:
    class _GUID(ctypes.Structure):
        _fields_ = [
            ('Data1', wintypes.DWORD),
            ('Data2', wintypes.WORD),
            ('Data3', wintypes.WORD),
            ('Data4', wintypes.BYTE * 8),
        ]

        def __init__(self, uuid_str: str):
            uuid = UUID(uuid_str)
            super().__init__()
            self.Data1, self.Data2, self.Data3, \
                self.Data4[0], self.Data4[1], rest = uuid.fields
            for i in range(2, 8):
                self.Data4[i] = rest >> (8 - i - 1) * 8 & 0xFF

    class _WindowsDownloadPath:
        """Resolves the Windows 'Downloads' known folder path via SHGetKnownFolderPath."""

        _FOLDERID_Downloads = '{374DE290-123F-4565-9164-39C4925E467B}'

        def __init__(self):
            self._fn = windll.shell32.SHGetKnownFolderPath
            self._fn.argtypes = [
                ctypes.POINTER(_GUID), wintypes.DWORD,
                wintypes.HANDLE, ctypes.POINTER(ctypes.c_wchar_p),
            ]

        def _resolve(self, folder_id: str) -> str:
            ptr  = ctypes.c_wchar_p()
            guid = _GUID(folder_id)
            if self._fn(ctypes.byref(guid), 0, 0, ctypes.byref(ptr)):
                raise ctypes.WinError()
            return ptr.value

        def __call__(self) -> str:
            return self._resolve(self._FOLDERID_Downloads)

        def __repr__(self) -> str:
            return self._resolve(self._FOLDERID_Downloads)


class DataCleaner:
    """Clean TAQ CSV exports from WRDS SAS scripts into log-return matrices.

    Reads a CSV file produced by the SAS pipeline in sas_scripts/, applies
    intraday resampling, and exposes `data` (raw prices) and
    `transformed_data` (log-return matrix ready for `funcgarch.fit`).

    On Windows, the Downloads folder is auto-detected via SHGetKnownFolderPath.
    On other platforms, supply `parent_path` explicitly to `.clean()`.

    Example::

        cleaner = DataCleaner()
        cleaner.clean('my_taq_export', parent_path='/path/to/data/')
        mY = cleaner.transformed_data.values
    """

    def __init__(self, **kwargs) -> None:
        if _WINDOWS_AVAILABLE:
            self._path = _WindowsDownloadPath()
            self.path  = self._path()
        else:
            self._path = None
            self.path  = None
        for key, val in kwargs.items():
            self.__setattr__(key, val)

    @staticmethod
    def _resample(df: pd.DataFrame) -> pd.DataFrame:
        return pivot_intraday(df)

    @staticmethod
    def _log_diff_filled(df: pd.DataFrame) -> pd.DataFrame:
        """Log-returns with NaN filled by zero (preserves shape)."""
        return np.log(df).diff().fillna(0)

    @staticmethod
    def _log_diff_clean(df: pd.DataFrame) -> pd.DataFrame:
        """Log-returns with leading NaN dropped."""
        return np.log(df).diff().dropna()

    def clean(
        self,
        file_name: str,
        parent_path: str | None = None,
    ) -> None:
        """Load and clean a TAQ CSV export.

        Args:
            file_name: CSV filename (with or without .csv extension).
            parent_path: Directory containing the file; defaults to Downloads
                         on Windows.  Required on non-Windows platforms.
        """
        if parent_path is None:
            if self.path is None:
                raise OSError(
                    'parent_path is required on non-Windows platforms. '
                    'Pass the directory containing the CSV file.'
                )
            parent_path = self.path + '//'

        if not file_name.endswith('.csv'):
            file_name += '.csv'

        raw = pd.read_csv(parent_path + file_name, parse_dates=False)
        raw.index = pd.to_datetime(raw.DATE.astype(str) + ' ' + raw.itime_m.astype(str))
        raw.index.name = 'date'
        raw['log_returns'] = self._log_diff_filled(raw.iprice)

        self.data = raw[['iprice', 'SYM_ROOT', 'log_returns']].rename(
            columns={'SYM_ROOT': 'ticker', 'iprice': 'price'}
        )
        self.transformed_data = self._log_diff_clean(self._resample(self.data).astype(float))
        print('DataCleaner: dataset ready.')


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    cleaner = DataCleaner()
    cleaner.clean('mydata')
    data = cleaner.transformed_data

    plt.rcParams['figure.figsize'] = (24, 8)
    plt.plot(cleaner.data.price.resample('60S').asfreq())
    plt.show()
