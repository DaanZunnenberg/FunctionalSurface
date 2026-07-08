"""TAQ (Trade and Quote) data cleaning utilities.

Requires: WRDS access and Windows (uses ctypes.windll to locate the Downloads folder).
The `DataCleaner` class reads the output of the SAS scripts in sas/ and produces
a clean intraday return matrix ready for the funcgarch estimators.
"""

from typing import Any, NoReturn
import os
import sys
import ctypes
from ctypes import windll, wintypes
from uuid import UUID
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm


def generate_time_list(start_time: str, end_time: str, interval_minutes: int) -> list[str]:
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


def transform(
    data: pd.DataFrame,
    open: str = '09:30',
    close: str = '16:30',
    interval_minutes: int = 1,
) -> pd.DataFrame:
    """Pivot raw TAQ price series into a (time, date) DataFrame.

    Resamples `data` to `interval_minutes`-minute bars and fills a matrix
    whose columns are trading dates and rows are intraday timestamps.

    Args:
        data: Price series with a DatetimeIndex.
        open: Market open time, default '09:30'.
        close: Market close time, default '16:30'.
        interval_minutes: Bar size in minutes.

    Returns:
        DataFrame of shape (n_bars, n_days) with NaN columns dropped.
    """
    mapper = lambda x, y: pd.to_datetime(str(x).split(' ')[0] + ' ' + y)
    time_list = generate_time_list(open, close, interval_minutes)
    daily_dates = data.resample('1d').last().index.unique()
    result = pd.DataFrame(columns=daily_dates, index=time_list)
    data = data.resample(str(60 * interval_minutes) + 'S').last()
    for col in tqdm(result.columns[:-1]):
        start_dt = mapper(col, open)
        end_dt   = mapper(col, close)
        result[col] = data[(data.index >= start_dt) & (data.index <= end_dt)].values
    return result.iloc[:-1].dropna(axis=1)


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


class DataCleaner:
    """Clean TAQ CSV exports from WRDS SAS scripts into log-return matrices.

    Reads a CSV file produced by the SAS pipeline in sas/, applies intraday
    resampling, and exposes `data` (raw prices) and `transformed_data`
    (log-return matrix ready for `funcgarch.fit`).

    Example::

        cleaner = DataCleaner()
        cleaner.clean('my_taq_export')
        mY = cleaner.transformed_data.values
    """

    def __init__(self, **kwargs) -> None:
        self._path = _WindowsDownloadPath()
        self.path  = self._path()
        for key, val in kwargs.items():
            self.__setattr__(key, val)

    @staticmethod
    def _resample(df: pd.DataFrame) -> pd.DataFrame:
        return transform(df)

    @staticmethod
    def _log_returns(df: pd.DataFrame) -> pd.DataFrame:
        return np.log(df).diff().fillna(0)

    @staticmethod
    def _log_changes(df: pd.DataFrame) -> pd.DataFrame:
        return np.log(df).diff().dropna()

    def clean(
        self,
        file_name: str,
        parent_path: str | None = None,
    ) -> None:
        """Load and clean a TAQ CSV export.

        Args:
            file_name: CSV filename (with or without .csv extension).
            parent_path: Directory containing the file; defaults to Downloads.
        """
        if parent_path is None:
            parent_path = self.path + '//'
        if not file_name.endswith('.csv'):
            file_name += '.csv'

        raw = pd.read_csv(parent_path + file_name, parse_dates=False)
        raw.index = pd.to_datetime(raw.DATE.astype(str) + ' ' + raw.itime_m.astype(str))
        raw.index.name = 'date'
        raw['log_returns'] = self._log_returns(raw.iprice)

        self.data = raw[['iprice', 'SYM_ROOT', 'log_returns']].rename(
            columns={'SYM_ROOT': 'ticker', 'iprice': 'price'}
        )
        self.transformed_data = self._log_changes(self._resample(self.data).astype(float))
        print('DataCleaner: dataset ready.')


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    cleaner = DataCleaner()
    cleaner.clean('mydata')
    data = cleaner.transformed_data

    plt.rcParams['figure.figsize'] = (24, 8)
    plt.plot(cleaner.data.price.resample('60S').asfreq())
    plt.show()
