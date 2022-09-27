import pandas as pd


class yieldMatSelector:
    """Class to define the criteria for subsetting the yield data."""

    def __init__(self):
        pass

    def adjustYieldSerie(self, df: pd.DataFrame, yieldtype: str, FD: bool) -> pd.DataFrame():

        """Method to select the correct yield maturities both for nominal and real values (hardcoded) as
        per assignment instructions and then calculates first difference if required."""

        if yieldtype == 'nominal':

            nominal_yields_2_10y = df[['Date', 'SVENY02', 'SVENY03', 'SVENY05', 'SVENY07',
                                       'SVENY10']].apply(lambda x: x / 100 if x.name not in 'Date' else x)
            if FD is False:
                nominal_yields_2_10y_eom = nominal_yields_2_10y[
                    nominal_yields_2_10y['Date'].dt.is_month_end == True].reset_index(inplace=False, drop=True)
                return nominal_yields_2_10y_eom
            else:
                nominal_yields_2_10y_shifted = nominal_yields_2_10y.apply(
                    lambda x: x.diff(1) if x.name not in 'Date' else x).dropna()
                nominal_yields_2_10y_eom_shifted = nominal_yields_2_10y_shifted[
                    nominal_yields_2_10y_shifted['Date'].dt.is_month_end == True].reset_index(inplace=False, drop=True)
                return nominal_yields_2_10y_eom_shifted

        else:

            real_yields_2_10y = df[['Date', 'TIPSY02', 'TIPSY03', 'TIPSY05', 'TIPSY07', 'TIPSY10']].apply(
                lambda x: x / 100 if x.name not in 'Date' else x)
            if FD is False:
                real_yields_2_10y_eom = real_yields_2_10y[
                    real_yields_2_10y['Date'].dt.is_month_end == True].reset_index(inplace=False, drop=True)
                return real_yields_2_10y_eom
            else:
                real_yields_2_10y_shifted = real_yields_2_10y.apply(
                    lambda x: x.diff(1) if x.name not in 'Date' else x).dropna()
                real_yields_2_10y_eom_shifted = real_yields_2_10y_shifted[
                    real_yields_2_10y_shifted['Date'].dt.is_month_end == True].reset_index(inplace=False, drop=True)
                return real_yields_2_10y_eom_shifted
