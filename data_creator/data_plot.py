#%%
import mplfinance as mpf
import pandas as pd


class OHLCPlot:
    """ Allows for the creation of finance OHLC plots """

    def __init__(
        self,
        up_color: str = "lightgreen",
        down_color: str = "orangered",
        edge: str = "black",
        wick_up: str = "green",
        wick_down: str = "red",
        grid_color: str = "white",
        grid_style: str = "--",
        mpf_base: str = "mike",
    ):
        """
        Initialize the Theming of the plot

        Args:
            - up_color (str): Colour in MatplotLib
            - down_color (str): Colour in MatplotLib
            - edge (str): Colour in MatplotLib
            - wick_up (str): Colour in MatplotLib
            - wick_down (str): Colour in MatplotLib
            - grid_color (str): Colour in MatplotLib
            - grid_style(str): '-' for solid, '--' for dotted, '' for None
            - mpf_base(str): the base mpf style, https://github.com/matplotlib/mplfinance/blob/master/examples/styles.ipynb for available list
        """
        super(OHLCPlot).__init__()
        mc = mpf.make_marketcolors(
            up=up_color,
            down=down_color,
            edge=edge,
            wick={"up": wick_up, "down": wick_down},
            inherit=True,
        )
        s = mpf.make_mpf_style(
            base_mpf_style=mpf_base,
            gridcolor=grid_color,
            gridstyle=grid_style,
            marketcolors=mc,
            rc={
                "axes.labelcolor": "none",
                "axes.spines.bottom": False,
                "axes.spines.left": False,
                "axes.spines.right": False,
                "axes.spines.top": False,
            },
        )

        self.s = s

    def from_df(
        self,
        df: pd.DataFrame,
        save: bool = False,
        loc: str = "test.png",
        y_label: str = "Price",
        y_label_lower: str = "Volume",
        title: str = "",
        axis_off: bool = True,
    ):
        kwargs = dict(
            type="candlestick",
            volume=True,
            style=self.s,
            figratio=(6, 6),
            title=title,
            ylabel=y_label,
            ylabel_lower=y_label_lower,
            axisoff=axis_off,
            tight_layout=True,
        )
        if save:
            mpf.plot(df, **kwargs, savefig=loc)
        else:
            mpf.plot(df, **kwargs)
