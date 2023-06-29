import itertools
from copy import deepcopy
from math import ceil
from typing import Union, Optional, List, Tuple, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots

from .._holder import DataHolder
from ..metrics import ActivityMetric, TransitionMetric, IdMetric, TraceMetric, UserMetric


def get_continuous_color(colorscale: List[list], intermed: float):
    if intermed <= 0 or len(colorscale) == 1:
        return colorscale[0][1]
    if intermed >= 1:
        return colorscale[-1][1]
    for cutoff, color in colorscale:
        if intermed > cutoff:
            low_cutoff, low_color = cutoff, color
        else:
            high_cutoff, high_color = cutoff, color
            break
    return px.colors.find_intermediate_color(lowcolor=low_color,
                                             highcolor=high_color,
                                             intermed=((intermed - low_cutoff) / (high_cutoff - low_cutoff)),
                                             colortype='rgb')


class ChartPainter:
    """
    Creates different types of interactive graphs using the Plotly library.

    Parameters
    ----------
    data: pandas.DataFrame or sberpm.DataHolder or sberpm.metrics instance
        Data to use for visualization.

    template: str, default='plotly'
        Name of the figure template. The following themes are available:
        'ggplot2', 'seaborn', 'simple_white', 'plotly', 'plotly_white',
        'plotly_dark', 'presentation', 'xgridoff', 'ygridoff', 'gridon',
        'none'.

    palette: str, default='sequential.Sunset_r'
        Name of the graph color palette. Must be the name of the Plotly
        continuous (sequential, diverging or cylclical) color scale
        from 'plotly.express.colors' submodules.

    shape_color: str, default='lime'
        Name of the color to use to draw new shapes in the figure.

    Examples
    --------
    >>> from sberpm.visual import ChartPainter
    >>> from sberpm.metrics import ActivityMetric
    >>> activity_metric = ActivityMetric(data_holder, time_unit='d')
    >>> painter = ChartPainter(activity_metric)
    >>> painter.bar(x=data_holder.activity_column, y='total_duration',
    >>>             sort='total_duration', n=50)
    """

    def __init__(self, data: Union[pd.DataFrame, DataHolder, ActivityMetric, TransitionMetric, IdMetric,
                                   TraceMetric, UserMetric],
                 template: str = 'plotly',
                 palette: str = 'sequential.Sunset_r',
                 shape_color: str = 'lime') -> None:
        self._dh = None
        if type(data) is DataHolder:
            self._data = data.data
            self._dh = data
        elif type(data) == pd.DataFrame:
            self._data = data
        elif type(data) in [ActivityMetric, TransitionMetric, IdMetric, TraceMetric, UserMetric]:
            self._data = data
        else:
            raise TypeError
        pio.templates.default = template
        px.defaults.color_discrete_sequence = eval('px.colors.' + palette)
        px.defaults.color_continuous_scale = eval('px.colors.' + palette)
        self._colors, _ = px.colors.convert_colors_to_same_type(eval('px.colors.' + palette))
        self._colorscale = px.colors.make_colorscale(self._colors)
        self._newshape_line_color = shape_color

        self._config = dict(toImageButtonOptions=dict(format='png',
                                                      height=None,
                                                      width=None,
                                                      scale=3),
                            modeBarButtonsToAdd=['drawline',
                                                 'drawopenpath',
                                                 'drawclosedpath',
                                                 'drawcircle',
                                                 'drawrect',
                                                 'eraseshape'],
                            showLink=True)

    def hist(self, x: Union[str, List[str]],
             color: Optional[str] = None,
             subplots: Optional[Tuple[str, str, int]] = None,
             barmode: str = 'stack',
             nbins: int = 50,
             cumulative: bool = False,
             orientation: str = 'v',
             opacity: float = 0.8,
             edge: bool = False,
             title: str = 'auto',
             slider: bool = False,
             height: Optional[int] = None,
             width: Optional[int] = None,
             font_size: int = 12,
             **kwargs: Optional[Any]) -> None:
        """
        Plots a histogram.

        Parameters
        ----------
        x: str or list of str
            Name of the column to make graph for. If it takes a list of
            column names, the input data is considered as wide-form
            rather than long-form.

        color: str, default=None
            Name of the column used to set color to bars.

        subplots: (rows, cols, ncols), default=None
            Creates a set of subplots:
            - rows: name of the column to use for constructing subplots
              along the y-axis.
            - cols: name of the column to use for constructing subplots
              along the x-axis.
            - ncols: number of columns of the subplot grid.

        barmode: {'stack', 'overlay', 'group'}, default='stack'
            Display mode of the bars with the same position coordinate.
            - If 'stack', the bars are stacked on top of each other.
            - If 'overlay', the bars are plotted over each other.
            - If 'group', the bars are placed side by side.

        nbins: int, default=50
            Number of bins.

        cumulative: bool, default=False
            Whether to plot a cumulative histogram.

        orientation: {'v', 'h'}, default='v'
            Orientation of the graph: 'v' for vertical and 'h' for horizontal.
            If 'h', the values are drawn along the y-axis.

        opacity: float, default=0.8
            Opacity of the bars. Ranges from 0 to 1.

        edge: bool, default=False
            Whether to draw bar edges.

        title: str, default='auto'
            Title of the graph. When 'auto', the title is generated automatically.

        slider: bool, default=False
            Whether to add a range slider to the plot.

        height: int, default=None
            Height of the figure in pixels.

        width: int, default=None
            Width of the figure in pixels.

        font_size: int, default=12
            Size of the global font.

        **kwargs: optional
            See 'plotly.express.histogram' for other possible arguments.
        """
        cols_or_lists = [el for el in [x, color, list(subplots[:2]) if subplots is not None else []] if el is not None]
        data = self._get_data(list(itertools.chain(*[[el] if isinstance(el, str) else el for el in cols_or_lists])))
        # data = self._data
        if barmode == 'stack':
            barmode = 'relative'
        if orientation == 'v':
            y = None
        else:
            y = x
            x = None
        if subplots:
            facet_row, facet_col, facet_col_wrap = subplots[0], subplots[1], subplots[2]
        else:
            facet_row, facet_col, facet_col_wrap = None, None, None
        if color:
            len_labels = data[color].nunique()
            nums = np.linspace(0, len_labels, len_labels) / len_labels
            color_discrete_sequence = [get_continuous_color(self._colorscale, z) for z in nums]
        else:
            color_discrete_sequence = None
        fig = px.histogram(data_frame=data,
                           x=x,
                           y=y,
                           color=color,
                           facet_row=facet_row,
                           facet_col=facet_col,
                           facet_col_wrap=facet_col_wrap,
                           color_discrete_sequence=color_discrete_sequence,
                           barmode=barmode,
                           nbins=nbins,
                           cumulative=cumulative,
                           opacity=opacity,
                           orientation=orientation,
                           height=height,
                           width=width,
                           **kwargs)
        if edge:
            fig.update_traces(marker_line=dict(color='black', width=1))
        if title == 'auto':
            if type(x) != list:
                title = f'Histogram of {x}'
            else:
                title = 'Histogram'
        if type(x) == list:
            legend = dict(orientation='h', x=0.5, xanchor='center', y=1.01, yanchor='bottom', title=None)
            margin = dict(l=5, r=5, t=80, b=5)
        else:
            legend = {}
            margin = dict(l=5, r=5, t=50, b=5)
        fig.update_layout(title=dict(text=title, x=0.5, xref='paper'),
                          legend=legend,
                          margin=margin,
                          font_size=font_size,
                          newshape_line_color=self._newshape_line_color)
        if slider:
            fig.update_xaxes(rangeslider_visible=True)
        fig.show(config=self._config)

    def bar(self, x: Optional[Union[str, List[str]]] = None,
            y: Optional[Union[str, List[str]]] = None,
            sort: Optional[str] = None,
            n: Optional[int] = None,
            color: Optional[str] = None,
            subplots: Optional[Tuple[str, str, int]] = None,
            barmode: str = 'stack',
            agg: str = None,
            add_line: Optional[List[str]] = None,
            text: bool = False,
            decimals: int = 2,
            orientation: str = 'auto',
            opacity: float = 1.0,
            edge: bool = False,
            title: str = 'auto',
            slider: bool = False,
            height: Optional[int] = None,
            width: Optional[int] = None,
            font_size: int = 12,
            **kwargs: Optional[Any]) -> None:
        """
        Makes a bar chart.

        Parameters
        ----------
        x: str or list of str, default=None
            Name of the column to draw on the x-axis. If it takes a list of
            column names, the input data is considered as wide-form
            rather than long-form.

        y: str or list of str, default=None
            Name of the column to draw on the y-axis. If it takes a list of
            column names, the input data is considered as wide-form
            rather than long-form.

        sort: str, default=None
            Name of the column to sort values by in descending (ascending)
            order if 'n' is positive (negative).

        n: int, default=None
            Number of sorted rows to draw. If positive, the rows are sorted in
            descending order; if negative, the rows are sorted in ascending order.

        color: str, default=None
            Name of the column used to set color to bars.

        subplots: (rows, cols, ncols), default=None
            Creates a set of subplots:
            - rows: name of the column to use for constructing subplots
              along the y-axis.
            - cols: name of the column to use for constructing subplots
              along the x-axis.
            - ncols: number of columns of the subplot grid.

        barmode: {'stack', 'overlay', 'group'}, default='stack'
            Display mode of the bars with the same position coordinate.
            - If 'stack', the bars are stacked on top of each other.
            - If 'overlay', the bars are plotted over each other.
            - If 'group', the bars are placed side by side.

        agg: {'count', 'sum', 'avg', 'min', 'max'}, default=None
            Name of the function used to aggregate 'y' ('x') values if
            orientation is set to 'v' ('h').

        add_line: list of str, default=None
            List of column names to add line to the graph for. Each line
            will be drawn along a separate y-axis.

        text: bool, default=False
            Whether to show text labels in the figure.

        decimals: int, default=2
            Number of decimal places to round 'text' labels of a float dtype to.

        orientation: {'auto', 'v', 'h'}, default='auto'
            Orientation of the graph: 'v' for vertical and 'h' for horizontal.
            By default, it is determined automatically based on the input
            data types.

        opacity: float, default=1.0
            Opacity of the bars. Ranges from 0 to 1.

        edge: bool, default=False
            Whether to draw bar edges.

        title: str, default='auto'
            Title of the graph. When 'auto', the title is generated automatically.

        slider: bool, default=False
            Whether to add a range slider to the plot.

        height: int, default=None
            Height of the figure in pixels.

        width: int, default=None
            Width of the figure in pixels.

        font_size: int, default=12
            Size of the global font.

        **kwargs: optional
            See 'plotly.express.bar' for other possible arguments.
        """
        if x is None and y is None:
            raise ValueError("Either 'x' or 'y' must be given")

        cols_or_lists = [x, y, sort, color, list(subplots[:2]) if subplots is not None else [], add_line]
        cols_or_lists = [el for el in cols_or_lists if el is not None]
        data = self._get_data(list(itertools.chain(*[[el] if isinstance(el, str) else el for el in cols_or_lists])))
        if sort:
            if n > 0:
                data = data.sort_values(by=sort, ascending=False).head(n)
            elif n < 0:
                data = data.sort_values(by=sort, ascending=True).head(n)
        if type(x) != list and type(y) != list:
            if y and pd.api.types.is_numeric_dtype(data[y]):
                continuous, categorical = y, x
                autorange = True
                if pd.api.types.is_integer_dtype(data[y]):
                    texttemplate = '%{y:d}'
                else:
                    texttemplate = f'%{{y:.{decimals}f}}'
            else:
                continuous, categorical = x, y
                autorange = 'reversed'
                if pd.api.types.is_integer_dtype(data[x]):
                    texttemplate = '%{x:d}'
                else:
                    texttemplate = f'%{{x:.{decimals}f}}'
        else:
            continuous, categorical = None, None
            if y and type(y) == list and pd.api.types.is_numeric_dtype(data[y[0]]):
                if pd.api.types.is_integer_dtype(data[y]):
                    texttemplate = '%{y:d}'
                else:
                    texttemplate = f'%{{y:.{decimals}f}}'
                autorange = True
            else:
                if pd.api.types.is_integer_dtype(data[x]):
                    texttemplate = '%{x:d}'
                else:
                    texttemplate = f'%{{x:.{decimals}f}}'
                autorange = 'reversed'
        if barmode == 'stack':
            barmode = 'relative'
        if orientation == 'auto':
            orientation = None
        if subplots:
            facet_row, facet_col, facet_col_wrap = subplots[0], subplots[1], subplots[2]
        else:
            facet_row, facet_col, facet_col_wrap = None, None, None
        if color:
            len_labels = data[color].nunique()
            nums = np.linspace(0, len_labels, len_labels) / len_labels
            color_discrete_sequence = [get_continuous_color(self._colorscale, z) for z in nums]
            color_discrete_map = {}
        else:
            color_discrete_map = {}
            color_discrete_sequence = None
            if categorical and data[categorical].nunique() == len(data):
                len_labels = len(data[categorical])
                nums = np.linspace(0, len_labels, len_labels) / len_labels
                color = [get_continuous_color(self._colorscale, z) for z in nums]
                color_discrete_map = 'identity'
        if add_line:
            opacity = 0.8
            color = None
            color_discrete_map = {}
            color_discrete_sequence = None
        if agg:
            fig = px.histogram(data_frame=data,
                               x=x,
                               y=y,
                               color=color,
                               facet_row=facet_row,
                               facet_col=facet_col,
                               facet_col_wrap=facet_col_wrap,
                               color_discrete_sequence=color_discrete_sequence,
                               color_discrete_map=color_discrete_map,
                               barmode=barmode,
                               opacity=opacity,
                               histfunc=agg,
                               orientation=orientation,
                               height=height,
                               width=width,
                               **kwargs)
            fig.update_layout(bargap=0.2)
        else:
            fig = px.bar(data_frame=data,
                         x=x,
                         y=y,
                         color=color,
                         facet_row=facet_row,
                         facet_col=facet_col,
                         facet_col_wrap=facet_col_wrap,
                         color_discrete_sequence=color_discrete_sequence,
                         color_discrete_map=color_discrete_map,
                         barmode=barmode,
                         opacity=opacity,
                         orientation=orientation,
                         height=height,
                         width=width,
                         **kwargs)
            if add_line and continuous and categorical:
                fig.data[-1].name = continuous
                layout_a = {}
                for i, a in enumerate(add_line):
                    if categorical == x:
                        x_a = data[x]
                        y_a = data[a]
                    else:
                        y_a = data[y]
                        x_a = data[a]
                    fig.add_trace(go.Scatter(x=x_a,
                                             y=y_a,
                                             name=a,
                                             mode='lines',
                                             marker_color=px.colors.qualitative.Plotly[i + 1 % 10],
                                             line_width=3,
                                             yaxis='y' + str(i + 2)))
                    layout_a[f'yaxis{i + 2}'] = dict(title=a,
                                                     overlaying='y',
                                                     anchor='free',
                                                     side='right',
                                                     showgrid=False,
                                                     position=1 - i * 0.07,
                                                     titlefont_color=px.colors.qualitative.Plotly[i + 1 % 10],
                                                     tickfont_color=px.colors.qualitative.Plotly[i + 1 % 10])
                fig.update_traces(hovertemplate=None,
                                  showlegend=True)
                fig.update_layout(hovermode='x',
                                  xaxis_domain=[0, 1 - 0.07 * (len(add_line) - 1)],
                                  **layout_a)
        if text:
            fig.update_traces(texttemplate=texttemplate,
                              textposition='outside',
                              selector=dict(type='bar'))
        else:
            fig.update_traces(textposition='none',
                              selector=dict(type='bar'))
        if edge:
            fig.update_traces(marker_line=dict(color='black', width=0.5))
        if title == 'auto':
            if continuous:
                title = f'Bar Chart of {continuous}'
            else:
                title = 'Bar Chart'
        if type(x) == list or type(y) == list or add_line:
            legend = dict(orientation='h', x=0.5, xanchor='center', y=1.01, yanchor='bottom', title=None)
            margin = dict(l=5, r=5, t=80, b=5)
        else:
            legend = {}
            margin = dict(l=5, r=5, t=50, b=5)
        fig.update_layout(title=dict(text=title, x=0.5, xref='paper'),
                          legend=legend,
                          margin=margin,
                          yaxis_autorange=autorange,
                          font_size=font_size,
                          newshape_line_color=self._newshape_line_color)
        if slider:
            fig.update_xaxes(rangeslider_visible=True)
        fig.show(config=self._config)

    def box(self, x: Optional[Union[str, List[str]]] = None,
            y: Optional[Union[str, List[str]]] = None,
            color: Optional[str] = None,
            subplots: Optional[Tuple[str, str, int]] = None,
            boxmode: str = 'group',
            points: str = 'outliers',
            orientation: str = 'auto',
            title: str = 'auto',
            height: Optional[int] = None,
            width: Optional[int] = None,
            font_size: int = 12,
            **kwargs: Optional[Any]) -> None:
        """
        Makes a box plot.

        Parameters
        ----------
        x: str or list of str, default=None
            Name of the column to draw on the x-axis. If it takes a list of
            column names, the input data is considered as wide-form
            rather than long-form.

        y: str or list of str, default=None
            Name of the column to draw on the y-axis. If it takes a list of
            column names, the input data is considered as wide-form
            rather than long-form.

        color: str, default=None
            Name of the column used to set color to boxes.

        subplots: (rows, cols, ncols), default=None
            Creates a set of subplots:
            - rows: name of the column to use for constructing subplots
              along the y-axis.
            - cols: name of the column to use for constructing subplots
              along the x-axis.
            - ncols: number of columns of the subplot grid.

        boxmode: {'group', 'overlay'}, default='group'
            Display mode of the boxes with the same position coordinate.
            - If 'group', the boxes are placed side by side.
            - If 'overlay', the boxes are plotted over each other.

        points: {'all', 'outliers', 'suspectedoutliers', 'False'},
            default='outliers'
            Type of underlying data points to display. Can be either all points,
            outliers only, suspected outliers only, or none of them.

        orientation: {'auto', 'v', 'h'}, default='auto'
            Orientation of the graph: 'v' for vertical and 'h' for horizontal.
            By default, it is determined automatically based on the input
            data types.

        title: str, default='auto'
            Title of the graph. When 'auto', the title is generated automatically.

        height: int, default=None
            Height of the figure in pixels.

        width: int, default=None
            Width of the figure in pixels.

        font_size: int, default=12
            Size of the global font.

        **kwargs: optional
            See 'plotly.express.box' for other possible arguments.
        """
        if x is None and y is None:
            raise ValueError("Either 'x' or 'y' must be given")

        cols_or_lists = [x, y, color, list(subplots[:2]) if subplots is not None else []]
        cols_or_lists = [el for el in cols_or_lists if el is not None]
        data = self._get_data(list(itertools.chain(*[[el] if isinstance(el, str) else el for el in cols_or_lists])))

        if orientation == 'auto':
            orientation = None
        if subplots:
            facet_row, facet_col, facet_col_wrap = subplots[0], subplots[1], subplots[2]
        else:
            facet_row, facet_col, facet_col_wrap = None, None, None
        if color:
            len_labels = data[color].nunique()
            nums = np.linspace(0, len_labels, len_labels) / len_labels
            color_discrete_sequence = [get_continuous_color(self._colorscale, z) for z in nums]
        else:
            color_discrete_sequence = None
        fig = px.box(data_frame=data,
                     x=x,
                     y=y,
                     color=color,
                     facet_row=facet_row,
                     facet_col=facet_col,
                     facet_col_wrap=facet_col_wrap,
                     boxmode=boxmode,
                     color_discrete_sequence=color_discrete_sequence,
                     points=points,
                     orientation=orientation,
                     height=height,
                     width=width,
                     **kwargs)
        if title == 'auto':
            if y and type(y) != list and pd.api.types.is_numeric_dtype(data[y]):
                title = f'Box Plot of {y}'
            elif x and type(x) != list and pd.api.types.is_numeric_dtype(data[x]):
                title = f'Box Plot of {x}'
            else:
                title = 'Box Plot'
        fig.update_layout(title=dict(text=title, x=0.5, xref='paper'),
                          margin=dict(l=5, r=5, t=50, b=5),
                          font_size=font_size,
                          newshape_line_color=self._newshape_line_color)
        fig.show(config=self._config)

    def scatter(self, x: Optional[Union[str, List[str]]] = None,
                y: Optional[Union[str, List[str]]] = None,
                sort: Optional[str] = None,
                n: Optional[int] = None,
                color: Optional[str] = None,
                size: Optional[Union[str, int]] = None,
                symbol: Optional[str] = None,
                subplots: Optional[Tuple[str, str, int]] = None,
                text: Optional[str] = None,
                decimals: int = 2,
                size_max: int = 20,
                orientation: str = 'auto',
                opacity: float = 1.0,
                edge: bool = False,
                title: str = 'auto',
                slider: bool = False,
                height: Optional[int] = None,
                width: Optional[int] = None,
                font_size: int = 12,
                **kwargs: Optional[Any]) -> None:
        """
        Makes a scatter plot.

        Parameters
        ----------
        x: str or list of str, default=None
            Name of the column to draw on the x-axis. If it takes a list of
            column names, the input data is considered as wide-form
            rather than long-form.

        y: str or list of str, default=None
            Name of the column to draw on the y-axis. If it takes a list of
            column names, the input data is considered as wide-form
            rather than long-form.

        sort: str, default=None
            Name of the column to sort values by in descending (ascending)
            order if 'n' is positive (negative).

        n: int, default=None
            Number of sorted rows to draw. If positive, the rows are sorted in
            descending order; if negative, the rows are sorted in ascending order.

        color: str, default=None
            Name of the column used to set color to markers.

        size: str or int, default=None
            If str, it is a name of the column used to set marker sizes.
            If integer, it defines the marker size.

        symbol: str, default=None
            Name of the column used to set symbols to markers.

        subplots: (rows, cols, ncols), default=None
            Creates a set of subplots:
            - rows: name of the column to use for constructing subplots
              along the y-axis.
            - cols: name of the column to use for constructing subplots
              along the x-axis.
            - ncols: number of columns of the subplot grid.

        text: str, default=None
            Name of the column to use as text labels in the figure.

        decimals: int, default=2
            Number of decimal places to round 'text' labels of a float dtype to.

        size_max: int, default=20
            The maximum marker size. Used if 'size' is given.

        orientation: {'auto', 'v', 'h'}, default='auto'
            Orientation of the graph: 'v' for vertical and 'h' for horizontal.
            By default, it is determined automatically based on the input
            data types.

        opacity: float, default=1.0
            Opacity of the markers. Ranges from 0 to 1.

        edge: bool, default=False
            Whether to draw marker edges.

        title: str, default='auto'
            Title of the graph. When 'auto', the title is generated automatically.

        slider: bool, default=False
            Whether to add a range slider to the plot.

        height: int, default=None
            Height of the figure in pixels.

        width: int, default=None
            Width of the figure in pixels.

        font_size: int, default=12
            Size of the global font.

        **kwargs: optional
            See 'plotly.express.scatter' for other possible arguments.
        """
        if x is None and y is None:
            raise ValueError("Either 'x' or 'y' must be given")

        cols_or_lists = [x, y, sort, color, list(subplots[:2]) if subplots is not None else [], symbol, text]
        if isinstance(size, str):
            cols_or_lists.append(size)
        cols_or_lists = [el for el in cols_or_lists if el is not None]
        data = self._get_data(list(itertools.chain(*[[el] if isinstance(el, str) else el for el in cols_or_lists])))

        if sort:
            if n > 0:
                data = data.sort_values(by=sort, ascending=False).head(n)
            elif n < 0:
                data = data.sort_values(by=sort, ascending=True).head(n)
        if pd.api.types.is_number(size):
            marker_size = size
            size = None
        else:
            marker_size = None
        if orientation == 'auto':
            orientation = None
        if subplots:
            facet_row, facet_col, facet_col_wrap = subplots[0], subplots[1], subplots[2]
        else:
            facet_row, facet_col, facet_col_wrap = None, None, None
        if color:
            len_labels = data[color].nunique()
            nums = np.linspace(0, len_labels, len_labels) / len_labels
            color_discrete_sequence = [get_continuous_color(self._colorscale, z) for z in nums]
        else:
            color_discrete_sequence = None
        fig = px.scatter(data_frame=data,
                         x=x,
                         y=y,
                         color=color,
                         symbol=symbol,
                         size=size,
                         text=text,
                         facet_row=facet_row,
                         facet_col=facet_col,
                         facet_col_wrap=facet_col_wrap,
                         color_discrete_sequence=color_discrete_sequence,
                         opacity=opacity,
                         size_max=size_max,
                         orientation=orientation,
                         height=height,
                         width=width,
                         **kwargs)
        if edge:
            fig.update_traces(marker_line=dict(color='black', width=0.5))
        if marker_size:
            fig.update_traces(marker_size=marker_size)
        if text:
            if pd.api.types.is_integer_dtype(data[text]):
                texttemplate = '%{text:d}'
            elif pd.api.types.is_float_dtype(data[text]):
                texttemplate = f'%{{text:.{decimals}f}}'
            else:
                texttemplate = '%{text}'
            fig.update_traces(texttemplate=texttemplate,
                              textposition='middle right')
        if title == 'auto':
            if y and type(y) != list and pd.api.types.is_numeric_dtype(data[y]):
                title = f'Scatter Plot of {y}'
            elif x and type(x) != list and pd.api.types.is_numeric_dtype(data[x]):
                title = f'Scatter Plot of {x}'
            else:
                title = 'Scatter Plot'
        if (y and type(y) != list and pd.api.types.is_numeric_dtype(data[y])) or (
                y and type(y) == list and pd.api.types.is_numeric_dtype(data[y[0]])):
            autorange = True
        else:
            autorange = 'reversed'
        if type(x) == list or type(y) == list:
            legend = dict(orientation='h', x=0.5, xanchor='center', y=1.01, yanchor='bottom', title=None)
            margin = dict(l=5, r=5, t=80, b=5)
        else:
            legend = {}
            margin = dict(l=5, r=5, t=50, b=5)
        fig.update_layout(title=dict(text=title, x=0.5, xref='paper'),
                          legend=legend,
                          margin=margin,
                          yaxis_autorange=autorange,
                          font_size=font_size,
                          newshape_line_color=self._newshape_line_color)
        if slider:
            fig.update_xaxes(rangeslider_visible=True)
        fig.show(config=self._config)

    def line(self, x: Optional[Union[str, List[str]]] = None,
             y: Optional[Union[str, List[str]]] = None,
             sort: Optional[str] = None,
             n: Optional[int] = None,
             color: Optional[str] = None,
             group: Optional[str] = None,
             dash: Optional[str] = None,
             subplots: Optional[Tuple[str, str, int]] = None,
             text: Optional[str] = None,
             decimals: int = 2,
             orientation: str = 'auto',
             line_width: int = 2,
             title: str = 'auto',
             slider: bool = False,
             height: Optional[int] = None,
             width: Optional[int] = None,
             font_size: int = 12,
             **kwargs: Optional[Any]) -> None:
        """
        Makes a line plot.

        Parameters
        ----------
        x: str or list of str, default=None
            Name of the column to draw on the x-axis. If it takes a list of
            column names, the input data is considered as wide-form
            rather than long-form.

        y: str or list of str, default=None
            Name of the column to draw on the y-axis. If it takes a list of
            column names, the input data is considered as wide-form
            rather than long-form.

        sort: str, default=None
            Name of the column to sort values by in descending (ascending)
            order if 'n' is positive (negative).

        n: int, default=None
            Number of sorted rows to draw. If positive, the rows are sorted in
            descending order; if negative, the rows are sorted in ascending order.

        color: str, default=None
            Name of the column used to set color to lines.

        group: str, default=None
            Name of the column used to group data rows into lines.

        dash: str, default=None
            Name of the column used to set dash patterns to lines.

        subplots: (rows, cols, ncols), default=None
            Creates a set of subplots:
            - rows: name of the column to use for constructing subplots
              along the y-axis.
            - cols: name of the column to use for constructing subplots
              along the x-axis.
            - ncols: number of columns of the subplot grid.

        text: str, default=None
            Name of the column to use as text labels in the figure.

        decimals: int, default=2
            Number of decimal places to round 'text' labels of a float dtype to.

        orientation: {'auto', 'v', 'h'}, default='auto'
            Orientation of the graph: 'v' for vertical and 'h' for horizontal.
            By default, it is determined automatically based on the input
            data types.

        line_width: int, default=2
            Width of the line(s).

        title: str, default='auto'
            Title of the graph. When 'auto', the title is generated automatically.

        slider: bool, default=False
            Whether to add a range slider to the plot.

        height: int, default=None
            Height of the figure in pixels.

        width: int, default=None
            Width of the figure in pixels.

        font_size: int, default=12
            Size of the global font.

        **kwargs: optional
            See 'plotly.express.line' for other possible arguments.
        """
        if x is None and y is None:
            raise ValueError("Either 'x' or 'y' must be given")

        cols_or_lists = [x, y, sort, color, list(subplots[:2]) if subplots is not None else [], group, dash, text]
        cols_or_lists = [el for el in cols_or_lists if el is not None]
        data = self._get_data(list(itertools.chain(*[[el] if isinstance(el, str) else el for el in cols_or_lists])))

        if sort:
            if n > 0:
                data = data.sort_values(by=sort, ascending=False).head(n)
            elif n < 0:
                data = data.sort_values(by=sort, ascending=True).head(n)
        if orientation == 'auto':
            orientation = None
        if subplots:
            facet_row, facet_col, facet_col_wrap = subplots[0], subplots[1], subplots[2]
        else:
            facet_row, facet_col, facet_col_wrap = None, None, None
        if color:
            len_labels = data[color].nunique()
            nums = np.linspace(0, len_labels, len_labels) / len_labels
            color_discrete_sequence = [get_continuous_color(self._colorscale, z) for z in nums]
        else:
            color_discrete_sequence = None
        fig = px.line(data_frame=data,
                      x=x,
                      y=y,
                      color=color,
                      line_group=group,
                      line_dash=dash,
                      text=text,
                      facet_row=facet_row,
                      facet_col=facet_col,
                      facet_col_wrap=facet_col_wrap,
                      color_discrete_sequence=color_discrete_sequence,
                      orientation=orientation,
                      height=height,
                      width=width,
                      **kwargs)
        fig.update_traces(line_width=line_width)
        if text:
            if pd.api.types.is_integer_dtype(data[text]):
                texttemplate = '%{text:d}'
            elif pd.api.types.is_float_dtype(data[text]):
                texttemplate = f'%{{text:.{decimals}f}}'
            else:
                texttemplate = '%{text}'
            fig.update_traces(texttemplate=texttemplate,
                              textposition='top right')
        if title == 'auto':
            if y and type(y) != list and pd.api.types.is_numeric_dtype(data[y]):
                title = f'Line Plot of {y}'
            elif x and type(x) != list and pd.api.types.is_numeric_dtype(data[x]):
                title = f'Line Plot of {x}'
            else:
                title = 'Line Plot'
        if (y and type(y) != list and pd.api.types.is_numeric_dtype(data[y])) or (
                y and type(y) == list and pd.api.types.is_numeric_dtype(data[y[0]])):
            autorange = True
        else:
            autorange = 'reversed'
        if type(x) == list or type(y) == list:
            legend = dict(orientation='h', x=0.5, xanchor='center', y=1.01, yanchor='bottom', title=None)
            margin = dict(l=5, r=5, t=80, b=5)
        else:
            legend = {}
            margin = dict(l=5, r=5, t=50, b=5)
        fig.update_layout(title=dict(text=title, x=0.5, xref='paper'),
                          legend=legend,
                          margin=margin,
                          yaxis_autorange=autorange,
                          font_size=font_size,
                          newshape_line_color=self._newshape_line_color)
        if slider:
            fig.update_xaxes(rangeslider_visible=True)
        fig.show(config=self._config)

    def pie(self, labels: str,
            values: Optional[str] = None,
            color: Optional[str] = None,
            n: Optional[int] = None,
            remainder: bool = True,
            text: str = 'percent',
            text_orientation: str = 'auto',
            hole: float = 0.4,
            opacity: float = 1.0,
            edge: bool = True,
            title: str = 'auto',
            height: Optional[int] = None,
            width: Optional[int] = None,
            font_size: int = 12,
            **kwargs: Optional[Any]) -> None:
        """
        Makes a pie chart.

        Parameters
        ----------
        labels: str
            Name of the column to use as labels for sectors.

        values: str, default=None
            Name of the column used to set values to sectors.

        color: str, default=None
            Name of the column used to set color to sectors.

        n: int, default=None
            Number of sorted rows to draw. If positive, the rows are sorted in
            descending order; if negative, the rows are sorted in ascending order.

        remainder: bool, default=True
            Whether to put the remaining values other than 'n' selected into
            a separate sector.

        text: {'percent', 'value'}, default='percent'
            Text information to display inside sectors.
            Can be either 'percent' or 'value'.

        text_orientation: {'auto', 'horizontal', 'radial', 'tangential'},
            default='auto'
            Orientation of text inside sectors.
            - If 'auto', text is oriented to be as big as possible in the middle
              of the sector.
            - If 'horizontal', text is oriented to be parallel with the bottom
              of the chart.
            - If 'radial', text is oriented along the radius of the sector.
            - If 'tangential', text is oriented perpendicular to the radius
              of the sector.

        hole: float, default=0.4
            Fraction of the radius to cut out of the pie to create a donut chart.
            Ranges from 0 to 1.

        opacity: float, default=1.0
            Opacity of the sectors. Ranges from 0 to 1.

        edge: bool, default=True
            Whether to draw sector edges.

        title: str, default='auto'
            Title of the graph. When 'auto', the title is generated automatically.

        height: int, default=None
            Height of the figure in pixels.

        width: int, default=None
            Width of the figure in pixels.

        font_size: int, default=12
            Size of the global font.

        **kwargs: optional
            See 'plotly.express.pie' for other possible arguments.
        """
        cols_or_lists = [labels, values, color]
        cols_or_lists = [el for el in cols_or_lists if el is not None]
        data = self._get_data(list(itertools.chain(*[[el] if isinstance(el, str) else el for el in cols_or_lists])))

        labels_input = labels
        values_input = values
        if not values:
            data = data[labels].value_counts()
            data_copy = data.copy()
            if n and n > 0:
                data = data.head(n)
                if remainder:
                    if data_copy[n:].sum() > 0:
                        data['Other'] = data_copy[n:].sum()
            elif n and n < 0:
                data = data.tail(-n)
                if remainder:
                    if data_copy[:-n].sum() > 0:
                        data['Other'] = data_copy[:-n].sum()
            data = pd.DataFrame(data).reset_index()
            values = labels
            labels = 'index'
            if text == 'percent':
                hovertemplate = values + '=%{label}<br>count=%{value}'
            else:  # text == 'value'
                hovertemplate = values + '=%{label}<br>percent=%{percent}'
        else:
            if n:
                data = data.sort_values(by=values, ascending=False).set_index(labels)[values]
                data_copy = data.copy()
                if n > 0:
                    data = data.head(n)
                    if remainder:
                        if data_copy[n:].sum() > 0:
                            data['Other'] = data_copy[n:].sum()
                elif n < 0:
                    data = data.tail(-n)
                    if remainder:
                        if data_copy[:-n].sum() > 0:
                            data['Other'] = data_copy[:-n].sum()
            data = pd.DataFrame(data).reset_index()
        len_labels = data[labels].nunique()
        nums = np.linspace(0, len_labels, len_labels) / len_labels
        color_discrete_sequence = [get_continuous_color(self._colorscale, z) for z in nums]
        fig = px.pie(data_frame=data,
                     names=labels,
                     values=values,
                     color=color,
                     color_discrete_sequence=color_discrete_sequence,
                     opacity=opacity,
                     hole=hole,
                     height=height,
                     width=width,
                     **kwargs)
        fig.update_traces(sort=False,
                          textinfo=text,
                          insidetextorientation=text_orientation)
        if edge:
            fig.update_traces(marker_line=dict(color='white', width=1))
        if not values_input:
            fig.update_traces(hovertemplate=hovertemplate)
        if title == 'auto':
            title = f'Pie Chart of {labels_input}'
        fig.update_layout(title=dict(text=title, x=0.5, xref='paper'),
                          legend_title=labels_input,
                          margin=dict(l=5, r=5, t=50, b=5),
                          font_size=font_size,
                          newshape_line_color=self._newshape_line_color)
        fig.show(config=self._config)

    def sunburst(self, path: List[str],
                 values: Optional[str] = None,
                 color: Optional[str] = None,
                 maxdepth: int = -1,
                 text_orientation: str = 'auto',
                 title: str = 'auto',
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 font_size: int = 12,
                 **kwargs: Optional[Any]) -> None:
        """
        Makes a sunburst plot.

        Parameters
        ----------
        path: list of str
            Names of the columns that correspond to different levels of
            the hierarchy of sectors, from root to leaves.

        values: str, default=None
            Name of the column used to set values to sectors.

        color: str, default=None
            Name of the column used to set color to sectors.

        maxdepth: int, default=-1
            Number of displayed sectors from any level. If -1, all levels
            in the hierarchy are shown.

        text_orientation: {'auto', 'horizontal', 'radial', 'tangential'},
            default='auto'
            Orientation of text inside sectors.
            - If 'auto', text is oriented to be as big as possible in the middle
              of the sector.
            - If 'horizontal', text is oriented to be parallel with the bottom
              of the chart.
            - If 'radial', text is oriented along the radius of the sector.
            - If 'tangential', text is oriented perpendicular to the radius
              of the sector.

        title: str, default='auto'
            Title of the graph. When 'auto', the title is generated automatically.

        height: int, default=None
            Height of the figure in pixels.

        width: int, default=None
            Width of the figure in pixels.

        font_size: int, default=12
            Size of the global font.

        **kwargs: optional
            See 'plotly.express.sunburst' for other possible arguments.
        """
        cols_or_lists = [path, values, color]
        cols_or_lists = [el for el in cols_or_lists if el is not None]
        data = self._get_data(list(itertools.chain(*[[el] if isinstance(el, str) else el for el in cols_or_lists])))

        len_labels = data[path[0]].nunique()
        nums = np.linspace(0, len_labels, len_labels) / len_labels
        color_discrete_sequence = [get_continuous_color(self._colorscale, z) for z in nums]
        fig = px.sunburst(data_frame=data,
                          path=path,
                          values=values,
                          color=color,
                          color_discrete_sequence=color_discrete_sequence,
                          maxdepth=maxdepth,
                          height=height,
                          width=width,
                          **kwargs)
        fig.update_traces(insidetextorientation=text_orientation)
        if title == 'auto':
            title = f'Sunburst Plot of {path[0]}'
        fig.update_layout(title=dict(text=title, x=0.5, xref='paper'),
                          margin=dict(l=5, r=5, t=50, b=5),
                          font_size=font_size,
                          newshape_line_color=self._newshape_line_color)
        fig.show(config=self._config)

    def heatmap(self, labels: Optional[Tuple[str, str, str]] = None,
                text: bool = False,
                decimals: int = 2,
                xaxis_side: str = 'bottom',
                title: str = 'auto',
                height: Optional[int] = None,
                width: Optional[int] = None,
                font_size: int = 12,
                **kwargs: Optional[Any]) -> None:
        """
        Makes a heatmap.

        Parameters
        ----------
        labels: (x, y, color), default=None
            Label names to display in the figure for axis (x and y) and
            colorbar (color) titles and hover boxes.

        text: bool, default=False
            Whether to show annotation text in the figure.

        decimals: int, default=2
            Number of decimal places to round annotations to.

        xaxis_side: {'bottom', 'top'}, default='bottom'
            Position of the x-axis in the figure.

        title: str, default='auto'
            Title of the graph. When 'auto', the title is generated automatically.

        height: int, default=None
            Height of the figure in pixels.

        width: int, default=None
            Width of the figure in pixels.

        font_size: int, default=12
            Size of the global font.

        **kwargs: optional
            See 'plotly.figure_factory.create_annotated_heatmap' for other
            possible arguments.
        """
        if not isinstance(self._data, pd.DataFrame):
            raise TypeError('Input data must be given as a pandas.DataFrame')
        data = self._data
        if labels:
            xaxis_title, yaxis_title, color_title = labels
        else:
            xaxis_title, yaxis_title, color_title = None, None, None
            if data.columns.name:
                xaxis_title = data.columns.name
            if data.index.name:
                yaxis_title = data.index.name
        x = xaxis_title if xaxis_title is not None else 'x'
        y = yaxis_title if yaxis_title is not None else 'y'
        z = color_title if color_title is not None else 'value'
        if text:
            annotation_text = np.around(data.values[::-1], decimals=decimals)
        else:
            annotation_text = np.empty(data.shape, dtype=str)
        fig = ff.create_annotated_heatmap(data.values[::-1],
                                          x=list(data.columns),
                                          y=list(data.index)[::-1],
                                          showscale=True,
                                          colorscale=self._colorscale,
                                          annotation_text=annotation_text,
                                          colorbar_title_text=color_title,
                                          **kwargs)
        fig.update_xaxes(side=xaxis_side)
        fig.update_traces(hovertemplate=f'{x}=%{{x}}<br>{y}=%{{y}}<br>{z}=%{{z}}<extra></extra>')
        if title == 'auto':
            title = 'Heatmap'
        fig.update_layout(title=dict(text=title, x=0.5, xref='paper'),
                          xaxis_title=xaxis_title,
                          yaxis_title=yaxis_title,
                          margin=dict(l=5, r=5, t=50, b=5),
                          height=height,
                          width=width,
                          font_size=font_size,
                          newshape_line_color=self._newshape_line_color)
        fig.show(config=self._config)

    def density_heatmap(self, x: Optional[Union[str, List[str]]] = None,
                        y: Optional[Union[str, List[str]]] = None,
                        color: Optional[str] = None,
                        subplots: Optional[Tuple[str, str, int]] = None,
                        nbins: Optional[Tuple[int, int]] = None,
                        agg: str = None,
                        orientation: str = 'auto',
                        title: str = 'auto',
                        height: Optional[int] = None,
                        width: Optional[int] = None,
                        font_size: int = 12,
                        **kwargs: Optional[Any]) -> None:
        """
        Makes a density heatmap.

        Parameters
        ----------
        x: str or list of str, default=None
            Name of the column to draw on the x-axis. If it takes a list of
            column names, the input data is considered as wide-form
            rather than long-form.

        y: str or list of str, default=None
            Name of the column to draw on the y-axis. If it takes a list of
            column names, the input data is considered as wide-form
            rather than long-form.

        color: str, default=None
            Name of the column to aggregate and set color to blocks.

        subplots: (rows, cols, ncols), default=None
            Creates a set of subplots:
            - rows: name of the column to use for constructing subplots
              along the y-axis.
            - cols: name of the column to use for constructing subplots
              along the x-axis.
            - ncols: number of columns of the subplot grid.

        nbins: (nbinsx, nbinsy), default=None
            Number of bins along the x-axis and y-axis.

        agg: {'count', 'sum', 'avg', 'min', 'max'}, default=None
            Name of the function used to aggregate values of 'color'.

        orientation: {'auto', 'v', 'h'}, default='auto'
            Orientation of the graph: 'v' for vertical and 'h' for horizontal.
            By default, it is determined automatically based on the input
            data types.

        title: str, default='auto'
            Title of the graph. When 'auto', the title is generated automatically.

        height: int, default=None
            Height of the figure in pixels.

        width: int, default=None
            Width of the figure in pixels.

        font_size: int, default=12
            Size of the global font.

        **kwargs: optional
            See 'plotly.express.density_heatmap' for other possible arguments.
        """
        if x is None and y is None:
            raise ValueError("Either 'x' or 'y' must be given")

        cols_or_lists = [x, y, color, list(subplots[:2]) if subplots is not None else []]
        cols_or_lists = [el for el in cols_or_lists if el is not None]
        data = self._get_data(list(itertools.chain(*[[el] if isinstance(el, str) else el for el in cols_or_lists])))

        if orientation == 'auto':
            orientation = None
        if subplots:
            facet_row, facet_col, facet_col_wrap = subplots[0], subplots[1], subplots[2]
        else:
            facet_row, facet_col, facet_col_wrap = None, None, None
        if nbins:
            nbinsx, nbinsy = nbins[0], nbins[1]
        else:
            nbinsx, nbinsy = None, None
        fig = px.density_heatmap(data_frame=data,
                                 x=x,
                                 y=y,
                                 z=color,
                                 facet_row=facet_row,
                                 facet_col=facet_col,
                                 facet_col_wrap=facet_col_wrap,
                                 nbinsx=nbinsx,
                                 nbinsy=nbinsy,
                                 orientation=orientation,
                                 histfunc=agg,
                                 height=height,
                                 width=width,
                                 **kwargs)
        if title == 'auto':
            if x and y:
                title = f'Density Heatmap of {x} and {y}'
            elif x:
                title = f'Density Heatmap of {x}'
            elif y:
                title = f'Density Heatmap of {y}'
            else:
                title = 'Density Heatmap'
        fig.update_layout(title=dict(text=title, x=0.5, xref='paper'),
                          margin=dict(l=5, r=5, t=50, b=5),
                          font_size=font_size,
                          newshape_line_color=self._newshape_line_color)
        fig.show(config=self._config)

    def gantt(self, x_start: str,
              x_end: str,
              y: Optional[str] = None,
              color: Optional[str] = None,
              subplots: Optional[Tuple[str, str, int]] = None,
              text: Optional[str] = None,
              decimals: int = 2,
              opacity: float = 1.0,
              title: str = 'auto',
              height: Optional[int] = None,
              width: Optional[int] = None,
              font_size: int = 12,
              **kwargs: Optional[Any]) -> None:
        """
        Makes a Gantt chart.

        Parameters
        ----------
        x_start: str
            Name of the start date column to draw on the x-axis.

        x_end: str
            Name of the end date column to draw on the x-axis.

        y: str, default=None
            Name of the task column to draw on the y-axis.

        color: str, default=None
            Name of the column used to set color to bars.

        subplots: (rows, cols, ncols), default=None
            Creates a set of subplots:
            - rows: name of the column to use for constructing subplots
              along the y-axis.
            - cols: name of the column to use for constructing subplots
              along the x-axis.
            - ncols: number of columns of the subplot grid.

        text: str, default=None
            Name of the column to use as text labels in the figure.

        decimals: int, default=2
            Number of decimal places to round 'text' labels of a float dtype to.

        opacity: float, default=1.0
            Opacity of the bars. Ranges from 0 to 1.

        title: str, default='auto'
            Title of the graph. When 'auto', the title is generated automatically.

        height: int, default=None
            Height of the figure in pixels.

        width: int, default=None
            Width of the figure in pixels.

        font_size: int, default=12
            Size of the global font.

        **kwargs: optional
            See 'plotly.express.timeline' for other possible arguments.
        """
        cols_or_lists = [x_start, x_end, y, color, list(subplots[:2]) if subplots is not None else []]
        cols_or_lists = [el for el in cols_or_lists if el is not None]
        data = self._get_data(list(itertools.chain(*[[el] if isinstance(el, str) else el for el in cols_or_lists])))

        if subplots:
            facet_row, facet_col, facet_col_wrap = subplots[0], subplots[1], subplots[2]
        else:
            facet_row, facet_col, facet_col_wrap = None, None, None
        if color:
            len_labels = data[color].nunique()
            nums = np.linspace(0, len_labels, len_labels) / len_labels
            color_discrete_sequence = [get_continuous_color(self._colorscale, z) for z in nums]
        else:
            color_discrete_sequence = None
        fig = px.timeline(data_frame=data,
                          x_start=x_start,
                          x_end=x_end,
                          y=y,
                          color=color,
                          facet_row=facet_row,
                          facet_col=facet_col,
                          facet_col_wrap=facet_col_wrap,
                          color_discrete_sequence=color_discrete_sequence,
                          text=text,
                          opacity=opacity,
                          height=height,
                          width=width,
                          **kwargs)
        if text:
            if pd.api.types.is_integer_dtype(data[text]):
                texttemplate = '%{text:d}'
            elif pd.api.types.is_float_dtype(data[text]):
                texttemplate = f'%{{text:.{decimals}f}}'
            else:
                texttemplate = '%{text}'
            fig.update_traces(texttemplate=texttemplate,
                              textposition='auto')  # ['inside', 'outside', 'auto', 'none']
        if title == 'auto':
            title = f'Gantt Chart'
        fig.update_layout(title=dict(text=title, x=0.5, xref='paper'),
                          margin=dict(l=5, r=5, t=50, b=5),
                          yaxis_autorange='reversed',
                          font_size=font_size,
                          newshape_line_color=self._newshape_line_color)
        fig.show(config=self._config)

    def pareto(self, x: str,
               bins: Union[List[int], str] = 'auto',
               text: bool = False,
               decimals: int = 2,
               opacity: float = 0.8,
               edge: bool = False,
               title: str = 'auto',
               height: Optional[int] = None,
               width: Optional[int] = None,
               font_size: int = 12) -> None:
        """
        Makes a Pareto chart.

        Parameters
        ----------
        x: str
            Name of the column to make graph for.

        bins: list of int or 'auto', default='auto'
            List of the x coordinates of the bars. If 'auto', bins are determined
            automatically.

        text: bool, default=False
            Whether to show text labels in the figure.

        decimals: int, default=2
            Number of decimal places to round cumulative percentage to.

        opacity: float, default=0.8
            Opacity of the bars. Ranges from 0 to 1.

        edge: bool, default=False
            Whether to draw bar edges.

        title: str, default='auto'
            Title of the graph. When 'auto', the title is generated automatically.

        height: int, default=None
            Height of the figure in pixels.

        width: int, default=None
            Width of the figure in pixels.

        font_size: int, default=12
            Size of the global font.
        """
        _data = self._get_data(x)

        if pd.api.types.is_numeric_dtype(_data[x]):
            if bins == 'auto':
                bins = np.arange(_data[x].max() + 1)
            data = pd.cut(x=_data[x], bins=bins, right=False).value_counts().sort_index()
            labels = bins  # [str(i) for i in data.index]
        else:
            data = _data[x].value_counts()
            labels = data.index
        values = data.values
        shares = np.cumsum(data / data.sum()).values * 100
        nums = np.linspace(0, len(labels), len(labels)) / len(labels)
        cols = [get_continuous_color(self._colorscale, z) for z in nums]
        if edge:
            line = dict(color='black', width=0.6)
        else:
            line = {}
        fig = go.Figure()
        fig.add_trace(go.Bar(x=labels,
                             y=values,
                             marker=dict(color=cols, line=line),
                             opacity=opacity,
                             name='Frequency',
                             text=values))
        fig.add_trace(go.Scatter(x=labels,
                                 y=shares,
                                 yaxis='y2',
                                 name='Cumulative Percentage',
                                 mode='lines',
                                 marker_color=px.colors.qualitative.Plotly[0],
                                 line_width=3))
        if text:
            fig.update_traces(texttemplate='%{text:d}',
                              textposition='outside',
                              selector=dict(type='bar'))
        fig.update_traces(hovertemplate='%{y}',
                          selector=dict(type='bar'))
        fig.update_traces(hovertemplate=f'%{{y:.{decimals}f}}%',
                          selector=dict(type='scatter'))
        if title == 'auto':
            title = f'Pareto Chart of {x}'
        fig.update_layout(title=dict(text=title, x=0.5, xref='paper'),
                          yaxis_title='Frequency',
                          xaxis_title=x,
                          hovermode='x',
                          legend=dict(orientation='h', x=0.5, xanchor='center', y=1.01, yanchor='bottom'),
                          margin=dict(l=5, r=5, t=80, b=5),
                          height=height,
                          width=width,
                          yaxis2=dict(title='Cumulative Percentage',
                                      anchor='x',
                                      overlaying='y',
                                      side='right',
                                      showgrid=False,
                                      titlefont_color=px.colors.qualitative.Plotly[0],
                                      tickfont_color=px.colors.qualitative.Plotly[0]),
                          font_size=font_size,
                          newshape_line_color=self._newshape_line_color)
        fig.show(config=self._config)

    def sankey(self, n: int = 10,
               sort_labels: bool = False,
               colored_links: bool = True,
               opacity: float = 0.5,
               orientation: str = 'h',
               title: str = 'auto',
               height: Optional[int] = None,
               width: Optional[int] = None,
               font_size: int = 10,
               **kwargs: Optional[Any]) -> None:
        """
        Makes a Sankey diagram.

        Parameters
        ----------
        n: int, default=10
            Number of the most frequent process traces to make graph for.

        sort_labels: bool, default=False
            Whether to sort labels to rearrange nodes in the figure.

        colored_links: bool, default=True
            Whether to set colors to links. If False, a translucent grey is used.
            If True, links are colored according to the source nodes.

        opacity: float, default=0.5
            Opacity of the links. Ranges from 0 to 1.

        orientation: {'v', 'h'}, default='h'
            Orientation of the graph: 'v' for vertical and 'h' for horizontal.

        title: str, default='auto'
            Title of the graph. When 'auto', the title is generated automatically.

        height: int, default=None
            Height of the figure in pixels.

        width: int, default=None
            Width of the figure in pixels.

        font_size: int, default=12
            Size of the global font.

        **kwargs: optional
            See 'plotly.graph_objects.Sankey' for other possible arguments.
        """
        if not isinstance(self._dh, DataHolder):
            raise TypeError('Input data must be given as a sberpm.DataHolder')

        top_traces = TraceMetric(self._dh).calc_metrics(*['count', 'ids']).sort_values('count', ascending=False).head(n)
        data_grouped = self._data.groupby(self._dh.id_column, as_index=False).agg({self._dh.activity_column: list})
        data_grouped[self._dh.activity_column] = data_grouped[self._dh.activity_column].apply(
            lambda x: ['start'] + x + ['end'])
        data = data_grouped.apply(pd.Series.explode)
        dh = deepcopy(self._dh)
        dh.data = data[data[dh.id_column].isin(list(set.union(*top_traces['ids'].values)))]

        labels = list(dh.data[dh.activity_column].unique())
        if sort_labels:
            labels = sorted(labels)
        act2ind = {k: v for v, k in enumerate(labels)}

        t = pd.DataFrame(TransitionMetric(dh).count()).reset_index()
        t['source'] = t['transition'].apply(lambda x: act2ind[x[0]])
        t['target'] = t['transition'].apply(lambda x: act2ind[x[1]])

        node_color = px.colors.convert_colors_to_same_type(px.colors.qualitative.Plotly)[0]
        node_color *= ceil(len(labels) / len(node_color))
        link_color = None
        if colored_links:
            link_color = t['source'].apply(lambda x: node_color[x % len(node_color)])
            link_color = link_color.apply(lambda x: f'rgba{x[3:-1]}, {opacity})')

        fig = go.Figure()
        fig.add_trace(go.Sankey(node=dict(pad=10,
                                          thickness=10,
                                          line=dict(color='black', width=0.5),
                                          label=labels,
                                          color=node_color),
                                link=dict(source=t['source'],
                                          target=t['target'],
                                          value=t['count'],
                                          color=link_color),
                                orientation=orientation,
                                valueformat='d',
                                **kwargs))
        if title == 'auto':
            title = f'Sankey Diagram of top {n} traces'
        fig.update_layout(title=dict(text=title, x=0.5, xref='paper'),
                          margin=dict(l=5, r=5, t=80, b=5),
                          height=height,
                          width=width,
                          font_size=font_size,
                          newshape_line_color=self._newshape_line_color)
        fig.show(config=self._config)

    def _get_data(self, metric_names: Union[str, List[str]]) -> pd.DataFrame:
        """
        If self._data is pd.Dataframe, just returns it.
        If self._data is a metric object, calls the given methods
        and returns the result as pandas.DataFrame.

        Parameters
        ----------
        metric_names: str or list of str
            Names of the metrics to calculate.

        Returns
        -------
        result: pandas.DataFrame
        """
        if isinstance(self._data, pd.DataFrame):
            return self._data
        elif isinstance(self._data, (ActivityMetric, TransitionMetric, IdMetric, TraceMetric, UserMetric)):
            metric_names = list(set(metric_names)) if not isinstance(metric_names, str) else [metric_names]
            res = self._data.calc_metrics(*metric_names, raise_no_method=False)

            if len(res.columns) == 0:
                res = self._data.calc_metrics(*metric_names, raise_no_method=False)  # just for raising error

            # make index a column and convert to str
            index_name = res.index.name
            if index_name not in res.columns:  # just in case
                res[index_name] = res.index
                res = res.reset_index(drop=True)
            if pd.api.types.is_object_dtype(res[index_name]):
                res[index_name] = res[index_name].astype(str)

            for name in metric_names:
                if name not in res.columns:
                    self._data.calc_metrics(name, raise_no_method=True)  # just for raising error
            return res
        else:
            raise RuntimeError()

    def hist_activity_of_dur(self,
                             by_activity: Optional[str] = None,
                             use_median: bool = True,
                             top: bool = False,
                             barmode: str = 'stack',
                             time_unit='s',
                             height: Optional[int] = None,
                             width: Optional[int] = None,
                             font_size: Optional[int] = None,
                             title: Optional[str]  = None,
                             round_n: int = 3):
        dh = self._dh.copy()
        if top:
            col = 0
            if not dh.duration_column:
                dh.check_or_calc_duration()
                if time_unit in ('week', 'w'):
                    time_unit = 604800
                elif time_unit in ('day', 'd'):
                    time_unit = 86400
                elif time_unit in ('hour', 'h'):
                    time_unit = 3600
                elif time_unit in ('minute', 'm'):
                    time_unit = 60
                elif time_unit in ('second', 's'):
                    time_unit = 1
                else:
                    raise ValueError(f'Unknown time unit: "{time_unit}"')
                dh.data[dh.duration_column] = dh.data[dh.duration_column] / time_unit

            fig = make_subplots(rows=1, cols=3)
            act_most_freq = dh.data[dh.activity_column].value_counts()[:1].index[0]
            tmp_data = dh.data[dh.data[dh.activity_column] == act_most_freq]
            longest_stages = dh.data.groupby([dh.activity_column])[dh.duration_column].max().sort_values().tail(2)

            if use_median:
                avg_or_median = tmp_data[dh.duration_column].median()
            else:
                max_v = tmp_data[dh.duration_column].max()
                min_v = tmp_data[dh.duration_column].min()
                avg_or_median = (max_v + min_v) / 2

            list_v = [round(avg_or_median * i / 5.0, round_n) for i in range(1, 10)]
            tmp_data[dh.duration_column][tmp_data[dh.duration_column] > list_v[-1]] = list_v[-1] + list_v[0] / 2

            fig_count = go.Histogram(x=tmp_data[dh.duration_column],
                                     histfunc='count',
                                     name=act_most_freq + '_the_most_frequent',
                                     nbinsx=10,
                                     showlegend=True,
                                     xbins=dict(start=0.0, size=list_v[0]))
            col += 1
            fig.append_trace(fig_count, row=1, col=col)

            for act in longest_stages.index:
                tmp_data = dh.data[dh.data[dh.activity_column] == act]

                if use_median:
                    avg_or_median = tmp_data[dh.duration_column].median()
                else:
                    max_v = tmp_data[dh.duration_column].max()
                    min_v = tmp_data[dh.duration_column].min()
                    avg_or_median = (max_v + min_v) / 2

                list_v = [round(avg_or_median * i / 5.0, round_n) for i in range(1, 10)]
                tmp_data[dh.duration_column][tmp_data[dh.duration_column] > list_v[-1]] = list_v[-1] + list_v[
                    0] / 2

                fig_count = go.Histogram(x=tmp_data[dh.duration_column],
                                         histfunc='count',
                                         name=act + '_the_most_longest',
                                         nbinsx=10,
                                         showlegend=True,
                                         xbins=dict(start=0.0, size=list_v[0]))
                col += 1
                fig.append_trace(fig_count, row=1, col=col)

            fig.update_layout(legend=dict(yanchor='bottom', xanchor="center", orientation="v"),
                              title=title,
                              height=height,
                              width=width,
                              font_size=font_size
                              )
            fig.update_xaxes(title_text='duration')
            fig.update_yaxes(title_text='count',title_standoff=1)

        else:
            if by_activity is not None:
                dh.data = dh.data[dh.data[dh.activity_column] == by_activity]
                if dh.data.empty:
                    raise ValueError(
                        f"The DataHolder is empty after filtering by the 'by_activity' parameter. Plese check your "
                        f"data and the entered parameters.")
            if not dh.duration_column:
                dh.check_or_calc_duration()
                if time_unit in ('week', 'w'):
                    time_unit = 604800
                elif time_unit in ('day', 'd'):
                    time_unit = 86400
                elif time_unit in ('hour', 'h'):
                    time_unit = 3600
                elif time_unit in ('minute', 'm'):
                    time_unit = 60
                elif time_unit in ('second', 's'):
                    time_unit = 1
                else:
                    raise ValueError(f'Unknown time unit: "{time_unit}"')
                dh.data[dh.duration_column] = dh.data[dh.duration_column] / time_unit

            if use_median:
                avg_or_median = dh.data[dh.duration_column].median()
            else:
                max_v = dh.data[dh.duration_column].max()
                min_v = dh.data[dh.duration_column].min()
                avg_or_median = (max_v + min_v) / 2

            list_v = [round(avg_or_median * i / 5.0, 2) for i in range(1, 10)]

            dh.data[dh.duration_column][dh.data[dh.duration_column] > list_v[-1]] = list_v[-1] + list_v[0] / 2

            len_labels = dh.data[dh.activity_column].nunique()
            nums = np.linspace(0, len_labels, len_labels) / len_labels
            color_discrete_sequence = [get_continuous_color(self._colorscale, z) for z in nums]

            fig = px.histogram(dh.data, x=dh.duration_column, histfunc='count', color=dh.activity_column,
                               nbins=10,
                               hover_data={dh.duration_column: False},
                               barmode=barmode,
                               color_discrete_sequence=color_discrete_sequence
                               )

            fig.update_traces(xbins=dict(start=0.0, size=list_v[0]))
            fig.update_xaxes(title_text='duration',
                             fixedrange=True,
                             dtick=list_v[0],
                             range=[0, list_v[0] + list_v[-1] - list_v[0] / 100],
                             nticks=10
                             )
            fig.update_yaxes(title_text='count',
                             fixedrange=True)

            fig.update_layout(title= title,
                              height=height,
                              width=width,
                              font_size=font_size
                              )
        fig.show(config=self._config)
