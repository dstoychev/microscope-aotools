import dataclasses
import json
import time
import numpy
import wx
import matplotlib.ticker
import matplotlib.figure
import matplotlib.backends.backend_wxagg
import matplotlib.backends.backend_wx
import tifffile
import typing

import cockpit
import microAO.aoMetrics
import microAO.events
import microAO.aoAlg


@dataclasses.dataclass(frozen=True)
class ConventionalResults:
    metrics: list
    image_stack: list
    metric_diagnostics: list
    modes: numpy.ndarray
    mode_label: str
    peak: numpy.ndarray
    fitting_name: str
    fitting_data: typing.Optional[microAO.aoAlg.MetricFittingData]


@dataclasses.dataclass(frozen=True)
class _ConventionalMetricPlotData:
    peak: numpy.ndarray
    metrics: numpy.ndarray
    modes: numpy.ndarray
    mode_label: str
    fitting_name: str
    fitting_data: typing.Optional[microAO.aoAlg.MetricFittingData]


class _ConventionalMetricPlotPanel(wx.Panel):
    _MODE_SPACING_FRACTION = 0.5

    def __init__(self, parent):
        super().__init__(parent)

        self._x_position = 0
        self._x_tick_positions = []
        self._x_tick_labels = []
        self._max_scan_range = 0
        self._margin_x = 0

        self._figure = matplotlib.figure.Figure(constrained_layout=True)
        self._axes = None
        self._canvas = matplotlib.backends.backend_wxagg.FigureCanvasWxAgg(
            self, wx.ID_ANY, self._figure
        )
        toolbar = matplotlib.backends.backend_wx.NavigationToolbar2Wx(
            self._canvas
        )
        save_images_button = wx.Button(self, label="Save raw images")
        save_data_button = wx.Button(self, label="Save data")

        toolbar.Show()
        save_images_button.Bind(wx.EVT_BUTTON, self._on_save_images)
        save_data_button.Bind(wx.EVT_BUTTON, self._on_save_data)

        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        button_sizer.Add(save_images_button, 0)
        button_sizer.Add(save_data_button, 0, wx.LEFT, 5)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self._canvas, 1, wx.EXPAND)
        sizer.Add(toolbar, 0, wx.EXPAND)
        sizer.Add(button_sizer, 0, wx.EXPAND | wx.ALL, 5)
        self.SetSizerAndFit(sizer)

    def _on_save_images(self, _: wx.CommandEvent):
        self.GetParent().GetParent().save_images()

    def _on_save_data(self, _: wx.CommandEvent):
        self.GetParent().GetParent().save_data()

    def initialise(self, max_scan_range):
        # Clear figure
        self._figure.clear()

        # Initialise attributes
        self._x_position = 0
        self._x_tick_positions = []
        self._x_tick_labels = []
        self._max_scan_range = max_scan_range
        self._margin_x = max_scan_range * self._MODE_SPACING_FRACTION

        # Create axes and set their labels
        self._axes = self._figure.add_subplot()
        self._axes.set_xlabel("Mode")
        self._axes.set_ylabel("Metric")

    def update(self):
        data = self.GetParent().GetParent().metric_data[-1]
        # Calculate parameters
        x_range = (self._x_position, self._x_position + self._max_scan_range)

        # Draw vertical line
        if self._x_position > 0:
            # This is not the first iteration
            x_line = self._x_position - self._margin_x / 2
            spine = list(self._axes.spines.values())[0]
            self._axes.axvline(
                x=x_line,
                color=spine.get_edgecolor(),
                linewidth=spine.get_linewidth(),
            )

        # Plot measurement
        self._axes.plot(
            numpy.interp(
                data.modes,
                (min(data.modes), max(data.modes)),
                x_range,
            ),
            data.metrics,
            marker="o",
            color="skyblue",
        )

        # Plot fit
        if data.fitting_data:
            xs = numpy.linspace(*data.fitting_data.range_x, 100)
            ys = data.fitting_data.curve(xs, *data.fitting_data.curve_params)
            self._axes.plot(
                numpy.interp(
                    xs,
                    (min(data.modes), max(data.modes)),
                    x_range,
                ),
                ys,
                color="green",
                alpha=0.5,
            )

        # Plot peak
        self._axes.plot(
            numpy.interp(
                data.peak[0],
                (min(data.modes), max(data.modes)),
                x_range,
            ),
            data.peak[1],
            marker="+",
            markersize=20,
            color="crimson",
        )

        # Configure ticks
        tick_position = x_range[0] + self._max_scan_range / 2
        self._x_tick_positions += [tick_position]
        self._x_tick_labels += [data.mode_label]
        self._axes.xaxis.set_major_locator(
            matplotlib.ticker.FixedLocator(self._x_tick_positions)
        )
        self._axes.xaxis.set_major_formatter(
            matplotlib.ticker.FixedFormatter(self._x_tick_labels)
        )

        # Update x position
        self._x_position = x_range[1] + self._margin_x

        # Set x-axis limits
        self._axes.set_xlim(
            left=-self._margin_x / 2,
            right=self._x_position - self._margin_x / 2,
        )

        # Refresh canvas
        self._canvas.draw()


class ConventionalResultsViewer(wx.Frame):
    def __init__(self, parent):
        super().__init__(parent, title="Metric viewer")

        # Instance attributes
        self._metric_images = []
        self.metric_data = []
        self.metric_diagnostics = []
        self._metric_name = ""
        self._metric_params = {}
        self._metric_fitting = ""

        self._notebook = wx.Notebook(self)

        # Sizing
        frame_sizer = wx.BoxSizer(wx.VERTICAL)
        frame_sizer.Add(self._notebook, 1, wx.EXPAND)
        self.SetSizerAndFit(frame_sizer)

        # Subscribe to pubsub events
        cockpit.events.subscribe(
            microAO.events.PUBSUB_SENSORLESS_START, self._on_start
        )
        cockpit.events.subscribe(
            microAO.events.PUBSUB_SENSORLESS_RESULTS, self._update
        )

        # Bind to close event
        self.Bind(wx.EVT_CLOSE, self._on_close)

    def save_images(self):
        # Ask the user to select file
        fpath = None
        with wx.FileDialog(
            self,
            "Save image stack",
            wildcard="TIFF file (*.tif; *.tiff)|*.tif;*.tiff",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        ) as fileDialog:
            if fileDialog.ShowModal() != wx.ID_OK:
                return
            fpath = fileDialog.GetPath()
        # Modes can have different scanning ranges and therefore it is not
        # possible to order the images in a proper 2D array => save them as a
        # linear sequence
        images = numpy.array(
            [
                image
                for image_stack in self._metric_images
                for image in image_stack
            ]
        )
        tifffile.imwrite(fpath, images)

    def save_data(self):
        def recursive_string_conversion(d):
            # Used to convert objects that are not JSON serialisable to objects
            # that are, e.g. NumPy's ndarray to list
            for k, v in d.items():
                if isinstance(v, dict):
                    recursive_string_conversion(v)
                else:
                    if isinstance(v, numpy.ndarray):
                        d[k] = v.tolist()
                    elif isinstance(v, numpy.polynomial.Polynomial):
                        d[k] = str(v)
                    elif callable(v):
                        d[k] = v.__name__

        # Ask the user to select file
        fpath = None
        with wx.FileDialog(
            self,
            "Save metric data",
            wildcard="JSON file (*.json)|*.json",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        ) as fileDialog:
            if fileDialog.ShowModal() != wx.ID_OK:
                return
            fpath = fileDialog.GetPath()
        # Convert data to dicts and save them as JSON
        data_dicts = []
        for data in self.metric_data:
            data_dict = dataclasses.asdict(data)
            recursive_string_conversion(data_dict)
            data_dicts.append(data_dict)
        json_dict = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metric name": self._metric_name,
            "metric parameters": self._metric_params,
            "metric fitting": self._metric_fitting,
            "correction results": data_dicts,
        }
        with open(fpath, "w", encoding="utf-8") as fo:
            json.dump(json_dict, fo, sort_keys=True, indent=4)

    def _on_start(self, routine):
        # Initialise attributes
        self._metric_images = []
        self.metric_data = []
        self.metric_diagnostics = []
        self._metric_name = routine.params["metric"]
        self._metric_params = {
            "wavelength": routine.params["wavelength"],
            "NA": routine.params["NA"],
            "pixel_size": routine.params["pixel_size"],
        }
        self._metric_fitting = routine.params["metric_fitting"]

        # Calculate required parameters
        max_scan_range = max(
            [
                mode.offsets.max() - mode.offsets.min()
                for mode in routine.params["modes"]
            ]
        )

        # Delete existing notebook pages
        self._notebook.DeleteAllPages()

        # Add new notebook pages and initialise them
        for panel_class, name, init_args in (
            (_ConventionalMetricPlotPanel, "Metric plot", (max_scan_range,)),
            (
                microAO.aoMetrics.metrics[self._metric_name].diagnostic_panel,
                "Metric diagnostics",
                (),
            ),
        ):
            if panel_class:
                panel = panel_class(self._notebook)
                self._notebook.AddPage(panel, name)
                panel.initialise(*init_args)

        # Re-fit the frame after the notebook has been updated
        self.Fit()

    def _update(
        self,
        results: ConventionalResults,
    ):
        # Save data
        self._metric_images.append(results.image_stack)
        metric_plot_data = _ConventionalMetricPlotData(
            peak=results.peak,
            metrics=results.metrics,
            modes=results.modes,
            mode_label=results.mode_label,
            fitting_name=results.fitting_name,
            fitting_data=results.fitting_data,
        )
        self.metric_data.append(metric_plot_data)
        self.metric_diagnostics.append(results.metric_diagnostics)

        # Update pages
        for page_id in range(self._notebook.GetPageCount()):
            self._notebook.GetPage(page_id).update()

    def _on_close(self, evt: wx.CloseEvent):
        # Unsubscribe from events
        cockpit.events.unsubscribe(
            microAO.events.PUBSUB_SENSORLESS_START, self._on_start
        )
        cockpit.events.unsubscribe(
            microAO.events.PUBSUB_SENSORLESS_RESULTS, self._update
        )

        # Continue + destroy frame
        evt.Skip()
