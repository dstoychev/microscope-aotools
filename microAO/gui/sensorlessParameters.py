import wx
import numpy as np
import dataclasses
from microAO.aoMetrics import metrics
from wx.lib.filebrowsebutton import DirBrowseButton


@dataclasses.dataclass(frozen=True)
class ConventionalParamsMode:
    # Noll index
    index_noll: int
    # The amplitude offsets used for scanning the mode
    offsets: np.ndarray


class ConventionalParametersDialog(wx.Dialog):
    def __init__(self, parent):
        super().__init__(
            parent,
            title="Sensorless parameters selection",
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
        )

        self._device = parent._device

        panel = wx.Panel(self)

        # Create the configurable controls
        params = self._device.sensorless_routine.params
        self._textctrl_reps = wx.TextCtrl(panel, value=str(params["num_reps"]))
        self._textctrl_na = wx.TextCtrl(panel, value=str(params["NA"]))
        self._textctrl_wavelength = wx.TextCtrl(
            panel, value=str(int(params["wavelength"] * 1e9))
        )
        self._cbox_metric = wx.ComboBox(
            panel,
            value=params["metric"],
            choices=list(metrics.keys()),
            style=wx.CB_READONLY,
        )
        self._textctrl_ranges = wx.TextCtrl(
            panel,
            value=self._params2text(params),
            size=wx.Size(400, 200),
            style=wx.TE_MULTILINE,
        )
        self._textctrl_logpath = DirBrowseButton(panel, labelText="Log path:")
        self._checkbox_dp_save = wx.CheckBox(panel)
        self._checkbox_dp_save.SetValue(params["save_as_datapoint"])

        # Configure the font of the scan ranges' text control
        self._textctrl_ranges.SetFont(
            wx.Font(
                14,
                wx.FONTFAMILY_MODERN,
                wx.FONTSTYLE_NORMAL,
                wx.FONTWEIGHT_NORMAL,
            )
        )

        # Define the grid and the text widgets
        widgets_data = (
            (
                wx.StaticText(panel, label="Number of repeats:"),
                wx.GBPosition(0, 0),
                wx.GBSpan(1, 1),
                wx.ALL,
                5,
            ),
            (
                self._textctrl_reps,
                wx.GBPosition(0, 1),
                wx.GBSpan(1, 1),
                wx.ALL,
                5,
            ),
            (
                wx.StaticText(panel, label="Numerical aperture:"),
                wx.GBPosition(1, 0),
                wx.GBSpan(1, 1),
                wx.ALL,
                5,
            ),
            (
                self._textctrl_na,
                wx.GBPosition(1, 1),
                wx.GBSpan(1, 1),
                wx.ALL,
                5,
            ),
            (
                wx.StaticText(panel, label="Excitation wavelength:"),
                wx.GBPosition(2, 0),
                wx.GBSpan(1, 1),
                wx.ALL,
                5,
            ),
            (
                self._textctrl_wavelength,
                wx.GBPosition(2, 1),
                wx.GBSpan(1, 1),
                wx.ALL,
                5,
            ),
            (
                wx.StaticText(panel, label="Metric:"),
                wx.GBPosition(3, 0),
                wx.GBSpan(1, 1),
                wx.ALL,
                5,
            ),
            (
                self._cbox_metric,
                wx.GBPosition(3, 1),
                wx.GBSpan(1, 1),
                wx.ALL,
                5,
            ),
            (
                wx.StaticText(panel, label="Save results as datapoint?"),
                wx.GBPosition(4, 0),
                wx.GBSpan(1, 1),
                wx.ALL,
                5,
            ),
            (
                self._checkbox_dp_save,
                wx.GBPosition(4, 1),
                wx.GBSpan(1, 1),
                wx.ALL,
                5,
            ),
            (
                wx.StaticText(
                    panel,
                    label=(
                        "Define the scanning ranges in the text field below.\n"
                        "Columns should be separated by four or more SPACE "
                        "characters.\nThe columns are:\n\tModes (Noll indices)"
                        "\n\tScan range min amplitude\n\tScan range max "
                        "amplitude\n\tScan range steps"
                    ),
                ),
                wx.GBPosition(5, 0),
                wx.GBSpan(1, 2),
                wx.ALL,
                5,
            ),
            (
                self._textctrl_ranges,
                wx.GBPosition(6, 0),
                wx.GBSpan(1, 2),
                wx.ALL | wx.EXPAND,
                5,
            ),
            (
                self._textctrl_logpath,
                wx.GBPosition(7, 0),
                wx.GBSpan(1, 2),
                wx.ALL | wx.EXPAND,
                5,
            ),
        )

        # Construct the grid
        panel_sizer = wx.GridBagSizer(vgap=0, hgap=0)
        panel_sizer.SetCols(2)
        panel_sizer.AddGrowableCol(1)
        for widget_data in widgets_data:
            panel_sizer.Add(*widget_data)
        panel.SetSizer(panel_sizer)

        # Create the standard buttons
        sizer_stdbuttons = wx.StdDialogButtonSizer()
        button_ok = wx.Button(self, wx.ID_OK)
        button_ok.Bind(wx.EVT_BUTTON, self._on_ok)
        sizer_stdbuttons.Add(button_ok)
        button_cancel = wx.Button(self, wx.ID_CANCEL)
        sizer_stdbuttons.Add(button_cancel)
        sizer_stdbuttons.Realize()

        # Finalise layout
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(panel, 1, wx.EXPAND)
        sizer.Add(sizer_stdbuttons, 0, wx.ALL, 5)
        self.SetSizerAndFit(sizer)

    def _params2text(self, params):
        mode_sets = []
        # Split into sets of sequential mode entries with same offsets
        for index in range(0, len(params["modes"])):
            if len(mode_sets) > 0 and np.array_equal(
                params["modes"][index - 1].offsets,
                params["modes"][index].offsets,
            ):
                mode_sets[-1][0].append(params["modes"][index].index_noll)
            else:
                mode_sets.append(
                    [
                        # Modes in the set
                        [params["modes"][index].index_noll],
                        # Scanning parameters of the set
                        (
                            params["modes"][index].offsets.min(),
                            params["modes"][index].offsets.max(),
                            params["modes"][index].offsets.shape[0],
                        ),
                    ]
                )
        # Split modes further into range sequences, i.e. list of lists
        for mode_set in mode_sets:
            mode_ranges = [[mode_set[0][0]]]
            for mode in mode_set[0][1:]:
                if mode == mode_ranges[-1][-1] + 1:
                    mode_ranges[-1].append(mode)
                else:
                    mode_ranges.append([mode])
            mode_set[0] = mode_ranges
        # Convert the modes to a string
        for mode_set in mode_sets:
            # Build the mode string
            mode_string = ", ".join(
                [
                    f"{r[0]}" if len(r) == 1 else f"{r[0]}-{r[-1]}"
                    for r in mode_set[0]
                ]
            )
            mode_set[0] = mode_string
        # Find the longest mode string
        max_mod_string = max([len(mode_set[0]) for mode_set in mode_sets])
        # Convert all sets into a multiline formatted string
        sets_string = ""
        col_sep = "    "
        for mode_set in mode_sets:
            range_string = col_sep.join([str(x) for x in mode_set[1]])
            # Append a new row
            sets_string += (
                f"{mode_set[0]: <{max_mod_string}}{col_sep}{range_string}\n"
            )
        return sets_string

    def _on_ok(self, event: wx.CommandEvent):
        # Parse the simple single-line widgets first
        widgets_data = [
            # widget, label, value, parsing function
            [self._textctrl_reps, "repeats", 0, int],
            [self._textctrl_na, "numerical aperture", 0, float],
            [self._textctrl_wavelength, "wavelength", 0, int],
            [self._cbox_metric, "metric", "", str],
            [self._textctrl_logpath, "logpath", "", str],
        ]
        for widget_data in widgets_data:
            try:
                widget_data[2] = widget_data[3](widget_data[0].GetValue())
            except ValueError:
                with wx.MessageDialog(
                    self,
                    f"Error! Cannot convert {widget_data[1]} to a number of "
                    f"type {widget_data[3].__name__}!",
                    "Parsing error",
                    wx.OK | wx.ICON_ERROR,
                ) as dlg:
                    dlg.ShowModal()
                return
        # Do widget-specific parsing for each of the single-line widgets
        ## Parse repeats
        if widgets_data[0][2] < 1:
            with wx.MessageDialog(
                self,
                f"Error! Repeats must be 1 or greater!",
                "Parsing error",
                wx.OK | wx.ICON_ERROR,
            ) as dlg:
                dlg.ShowModal()
            return
        ## Parse NA
        if widgets_data[1][2] <= 0.0:
            with wx.MessageDialog(
                self,
                f"Error! Numerical aperture must be greater than 0.0!",
                "Parsing error",
                wx.OK | wx.ICON_ERROR,
            ) as dlg:
                dlg.ShowModal()
            return
        ## Parse wavelength
        if widgets_data[2][2] <= 0:
            with wx.MessageDialog(
                self,
                f"Error! Wavelength must be greater than 0!",
                "Parsing error",
                wx.OK | wx.ICON_ERROR,
            ) as dlg:
                dlg.ShowModal()
            return
        # Parse the multi-line widget
        mode_params = []
        lines = [
            line.strip()
            for line in self._textctrl_ranges.GetValue().splitlines()
        ]
        lines = [line for line in lines if line]
        if len(lines) == 0:
            with wx.MessageDialog(
                self,
                f"Error! At least one scanning range needs to be defined!",
                "Parsing error",
                wx.OK | wx.ICON_ERROR,
            ) as dlg:
                dlg.ShowModal()
            return
        for line_index, line in enumerate(lines):
            columns = [column.strip() for column in line.split("    ")]
            columns = [column for column in columns if column]
            # Parse number of columns
            if len(columns) != 4:
                with wx.MessageDialog(
                    self,
                    f"Error! Improper formatting on line {line_index + 1} of "
                    f"scan ranges! Expected 4 column but got {len(columns)} "
                    "instead.",
                    "Parsing error",
                    wx.OK | wx.ICON_ERROR,
                ) as dlg:
                    dlg.ShowModal()
                return
            # Parse modes
            modes = []
            mode_ranges = [x.strip() for x in columns[0].split(",")]
            mode_ranges = [x for x in mode_ranges if x]
            for mode_range in mode_ranges:
                if "-" in mode_range:
                    bounds = [x.strip() for x in mode_range.split("-")]
                    if len(bounds) != 2:
                        with wx.MessageDialog(
                            self,
                            "Error! Improper formatting of modes on line "
                            f"{line_index + 1} of scan ranges!",
                            "Parsing error",
                            wx.OK | wx.ICON_ERROR,
                        ) as dlg:
                            dlg.ShowModal()
                        return
                    try:
                        range_start = int(bounds[0])
                        range_end = int(bounds[1]) + 1
                        modes.extend(list(range(range_start, range_end)))
                    except TypeError:
                        with wx.MessageDialog(
                            self,
                            "Error! Improper formatting of modes on line "
                            f"{line_index + 1} of scan ranges! Modes need to "
                            "be integers.",
                            "Parsing error",
                            wx.OK | wx.ICON_ERROR,
                        ) as dlg:
                            dlg.ShowModal()
                        return
                else:
                    try:
                        modes.append(int(mode_range))
                    except ValueError:
                        with wx.MessageDialog(
                            self,
                            "Error! Improper formatting of modes on line "
                            f"{line_index + 1} of scan ranges! Modes need to "
                            "be integers.",
                            "Parsing error",
                            wx.OK | wx.ICON_ERROR,
                        ) as dlg:
                            dlg.ShowModal()
                        return
            if min(modes) <= 0:
                with wx.MessageDialog(
                    self,
                    "Error! Improper formatting of modes on line "
                    f"{line_index + 1} of scan ranges! Modes need to be "
                    "specified as Noll indices, therefore integers greater "
                    "than 0.",
                    "Parsing error",
                    wx.OK | wx.ICON_ERROR,
                ) as dlg:
                    dlg.ShowModal()
                return
            # Parse range bounds
            for index, label in ((1, "range min"), (2, "range max")):
                try:
                    columns[index] = float(columns[index])
                except ValueError:
                    with wx.MessageDialog(
                        self,
                        f"Error! Cannot convert {label} on line "
                        f"{line_index + 1} to a floating-point number!",
                        "Parsing error",
                        wx.OK | wx.ICON_ERROR,
                    ) as dlg:
                        dlg.ShowModal()
                    return
            # Parse steps
            try:
                columns[3] = int(columns[3])
            except ValueError:
                with wx.MessageDialog(
                    self,
                    f"Error! Cannot convert steps on line {line_index + 1} to "
                    "an integer number!",
                    "Parsing error",
                    wx.OK | wx.ICON_ERROR,
                ) as dlg:
                    dlg.ShowModal()
                return
            # Create the mode parameters
            for mode in modes:
                mode_params.append(
                    ConventionalParamsMode(
                        mode, np.linspace(columns[1], columns[2], columns[3])
                    )
                )
        # Update the sensorless AO parameters
        self._device.sensorless_routine.params["num_reps"] = widgets_data[0][2]
        self._device.sensorless_routine.params["modes"] = mode_params
        self._device.sensorless_routine.params["NA"] = widgets_data[1][2]
        self._device.sensorless_routine.params["wavelength"] = (
            widgets_data[2][2] * 1e-9
        )
        self._device.sensorless_routine.params["metric"] = widgets_data[3][2]
        self._device.sensorless_routine.params["log_path"] = widgets_data[4][2]
        self._device.sensorless_routine.params[
            "save_as_datapoint"
        ] = self._checkbox_dp_save.GetValue()
        # Propagate event
        event.Skip()


class MLParametersDialog(wx.Dialog):
    def __init__(self, parent):
        super().__init__(
            parent,
            title="Sensorless parameters selection",
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
        )

        panel = wx.Panel(self)

        panel_sizer = wx.BoxSizer(wx.VERTICAL)
        panel_sizer.Add(
            wx.StaticText(
                panel,
                label="This routine has no parameters to set.",
            ),
            1,
            wx.EXPAND | wx.ALL,
            20,
        )
        panel.SetSizerAndFit(panel_sizer)

        # Create the standard buttons
        sizer_stdbuttons = wx.StdDialogButtonSizer()
        button_ok = wx.Button(self, wx.ID_OK)
        button_ok.Bind(wx.EVT_BUTTON, self._on_ok)
        sizer_stdbuttons.Add(button_ok)
        button_cancel = wx.Button(self, wx.ID_CANCEL)
        sizer_stdbuttons.Add(button_cancel)
        sizer_stdbuttons.Realize()

        # Finalise layout
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(panel, 1, wx.EXPAND)
        sizer.Add(sizer_stdbuttons, 0, wx.ALL, 5)
        self.SetSizerAndFit(sizer)

    def _on_ok(self, event: wx.CommandEvent):
        # Propagate event
        event.Skip()
