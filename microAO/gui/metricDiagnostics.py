import numpy
import wx
import matplotlib.pyplot
import matplotlib.figure
import matplotlib.backends.backend_wxagg
import matplotlib.backends.backend_wx
import skimage.exposure


class DiagnosticsPanelBase(wx.Panel):
    _DEFAULT_CMAP = "inferno"

    def __init__(self, parent):
        super().__init__(parent)

        # Create figure
        self._figure = matplotlib.figure.Figure(constrained_layout=True)
        self._axes = None
        self._canvas = matplotlib.backends.backend_wxagg.FigureCanvasWxAgg(
            self, wx.ID_ANY, self._figure
        )
        toolbar = matplotlib.backends.backend_wx.NavigationToolbar2Wx(
            self._canvas
        )
        toolbar.Show()

        # Create widgets
        stext_mode = wx.StaticText(self, label="Mode:")
        self._slider_mode = wx.Slider(self, style=wx.SL_LABELS)
        self._slider_mode.Bind(wx.EVT_SLIDER, self._on_slider_mode)
        stext_meas = wx.StaticText(self, label="Measurement:")
        self._slider_meas = wx.Slider(self, style=wx.SL_LABELS)
        self._slider_meas.Bind(wx.EVT_SLIDER, self._on_slider_meas)
        stext_noll_label = wx.StaticText(self, label="Noll index:")
        self._stext_noll = wx.StaticText(self, label="")
        stext_cmap = wx.StaticText(self, label="Colourmap:")
        self._cmap_choice = wx.Choice(
            self, choices=sorted(matplotlib.pyplot.colormaps())
        )
        self._cmap_choice.SetStringSelection(self._DEFAULT_CMAP)
        self._cmap_choice.Bind(wx.EVT_CHOICE, self._on_cmap_choice)

        # Lay out the widgets
        widgets_sizer = wx.GridBagSizer(vgap=0, hgap=10)
        widgets_sizer.SetCols(2)
        widgets_sizer.AddGrowableCol(1)
        widgets_sizer.Add(
            stext_mode,
            wx.GBPosition(0, 0),
            wx.GBSpan(1, 1),
            wx.ALIGN_CENTRE_VERTICAL | wx.ALL,
            5,
        )
        widgets_sizer.Add(
            self._slider_mode, wx.GBPosition(0, 1), wx.GBSpan(1, 1), wx.EXPAND
        )
        widgets_sizer.Add(
            stext_meas,
            wx.GBPosition(1, 0),
            wx.GBSpan(1, 1),
            wx.ALIGN_CENTRE_VERTICAL | wx.ALL,
            5,
        )
        widgets_sizer.Add(
            self._slider_meas, wx.GBPosition(1, 1), wx.GBSpan(1, 1), wx.EXPAND
        )
        widgets_sizer.Add(
            stext_noll_label,
            wx.GBPosition(2, 0),
            wx.GBSpan(1, 1),
            wx.ALIGN_CENTRE_VERTICAL | wx.ALL,
            5,
        )
        widgets_sizer.Add(
            self._stext_noll,
            wx.GBPosition(2, 1),
            wx.GBSpan(1, 1),
            wx.ALIGN_CENTRE_VERTICAL | wx.ALIGN_LEFT,
        )
        widgets_sizer.Add(
            stext_cmap,
            wx.GBPosition(3, 0),
            wx.GBSpan(1, 1),
            wx.ALIGN_CENTRE_VERTICAL | wx.ALL,
            5,
        )
        widgets_sizer.Add(
            self._cmap_choice,
            wx.GBPosition(3, 1),
            wx.GBSpan(1, 1),
            wx.ALIGN_CENTRE_VERTICAL | wx.ALIGN_LEFT,
        )

        # Finalise layout
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self._canvas, 1, wx.EXPAND)
        sizer.Add(toolbar, 0, wx.EXPAND)
        sizer.Add(widgets_sizer, 0, wx.EXPAND)
        self.SetSizerAndFit(sizer)

    def _update_plot(self):
        raise NotImplementedError()

    def _on_slider_mode(self, event: wx.CommandEvent):
        mode_index = event.GetInt() - 1
        current_diagnostics = (
            self.GetParent().GetParent().metric_diagnostics[mode_index]
        )
        current_data = self.GetParent().GetParent().metric_data[mode_index]
        # Update the measurement slider if necessary
        new_meas_max = len(current_diagnostics)
        if new_meas_max < self._slider_meas.GetValue():
            # The new max value is less than the current value => clamp to max
            self._slider_meas.SetValue(new_meas_max)
        self._slider_meas.SetMax(new_meas_max)
        # Update the Noll index
        self._stext_noll.SetLabel(current_data.mode_label)
        # Update the plot
        self._update_plot()

    def _on_slider_meas(self, _: wx.CommandEvent):
        self._update_plot()

    def _on_cmap_choice(self, _: wx.CommandEvent):
        self._update_plot()

    def initialise(self):
        # Clear the figure and create axes
        self._figure.clear()
        self._axes = self._figure.add_subplot()
        # Reset sliders
        self._slider_mode.SetValue(1)
        self._slider_mode.SetMin(1)
        self._slider_mode.SetMax(1)
        self._slider_meas.SetValue(1)
        self._slider_meas.SetMin(1)
        self._slider_meas.SetMax(1)

    def update(self):
        diagnostics = self.GetParent().GetParent().metric_diagnostics
        mode_index = self._slider_mode.GetValue() - 1
        data = self.GetParent().GetParent().metric_data[mode_index]
        # Update the mode slider
        self._slider_mode.SetMax(len(diagnostics))
        # Update the measurement slider
        self._slider_meas.SetMax(len(diagnostics[mode_index]))
        # Update Noll index
        self._stext_noll.SetLabel(data.mode_label)
        # Update the plot
        self._update_plot()


class DiagnosticsPanelFourier(DiagnosticsPanelBase):
    def _update_plot(self):
        index_mode = self._slider_mode.GetValue() - 1
        index_meas = self._slider_meas.GetValue() - 1
        diagnostics = (
            self.GetParent()
            .GetParent()
            .metric_diagnostics[index_mode][index_meas]
        )
        # Clear axes
        self._figure.clear()
        # Update images
        self._axes = self._figure.subplots(1, 2, sharex=True, sharey=True)
        self._axes[0].imshow(
            diagnostics.fft_sq_log, cmap=self._cmap_choice.GetStringSelection()
        )
        self._axes[1].imshow(
            diagnostics.freq_above_noise,
            cmap=self._cmap_choice.GetStringSelection(),
        )
        for a in self._axes:
            a.axis("off")
        # Update canvas
        self._canvas.draw()


class DiagnosticsPanelContrast(DiagnosticsPanelBase):
    def _update_plot(self):
        index_mode = self._slider_mode.GetValue() - 1
        index_meas = self._slider_meas.GetValue() - 1
        diagnostics = (
            self.GetParent()
            .GetParent()
            .metric_diagnostics[index_mode][index_meas]
        )
        # Clear axes
        self._figure.clear()
        # Update image and table
        self._axes = self._figure.subplots(1, 2)
        self._axes[0].imshow(
            skimage.exposure.rescale_intensity(
                diagnostics.image_raw,
                in_range=tuple(
                    numpy.percentile(diagnostics.image_raw, (1, 99))
                ),
            ),
            cmap=self._cmap_choice.GetStringSelection(),
        )
        self._axes[1].table(
            [
                ["Mean top", "Mean bottom"],
                [diagnostics.mean_top, diagnostics.mean_bottom],
            ],
            cellLoc="center",
            loc="center",
        )
        for a in self._axes:
            a.axis("off")
        # Update canvas
        self._canvas.draw()


class DiagnosticsPanelGradient(DiagnosticsPanelBase):
    def _update_plot(self):
        index_mode = self._slider_mode.GetValue() - 1
        index_meas = self._slider_meas.GetValue() - 1
        diagnostics = (
            self.GetParent()
            .GetParent()
            .metric_diagnostics[index_mode][index_meas]
        )
        # Clear axes
        self._figure.clear()
        # Update images
        self._axes = self._figure.subplots(1, 4, sharex=True, sharey=True)
        self._axes[0].imshow(
            skimage.exposure.rescale_intensity(
                diagnostics.image_raw,
                in_range=tuple(
                    numpy.percentile(diagnostics.image_raw, (1, 99))
                ),
            ),
            cmap=self._cmap_choice.GetStringSelection(),
        )
        self._axes[0].set_title("Raw image")
        self._axes[1].imshow(diagnostics.grad_mask_x)
        self._axes[1].set_title("Grad. mask X")
        self._axes[2].imshow(diagnostics.grad_mask_y)
        self._axes[2].set_title("Grad. mask Y")
        self._axes[3].imshow(diagnostics.correction_grad)
        self._axes[3].set_title("Gradient")
        for a in self._axes:
            a.axis("off")
        # Update canvas
        self._canvas.draw()


class DiagnosticsPanelFourierPower(DiagnosticsPanelBase):
    def _update_plot(self):
        index_mode = self._slider_mode.GetValue() - 1
        index_meas = self._slider_meas.GetValue() - 1
        diagnostics = (
            self.GetParent()
            .GetParent()
            .metric_diagnostics[index_mode][index_meas]
        )
        # Clear axes
        self._figure.clear()
        # Update images
        self._axes = self._figure.subplots(1, 2, sharex=True, sharey=True)
        self._axes[0].imshow(
            diagnostics.fftarray_sq_log,
            cmap=self._cmap_choice.GetStringSelection(),
        )
        self._axes[1].imshow(
            diagnostics.freq_above_noise,
            cmap=self._cmap_choice.GetStringSelection(),
        )
        for a in self._axes:
            a.axis("off")
        # Update canvas
        self._canvas.draw()


class DiagnosticsPanelSecondMoment(DiagnosticsPanelBase):
    def _update_plot(self):
        index_mode = self._slider_mode.GetValue() - 1
        index_meas = self._slider_meas.GetValue() - 1
        diagnostics = (
            self.GetParent()
            .GetParent()
            .metric_diagnostics[index_mode][index_meas]
        )
        # Clear axes
        self._figure.clear()
        # Update images
        self._axes = self._figure.subplots(1, 2, sharex=True, sharey=True)
        self._axes[0].imshow(
            diagnostics.fftarray_sq_log,
            cmap=self._cmap_choice.GetStringSelection(),
        )
        self._axes[1].imshow(
            diagnostics.fftarray_sq_log_masked,
            cmap=self._cmap_choice.GetStringSelection(),
        )
        for a in self._axes:
            a.axis("off")
        # Update canvas
        self._canvas.draw()
