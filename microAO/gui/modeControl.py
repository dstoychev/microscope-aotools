
from cockpit import events
from cockpit.gui.guiUtils import FLOATVALIDATOR

import wx
import wx.lib.newevent
import wx.lib.scrolledpanel

import numpy as np

from microAO.events import *


_DEFAULT_ZERNIKE_MODE_NAMES = {
    1: "Piston",
    2: "Tip",
    3: "Tilt",
    4: "Defocus",
    5: "Astig (O)",
    6: "Astig (V)",
    7: "Coma (V)",
    8: "Coma (H)",
    9: "Trefoil (V)",
    10: "Trefoil (O)",
    11: "Spherical",
    12: "Astig 2 (V)",
    13: "Astig 2 (O)",
    14: "Quadrafoil (V)",
    15: "Quadrafoil (O)",
}

ModeChangeEvent, EVT_MODE_CHANGED = wx.lib.newevent.NewEvent()

class _FloatCtrl(wx.TextCtrl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def value(self):
        try:
            val = float(self.GetValue())
        except Exception as e:
            val = None

        return val

class _Mode(wx.Panel):
    """Manual mode selection GUI."""

    def __init__(self, parent, id, value=0):
        super().__init__(parent)

        # id to identify mode
        self.id = id

        # Mode value
        self.value = value

        # Store focus state
        self.focus = False

        row_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Label for mode
        mode_index = self.id+1
        try:
            mode_label = _DEFAULT_ZERNIKE_MODE_NAMES[mode_index]
        except KeyError:
            mode_label = ""

        self._mode_index = wx.StaticText(self, label=str(mode_index), size=wx.Size(30,-1), style=(wx.ALIGN_CENTRE_HORIZONTAL|wx.ST_NO_AUTORESIZE))
        self._mode_label = wx.StaticText(self, label=mode_label, size=wx.Size(120,-1), style=(wx.ALIGN_CENTRE_HORIZONTAL|wx.ST_NO_AUTORESIZE))

        # Mode slider: drag to set mode
        default_range = 1.5   # range of slider
        self._slider = wx.Slider(self, value=0, minValue=-100, maxValue=100, size=wx.Size(200,-1))
        self._slider.Bind(wx.EVT_SCROLL, self.OnSlider)

        # Adjust mode adjustment range. Influences range of slider.
        self._slider_min = _FloatCtrl(self, wx.ID_ANY, "{}".format(-default_range), validator=FLOATVALIDATOR, size=wx.Size(50,-1), style=(wx.ALIGN_CENTRE_HORIZONTAL|wx.ST_NO_AUTORESIZE))
        self._slider_max = _FloatCtrl(self, wx.ID_ANY, "{}".format(default_range), validator=FLOATVALIDATOR, size=wx.Size(50,-1), style=(wx.ALIGN_CENTRE_HORIZONTAL|wx.ST_NO_AUTORESIZE))
        
        self._slider_min.Bind(wx.EVT_TEXT, self.UpdateValueRanges)
        self._slider_max.Bind(wx.EVT_TEXT, self.UpdateValueRanges)

        # Current mode value
        self._val = wx.SpinCtrlDouble(self, initial=0, inc=0.001, size=wx.Size(160,-1), style=(wx.ALIGN_CENTRE_HORIZONTAL|wx.ST_NO_AUTORESIZE))
        self._val.SetDigits(4)
        self.UpdateValueRanges()
        self._val.SetValue(self.value)
        self._val.Bind(wx.EVT_SPINCTRLDOUBLE, self.OnModeValueChange)
        self._val.Bind(wx.EVT_SET_FOCUS, self.OnModeGetFocus)
        self._val.Bind(wx.EVT_KILL_FOCUS, self.OnModeLoseFocus)

        # Layout
        row_sizer.Add(self._mode_index, wx.SizerFlags().CentreVertical())
        row_sizer.Add(self._mode_label, wx.SizerFlags().CentreVertical())
        row_sizer.Add(self._slider_min, wx.SizerFlags().Expand())
        row_sizer.Add(self._slider, wx.SizerFlags().Expand())
        row_sizer.Add(self._slider_max, wx.SizerFlags().Expand())
        row_sizer.Add(self._val, wx.SizerFlags().Expand())

        # Set widget sizer
        self.SetSizerAndFit(row_sizer)

    def OnSlider(self, evt):
        # Assign focus to control if sliding
        self.focus = True

        # Set value
        try:
            val = (self._slider.GetValue()+100)/200 * (self._slider_max.value - self._slider_min.value) + self._slider_min.value
            self.SetValue(val)
        except TypeError:
            pass

        # Reset slider and ranges when slide end (mouse released)
        if not wx.GetMouseState().LeftIsDown():
            # Lose focus as released
            self.focus = False

    def OnModeValueChange(self, evt):
        new_val = self._val.GetValue()
        self.UpdateValueRanges(new_val)
        self.SetValue(new_val)

    def OnRangeChange(self, evt):
        self.UpdateValueRanges()
        evt.Skip()
    
    def OnModeGetFocus(self, evt):
        self.focus = True

    def OnModeLoseFocus(self, evt):
        self.focus = False

    def UpdateValueRanges(self, middle=None, range=None):
        min_val =  self._slider_min.value
        if min_val is not None:
            self._val.SetMin(min_val)

        max_val = self._slider_max.value
        if max_val is not None:
            self._val.SetMax(max_val)

        self.SetValue(self.value, quiet=True)

    def GetValue(self):
        return self._val.GetValue()

    def SetValue(self, val, quiet=False):
        """ Set control value 

            Sets control value. Emits mode change event by default, a quiet flag can be 
            used to override this behaviour.        
        """
        # Set value property
        self.value = val

        # Set value control
        self._val.SetValue(val)
        
        # Set slider control value
        slider_val = 200 * (val - self._slider_min.value)/(self._slider_max.value - self._slider_min.value) - 100 
        self._slider.SetValue(slider_val)

        # Emit mode change event, if required
        if not quiet:
            evt = ModeChangeEvent(mode=self.id, value= self.value)
            wx.PostEvent(self, evt)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self._value = val

class _ModesPanel(wx.lib.scrolledpanel.ScrolledPanel):
    def __init__(self, parent, device):
        wx.lib.scrolledpanel.ScrolledPanel.__init__(self, parent, -1)

        # Set attributes
        self._device = device
        control_matrix = self._device.proxy.get_controlMatrix()
        self._n_modes = control_matrix.shape[1]

        # Create root panel and sizer
        root_panel = wx.Panel(self)
        root_sizer = wx.BoxSizer(wx.VERTICAL)

        # Set headings
        heading_panel = wx.Panel(root_panel)
        heading_sizer = wx.BoxSizer(wx.HORIZONTAL)

        headings = [
            ("", 150), 
            ("Min", 50), 
            ("Control", 200), 
            ("Max", 50), 
            ("Value", 160)
        ]

        font = wx.Font( wx.FontInfo(10).Bold())
        flags = wx.SizerFlags().Centre()

        for heading in headings:
            heading_widget = wx.StaticText(heading_panel, label=heading[0], size=wx.Size(heading[1],-1), style=(wx.ALIGN_CENTRE_HORIZONTAL|wx.ST_NO_AUTORESIZE))
            heading_widget.SetFont(font)
            heading_sizer.Add(heading_widget, flags)

        heading_panel.SetSizer(heading_sizer)
        
        root_sizer.Add(heading_panel, flags=wx.SizerFlags().Border(wx.BOTTOM,8))

        # Add control per mode
        modes = np.zeros(self._n_modes)
        last_modes = self._device.proxy.get_last_modes()
        if last_modes is not None:
            modes += last_modes

        self._mode_controls = []
        for i, mode in enumerate(modes):
            mode_control = _Mode(root_panel, id=i, value=mode)
            mode_control.Bind(EVT_MODE_CHANGED, self.OnMode)
            self._mode_controls.append(mode_control)
            root_sizer.Add(mode_control)

        root_panel.SetSizer(root_sizer)
        
        # Set frame sizer
        frame_sizer = wx.BoxSizer(wx.VERTICAL)
        frame_sizer.Add(root_panel, wx.SizerFlags().Expand().Border(wx.ALL, 10))
        self.SetSizerAndFit(frame_sizer)

        # Subscribe to pubsub events
        events.subscribe(PUBSUB_SET_PHASE, self.HandleSetPhase)

    def OnMode(self, evt):
        # self._modes[evt.mode] = evt.value
        modes = self.GetModes()
        self._device.set_phase(
            modes, 
            offset=self._device.proxy.get_system_flat()
        )

    def GetModes(self):
        modes = []
        for mode_control in self._mode_controls:
            modes.append(mode_control.value)
        
        return modes

    def UpdateModes(self, modes):
        # Update each mode
        for i, value in enumerate(modes):
            mode_control = self._mode_controls[i]
            if value != mode_control.value and not mode_control.focus:
                mode_control.SetValue(value, quiet=True)
                mode_control.UpdateValueRanges()
    
    def Reset(self, quiet=False):
        for mode_control in self._mode_controls:
            mode_control.SetValue(0, quit=quiet)

    def HandleSetPhase(self, modes, actuator_values):
        if modes is not None:
            self.UpdateModes(modes)

class ModesControl(wx.Frame):
    def __init__(self, parent, device):
        super().__init__(parent)
        self._panel = _ModesPanel(self, device)
        self._panel.SetupScrolling()
        self._sizer = wx.BoxSizer(wx.VERTICAL)
        self._sizer.Add(self._panel)
        self.SetSizerAndFit(self._sizer)
        self.SetTitle('DM mode control')