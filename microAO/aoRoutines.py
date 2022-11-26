import abc
from dataclasses import dataclass
import numpy as np
import wx
import typing

import cockpit.events
import microAO.events
from microAO.aoAlg import AdaptiveOpticsFunctions
import microAO.gui.sensorlessParameters
import microAO.gui.sensorlessViewer


class _Routine(metaclass=abc.ABCMeta):
    """An abstract base class for an AO routine

    A setup and image processing method must be defined.

    """

    def __init__(self, name) -> None:
        self.name = name
        self.params = {}

    @abc.abstractmethod
    def setup(self, sensorless_data) -> np.ndarray:
        """Perform routine setup and determine first set of modes to apply."""
        pass

    @abc.abstractmethod
    def process(self, images_all) -> tuple[bool, np.ndarray]:
        """Process the current stack of images."""
        pass


class _ConventionalRoutine(_Routine):
    def __init__(self) -> None:
        super().__init__("Conventional")
        self._index_mode = 0
        self._index_offset = 0
        self._total_measurements = 0
        self._corrections = None
        self.params.update(
            {
                "num_reps": 1,
                "NA": 1.1,
                "wavelength": 560e-9,
                "pixel_size": 0.0,
                "metric": "Fourier",
                "metric_fitting": "Gaussian",
                "modes": (
                    microAO.gui.sensorlessParameters.ConventionalParamsMode(
                        11, np.linspace(-1.5, 1.5, 5)
                    ),
                    microAO.gui.sensorlessParameters.ConventionalParamsMode(
                        22, np.linspace(-1.5, 1.5, 5)
                    ),
                    microAO.gui.sensorlessParameters.ConventionalParamsMode(
                        5, np.linspace(-1.5, 1.5, 5)
                    ),
                    microAO.gui.sensorlessParameters.ConventionalParamsMode(
                        6, np.linspace(-1.5, 1.5, 5)
                    ),
                    microAO.gui.sensorlessParameters.ConventionalParamsMode(
                        7, np.linspace(-1.5, 1.5, 5)
                    ),
                    microAO.gui.sensorlessParameters.ConventionalParamsMode(
                        8, np.linspace(-1.5, 1.5, 5)
                    ),
                    microAO.gui.sensorlessParameters.ConventionalParamsMode(
                        9, np.linspace(-1.5, 1.5, 5)
                    ),
                    microAO.gui.sensorlessParameters.ConventionalParamsMode(
                        10, np.linspace(-1.5, 1.5, 5)
                    ),
                ),
                "save_as_datapoint": False,
                "log_path": "",
            }
        )

    def setup(self, initial_corrections) -> np.ndarray:
        # Define additional data required for routine
        self._index_mode = 0
        self._index_offset = 0
        self._total_measurements = (
            sum([len(mode.offsets) for mode in self.params["modes"]])
            * self.params["num_reps"]
        )
        self._corrections = initial_corrections
        self.params["pixel_size"] = (
            wx.GetApp().Objectives.GetPixelSize() * 10 ** -6
        )

        # Define the first correction to apply
        new_modes = initial_corrections.copy()
        new_modes[
            self.params["modes"][self._index_mode].index_noll - 1
        ] += self.params["modes"][self._index_mode].offsets[self._index_offset]

        return new_modes

    def process(self, images_all) -> tuple[bool, np.ndarray]:
        # Correct mode if enough measurements have been taken
        if (
            self._index_offset
            == self.params["modes"][self._index_mode].offsets.shape[0] - 1
        ):
            # Calculate required parameters
            mode_index_noll_0 = (
                self.params["modes"][self._index_mode].index_noll - 1
            )
            modes = (
                self._corrections[mode_index_noll_0]
                + self.params["modes"][self._index_mode].offsets
            )
            images_mode = images_all[-modes.shape[0] :]

            # Find aberration amplitudes and correct
            (
                peak,
                metrics,
                fitting_data,
                metric_diagnostics,
            ) = AdaptiveOpticsFunctions.find_zernike_amp_sensorless(
                image_stack=images_mode,
                modes=modes,
                metric_name=self.params["metric"],
                metric_fitting_name=self.params["metric_fitting"],
                wavelength=self.params["wavelength"],
                NA=self.params["NA"],
                pixel_size=self.params["pixel_size"],
            )

            self._corrections[mode_index_noll_0] = peak[0]

            # Publish results
            cockpit.events.publish(
                microAO.events.PUBSUB_SENSORLESS_RESULTS,
                microAO.gui.sensorlessViewer.ConventionalResults(
                    metrics=metrics,
                    image_stack=images_mode,
                    metric_diagnostics=metric_diagnostics,
                    modes=modes,
                    mode_label=f"Z{mode_index_noll_0 + 1}",
                    peak=peak,
                    fitting_name=self.params["metric_fitting"],
                    fitting_data=fitting_data,
                ),
            )

            # Update indices
            self._index_offset = 0
            self._index_mode += 1
            if self._index_mode == len(self.params["modes"]):
                self._index_mode = 0
        else:
            # Increment offset index
            self._index_offset += 1

        # Set next mode and return data, unless all measurements acquired
        done = False
        new_modes = self._corrections.copy()
        if len(images_all) < self._total_measurements:
            # Apply next set of offsets
            new_modes[
                self.params["modes"][self._index_mode].index_noll - 1
            ] += self.params["modes"][self._index_mode].offsets[
                self._index_offset
            ]
        else:
            # All data has been acquired => set completion flag
            done = True

        # Update status light
        cockpit.events.publish(
            cockpit.events.UPDATE_STATUS_LIGHT,
            "image count",
            "Sensorless AO: image {n}/{N}, mode {n_md}, meas. {n_ms}".format(
                n=len(images_all) + 1,
                N=self._total_measurements,
                n_md=self.params["modes"][self._index_mode].index_noll,
                n_ms=self._index_offset + 1,
            ),
        )

        # Format return data
        return (done, new_modes)


class _MLRoutine(_Routine):
    def __init__(self) -> None:
        super().__init__("ML")

    def setup(self, initial_corrections) -> np.ndarray:
        # Define additional data required for routine
        self._index_image = 0

        # Define the first correction to apply
        new_modes = initial_corrections.copy()

        return new_modes

    def process(self, images_all) -> tuple[bool, np.ndarray]:
        # Determine the new modes
        # new_modes = model.predict()
        new_modes = self._corrections.copy()

        # Update the image index
        self._index_image += 1

        # Determine completion state
        done = False
        if self._index_image >= 5:
            done = True

        return (done, new_modes)


@dataclass(frozen=True)
class _RoutineObjects:
    parameter_dialog: wx.Dialog
    results_viewer: typing.Optional[wx.Frame]


routines = {
    _ConventionalRoutine(): _RoutineObjects(
        microAO.gui.sensorlessParameters.ConventionalParametersDialog,
        microAO.gui.sensorlessViewer.ConventionalResultsViewer,
    ),
    _MLRoutine(): _RoutineObjects(
        microAO.gui.sensorlessParameters.MLParametersDialog, None
    ),
}
