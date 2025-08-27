"""Sinusoidal wave generator"""

import numpy as np

from ..graphics.shape import Shape
from ..graphics.all_enums import Types


class SineWave(Shape):
    """Sinusoidal wave generator

    Args:
        period (float, optional): Period of the sine wave. Defaults to 40.
        amplitude (float, optional): Amplitude of the sine wave. Defaults to 20.
        duration (float, optional): Duration of the sine wave. Defaults to 1.
        n_points (int, optional): Sampling rate. Defaults to 100.
        phase_angle (float, optional): Phase angle of the sine wave. Defaults to 0.
        damping (float, optional): Damping coefficient. .001-.005 is usual. Defaults to 0.
        rot_angle (float, optional): Rotation angle of the sine wave.. Defaults to 0.
        xform_matrix (ndarray, optional): Transformation matrix. Defaults to None.

    Returns:
        Shape: _description_
    """

    def __init__(
        self,
        period: float = 40,
        amplitude: float = 20,
        duration: float = 40,
        n_points: int = 100,
        phase_angle: float = 0,
        damping: float = 0,
        rot_angle: float = 0,
        xform_matrix: "ndarray" = None,
        **kwargs,
    ) -> Shape:
        phase = phase_angle
        freq = 1 / period
        n_cycles = duration / period
        x = np.linspace(0, duration, int(n_points * n_cycles))
        y = amplitude * np.sin(2 * np.pi * freq * x + phase)
        if damping:
            y *= np.exp(-damping * x)
        vertices = np.column_stack((x, y)).tolist()
        super().__init__(vertices, xform_matrix=xform_matrix, **kwargs)
        self.subtype = Types.SINE_WAVE
        self.period = (period,)
        self.amplitude = (amplitude,)
        self.duration = (duration,)
        self.n_points = (n_points,)
        self.phase = (phase,)
        self.damping = (damping,)
        self.rot_angle = (rot_angle,)

    def copy_(self):
        """_description_

        Returns:
            SineWave: _description_
        """
        return SineWave(
            self.period,
            self.amplitude,
            self.duration,
            self.n_points,
            self.phase,
            self.damping,
            self.rot_angle,
            self.xform_matrix,
            **self.kwargs,
        )
