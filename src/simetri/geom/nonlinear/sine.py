"""Sinusoidal wave shape generator."""

from math import exp

import numpy as np
from numpy.typing import NDArray

from ...base.all_enums import Types
from ...shapes.shape import Shape


class SineWave(Shape):
    """A sampled sine wave as a ``Shape`` polyline.

    Attributes:
        period: Period of the sine wave.
        amplitude: Amplitude of the sine wave.
        duration: Horizontal length of the sampled wave.
        n_points: Points sampled per period.
        phase: Phase angle in radians.
        damping: Exponential damping coefficient (typical range 0.001–0.005).
        rot_angle: Rotation angle stored with the instance.

    Examples:
        ::

            import simetri.graphics as sg

            wave = sg.SineWave(period=40, amplitude=20, duration=80)
            canvas = sg.Canvas()
            canvas.draw(wave)
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
        """Create a sine-wave shape from sampled points.

        Args:
            period: Period of the sine wave. Defaults to 40.
            amplitude: Amplitude of the sine wave. Defaults to 20.
            duration: Duration (x-span) of the sine wave. Defaults to 40.
            n_points: Sampling rate per period. Defaults to 100.
            phase_angle: Phase angle in radians. Defaults to 0.
            damping: Damping coefficient; 0.001–0.005 is typical. Defaults to 0.
            rot_angle: Rotation angle stored on the instance. Defaults to 0.
            xform_matrix: Optional transformation matrix. Defaults to None.
            **kwargs: Additional keyword arguments passed to ``Shape``.
        """
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
        """Return a new ``SineWave`` with the same parameters.

        Returns:
            SineWave: A copy of this sine wave.
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


def sine_wave(
    amplitude: float,
    frequency: float,
    duration: float,
    sample_rate: float,
    phase: float = 0,
) -> NDArray:
    """
    Generate a sine wave.

    Args:
        amplitude (float): Amplitude of the wave.
        frequency (float): Frequency of the wave.
        duration (float): Duration of the wave.
        sample_rate (float): Sample rate.
        phase (float, optional): Phase angle of the wave. Defaults to 0.

    Returns:
        np.ndarray: Time and signal arrays representing the sine wave.
    """
    time = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = amplitude * np.sin(2 * np.pi * frequency * time + phase)
    # plt.plot(time, signal)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.title('Discretized Sine Wave')
    # plt.grid(True)
    # plt.show()
    return time, signal


def damping_function(amplitude, duration, sample_rate):
    """
    Generates a damping function based on the given amplitude, duration, and sample rate.

    Args:
        amplitude (float): The initial amplitude of the damping function.
        duration (float): The duration over which the damping occurs, in seconds.
        sample_rate (float): The number of samples per second.

    Returns:
        list: A list of float values representing the damping function over time.
    """
    damping = []
    for i in range(int(duration * sample_rate)):
        damping.append(amplitude * exp(-i / (duration * sample_rate)))
    return damping


def sine_points(
    period: float = 40,
    amplitude: float = 20,
    duration: float = 40,
    n_points: int = 100,
    phase_angle: float = 0,
    damping: float = 0,
) -> NDArray:
    """
    Generate sine wave points.

    Args:
        amplitude (float): Amplitude of the wave.
        frequency (float): Frequency of the wave.
        duration (float): Duration of the wave.
        sample_rate (float): Sample rate.
        phase (float, optional): Phase angle of the wave. Defaults to 0.
        damping (float, optional): Damping coefficient. Defaults to 0.
    Returns:
        np.ndarray: Array of points representing the sine wave.
    """
    phase = phase_angle
    freq = 1 / period
    n_cycles = duration / period
    x = np.linspace(0, duration, int(n_points * n_cycles))
    y = amplitude * np.sin(2 * np.pi * freq * x + phase)
    if damping:
        y *= np.exp(-damping * x)
    vertices = np.column_stack((x, y)).tolist()

    return vertices
