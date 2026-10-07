"""Build sinusoidal wave polylines as Simetri shapes."""

from __future__ import annotations

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

        >>> import simetri.graphics as sg
        >>> wave = sg.SineWave(period=40, amplitude=20, duration=80, n_points=4)
        >>> [[round(float(c), 6) or 0.0 for c in q[:2]] for q in wave.vertices]
        [[0.0, 0.0], [11.428571, 19.498558], [22.857143, -8.677675], [34.285714, -15.63663], [45.714286, 15.63663], [57.142857, 8.677675], [68.571429, -19.498558], [80.0, 0.0]]
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
        xform_matrix: NDArray | None = None,
        **kwargs: object,
    ) -> None:
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
        self.period = period
        self.amplitude = amplitude
        self.duration = duration
        self.n_points = n_points
        self.phase = phase
        self.damping = damping
        self.rot_angle = rot_angle

    def __repr__(self) -> str:
        """Return a SineWave string from this wave's vertices.

        Examples:
            >>> import simetri.graphics as sg
            >>> wave = sg.SineWave(
            ...     period=40, amplitude=20, duration=80, n_points=4
            ... )
            >>> repr(wave).startswith("SineWave([")
            True
            >>> str(wave).startswith("Shape")
            True
        """
        if len(self.primary_points) == 0:
            return "SineWave()"
        if len(self.primary_points) < 4:
            return f"SineWave({self.vertices})"
        return f"SineWave([{self.vertices[0]}, ..., {self.vertices[-1]}])"

    def copy_(self) -> SineWave:
        """Return a new ``SineWave`` with the same parameters.

        Returns:
            SineWave: A copy of this sine wave.

        Examples:
            >>> import simetri.graphics as sg
            >>> wave = sg.SineWave(period=40, amplitude=20, duration=80, n_points=4)
            >>> copy = wave.copy_()
            >>> copy.period, copy.amplitude, copy.duration, copy.n_points
            (40, 20, 80, 4)
            >>> copy.vertices == wave.vertices
            True
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
        )


def sine_wave(
    amplitude: float,
    frequency: float,
    duration: float,
    sample_rate: float,
    phase: float = 0,
) -> tuple[NDArray, NDArray]:
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

    Examples:
        >>> import simetri.graphics as sg
        >>> time, signal = sg.sine_wave(1.0, 1.0, 1.0, 4.0)
        >>> [round(float(x), 6) for x in time]
        [0.0, 0.25, 0.5, 0.75]
        >>> [round(float(x), 6) for x in signal]
        [0.0, 1.0, 0.0, -1.0]
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


def damping_function(
    amplitude: float, duration: float, sample_rate: float
) -> list[float]:
    """
    Generates a damping function based on the given amplitude, duration, and sample rate.

    Args:
        amplitude (float): The initial amplitude of the damping function.
        duration (float): The duration over which the damping occurs, in seconds.
        sample_rate (float): The number of samples per second.

    Returns:
        list[float]: Damping samples over time.

    Examples:
        >>> import simetri.graphics as sg
        >>> vals = sg.damping_function(10.0, 1.0, 4.0)
        >>> [round(v, 6) for v in vals]
        [10.0, 7.788008, 6.065307, 4.723666]
    """
    return [
        amplitude * exp(-i / (duration * sample_rate))
        for i in range(int(duration * sample_rate))
    ]


def sine_points(
    period: float = 40,
    amplitude: float = 20,
    duration: float = 40,
    n_points: int = 100,
    phase_angle: float = 0,
    damping: float = 0,
) -> list[list[float]]:
    """
    Generate sine wave points.

    Args:
        period: Period of the wave. Defaults to 40.
        amplitude: Amplitude of the wave. Defaults to 20.
        duration: Duration of the wave. Defaults to 40.
        n_points: Samples per period. Defaults to 100.
        phase_angle: Phase angle in radians. Defaults to 0.
        damping: Damping coefficient. Defaults to 0.

    Returns:
        list[list[float]]: ``(x, y)`` samples of the sine wave.

    Examples:
        >>> import simetri.graphics as sg
        >>> pts = sg.sine_points(period=10, amplitude=5, duration=10, n_points=4)
        >>> [[round(float(c), 6) or 0.0 for c in q[:2]] for q in pts]
        [[0.0, 0.0], [3.333333, 4.330127], [6.666667, -4.330127], [10.0, 0.0]]
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
