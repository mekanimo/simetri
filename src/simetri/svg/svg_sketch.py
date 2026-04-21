'''SVG related sketches are handled here.'''

from dataclasses import dataclass

from ..graphics.common import common_properties


@dataclass
class SVG_Mask:
    """SVG_Mask is used to configure SVG opacity mask attributes.

    Attributes mirror SVG gradient-style masks for alpha/luminance masking.
    """

    mask_type: str = None  # 'linear' or 'radial'
    x1: float = None
    y1: float = None
    x2: float = None
    y2: float = None
    cx: float = None
    cy: float = None
    r: float = None
    fx: float = None
    fy: float = None
    units: str = None  # gradient units
    spread_method: str = None
    transform: str = None
    stops: object = (
        None  # mask stops: [(offset, opacity)] or [(offset, color, opacity)]
    )
    mask_units: str = None  # maskUnits
    mask_content_units: str = None  # maskContentUnits

    def __post_init__(self):
        """Initialize the SVG_Mask object."""
        common_properties(self, id_only=True)

    def __str__(self):
        return f"SVG_Mask: {self.id}"

    def __repr__(self):
        return f"SVG_Mask: {self.id}"