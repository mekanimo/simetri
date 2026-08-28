## ✅ Goal & approach 🧩
"""

- 🏗️ Use `@dataclass` for each primitive with explicit attributes
- 🧰 Include an `extra: dict` on every element to support *any additional attributes* (presentation attributes, future SVG2 attributes, vendor quirks, etc.)
- 🖨️ Provide `to_element()` + `to_string()` to emit valid SVG markup

"""

import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ..graphics.all_enums import ColorMatrix, FilterType
from ..settings.settings import defaults

Number = int | float
NumOrStr = Number | str
MaybeSeq = None | str | Number | Sequence[NumOrStr]

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"


def _fmt_value(v: Any) -> str:
    """Convert Python values to SVG attribute strings."""
    if v is None:
        raise ValueError("internal: _fmt_value called with None")

    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, (list, tuple)):
        return " ".join(_fmt_value(x) for x in v)
    return str(v)


def _set_attrib(el: ET.Element, name: str, value: Any) -> None:
    if value is None:
        return
    el.set(name, _fmt_value(value))


@dataclass
class SVGElement:
    """Base element with common SVG-ish attributes."""

    id: str | None = None
    class_: str | None = None  # maps to "class"
    style: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def _apply_common(self, el: ET.Element) -> None:
        _set_attrib(el, "id", self.id)
        _set_attrib(el, "class", self.class_)
        _set_attrib(el, "style", self.style)
        for k, v in self.extra.items():
            _set_attrib(el, k, v)


# ----------------------------
# Filter container (<filter>)
# ----------------------------
@dataclass
class SVG_Filter(SVGElement):
    """
    Represents an SVG <filter> element containing filter primitives.

    Common <filter> attributes:
      - x, y, width, height
      - filterUnits, primitiveUnits
      - color-interpolation-filters
      - filterRes (pair or string)
      - href (SVG2) or xlink:href (legacy)
    """

    x: NumOrStr | None = None
    y: NumOrStr | None = None
    width: NumOrStr | None = None
    height: NumOrStr | None = None

    filterUnits: str | None = None  # userSpaceOnUse | objectBoundingBox
    primitiveUnits: str | None = None  # userSpaceOnUse | objectBoundingBox
    color_interpolation_filters: str | None = None  # "sRGB" | "linearRGB"
    filterRes: str | tuple[int, int] | None = None

    href: str | None = None  # SVG2
    xlink_href: str | None = None  # SVG1.1 legacy

    primitives: list["FilterPrimitive"] = field(default_factory=list)

    def add(self, *prims: "FilterPrimitive") -> "SVG_Filter":
        self.primitives.extend(prims)
        return self

    def to_element(self) -> ET.Element:
        # Note: ElementTree namespaces are easiest if we emit plain tags here and
        # include xmlns at the top-level string creation (see to_string()).
        el = ET.Element("filter")

        self._apply_common(el)
        _set_attrib(el, "x", self.x)
        _set_attrib(el, "y", self.y)
        _set_attrib(el, "width", self.width)
        _set_attrib(el, "height", self.height)
        _set_attrib(el, "filterUnits", self.filterUnits)
        _set_attrib(el, "primitiveUnits", self.primitiveUnits)
        _set_attrib(
            el, "color-interpolation-filters", self.color_interpolation_filters
        )

        if isinstance(self.filterRes, tuple):
            _set_attrib(
                el, "filterRes", f"{self.filterRes[0]} {self.filterRes[1]}"
            )
        else:
            _set_attrib(el, "filterRes", self.filterRes)

        _set_attrib(el, "href", self.href)
        if self.xlink_href:
            el.set(f"{{{XLINK_NS}}}href", self.xlink_href)

        for p in self.primitives:
            el.append(p.to_element())

        return el

    def to_string(
        self,
        pretty: bool = True,
        include_defs: bool = True,
        include_xmlns: bool = True,
    ) -> str:
        # Register xlink prefix if we might use it
        ET.register_namespace("xlink", XLINK_NS)

        root = self.to_element()

        if include_xmlns:
            # Ensure SVG namespace is present on the top element (works fine when embedding in <svg>)
            root.set("xmlns", SVG_NS)

        if include_defs:
            defs = ET.Element("defs")
            if include_xmlns:
                defs.set("xmlns", SVG_NS)
                defs.set("xmlns:xlink", XLINK_NS)
            defs.append(root)
            root = defs

        if pretty:
            _indent_xml(root)

        return ET.tostring(root, encoding="unicode")


def _indent_xml(elem: ET.Element, level: int = 0) -> None:
    """In-place pretty-printer for ElementTree."""
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            _indent_xml(child, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


# -----------------------------------
# Base filter primitive + common attrs
# -----------------------------------
@dataclass
class FilterPrimitive(SVGElement):
    """Base class for primitives with common primitive-region attributes."""

    # Common filter primitive attributes
    x: NumOrStr | None = None
    y: NumOrStr | None = None
    width: NumOrStr | None = None
    height: NumOrStr | None = None

    in_: str | None = None  # maps to "in"
    result: str | None = None
    primitive_type: ClassVar[FilterType] = None

    @property
    def type(self) -> FilterType:
        return self.primitive_type

    def _apply_primitive_common(self, el: ET.Element) -> None:
        self._apply_common(el)
        _set_attrib(el, "x", self.x)
        _set_attrib(el, "y", self.y)
        _set_attrib(el, "width", self.width)
        _set_attrib(el, "height", self.height)
        _set_attrib(el, "in", self.in_)
        _set_attrib(el, "result", self.result)


# -------------------
# 1) feBlend
# -------------------
@dataclass
class feBlend(FilterPrimitive):
    primitive_type: ClassVar[FilterType] = FilterType.BLEND
    in2: str | None = None
    mode: str | None = None  # normal|multiply|screen|darken|lighten|...

    def to_element(self) -> ET.Element:
        el = ET.Element("feBlend")
        self._apply_primitive_common(el)
        _set_attrib(el, "in2", self.in2)
        _set_attrib(el, "mode", self.mode)
        return el


# -------------------
# 2) feColorMatrix
# -------------------
@dataclass
class feColorMatrix(FilterPrimitive):
    primitive_type: ClassVar[FilterType] = FilterType.COLOR_MATRIX
    matrix_type: ColorMatrix | None = None
    values: MaybeSeq | None = None

    def __post_init__(self):
        if self.matrix_type is None:
            self.matrix_type = defaults["filter_color_matrix_type"]

        if self.values is None:
            if self.matrix_type == ColorMatrix.SATURATE:
                self.values = defaults["filter_color_matrix_saturate"]
            elif self.matrix_type == ColorMatrix.HUE_ROTATE:
                self.values = defaults["filter_color_matrix_hue_rotate"]
            else:
                self.values = defaults["filter_color_matrix_values"]

    def to_element(self) -> ET.Element:
        el = ET.Element("feColorMatrix")
        self._apply_primitive_common(el)
        _set_attrib(el, "type", self.matrix_type)
        _set_attrib(el, "values", self.values)
        return el


# -------------------
# 3) feComponentTransfer (+ feFuncR/G/B/A)
# -------------------
@dataclass
class feFunc(SVGElement):
    """Base for feFuncR/G/B/A"""

    type: str | None = None  # identity|table|discrete|linear|gamma
    tableValues: MaybeSeq | None = None
    slope: Number | None = None
    intercept: Number | None = None
    amplitude: Number | None = None
    exponent: Number | None = None
    offset: Number | None = None

    TAG: str = "feFuncR"  # overridden

    def to_element(self) -> ET.Element:
        el = ET.Element(self.TAG)
        self._apply_common(el)
        _set_attrib(el, "type", self.type)
        _set_attrib(el, "tableValues", self.tableValues)
        _set_attrib(el, "slope", self.slope)
        _set_attrib(el, "intercept", self.intercept)
        _set_attrib(el, "amplitude", self.amplitude)
        _set_attrib(el, "exponent", self.exponent)
        _set_attrib(el, "offset", self.offset)
        return el


@dataclass
class feFuncR(feFunc):
    TAG: str = "feFuncR"


@dataclass
class feFuncG(feFunc):
    TAG: str = "feFuncG"


@dataclass
class feFuncB(feFunc):
    TAG: str = "feFuncB"


@dataclass
class feFuncA(feFunc):
    TAG: str = "feFuncA"


@dataclass
class feComponentTransfer(FilterPrimitive):
    primitive_type: ClassVar[FilterType] = FilterType.COMPONENT_TRANSFER
    funcR: feFuncR | None = None
    funcG: feFuncG | None = None
    funcB: feFuncB | None = None
    funcA: feFuncA | None = None

    def to_element(self) -> ET.Element:
        el = ET.Element("feComponentTransfer")
        self._apply_primitive_common(el)
        for f in (self.funcR, self.funcG, self.funcB, self.funcA):
            if f is not None:
                el.append(f.to_element())
        return el


# -------------------
# 4) feComposite
# -------------------
@dataclass
class feComposite(FilterPrimitive):
    primitive_type: ClassVar[FilterType] = FilterType.COMPOSITE
    in2: str | None = None
    operator: str | None = None  # over|in|out|atop|xor|arithmetic
    k1: Number | None = None
    k2: Number | None = None
    k3: Number | None = None
    k4: Number | None = None

    def to_element(self) -> ET.Element:
        el = ET.Element("feComposite")
        self._apply_primitive_common(el)
        _set_attrib(el, "in2", self.in2)
        _set_attrib(el, "operator", self.operator)
        _set_attrib(el, "k1", self.k1)
        _set_attrib(el, "k2", self.k2)
        _set_attrib(el, "k3", self.k3)
        _set_attrib(el, "k4", self.k4)
        return el


# -------------------
# 5) feConvolveMatrix
# -------------------
@dataclass
class feConvolveMatrix(FilterPrimitive):
    primitive_type: ClassVar[FilterType] = FilterType.CONVOLVE_MATRIX
    order: int | tuple[int, int] | None = None
    kernelMatrix: MaybeSeq | None = None
    divisor: Number | None = None
    bias: Number | None = None
    targetX: int | None = None
    targetY: int | None = None
    edgeMode: str | None = None  # duplicate|wrap|none
    kernelUnitLength: Number | tuple[Number, Number] | None = None
    preserveAlpha: bool | None = None

    def to_element(self) -> ET.Element:
        el = ET.Element("feConvolveMatrix")
        self._apply_primitive_common(el)

        if isinstance(self.order, tuple):
            _set_attrib(el, "order", f"{self.order[0]} {self.order[1]}")
        else:
            _set_attrib(el, "order", self.order)

        _set_attrib(el, "kernelMatrix", self.kernelMatrix)
        _set_attrib(el, "divisor", self.divisor)
        _set_attrib(el, "bias", self.bias)
        _set_attrib(el, "targetX", self.targetX)
        _set_attrib(el, "targetY", self.targetY)
        _set_attrib(el, "edgeMode", self.edgeMode)

        if isinstance(self.kernelUnitLength, tuple):
            _set_attrib(
                el,
                "kernelUnitLength",
                f"{self.kernelUnitLength[0]} {self.kernelUnitLength[1]}",
            )
        else:
            _set_attrib(el, "kernelUnitLength", self.kernelUnitLength)

        _set_attrib(el, "preserveAlpha", self.preserveAlpha)
        return el


# -------------------
# Lighting subelements
# -------------------
@dataclass
class feDistantLight(SVGElement):
    azimuth: Number | None = None
    elevation: Number | None = None

    def to_element(self) -> ET.Element:
        el = ET.Element("feDistantLight")
        self._apply_common(el)
        _set_attrib(el, "azimuth", self.azimuth)
        _set_attrib(el, "elevation", self.elevation)
        return el


@dataclass
class fePointLight(SVGElement):
    x: Number | None = None
    y: Number | None = None
    z: Number | None = None

    def to_element(self) -> ET.Element:
        el = ET.Element("fePointLight")
        self._apply_common(el)
        _set_attrib(el, "x", self.x)
        _set_attrib(el, "y", self.y)
        _set_attrib(el, "z", self.z)
        return el


@dataclass
class feSpotLight(SVGElement):
    x: Number | None = None
    y: Number | None = None
    z: Number | None = None
    pointsAtX: Number | None = None
    pointsAtY: Number | None = None
    pointsAtZ: Number | None = None
    specularExponent: Number | None = None
    limitingConeAngle: Number | None = None

    def to_element(self) -> ET.Element:
        el = ET.Element("feSpotLight")
        self._apply_common(el)
        _set_attrib(el, "x", self.x)
        _set_attrib(el, "y", self.y)
        _set_attrib(el, "z", self.z)
        _set_attrib(el, "pointsAtX", self.pointsAtX)
        _set_attrib(el, "pointsAtY", self.pointsAtY)
        _set_attrib(el, "pointsAtZ", self.pointsAtZ)
        _set_attrib(el, "specularExponent", self.specularExponent)
        _set_attrib(el, "limitingConeAngle", self.limitingConeAngle)
        return el


# -------------------
# 6) feDiffuseLighting
# -------------------
@dataclass
class feDiffuseLighting(FilterPrimitive):
    primitive_type: ClassVar[FilterType] = FilterType.DIFFUSE_LIGHTING
    surfaceScale: Number | None = None
    diffuseConstant: Number | None = None
    kernelUnitLength: Number | tuple[Number, Number] | None = None
    lighting_color: str | None = None  # maps to "lighting-color"

    light: feDistantLight | fePointLight | feSpotLight | None = None

    def to_element(self) -> ET.Element:
        el = ET.Element("feDiffuseLighting")
        self._apply_primitive_common(el)
        _set_attrib(el, "surfaceScale", self.surfaceScale)
        _set_attrib(el, "diffuseConstant", self.diffuseConstant)

        if isinstance(self.kernelUnitLength, tuple):
            _set_attrib(
                el,
                "kernelUnitLength",
                f"{self.kernelUnitLength[0]} {self.kernelUnitLength[1]}",
            )
        else:
            _set_attrib(el, "kernelUnitLength", self.kernelUnitLength)

        _set_attrib(el, "lighting-color", self.lighting_color)

        if self.light is not None:
            el.append(self.light.to_element())

        return el


# -------------------
# 7) feDisplacementMap
# -------------------
@dataclass
class feDisplacementMap(FilterPrimitive):
    primitive_type: ClassVar[FilterType] = FilterType.DISPLACEMENT_MAP
    in2: str | None = None
    scale: Number | None = None
    xChannelSelector: str | None = None  # R|G|B|A
    yChannelSelector: str | None = None  # R|G|B|A

    def to_element(self) -> ET.Element:
        el = ET.Element("feDisplacementMap")
        self._apply_primitive_common(el)
        _set_attrib(el, "in2", self.in2)
        _set_attrib(el, "scale", self.scale)
        _set_attrib(el, "xChannelSelector", self.xChannelSelector)
        _set_attrib(el, "yChannelSelector", self.yChannelSelector)
        return el


# -------------------
# 8) feDropShadow
# -------------------
@dataclass
class feDropShadow(FilterPrimitive):
    primitive_type: ClassVar[FilterType] = FilterType.DROP_SHADOW
    dx: Number | None = None
    dy: Number | None = None
    stdDeviation: Number | tuple[Number, Number] | None = None
    flood_color: str | None = None  # "flood-color"
    flood_opacity: Number | None = None  # "flood-opacity"

    def to_element(self) -> ET.Element:
        el = ET.Element("feDropShadow")
        self._apply_primitive_common(el)
        _set_attrib(el, "dx", self.dx)
        _set_attrib(el, "dy", self.dy)
        if isinstance(self.stdDeviation, tuple):
            _set_attrib(
                el,
                "stdDeviation",
                f"{self.stdDeviation[0]} {self.stdDeviation[1]}",
            )
        else:
            _set_attrib(el, "stdDeviation", self.stdDeviation)
        _set_attrib(el, "flood-color", self.flood_color)
        _set_attrib(el, "flood-opacity", self.flood_opacity)
        return el


# -------------------
# 9) feFlood
# -------------------
@dataclass
class feFlood(FilterPrimitive):
    primitive_type: ClassVar[FilterType] = FilterType.FLOOD
    flood_color: str | None = None
    flood_opacity: Number | None = None

    def to_element(self) -> ET.Element:
        el = ET.Element("feFlood")
        # feFlood doesn't use "in" in the usual sense, but allowing it doesn't hurt.
        self._apply_primitive_common(el)
        _set_attrib(el, "flood-color", self.flood_color)
        _set_attrib(el, "flood-opacity", self.flood_opacity)
        return el


# -------------------
# 10) feGaussianBlur
# -------------------
@dataclass
class feGaussianBlur(FilterPrimitive):
    primitive_type: ClassVar[FilterType] = FilterType.GAUSSIAN_BLUR
    stdDeviation: Number | tuple[Number, Number] | None = None
    edgeMode: str | None = None  # duplicate|wrap|none

    def to_element(self) -> ET.Element:
        el = ET.Element("feGaussianBlur")
        self._apply_primitive_common(el)
        if isinstance(self.stdDeviation, tuple):
            _set_attrib(
                el,
                "stdDeviation",
                f"{self.stdDeviation[0]} {self.stdDeviation[1]}",
            )
        else:
            _set_attrib(el, "stdDeviation", self.stdDeviation)
        _set_attrib(el, "edgeMode", self.edgeMode)
        return el


# -------------------
# 11) feImage
# -------------------
@dataclass
class feImage(FilterPrimitive):
    primitive_type: ClassVar[FilterType] = FilterType.IMAGE
    href: str | None = None  # SVG2
    xlink_href: str | None = None  # legacy
    preserveAspectRatio: str | None = None
    crossOrigin: str | None = None

    def to_element(self) -> ET.Element:
        el = ET.Element("feImage")
        self._apply_primitive_common(el)
        _set_attrib(el, "href", self.href)
        if self.xlink_href:
            el.set(f"{{{XLINK_NS}}}href", self.xlink_href)
        _set_attrib(el, "preserveAspectRatio", self.preserveAspectRatio)
        _set_attrib(el, "crossOrigin", self.crossOrigin)
        return el


# -------------------
# 12) feMerge (+ feMergeNode)
# -------------------
@dataclass
class feMergeNode(SVGElement):
    in_: str | None = None

    def to_element(self) -> ET.Element:
        el = ET.Element("feMergeNode")
        self._apply_common(el)
        _set_attrib(el, "in", self.in_)
        return el


@dataclass
class feMerge(FilterPrimitive):
    primitive_type: ClassVar[FilterType] = FilterType.MERGE
    nodes: list[feMergeNode] = field(default_factory=list)

    def add_node(self, in_: str) -> "feMerge":
        self.nodes.append(feMergeNode(in_=in_))
        return self

    def to_element(self) -> ET.Element:
        el = ET.Element("feMerge")
        self._apply_primitive_common(el)
        for n in self.nodes:
            el.append(n.to_element())
        return el


# -------------------
# 13) feMorphology
# -------------------
@dataclass
class feMorphology(FilterPrimitive):
    primitive_type: ClassVar[FilterType] = FilterType.MORPHOLOGY
    operator: str | None = None  # erode|dilate
    radius: Number | tuple[Number, Number] | None = None

    def to_element(self) -> ET.Element:
        el = ET.Element("feMorphology")
        self._apply_primitive_common(el)
        _set_attrib(el, "operator", self.operator)
        if isinstance(self.radius, tuple):
            _set_attrib(el, "radius", f"{self.radius[0]} {self.radius[1]}")
        else:
            _set_attrib(el, "radius", self.radius)
        return el


# -------------------
# 14) feOffset
# -------------------
@dataclass
class feOffset(FilterPrimitive):
    primitive_type: ClassVar[FilterType] = FilterType.OFFSET
    dx: Number | None = None
    dy: Number | None = None

    def to_element(self) -> ET.Element:
        el = ET.Element("feOffset")
        self._apply_primitive_common(el)
        _set_attrib(el, "dx", self.dx)
        _set_attrib(el, "dy", self.dy)
        return el


# -------------------
# 15) feSpecularLighting
# -------------------
@dataclass
class feSpecularLighting(FilterPrimitive):
    primitive_type: ClassVar[FilterType] = FilterType.SPECULAR_LIGHTING
    surfaceScale: Number | None = None
    specularConstant: Number | None = None
    specularExponent: Number | None = None
    kernelUnitLength: Number | tuple[Number, Number] | None = None
    lighting_color: str | None = None

    light: feDistantLight | fePointLight | feSpotLight | None = None

    def to_element(self) -> ET.Element:
        el = ET.Element("feSpecularLighting")
        self._apply_primitive_common(el)
        _set_attrib(el, "surfaceScale", self.surfaceScale)
        _set_attrib(el, "specularConstant", self.specularConstant)
        _set_attrib(el, "specularExponent", self.specularExponent)

        if isinstance(self.kernelUnitLength, tuple):
            _set_attrib(
                el,
                "kernelUnitLength",
                f"{self.kernelUnitLength[0]} {self.kernelUnitLength[1]}",
            )
        else:
            _set_attrib(el, "kernelUnitLength", self.kernelUnitLength)

        _set_attrib(el, "lighting-color", self.lighting_color)

        if self.light is not None:
            el.append(self.light.to_element())

        return el


# -------------------
# 16) feTile
# -------------------
@dataclass
class feTile(FilterPrimitive):
    primitive_type: ClassVar[FilterType] = FilterType.TILE

    def to_element(self) -> ET.Element:
        el = ET.Element("feTile")
        self._apply_primitive_common(el)
        return el


# -------------------
# 17) feTurbulence
# -------------------
@dataclass
class feTurbulence(FilterPrimitive):
    primitive_type: ClassVar[FilterType] = FilterType.TURBULENCE
    baseFrequency: Number | tuple[Number, Number] | None = None
    numOctaves: int | None = None
    seed: Number | None = None
    stitchTiles: str | None = None  # stitch|noStitch
    turbulence_type: str | None = None  # turbulence|fractalNoise

    def to_element(self) -> ET.Element:
        el = ET.Element("feTurbulence")
        self._apply_primitive_common(el)

        if isinstance(self.baseFrequency, tuple):
            _set_attrib(
                el,
                "baseFrequency",
                f"{self.baseFrequency[0]} {self.baseFrequency[1]}",
            )
        else:
            _set_attrib(el, "baseFrequency", self.baseFrequency)

        _set_attrib(el, "numOctaves", self.numOctaves)
        _set_attrib(el, "seed", self.seed)
        _set_attrib(el, "stitchTiles", self.stitchTiles)
        _set_attrib(el, "type", self.turbulence_type)
        return el


# Usage
"""
## 🧪 Example usage (build a filter + emit `<defs>...</defs>`) 🧪

f = SVG_Filter(id="goo", x="-20%", y="-20%", width="140%", height="140%")

f.add(
    feGaussianBlur(in_="SourceGraphic", stdDeviation=8, result="blur"),
    feColorMatrix(
        in_="blur",
        matrix_type=ColorMatrix.MATRIX,
        values=[
            1,0,0,0,0,
            0,1,0,0,0,
            0,0,1,0,0,
            0,0,0,18,-7
        ],
        result="goo"
    ),
    feComposite(in_="SourceGraphic", in2="goo", operator="over", result="final"),
)

print(f.to_string(pretty=True, include_defs=True))
"""
