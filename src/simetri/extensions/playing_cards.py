"""Playing cards as custom widgets. Self-contained; not part of the core library.

Use from a script::

    from simetri.extensions.playing_cards import PlayingCard, Rank, Suit
    import simetri.graphics as sg

    canvas = sg.Canvas()
    canvas.draw(PlayingCard(Rank.ACE, Suit.SPADES, center=(40, 50)))
    canvas.save("ace.svg", overwrite=True)
"""

import random
from enum import StrEnum
from math import pi

import simetri.graphics as sg
from simetri.base.core import _update_inplace

CARD_WIDTH = 62
CARD_HEIGHT = 88
CARD_FILLET_RADIUS = 6
CARD_INNER_BOX_INSET = CARD_FILLET_RADIUS + 3
CARD_FONT_SIZE = 10
CARD_PIP_SCALE = 5
CARD_PIP_BOX_INSET = CARD_FONT_SIZE
CARD_INDEX_GAP = CARD_FONT_SIZE / 5


class Rank(StrEnum):
    """Playing-card ranks, Ace through King."""

    ACE = "ACE"
    TWO = "TWO"
    THREE = "THREE"
    FOUR = "FOUR"
    FIVE = "FIVE"
    SIX = "SIX"
    SEVEN = "SEVEN"
    EIGHT = "EIGHT"
    NINE = "NINE"
    TEN = "TEN"
    JACK = "JACK"
    QUEEN = "QUEEN"
    KING = "KING"


class Suit(StrEnum):
    """Playing-card suits."""

    CLUBS = "CLUBS"
    DIAMONDS = "DIAMONDS"
    HEARTS = "HEARTS"
    SPADES = "SPADES"


_RANK_LABEL = {
    Rank.ACE: "A",
    Rank.TWO: "2",
    Rank.THREE: "3",
    Rank.FOUR: "4",
    Rank.FIVE: "5",
    Rank.SIX: "6",
    Rank.SEVEN: "7",
    Rank.EIGHT: "8",
    Rank.NINE: "9",
    Rank.TEN: "10",
    Rank.JACK: "J",
    Rank.QUEEN: "Q",
    Rank.KING: "K",
}

_FACE_RANKS = frozenset({Rank.JACK, Rank.QUEEN, Rank.KING})
_RED_SUITS = frozenset({Suit.HEARTS, Suit.DIAMONDS})

_SUIT_SVG = {
    Suit.HEARTS: (
        "M0 41 C-6 32-38 13-38-13 C-38-31-16-40 0-22 "
        "C16-40 38-31 38-13 C38 13 6 32 0 41Z"
    ),
    Suit.DIAMONDS: "M0-43L40 0L0 43L-40 0Z",
    Suit.CLUBS: (
        "M0-44 C-11.5-44-20-34.8-20-23.5 C-20-18.7-18.2-14.4-15-11 "
        "C-18.6-13.3-22.7-14.5-27-14.5 C-38-14.5-46-5.8-46 5 "
        "C-46 16.2-37.3 25-26 25 C-17.6 25-10.6 20.1-6.8 13.6 "
        "C-6.7 26-10.7 34.6-19.5 44 L19.5 44 "
        "C10.7 34.6 6.7 26 6.8 13.6 C10.6 20.1 17.6 25 26 25 "
        "C37.3 25 46 16.2 46 5 C46-5.8 38-14.5 27-14.5 "
        "C22.7-14.5 18.6-13.3 15-11 C18.2-14.4 20-18.7 20-23.5 "
        "C20-34.8 11.5-44 0-44Z"
    ),
    Suit.SPADES: (
        "M0-43 C-6-31-34-13-34 7 C-34 20-26 27-15 27 "
        "C-10 27-6 24-3 19 C-4 29-9 37-17 43 L17 43 "
        "C9 37 4 29 3 19 C6 24 10 27 15 27 "
        "C26 27 34 20 34 7 C34-13 6-31 0-43Z"
    ),
}


def _suit_color(suit: Suit):
    """Return the ink color for ``suit``."""
    if suit in _RED_SUITS:
        return sg.red
    return sg.black


def _suit_pip(suit: Suit, scale: float, fill_color):
    """Return a closed suit pip centered at the origin."""
    if suit not in _SUIT_SVG:
        raise ValueError(f"Unknown suit: {suit!r}")
    pip = sg.svg_path_to_path2d(_SUIT_SVG[suit])
    mid_x, mid_y = pip.b_box.midpoint[:2]
    pip.translate(-mid_x, -mid_y)
    pip.scale(1, -1)
    height = pip.b_box.height
    pip.scale((2 * scale) / height)
    pip.fill = True
    pip.fill_color = fill_color
    pip.line_color = fill_color
    return pip


def _rank_tag(
    label: str, pos, ink, font_size: float = CARD_FONT_SIZE
) -> sg.Tag:
    """Return a frameless centered rank letter at ``pos``."""
    return sg.Tag(
        label,
        pos,
        font_size=font_size,
        font_color=ink,
        fill=False,
        align=sg.Align.CENTER,
        frame_inner_sep=0,
    )


def _pip_positions(pip_box: sg.BoundingBox, rank: Rank) -> list:
    """Return pip centers from ``pip_box`` anchors and offset lines."""
    north = pip_box.offset_point(sg.Anchor.NORTH, 0, 0)
    south = pip_box.offset_point(sg.Anchor.SOUTH, 0, 0)
    east = pip_box.offset_point(sg.Anchor.EAST, 0, 0)
    west = pip_box.offset_point(sg.Anchor.WEST, 0, 0)
    center = pip_box.offset_point(sg.Anchor.CENTER, 0, 0)
    northwest = pip_box.offset_point(sg.Anchor.NORTHWEST, 0, 0)
    northeast = pip_box.offset_point(sg.Anchor.NORTHEAST, 0, 0)
    southwest = pip_box.offset_point(sg.Anchor.SOUTHWEST, 0, 0)
    southeast = pip_box.offset_point(sg.Anchor.SOUTHEAST, 0, 0)
    mid_top = sg.midpoint(north, center)
    mid_bottom = sg.midpoint(south, center)
    four = [northwest, northeast, southwest, southeast]
    if rank == Rank.ACE:
        return [center]
    if rank == Rank.TWO:
        return [north, south]
    if rank == Rank.THREE:
        return [north, center, south]
    if rank == Rank.FOUR:
        return four
    if rank == Rank.FIVE:
        return four + [center]
    if rank == Rank.SIX:
        return four + [west, east]
    if rank == Rank.SEVEN:
        return four + [west, east, mid_top]
    if rank == Rank.EIGHT:
        return four + [west, east, mid_top, mid_bottom]
    if rank == Rank.NINE:
        return four + [
            sg.midpoint(northwest, west),
            sg.midpoint(northeast, east),
            sg.midpoint(southwest, west),
            sg.midpoint(southeast, east),
            center,
        ]
    if rank == Rank.TEN:
        return four + [
            sg.midpoint(northwest, west),
            sg.midpoint(northeast, east),
            sg.midpoint(southwest, west),
            sg.midpoint(southeast, east),
            north,
            south,
        ]
    return []


def _card_graphics(
    rank: Rank, suit: Suit, width: float, height: float
) -> sg.Group:
    """Build the face and marks at the origin from ``width`` and ``height``."""
    face = sg.Rectangle(
        (0, 0),
        width,
        height,
        fill=True,
        fill_color=sg.white,
        line_color=sg.black,
        draw_fillets=True,
        fillet_radius=CARD_FILLET_RADIUS,
    )
    marks_box = face.b_box.get_inflated_b_box(-CARD_INNER_BOX_INSET)
    pip_box = marks_box.get_inflated_b_box(-CARD_PIP_BOX_INSET)
    ink = _suit_color(suit)
    label = _RANK_LABEL[rank]
    pip_scale = CARD_PIP_SCALE * 0.45
    items = [face]

    top_pos = marks_box.offset_point(sg.Anchor.NORTHWEST, 0, 0)
    rank_top = _rank_tag(label, top_pos, ink)
    suit_top = _suit_pip(suit, pip_scale, ink)
    line = rank_top.b_box.offset_line(
        sg.Side.BOTTOM, CARD_INDEX_GAP + suit_top.b_box.height / 2
    )
    suit_x, suit_y = sg.midpoint(line[0], line[1])[:2]
    mid_x, mid_y = suit_top.b_box.midpoint[:2]
    suit_top.translate(suit_x - mid_x, suit_y - mid_y)
    items.extend([rank_top, suit_top])

    bottom_pos = marks_box.offset_point(sg.Anchor.SOUTHEAST, 0, 0)
    rank_bottom = _rank_tag(label, bottom_pos, ink).rotate(pi, about=bottom_pos)
    suit_bottom = _suit_pip(suit, pip_scale, ink)
    line = rank_bottom.b_box.offset_line(
        sg.Side.BOTTOM, CARD_INDEX_GAP + suit_bottom.b_box.height / 2
    )
    suit_x, suit_y = sg.midpoint(line[0], line[1])[:2]
    mid_x, mid_y = suit_bottom.b_box.midpoint[:2]
    suit_bottom.translate(suit_x - mid_x, suit_y - mid_y)
    suit_bottom.rotate(pi, about=bottom_pos)
    items.extend([rank_bottom, suit_bottom])

    if rank in _FACE_RANKS:
        items.append(
            _rank_tag(
                label,
                marks_box.offset_point(sg.Anchor.CENTER, 0, 0),
                ink,
                CARD_FONT_SIZE * 3,
            )
        )
        return sg.Group(items)

    center_scale = CARD_PIP_SCALE
    if rank == Rank.ACE:
        center_scale = CARD_PIP_SCALE * 2
    _, box_mid_y = pip_box.midpoint[:2]
    for pos in _pip_positions(pip_box, rank):
        pip = _suit_pip(suit, center_scale, ink)
        mid_x, mid_y = pip.b_box.midpoint[:2]
        pip_x, pip_y = pos[:2]
        pip.translate(pip_x - mid_x, pip_y - mid_y)
        if pip_y < box_mid_y:
            pip.rotate(pi, about=pos)
        items.append(pip)
    return sg.Group(items)


class PlayingCard(sg.Group):
    """A single playing card drawn through ``draw_list``.

    Args:
        rank: ``Rank`` of the card.
        suit: ``Suit`` of the card.
        center: Card center. Defaults to ``(0, 0)``.
        width: Card width. Defaults to ``CARD_WIDTH``.
        height: Card height. Defaults to ``CARD_HEIGHT``.
    """

    def __init__(
        self,
        rank: Rank,
        suit: Suit,
        center=(0, 0),
        width: float | None = None,
        height: float | None = None,
    ):
        if rank not in Rank:
            raise ValueError(f"rank must be a Rank, got {rank!r}")
        if suit not in Suit:
            raise ValueError(f"suit must be a Suit, got {suit!r}")
        super().__init__()
        self.rank = rank
        self.suit = suit
        self.center = center[:2]
        if width is None:
            width = CARD_WIDTH
        if height is None:
            height = CARD_HEIGHT
        self.width = width
        self.height = height
        self._rebuild_draw_list()

    def _rebuild_draw_list(self) -> None:
        """Build the card at the origin, then sit it on ``self.center``."""
        card_graphics = _card_graphics(
            self.rank, self.suit, self.width, self.height
        )
        cx, cy = self.center[:2]
        card_graphics.translate(cx, cy)
        self.clear()
        self.append(card_graphics)
        self.draw_list = [card_graphics]

    def _update(
        self,
        xform_matrix,
        reps=0,
        take=None,
        incr=None,
        merge: bool = False,
        xform_type=None,
    ):
        """Move the card center, then rebuild the face."""
        if take is not None:
            raise ValueError(
                "PlayingCard._update does not support take=; "
                "transform the whole card."
            )
        if merge:
            raise ValueError("PlayingCard._update does not support merge=True.")
        if reps == 0:
            transformed = sg.homogenize([self.center]) @ xform_matrix
            x, y = transformed[0][:2]
            self.center = (float(x), float(y))
            self._rebuild_draw_list()
            return self

        cards = []
        for i in range(reps):
            if incr is not None and i > 0:
                xform_matrix = _update_inplace(xform_matrix, xform_type, incr)
            card_copy = self.copy()
            card_copy._update(xform_matrix)
            cards.append(card_copy)
        return cards

    def copy(self) -> "PlayingCard":
        """Return a card with the same rank, suit, center, and size."""
        return PlayingCard(
            self.rank,
            self.suit,
            center=self.center,
            width=self.width,
            height=self.height,
        )


class Deck(list):
    """A list of ``PlayingCard`` objects with shuffle and deal from the top."""

    def shuffle(
        self,
        seed: int | None = None,
        *,
        rng: random.Random | None = None,
    ) -> "Deck":
        """Shuffle this deck in place (mutated).

        Args:
            seed: Seed for a local RNG. Defaults to None.
            rng: Existing generator to use. Defaults to None.
                When set, ``seed`` is ignored.

        Returns:
            Deck: This deck, shuffled.
        """
        if rng is None:
            rng = random.Random(seed)
        rng.shuffle(self)
        return self

    def deal_card(self) -> PlayingCard:
        """Remove and return the top card (mutated).

        The top card is the first item in the deck. After ``shuffle`` that
        order is random. Without ``shuffle`` cards come in deck order.

        Returns:
            PlayingCard: The dealt card.

        Raises:
            ValueError: If the deck is empty.
        """
        if not self:
            raise ValueError("Cannot deal from an empty deck.")
        return self.pop(0)


def standard_deck(origin=(0, 0), gap: float | None = None) -> Deck:
    """Return 52 cards in rank-major order, laid out in four suit rows.

    Args:
        origin: Center of the first card (Ace of Clubs). Defaults to ``(0, 0)``.
        gap: Extra space between cards. Defaults to ``CARD_WIDTH / 5``.

    Returns:
        Deck: All 52 cards.
    """
    if gap is None:
        gap = CARD_WIDTH / 5
    origin_x, origin_y = origin[:2]
    cards = Deck()
    suits = (Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS, Suit.SPADES)
    ranks = (
        Rank.ACE,
        Rank.TWO,
        Rank.THREE,
        Rank.FOUR,
        Rank.FIVE,
        Rank.SIX,
        Rank.SEVEN,
        Rank.EIGHT,
        Rank.NINE,
        Rank.TEN,
        Rank.JACK,
        Rank.QUEEN,
        Rank.KING,
    )
    for row, suit in enumerate(suits):
        for column, rank in enumerate(ranks):
            center = (
                origin_x + column * (CARD_WIDTH + gap),
                origin_y - row * (CARD_HEIGHT + gap),
            )
            cards.append(PlayingCard(rank, suit, center=center))
    return cards
