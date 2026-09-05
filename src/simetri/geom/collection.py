"""Used for grouping geometric entities.
Similar to the Group objects but used for geometric operations,
such as convex and concave hulls, etc.
It can have cosmetic properties that can be applied to all members.
"""

from simetri.base.common import get_unique_id


class Collection:
    def __init__(self, entities=None):
        self.entities = entities
        self.line_width = None
        self.line_color = None
        self.fill_color = None
        self.id = get_unique_id(self)
        # finish this.

    @property
    def convex_hull(self):
        pass

    @property
    def concave_hull(self):
        pass
