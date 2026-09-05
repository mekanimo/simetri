"""
Doubly Connected Edge List (DCEL)
Half-Edge Data Structure
"""


class Vertex:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.half_edge = None  # One outgoing half-edge


class Face:
    def __init__(self):
        self.half_edge = None  # One half-edge bounding this face


class HalfEdge:
    def __init__(self):
        self.vertex = None  # Destination vertex
        self.pair = None  # Opposite half-edge
        self.next = None  # Next half-edge around the face
        self.prev = None  # Previous half-edge around the face
        self.face = None  # Face on the left side


def get_face_vertices(face):
    coords = []
    start_he = face.half_edge
    he = start_he
    while True:
        # The vertex field stores the target of the half-edge
        # So the origin of 'he' is he.prev.vertex
        origin_vertex = he.prev.vertex
        coords.append((origin_vertex.x, origin_vertex.y))
        he = he.next
        if he == start_he:
            break
    return coords


def create_square_patch():
    # 1. Create vertices
    v0 = Vertex(0.0, 0.0)
    v1 = Vertex(1.0, 0.0)
    v2 = Vertex(1.0, 1.0)
    v3 = Vertex(0.0, 1.0)

    vertices = [v0, v1, v2, v3]

    # 2. Create half-edges for a single square face
    he0 = HalfEdge()
    he1 = HalfEdge()
    he2 = HalfEdge()
    he3 = HalfEdge()

    # Link face
    face = Face()
    face.half_edge = he0

    # Wire next/prev and vertex targets
    he0.vertex, he0.next, he0.prev, he0.face = v1, he1, he3, face
    he1.vertex, he1.next, he1.prev, he1.face = v2, he2, he0, face
    he2.vertex, he2.next, he2.prev, he2.face = v3, he3, he1, face
    he3.vertex, he3.next, he3.prev, he3.face = v0, he0, he2, face

    # Attach outgoing pointers to vertices
    v0.half_edge = he0
    v1.half_edge = he1
    v2.half_edge = he2
    v3.half_edge = he3

    return face, vertices
