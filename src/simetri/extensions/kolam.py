import numpy as np

from simetri.graphics import Canvas, LinPath, light_gray, Shape, Batch, Circle



def generate_board(n_rows=5, n_cols=5):
    choices = [1,   0]
    choices2 = [1,   0]
    # create two n_rows x n_cols matrices with random 1 and 0 values
    nodes = np.random.choice(choices, (n_rows, n_cols)).astype(int)
    squares = np.random.choice(choices2, (n_rows-1, n_cols-1)).astype(int)
    board = np.zeros((n_rows+1, n_cols+1)).astype(int)

    # insert the squares into the board in the middle of the board
    # leaving a border of zeros around the squares
    board[1:-1, 1:-1] = squares
    return board, nodes

def draw_kolam(board, nodes, canvas, size=40):
    n_rows, n_cols = board.shape
    # draw the squares
    size2 = size / 2
    for j in range(n_rows-1):
        for i in range(n_cols-1):
            if board[i, j] == 1:
                center = i * size - size2, j * size - size2
                # canvas.rectangle(center, size, size, fill_color=light_gray, alpha=0.5)
    # draw the nodes
    n_rows, n_cols = board.shape
    for j in range(n_rows-1):
        for i in range(n_cols-1):
            txt = f'{board[i, j+1]}{board[i+1, j+1]}{board[i, j]}{board[i+1, j]}'
            # if nodes[i, j] == 1:
            if True:
                canvas.circle((i*size, j*size), 2)
                if txt in d_path:
                    canvas.draw(d_path[txt].copy().translate(i*size, j*size), line_width=2, fill=False)

                # canvas.text(str(int(board[i, j])), (i*size-size2, j*size-size2), font_size=8, fill=False)
            # canvas.text(txt, (i*size-size2, j*size-size2), font_size=8, fill=False)

size = 40
size2 = size / 2
size4 = size / 4
x = size4
y = size2
path = LinPath((x, y))
path.line_to((x, 0))
path.arc(size4, size4, 0, -3*np.pi/2, 0)
path.forward(size2)

path4 = Batch([Shape([(-size2, -size4), (size2, -size4)]),
               Shape([(-size4, -size2), (-size4, size2)])
               ]).mirror(((size, -size), (-size, size)), reps=1)

x = -size4
y = -size2
path2_ = LinPath((x, y))
path2_.forward(size2)
path2_.arc(size4, size4, np.pi, -np.pi, 0)
path2_.forward(size2)
shape2_ = Shape([(-size2, -size4), (size2, -size4)])
path2 = Batch([path2_, shape2_])
path3 = LinPath((-size4, -size2))
path3.line_to((-size4, 0))
path3.arc(size4, size4, np.pi, -np.pi/4, 0)
path3 = Shape(path3.vertices).mirror(((1, -1), (-1, 1)), reps=1).mirror(((-1, -1), (1, 1)), reps=1)

path5_ = LinPath((-size2, -size4))
path5_.line_to((-size4, -size4))
path5_.arc(size2, size2, 3*np.pi/2, np.pi/2, 0)
path5_.forward(size4)
shape5_1 = Shape([(-size2, size4), (size2, size4)])
shape5_2 = Shape([(-size4, -size2), (-size4, size2)])
path5 = Batch([path5_, shape5_1, shape5_2])

circ = Circle((0, 0), size4)

d_path = {'0005': circ,
          '0001': Shape(path.vertices).rotate(-np.pi/2),
          '0010': Shape(path.vertices).mirror(((0, 0), (-1, 1))),
          '0011': path2,
          '0100': Shape(path.vertices),
          '0101': path2.copy().rotate(np.pi/2),
          '0110': path3,
          '0111': path5.copy().mirror(((-1, -1), (1, 1))),
          '1000': Shape(path.vertices).rotate(np.pi/2),

          '1111': path4,
          '1010': path2.copy().rotate(-np.pi/2),
          '1100': path2.copy().rotate(np.pi),
          '1001': path3.copy().rotate(np.pi/2),
          '1101': path5.copy().rotate(-np.pi/2),
          '1110': path5,
          '1011': path5.copy().mirror(((0, 0), (1, 0))),
}
canvas = Canvas()
canvas.rotate(np.pi/4)
# canvas.draw(path5.copy().mirror(((0, 0), (1, 0))))
# canvas.draw(path2_)
# canvas.translate(200, 0)
# canvas.draw(path2.copy().rotate(np.pi/2))
# canvas.draw(Shape(path2_.vertices))
for i in range(1):
    board, nodes = generate_board()
    draw_kolam(board, nodes, canvas)
    # canvas.translate(250, 0)

canvas.save('c:/tmp/kolam_test.pdf', overwrite=True)