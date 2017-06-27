"""
This code was adapted from here:
http://programarcadegames.com/index.php?chapter=array_backed_grids
"""

import pygame
import time

# set title of screen
TITLE = "Animating Gridworld"

# this sets the WIDTH and HEIGHT of each grid location
WIDTH = 40
HEIGHT = 40

# this sets the margin between each cell
MARGIN = 5

# define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


class Viewer:
    """
    A class to render the Gridworld.
    """

    def __init__(self, num_cols, num_rows):
        self.num_cols = num_cols
        self.num_rows = num_rows
        # some pygame initializations
        pygame.init()
        pygame.display.set_caption(TITLE)
        window_width = WIDTH * self.num_cols + MARGIN * (self.num_cols + 1)
        window_height = HEIGHT * self.num_rows + MARGIN * (self.num_rows + 1)
        window_size = [window_width, window_height]
        self.screen = pygame.display.set_mode(window_size)

    # update the screen with updated information
    def update(self, entity_map):
        # generate an updated representation of the grid
        grid = generate_grid(self.num_rows, self.num_cols, entity_map)
        # set the screen background
        self.screen.fill(BLACK)
        # draw each square in the grid
        for row in range(self.num_rows):
            for column in range(self.num_cols):
                color = WHITE
                # color a goal square green
                if grid[row][column] == 'agent':
                    color = BLUE
                elif grid[row][column] == 'goal':
                    color = GREEN
                # define the square (rectangle) to draw
                rectangle_left_edge = (MARGIN + WIDTH) * column + MARGIN
                rectangle_top_edge = (MARGIN + HEIGHT) * row + MARGIN
                rectangle = [rectangle_left_edge, rectangle_top_edge, WIDTH, HEIGHT]
                # draw the square (rectangle)
                pygame.draw.rect(screen, color, rectangle)
        # update screen
        pygame.display.flip()

    def close(self):
        pygame.quit()


# utility functions ------------------------------------------------------------

# re-generate the grid given its size and the positions of various entities
def generate_grid(num_rows, num_cols, entity_map):
    grid = [[0 for _ in range(num_cols)] for _ in range(num_rows)]
    for entity,(x,y) in entity_map.items():
        grid[-(y+1)][x] = entity
    return grid
