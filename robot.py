import numpy as np
from heapq import *

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import from_levels_and_colors


constants = {
    "air": {"id": 0, "color": "#ffffff"},
    "home": {"id": 8, "color": "#004225"},
    "block": {"id": 1, "color": "#c00000"},
    "dirt": {"id": 2, "color": "#b18667"},
    "vacuum": {"id": 3, "color": "#000000"},
    "unknown": {"id": 4, "color": "#a4dded"},
    "inaccessible": {"id": 5, "color": "#014f63"},
    "DFS": {"id": 6, "color": "#ffb5b6"},
    "A*": {"id": 7, "color": "#ff5556"}
}


class UI():
    """
    Class for displaying things on screen
    """

    def __init__(self, delay=1e-9, verbose=False):
        """
        Initialize the UI
        """
        self._delay = delay
        sorted_constants = sorted(constants.items(), key=lambda x: x[1]["id"])

        # Draw figure
        fig = plt.figure("Vacuum room & AI")
        patches = [Patch(color=v["color"], label=k) for k, v in sorted_constants]
        fig.legend(handles=patches, loc="center", bbox_to_anchor=(0.5, 0.1), ncol=4)

        # Draw two subplots
        ax = fig.subplots(ncols=2, sharex=True, sharey=True)
        ax[0].set_axis_off()
        ax[1].set_axis_off()
        ax[0].set_title("Room view")
        ax[1].set_title("AI knowledge and search")

        # Create artists for the room view and the AI view
        levels = [v["id"] for k, v in sorted_constants]
        levels += [levels[-1] + 1]
        colors = [v["color"] for k, v in sorted_constants]
        cmap, norm = from_levels_and_colors(levels, colors)
        self.world_artist = ax[0].matshow([[0]], cmap=cmap, norm=norm, interpolation="none")
        self.ai_artist = ax[1].matshow([[0]], cmap=cmap, norm=norm, interpolation="none")

        # Create text elements
        text_table = ax[1].table(cellText=[" ", " "], loc="bottom", cellLoc="left", edges="open")
        self.vacuum_position_text = type("dummy_log", (object,), dict(set_text=lambda x: None))
        self.log_text = type("dummy_log", (object,), dict(set_text=lambda x: None))
        if verbose:
            text_table[0, 0].set_height(0.1)
            self.vacuum_position_text = text_table[0, 0].get_text()
            self.log_text = text_table[1, 0].get_text()

    def display(self, m, plot=0, overlay=None):
        """
        Draw matrix `m` in a plot. If `overlay` is not `None`, then draw over
        the matrix with extra overlaid values
        """
        data = m.copy()
        if overlay is not None:
            for xy in overlay[0]:
                data[xy] = overlay[1]

        if plot == 0:
            self.world_artist.set_data(data)
        elif plot == 1:
            self.ai_artist.set_data(data)

        if self._delay > 0:
            plt.pause(self._delay)
        else:
            plt.draw()
            plt.waitforbuttonpress()

        return data

    def update_vacuum_position(self, pos):
        """
        Update the displayed vacuum cleaner position
        """
        self.vacuum_position_text.set_text(f"vacuum position: {pos}")

    def log(self, message):
        """
        Display `message` in the interface log.
        """
        self.log_text.set_text(message)


class VacuumWorld():
    """
    Class representing a world populated by vacuum cleaner. The world is
    encoded as a grid. Each square of the grid can be:
    - air : a navigable square for vacuum cleaners,
    - block : an obstructed square where vacuum cleaners cannot go,
    - dirt : a navigable square that can be cleaned by a vacuum cleaner.

    This class includes methods for generating elements in the world, as well
    as methods for controlling vacuum cleaners. Vacuum cleaners can only move
    in the four cardinal directions.
    """


    def __init__(self, shape, ui):
        """
        Create a world of a given shape
        """
        self.ui = ui
        self.vacuums = []

        self.grid = np.full((shape[0] + 2, shape[1] + 2), constants["block"]["id"], dtype=int)
        self._inner_grid = self.grid[1:-1, 1:-1]
        self.reset()

    def get_true_shape(self):

        return self.grid.shape

    def display(self):
        """
        Display current world grid in the UI
        """
        self.ui.display(self.grid, plot=0, overlay=(self.vacuums, constants["vacuum"]["id"]))


    # World creation methods

    def reset(self):
        """
        Clear the world.
        """
        self._inner_grid.fill(constants["air"]["id"])
        self.display()

    def gen_bernoulli(self, p, element):
        """
        Generate elements in the world following a Bernoulli distribution
        """
        rng = np.random.default_rng()
        blocks = rng.binomial(1, p, size=self._inner_grid.shape)
        self._inner_grid[blocks == 1] = constants[element]["id"]
        self.display()

    def gen_constant(self, n, element):
        """
        Generate a fixed number of elements in the world randomly
        """
        rng = np.random.default_rng()
        flat_idx = rng.choice(np.arange(self._inner_grid.size), size=n, replace=False)
        idx = np.unravel_index(flat_idx, self._inner_grid.shape)
        self._inner_grid[idx] = constants[element]["id"]
        self.display()

    def set_elements(self, positions, element):
        """
        Place elements at given positions
        """
        for xy in positions:
            self.grid[xy] = constants[element]["id"]
        self.display()

    def add_vacuums(self, n):
        """
        Add a specified number of vacuum cleaners, to be placed on distinct
        available squares
        """
        rng = np.random.default_rng()
        available = np.nonzero(np.logical_or(self.grid == constants["air"]["id"], self.grid == constants["dirt"]["id"]))
        idx = rng.choice(np.arange(available[0].size), size=n, replace=False)
        self.vacuums = list(zip(available[0][idx], available[1][idx]))
        #self.vacuums = [(10, 5)]
        self.grid[self.vacuums[0]] = constants["home"]["id"]
        self.display()
        return self.vacuums

    # Vacuum methods

    def vacuum_check(self, i):
        """
        Check if the vacuum cleaner is on a dirty square
        """
        return self.grid[self.vacuums[i]] == constants["dirt"]["id"]

    def vacuum_use(self, i):
        """
        Use the vacuum cleaner. We assume that the vacuum cleaner is in
        a valid position
        """
        self.grid[self.vacuums[i]] = constants["air"]["id"]

    def vacuum_move(self, i, direction):
        """
        Attempt to move the vacuum cleaner in a specified direction
        """
        x0, y0 = self.vacuums[i]
        x1, y1 = x0, y0
        if direction == 'N':
            x1 -= 1
        elif direction == 'E':
            y1 += 1
        elif direction == 'S':
            x1 += 1
        elif direction == 'W':
            y1 -= 1

        if self.grid[x1, y1] == constants["block"]["id"]:
            return False
        else:
            self.vacuums[i] = (x1, y1)
            self.display()
            return True


class Vacuum():
    """
    Class controlling a single vacuum cleaners
    """

    def __init__(self, world, i):
        """
        Initialize the object by attaching it to the vacuum cleaner of
        `world`
        """
        self._world = world
        self._i = i

    def check(self):

        return self._world.vacuum_check(self._i)

    def use(self):

        return self._world.vacuum_use(self._i)



    def move(self, direction):

        return self._world.vacuum_move(self._i, direction)


class SingleVacuumAI():
    """
    Class that controls a vacuum cleaner in a world of air, blocks, and dirt.
    The vacuum cleaner belongs to a VacuumWorld.
    """

    def __init__(self, vacuum, shape, vacuum_pos, ui):
        """
        Initialize the AI with a vacuum cleaner's initial position
        """
        self.vacuum = vacuum
        self.space = np.full((shape[0] + 2, shape[1] + 2), constants["block"]["id"], dtype=int)
        self.space[1:-1, 1:-1] = constants["inaccessible"]["id"]

        self.start = vacuum_pos
        self.pos = self.start

        print("[AI] Starting vacuum AI at", self.start)
        self.space[self.pos] = constants["air"]["id"]
        self._update_accessible(self.pos)

        self._space_overlaid = None
        self.ui = ui
        self.ui.display(self.space, plot=1)

    def reset(self):

        self.ui.log("Reset knowledge space")
        self.space[1:-1, 1:-1] = constants["inaccessible"]["id"]

    def clean(self):
        """
        Clean the entire world and return to the start
        """
        if self.vacuum.check():
            self.vacuum.use()
        self._dfs_exploration()
        print("[AI] Done cleaning, going home")
        self.ui.log("Searching for home with A*")
        self._go_to(self.start)
        self.ui.log("At home")
        self.ui.display(self.space, plot=1)

    def _dfs_exploration(self):
        """
        Explore the grid with a greedy depth-first search method. For the
        problem of online graph exploration (we don't know anything about the
        graph beforehand), greedy algorithms are optimal

        We code imperatively, maintaining a stack of squares in the current
        branch being explored. While possible, we use an exploration heuristic
        (see `_exploration_heuristic`) to explore unknown squares. If all
        neighboring squares are known, then we backtrack to the most recent
        square with an unknown neighbor. Backtracking is done using an A*
        search algorithm (see `_astar`). This is marginally better than the
        simpler method of backtracking step by step to the oldest visited
        neighboring square.
        """
        stack = [None]

        while stack:
            neighbors = self._unknown_neighbors(self.pos)
            if neighbors:
                unknown_square = max(neighbors, key=self._exploration_heuristic)
                moved = self.vacuum.move(self._cardinal_direction(self.pos, unknown_square))
                if moved:
                    self.ui.log("moved, updating knowledge")
                    if self.vacuum.check():
                        self.vacuum.use()
                    stack.append(self.pos)
                    self.pos = unknown_square
                    self.space[self.pos] = constants["air"]["id"]
                    self._update_accessible(self.pos)
                else:
                    self.ui.log("blocked, updating knowledge")
                    self.space[unknown_square] = constants["block"]["id"]
            else:
                self.ui.log("backtracking with A*")
                prev_square = None
                while not neighbors:
                    prev_square = stack.pop()
                    if prev_square is None:
                        return
                    neighbors = self._unknown_neighbors(prev_square)
                self._go_to(prev_square)
            self.ui.update_vacuum_position(self.pos)
            self._space_overlaid = self.ui.display(self.space, plot=1, overlay=(stack[1:], constants["DFS"]["id"]))

    def _update_accessible(self, pos):
        """
        Internal function updating inaccessible neighbors to unknown squares
        """
        x, y = pos
        for neighbor in [(x - 1, y), (x, y - 1), (x + 1, y), (x, y + 1)]:
            if self.space[neighbor] == constants["inaccessible"]["id"]:
                self.space[neighbor] = constants["unknown"]["id"]

    def _unknown_neighbors(self, pos):
        """
        Internal function getting list of unknown neighbors
        """
        x, y = pos
        neighbors = []
        for xy in [(x - 1, y), (x, y - 1), (x + 1, y), (x, y + 1)]:
            if self.space[xy] == constants["unknown"]["id"]:
                neighbors.append(xy)
        return neighbors

    def _cardinal_direction(self, from_pos, to_pos):

        x0, y0 = from_pos
        x1, y1 = to_pos

        return " SEWN"[x1 - x0 + 2 * (y1 - y0)]

    def _exploration_heuristic(self, candidate_pos):
        """
        Internal function computing a heuristic when exploring a branch in
        depth-first search. We use the following heuristics:
        - Favor squares with many known neighbors,
        - Favor increasing distance from the start.

        """
        x, y = candidate_pos
        n_known = 0
        for xy in [(x - 1, y), (x, y - 1), (x + 1, y), (x, y + 1)]:
            if self.space[xy] == constants["air"]["id"] or self.space[xy] == constants["block"]["id"]:
                n_known += 1

        return (n_known, abs(x - self.start[0]) + abs(y - self.start[1]))

    def _go_to(self, to_pos):
        """
        Follow a path in known territory. Uses an A* search algorithm
        """
        path = self._astar(self.pos, to_pos)
        for i in range(len(path) - 1):
            self.vacuum.move(self._cardinal_direction(path[i], path[i + 1]))
        self.pos = to_pos
        self.ui.update_vacuum_position(self.pos)

    def _astar(self, from_pos, to_pos):
        """
        A* algorithm from from_pos to to_pos using 1-norm distance as the
        heuristic. We only travel on known squares. Since the heuristic is
        admissible, we are guaranteed to get the optimal solution.

        """
        prev_square = np.full(self.space.shape + (2,), -1)
        prev_square[from_pos] = from_pos
        counter = 0
        candidates = []
        best_candidate = (0, counter, 0, from_pos)

        while best_candidate[-1] != to_pos:
            x, y = best_candidate[-1]
            neighbors = []
            for xy in [(x - 1, y), (x, y - 1), (x + 1, y), (x, y + 1)]:
                if self.space[xy] == constants["air"]["id"] and prev_square[xy][0] == -1:
                    neighbors.append(xy)
                    prev_square[xy] = (x, y)
            for xy in neighbors:
                backward_cost = best_candidate[2] + 1
                cost = backward_cost + abs(xy[0] - to_pos[0]) + abs(xy[1] - to_pos[1])
                counter += 1
                heappush(candidates, (cost, counter, backward_cost, xy))
                self.ui.display(self._space_overlaid, plot=1,
                                overlay=((c[-1] for c in candidates), constants["A*"]["id"]))
            if candidates:
                best_candidate = heappop(candidates)
            else:
                print("[AI] A* did not find a path, producing an artificial path")
                return [from_pos]

        pos = best_candidate[-1]
        best_path = [pos]
        while pos != from_pos:
            pos = tuple(prev_square[pos])
            best_path.append(pos)
        self.ui.display(self._space_overlaid, plot=1, overlay=(best_path, constants["A*"]["id"]))
        return list(reversed(best_path))


def simulation(shape, density, dirtiness, delay=1e-9, verbose=True):
    """
    Generate a randomized world and run the vacuum cleaner AI.
    """
    ui = UI(delay, verbose)

    world = VacuumWorld(shape, ui)

    world.gen_bernoulli(dirtiness, "dirt")
    #world.set_elements([(5, 1), (6, 1), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 10), (2, 4), (2, 5), (2, 6), (2, 7), (2, 10), (3, 10), (3, 4),(3, 5),(3, 6),(3,7),
                        #(4, 10), (4, 4),(4, 5),(4, 6),(4,7),(5,10),(6,10),(7,10)], "block")
    world.gen_bernoulli(density, "block")
    vacuum_pos = world.add_vacuums(1)[0]


    vacuum = Vacuum(world, 0)
    ai = SingleVacuumAI(vacuum, shape, vacuum_pos, ui)
    ai.clean()

    print("Enter anything to close the UI: ", end="")
    input()


if __name__ == "__main__":
    simulation((10, 10), .3, 1, delay=0.1)
