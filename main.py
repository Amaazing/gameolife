import asyncio
import math
from typing import Callable, Union, Any, Iterable

import numpy as np


class Counter:
    def __init__(self, n=0):
        self.counter = n

    def increment(self):
        self.counter += 1

    def decrement(self):
        _v = self.counter - 1
        if _v < 0:
            raise ValueError("counter cannot be negative")
        else:
            self.counter = _v

    def reset_counter(self):
        self.counter = 0

    def get_counter(self) -> int:
        return self.counter


class TriggerCounter(Counter):
    """
    Runs a specific function when the Counter reaches the trigger_value - then resets the counter to 0
    """

    def __init__(self,
                 n=0,
                 trigger_function: Callable[..., Any] = None,
                 trigger_params: list = None,
                 trigger_value: int = 0,
                 reset_function: Callable[..., Any] = None,
                 reset_params: list = None):
        super(TriggerCounter, self).__init__(n)
        if trigger_value == n:
            raise ValueError(f"trigger value ({trigger_value}) cannot be the same as n ({n}).")
        self.trigger_function = trigger_function
        self.trigger_params = trigger_params
        self.trigger_value = trigger_value
        self.reset_function = reset_function
        self.reset_params = reset_params

    def increment(self):
        _v = self.counter + 1
        if _v > self.trigger_value:
            raise ValueError(f"counter cannot be over the trigger value ({self.trigger_value})")
        self.counter = _v
        self.check_if_trigger_value_is_reached()

    def check_if_trigger_value_is_reached(self):
        if self.counter == self.trigger_value:
            self.trigger_function(*self.trigger_params)

    def reset_counter(self):
        super(TriggerCounter, self).reset_counter()
        if self.reset_function:
            self.reset_function(*self.reset_params)


class Model:

    def __init__(self):
        self.grid = None
        self.grid_size = 0
        self._number_of_cells = 0

        # contains a list of indexs for cells that have had their value changed in the update cycle
        # can be useful later on for a View class to grab just the changed cells and update those,
        # instead of the whole grid
        self._indexes_updated = []

        self.top_left: Union[bool, None] = lambda y, x: self.safe_index(y - 1, x - 1)
        self.top: Union[bool, None] = lambda y, x: self.safe_index(y - 1, x)
        self.top_right: Union[bool, None] = lambda y, x: self.safe_index(y - 1, x + 1)
        self.right: Union[bool, None] = lambda y, x: self.safe_index(y, x + 1)
        self.bottom_right: Union[bool, None] = lambda y, x: self.safe_index(y + 1, x + 1)
        self.bottom: Union[bool, None] = lambda y, x: self.safe_index(y + 1, x)
        self.bottom_left: Union[bool, None] = lambda y, x: self.safe_index(y + 1, x - 1)
        self.left: Union[bool, None] = lambda y, x: self.safe_index(y, x - 1)

    def _neighbours_gen(self, y: int, x: int) -> Iterable[int]:
        _f: list = [self.top_left, self.top, self.top_right, self.right, self.bottom_right, self.bottom,
                    self.bottom_left, self.left]
        return (f(y, x) or 0 for f in _f)  # get values for neighbours of [y, x]

    def init_from_array(self, _pattern: list):
        self.grid = np.asarray(_pattern)
        self.grid_size = len(_pattern)
        self._init()

    def init(self, grid_size: int):
        self.grid = np.zeros((grid_size, grid_size), bool)
        self.grid_size = grid_size
        self._init()

    def _init(self):
        """
        completes the initialisation of Model by setting values that can
        only be done after the initial init (init, init_from_array)
        """
        self._number_of_cells = int(math.pow(self.grid_size, 2))

        # this Event object is used when updating the grid to the
        # next iteration. first all the cells new value is  evaluated,
        # once that's done this Event object will be set, and then
        # the cells will update their value to the new value. This is
        # to prevent cells updating separately as that will impact the
        # checking methods.
        self._update_event_obj = asyncio.Event()
        # Set the Event object when the trigger is invoked
        # Clear the Event object when the counter is reset
        self._update_counter = TriggerCounter(
            trigger_function=(lambda e: e.set()),
            trigger_params=[self._update_event_obj],
            trigger_value=self._number_of_cells,
            reset_function=(lambda e: e.clear()),
            reset_params=[self._update_event_obj]
        )

    def index_to_row_col(self, index: int) -> tuple:
        x = (self.grid_size - 1 + index) % self.grid_size  # wolframalpha
        y = math.floor(index / (self.grid_size + 0.1))
        return y, x

    async def update_cell(self, i: int, rule: Callable[[int, int], bool] = None):
        _y, _x = self.index_to_row_col(i)
        current_value = self.grid[_y][_x]
        new_state = rule(_y, _x)
        self._update_counter.increment()

        # if the new state is None, do nothing
        # if there is no change in state, do nothing
        if new_state is None or new_state == current_value:
            return

        await self._update_event_obj.wait()
        self.grid[_y][_x] = new_state
        self._indexes_updated.append(i)

    def safe_index(self, y: int, x: int) -> Union[bool, None]:
        if (
                y >= self.grid_size or
                x >= self.grid_size or
                y < 0 or
                x < 0
        ):
            return None
        return self.grid[y][x]

    async def next_iter(self):
        # for each cell in the grid.
        # 1 index instead of 0 index
        self._indexes_updated = []
        self._update_counter.reset_counter()
        update_tasks = []
        for cell_index in range(1, self._number_of_cells + 1):
            # the update task is a coroutine so that it can
            # evaluate the value of the cell for the next iteration,
            # and apply the value once the whole grid has been evaluated
            task = asyncio.create_task(
                self.update_cell(cell_index, rule=self.rule_function))
            update_tasks.append(task)

        # wait for all the cells to have updated
        await asyncio.gather(*update_tasks)

    def rule_function(self, y: int, x: int) -> Union[bool, None]:
        """
            Any live cell with fewer than two live neighbours dies, as if by underpopulation.
            Any live cell with two or three live neighbours lives on to the next generation.
            Any live cell with more than three live neighbours dies, as if by overpopulation.
            Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
        :param y:
        :param x:
        :return: Should cell be alive?
                 True if the cell should be alive, False if the cell should die, None if nothing should happen
        """
        neighbour_count = sum(self._neighbours_gen(y, x))
        if self.is_cell_alive(y=y, x=x):
            if neighbour_count < 2:
                return False
            elif neighbour_count in [2, 3]:
                return True
            elif neighbour_count > 3:
                return False
        else:
            if neighbour_count == 3:
                return True

    def is_cell_alive(self, *p_args, i=None, y=None, x=None):
        if p_args:
            raise ValueError("is_cell_alive does not accept positional parameters")
        if None in [y, x] and i is None:
            raise ValueError(f"{'y' if y is None else 'x'} is None")
        if i:
            return bool(self.get_cell_by_index(i))
        else:
            return bool(self.get_cell_by_y_x(y, x))

    def get_cell_by_index(self, i):
        _y, _x = self.index_to_row_col(i)
        return self.get_cell_by_y_x(_y, _x)

    def get_cell_by_y_x(self, _y: int, _x: int):
        return self.safe_index(_y, _x)

    def get_grid(self) -> np.ndarray:
        return self.grid

    def get_updated_cell_indexs(self) -> list:
        """
        returns the list of indexs, of cells that were updated in the last
        `next_iter` invocation
        :return: list of indexs, of cells that had their value changed
        """
        return self._indexes_updated


class View:
    def __init__(self, model: Model):
        self.model = model
        self.frame = 0

    async def update(self):
        await self.model.next_iter()
        self.frame += 1

    def print_grid(self):
        print(self.model.get_grid())
        print(f"changed values:", *self.model.get_updated_cell_indexs())


class Loop:
    def __init__(self, fps=None, view: View = None):
        if None in [fps, view]:
            raise Exception()
        self.view = view
        self.fps = fps

    async def start_loop(self):
        while True:
            self.view.print_grid()
            await asyncio.sleep(float(1 / self.fps))
            await self.view.update()


async def main():
    pattern = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ]
    # pattern = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for i in range(10)]
    # pattern[0][4] = 1
    # grid must be a square
    model = Model()
    model.init_from_array(pattern)
    view = View(model)
    loop = Loop(fps=1, view=view)
    await loop.start_loop()


if __name__ == "__main__":
    asyncio.run(main())
