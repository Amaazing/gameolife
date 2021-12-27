import asyncio
import math
from typing import Callable, Union, Any

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

        self.top_left: Union[bool, None] = lambda y, x: self.safe_index(y - 1, x - 1)
        self.top: Union[bool, None] = lambda y, x: self.safe_index(y - 1, x)
        self.top_right: Union[bool, None] = lambda y, x: self.safe_index(y - 1, x + 1)
        self.right: Union[bool, None] = lambda y, x: self.safe_index(y, x + 1)
        self.bottom_right: Union[bool, None] = lambda y, x: self.safe_index(y + 1, x + 1)
        self.bottom: Union[bool, None] = lambda y, x: self.safe_index(y + 1, x)
        self.bottom_left: Union[bool, None] = lambda y, x: self.safe_index(y + 1, x - 1)
        self.left: Union[bool, None] = lambda y, x: self.safe_index(y, x - 1)

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
        new_state = rule(_y, _x)
        self._update_counter.increment()

        await self._update_event_obj.wait()
        self.grid[_y][_x] = new_state

    def safe_index(self, y: int, x: int) -> Union[bool, None]:
        r: Union[bool, None]
        if (
                y >= self.grid_size or
                x >= self.grid_size or
                y < 0 or
                x < 0
        ):
            return None
        return self.grid[y][x]

    async def next_iter(self):
        # TODO: this will be defined elsewhere, as it could get very complicated
        def rule_function(y: int, x: int) -> bool:
            return bool(self.top(y, x))

        # for each cell in the grind, 1 index instead of 0 index
        self._update_counter.reset_counter()
        update_tasks = []
        for cell_index in range(1, self._number_of_cells + 1):
            # the update task is a coroutine so that it can
            # evaluate the value of the cell for the next iteration,
            # and apply the value once the whole grid has been evaluated
            task = asyncio.create_task(
                self.update_cell(cell_index, rule=rule_function))
            update_tasks.append(task)

        # wait for all the cells to have updated
        await asyncio.gather(*update_tasks)

    def get_grid(self) -> np.ndarray:
        return self.grid


class View:
    def __init__(self, model: Model):
        self.model = model
        self.frame = 0

    async def update(self):
        await self.model.next_iter()
        self.frame += 1

    def print_grid(self):
        print(self.model.get_grid())


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
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]
    # grid must be a square
    model = Model()
    model.init_from_array(pattern)
    view = View(model)
    loop = Loop(fps=1, view=view)
    await loop.start_loop()


if __name__ == "__main__":
    asyncio.run(main())

'''
    for every cell in the gird:
        
'''