from unittest import TestCase
from main import Model


class TestModel(TestCase):
    def test__neighbours_gen(self):
        model = Model()
        pattern = [
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
        model.init_from_array(pattern)

        expect = [1, 0, 1,
                  1, 1, 1,
                  0, 0, 0]

        for i in range(1, 7 + 1):
            _y, _x = model.index_to_row_col(i)
            n = model._neighbours_gen(_y, _x)
            r = sum(n)
            try:
                self.assertEqual(expect[i - 1], r)
            except AssertionError as e:
                print(f"i = {i}")
                print(f"_y, _x = {_y, _x}")
                raise e

    def test_is_cell_alive(self):
        model = Model()
        pattern = [
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
        model.init_from_array(pattern)

        _i = list(range(1, model.grid_size + 1))
        _expect = [x == 2 for x in _i]  # only i=2 is True

        for (index, expect_value) in zip(_i, _expect):
            r = model.is_cell_alive(index)
            try:
                self.assertEqual(expect_value, r)
            except AssertionError as e:
                print(f"i = {index}")
                raise e
