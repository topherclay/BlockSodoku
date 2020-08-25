import keyboard
import time
import pyautogui as pag
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt






class BoardInstance:

    def __init__(self, grid=np.zeros([9, 9])):
        self.grid = grid
        self.is_valid = True
        self.instances = []
        self.placements = []
        self.scores = []


    def try_every_spot(self, piece):
        self.instances = []
        self.placements = []


        for column in range(0, 9):
            for row in range(0, 9):
                new_instance = BoardInstance()
                new_instance.grid = self.grid.copy()

                new_instance.attempt_to_place_piece(piece.info.piece_grid, [row, column])
                if new_instance.is_valid:
                    # print(f"Is this valid: {new_instance.is_valid}")
                    # print(new_instance.grid)
                    self.instances.append(new_instance)
                    self.placements.append([row, column])

        # print(f"{len(self.instances)} valid placements")


    def analyse_all_instances(self):
        self.scores = []
        for instance in self.instances:

            instance.resolve_scoring_event()


            score, _, _ = BoardAnalyzer(instance.grid).report()

            self.scores.append(score)



    def get_best_placement_and_update(self):

        master = zip(self.scores, self.placements, self.instances)


        best = sorted(master, reverse=True)


        self.grid = best[0][2].grid

        self.resolve_scoring_event()
        # print(f"grid will become this:")
        # print(self.grid)
        return best[0][1]



    def attempt_to_place_piece(self, piece_grid, coordinate):

        for row in range(0, 5):
            for column in range(0, 5):
                if piece_grid[column, row]:
                    cell_y = coordinate[0] + column
                    cell_x = coordinate[1] + row
                    self.attempt_add_cell([cell_y, cell_x])
                    if not self.is_valid:
                        return


        self.resolve_scoring_event()


    def attempt_add_cell(self, coordinate):

        if coordinate[0] < 0 or coordinate[0] > 8:
            self.is_valid = False
            return False
        if coordinate[1] < 0 or coordinate[1] > 8:
            self.is_valid = False
            return False

        # returns True only if placement is valid
        # places in format (y, x) in relation to board
        if self.grid[coordinate[0], coordinate[1]]:
            # coordinate is already occupied. Illegal move
            self.is_valid = False
            return False
        else:
            self.grid[coordinate[0], coordinate[1]] = 1
            return True

    def resolve_scoring_event(self):
        scoring_rows = []
        scoring_columns = []


        # print("scoring")
        for row in range(0, 9):
            if self.grid[row, :].all():
                # print(f"row {row} is a score")
                scoring_rows.append(row)

        for col in range(0, 9):
            if self.grid[:, col].all():
                # print(f"column {col} is a score")
                scoring_columns.append(col)

        # print(f"scoring_rows = {scoring_rows}")
        # print(f"scoring_columns = {scoring_columns}")


        region = 0
        scoring_regions = []

        for column in range(0, 7, 3):

            for row in range(0, 7, 3):

                if self.grid[column:column + 3, row: row + 3].all() == 1:
                    # print(f"found full region at col:{column} and row:{row}")
                    # print(f"found full region at {region}")
                    scoring_regions.append(region)
                    # print(f"region {region} scored?")
                # else:
                    # print(f"found empty region at {column} and {row}")
                region += 1
        # print(f"scoring regions = {scoring_regions}")
        # print("end scoring")



        #  Start clearing out scoring rows and columns!

        clearing_row_step = 0
        clearing_column_step = 0

        for row in range(0, 9):
            if clearing_row_step in scoring_rows:
                self.grid[row, :] = 0
                # print(f"cleared row {row}")
            clearing_row_step = clearing_row_step + 1

        for col in range(0, 9):

            if clearing_column_step in scoring_columns:
                self.grid[:, col] = 0
                # print(f"cleared column {col}")
            clearing_column_step = clearing_column_step + 1


        clearing_regions_step = 0
        for column in range(0, 7, 3):

            for row in range(0, 7, 3):

                if clearing_regions_step in scoring_regions:
                    self.grid[column: column + 3, row: row + 3] = 0
                    # print(f"cleared region {clearing_regions_step}")
                clearing_regions_step = clearing_regions_step + 1

        # if any([scoring_columns, scoring_rows, scoring_regions]):
        #     print(f"Potential Score on col, row, reg{[scoring_columns, scoring_rows, scoring_regions]}")
        #     # print(f"with placement {self.placements[-1]}")
        #     pag.sleep(.5)


class BoardAnalyzer:


    def __init__(self, grid):
        self.grid = grid
        self.list_of_cells = self.make_cells_list()
        self.cell_values = self.get_cell_values()
        self.filled_cells = 0



    def report(self, verbose=False):

        regions_score = 0


        # # incentives for filling up regions
        # regions_score = max(self.is_region_over_n(4), 4)
        # regions_score = regions_score + max(self.is_region_over_n(5), 4)
        # regions_score = regions_score + max(self.is_region_over_n(6), 4)
        # regions_score = regions_score + max(self.is_region_over_n(7), 4)


        # highest_region only counts the top one
        highest_region = self.get_highest_region_count()
        # highest_region = 0


        # line totals
        best_row_score = self.get_highest_line_count(line="row")
        best_column_score = self.get_highest_line_count(line="column")


        # individual cells handled by other class.
        cell_values = sum(self.cell_values)


        # fill full regions first
        region_fullness_list = self.get_all_region_fill_counts()
        region_fullness_list.sort(reverse=True)
        # fill the top one first then second etc.
        region_fullness_list = self.fullness_list_stagger_weights(region_fullness_list)


        region_fullness_score = round(sum(region_fullness_list) / 5)


        # get fullness of rows and columns
        row_fullness_list = self.get_line_fullness_counts(line="row")
        row_fullness_list.sort(reverse=True)
        row_fullness_list = self.fullness_list_stagger_weights(row_fullness_list)

        row_fullness_score = round(sum(row_fullness_list) / 5)

        column_fullness_list = self.get_line_fullness_counts(line="column")
        column_fullness_list.sort(reverse=True)
        column_fullness_list = self.fullness_list_stagger_weights(column_fullness_list)

        column_fullness_score = round(sum(column_fullness_list) / 5)


        row_chunkiness = sum(self.check_line_chunkiness(line="row")) * -5
        column_chunkiness = sum(self.check_line_chunkiness(line="column")) * -5





        bad_stuff = cell_values + column_chunkiness + row_chunkiness
        good_stuff = best_column_score + best_row_score + highest_region + region_fullness_score + column_fullness_score +\
            row_fullness_score


        total_for_board = cell_values + best_column_score + best_row_score + \
                          highest_region + region_fullness_score + column_fullness_score + row_fullness_score + \
                          column_chunkiness + row_chunkiness


        if verbose:
            print(f"cell values = {cell_values}")
            # print(f"best col = {best_column_score}")
            # print(f"best row = {best_row_score}")
            # print(f"best region = {highest_region}")
            print(f"region fullness list sorted = \n{region_fullness_list}")
            print(f"region_fullness_score = {region_fullness_score}")
            print(f"row chunkiness = {row_chunkiness}")
            print(f"column chunkiness = {column_chunkiness}")
            print(f"Total Score for board = {total_for_board}")

        # higher is better
        return total_for_board, bad_stuff, good_stuff


    @staticmethod
    def fullness_list_stagger_weights(region_list):


        staggered_weight = 18
        for item in range(0, len(region_list)):
            region_list[item] = region_list[item] * staggered_weight
            staggered_weight = staggered_weight - 2

        return region_list

    def check_line_chunkiness(self, line=""):
        is_on = False

        chunks = [0, 0, 0, 0, 0, 0, 0, 0, 0]



        for row in range(0, 9):
            for column in range(0, 9):

                if line == "row":
                    block = self.grid[row, column]
                if line == "column":
                    block = self.grid[column, row]


                if block:
                    if not is_on:
                        is_on = True
                        chunks[row] = chunks[row] + 1
                if not block:
                    if is_on:
                        is_on = False

        return chunks




    def get_cell_values(self):

        values_of_cells = []
        self.filled_cells = 0


        for cell in self.list_of_cells:
            value = CellAnalyzer(self.grid, cell).tally_score()
            values_of_cells.append(value)
            if CellAnalyzer(self.grid, cell).is_filled:
                self.filled_cells = self.filled_cells + 1




        return values_of_cells

    @staticmethod
    def make_cells_list():

        list_of_cells = []

        for column in range(0, 9):
            pass
            for row in range(0, 9):
                list_of_cells.append((row, column))

        return list_of_cells


    def is_region_over_n(self, n):

        regions_over_n = 0

        for column in range(0, 7, 3):
            for row in range(0, 7, 3):
                region = self.grid[column: column + 3, row: row + 3]
                all_cells = np.sum(region)
                if all_cells > n:
                    regions_over_n = regions_over_n + 1

        return regions_over_n

    def get_highest_region_count(self):

        highest_region_count = 0

        for column in range(0, 7, 3):
            for row in range(0, 7, 3):
                region = self.grid[column: column + 3, row: row + 3]
                this_region_count = np.sum(region)
                if this_region_count > highest_region_count:
                    highest_region_count = this_region_count

        return highest_region_count


    def get_all_region_fill_counts(self):

        all_region_counts = []

        for column in range(0, 7, 3):
            for row in range(0, 7, 3):
                region = self.grid[column: column + 3, row: row + 3]
                this_region_count = np.sum(region)
                all_region_counts.append(this_region_count)

        return all_region_counts


    def get_line_fullness_counts(self, line=""):
        all_counts = []

        for count in range(0, 9):
            if line == "column":
                count = sum(self.grid[:, count])
            elif line == "row":
                count = sum(self.grid[count, :])
            else:
                raise ValueError("parameter line must equal row or column")
            all_counts.append(count)

        return all_counts




        pass












    def get_highest_line_count(self, line=""):

        highest_count = 0
        if line == "column":
            for column in range(0, 9):
                count = np.sum(self.grid[:, column])
                if count > highest_count:
                    highest_count = count
            return highest_count
        elif line == "row":
            for row in range(0, 9):
                count = np.sum(self.grid[row, :])
                if count > highest_count:
                    highest_count = count
            return highest_count
        else:
            raise ValueError


class CellAnalyzer:
    def __init__(self, grid, coordinate):
        self.grid = grid
        self.coordinate = coordinate
        self.is_filled = self.check_if_filled()

        self.score = self.tally_score()

    def tally_score(self):

        score = self.is_filled * -30 + self.is_surrounded_by_filled() * -10



        return score


    def check_if_filled(self):
        filled_state = self.grid[self.coordinate[0], self.coordinate[1]]
        return filled_state


    def cell_above_address(self):

        neighbor_y = self.coordinate[0] - 1
        neighbor_x = self.coordinate[1]
        if neighbor_y < 0:
            return False
        return neighbor_y, neighbor_x

    def cell_below_address(self):
        neighbor_y = self.coordinate[0] + 1
        neighbor_x = self.coordinate[1]
        if neighbor_y > 8:
            return False
        return neighbor_y, neighbor_x

    def cell_east_address(self):
        neighbor_y = self.coordinate[0]
        neighbor_x = self.coordinate[1] + 1
        if neighbor_y > 8:
            return False
        return neighbor_y, neighbor_x

    def cell_west_address(self):
        neighbor_y = self.coordinate[0]
        neighbor_x = self.coordinate[1] - 1
        if neighbor_y < 0:
            return False
        return neighbor_y, neighbor_x





    def is_surrounded_by_filled(self):

        def skip():
            pass

        if self.is_filled:
            return False

        funcs = [self.cell_above_address, self.cell_below_address, self.cell_east_address, self.cell_west_address]

        if self.coordinate[0] == 0:
            # top row
            funcs[0] = skip
            # funcs = [self.cell_below_address, self.cell_east_address, self.cell_west_address]

        if self.coordinate[0] == 8:
            # bot row
            funcs[1] = skip
            # funcs = [self.cell_above_address, self.cell_east_address, self.cell_west_address]

        if self.coordinate[1] == 8:
            # right side
            funcs[2] = skip
            # funcs = [self.cell_above_address, self.cell_below_address, self.cell_west_address]


        if self.coordinate[1] == 0:
            # left side
            funcs[3] = skip
            # funcs = [self.cell_above_address, self.cell_below_address, self.cell_east_address]





        for func in funcs:
            if func == skip:
                continue
            try:
                if not self.grid[func()]:
                    return False
            except IndexError:
                continue
        return True


class ScreenLocator:

    def __init__(self, use_defaults=True, write_defaults=False):
        self.coordinate_zero = [0, 0]  # This will be a location on the screen.
        self.scale = 1.0
        standard_distance = 467  # this will be width of the grid when the original photos are taken
        self.distance = standard_distance

        if use_defaults:
            self.read_defaults_from_file()
        else:
            self.determine_scale()
        if write_defaults:
            self.write_new_default()

        self.region_zero = self.get_region_zero()
        self.region_one = self.get_region_one()
        self.region_two = self.get_region_two()

        self.region = [self.region_zero, self.region_one, self.region_two]












    def determine_scale(self, standard_scale=465):
        scale = standard_scale

        left_corner, right_corner = self.get_get_the_corners()


        print("Corners Found:")
        print(f"\tleft = {left_corner} \n\tright = {right_corner}")

        width = right_corner[0] - left_corner[0]

        scale = round(width / standard_scale, 2)
        print(f"\tWidth of grid is {width}")
        print(scale)
        self.scale = scale
        self.distance = int(round(standard_scale * scale))


        self.get_coordinate_zero_from_bottom_left_corner(left_corner)



    @staticmethod
    def get_get_the_corners():
        bottom_left_corner = None
        bottom_right_corner = None

        print("Getting Corners to determine location and scale")

        while not bottom_left_corner:
            input("\tPlace mouse on bottom left corner and press enter.")
            bottom_left_corner = pag.position()

        print(f"\tLeft Corner = {bottom_left_corner}")

        while not bottom_right_corner:
            input("\n \tPlace mouse on bottom right corner and press enter.")
            bottom_right_corner = pag.position()

        print(f"\tRight Corner = {bottom_right_corner}")

        return bottom_left_corner, bottom_right_corner

    def get_coordinate_zero_from_bottom_left_corner(self, bottom_left):

        coordinate_zero = (bottom_left[0], int(round(bottom_left[1] - self.distance)))

        print(f"coordinate_zero is {coordinate_zero}")
        self.coordinate_zero = coordinate_zero


    def read_defaults_from_file(self):
        defaults_file = "screen_defaults.txt"

        with open(defaults_file, "r") as file:
            lines = file.readlines()

        x_coord, y_coord = lines[1].split(",")
        x_coord = int(x_coord[1:])
        y_coord = int(y_coord[1:-2])

        self.coordinate_zero = (x_coord, y_coord)
        self.scale = float(lines[3])
        self.distance = int(lines[5])


        # print(self.coordinate_zero)
        # print(self.scale)
        # print(self.distance)


    def write_new_default(self):
        defaults_file = "screen_defaults.txt"

        with open(defaults_file, "w") as file:
            file.write("coordinate zero: \n")
            file.write(str(self.coordinate_zero))

            file.write("\nscale: \n")
            file.write(str(self.scale))

            file.write("\ndistance: \n")
            file.write(str(self.distance))

    def get_region_zero(self):
        top_left = [self.coordinate_zero[0], self.coordinate_zero[1] + self.distance]
        bottom_left = [top_left[0], top_left[1] + self.distance]


        top_left[1] = top_left[1] + round(self.distance * 0.2)
        bottom_left[1] = bottom_left[1] - round(self.distance * 0.3)


        top_right = top_left[0] + round(self.distance / 3), top_left[1]
        bottom_right = bottom_left[0] + round(self.distance / 3), bottom_left[1]




        # pag.moveTo(top_left, duration=.5)
        # pag.moveTo(bottom_left, duration=.5)
        # pag.moveTo(bottom_right, duration=.5)
        # pag.moveTo(top_right, duration=.5)

        return top_left, top_right, bottom_right, bottom_left


    def get_region_one(self):
        region_one = []

        for corner in self.region_zero:
            new_corner = corner[0] + round(self.distance/3), corner[1]
            region_one.append(new_corner)


        return region_one


    def get_region_two(self):
        region_two = []

        for corner in self.region_zero:
            new_corner = corner[0] + round(self.distance / 3) * 2, corner[1]
            region_two.append(new_corner)

        return region_two


class PieceIdentifier:
    def __init__(self, screen, slot):
        self.slot = slot
        self.scale = screen.scale
        self.region = screen.region[slot]
        self.image_folder = "./block_images"
        self.info = self.get_info()

    def get_info(self):
        name = self.identify_from_image()
        info = PieceInfo(name, self.slot)
        # print(info.piece_grid)
        return info

    def region_to_point_w_h(self):
        x = self.region[0][0]
        y = self.region[0][1]
        w = self.region[1][0] - self.region[0][0]
        h = self.region[2][1] - self.region[0][1]

        return x, y, w, h

    def get_image(self):

        area = self.region_to_point_w_h()

        im = pag.screenshot("area.jpg", region=area)

        return im

    @staticmethod
    def save_to_folder(image, name):

        image.save(f"block_images/{name}.png")

    def identify_from_image(self):
        needle = self.get_image()
        block_name = None



        for root, dire, file in os.walk(top=self.image_folder):
            for name in file:
                haystack = f"{self.image_folder}/{name}"
                image = cv2.imread(haystack)

                try:
                    found = pag.locate(needle, image, confidence=0.89)
                    if found:
                        block_name = name
                except ValueError as e:
                    print(e)

                    # needle.show()
                    # cv2.imshow("name", image)
                    # cv2.waitKey()
                    # cv2.destroyAllWindows()

                if block_name == "length_five_horz.png" or block_name == "length_four_horz.png" or block_name == "bottom_bottom_left_medium_L.png":
                    block_name = None
                    # print("\t\t\tRETRYING TO IDENTIFY THIS BLOCK")
                    try:
                        found = pag.locate(needle, image, confidence=.98)
                        if found:
                            block_name = name
                    except ValueError as e:
                        print(e)

                if found:
                    a = 0
                    break


        if not block_name:
            # needle.show()
            print("--Failed to find a real block!-- raising value error")
            raise ValueError




        block_name = block_name.split(".")[0]
        return block_name


class PieceMover:
    def __init__(self, screen, piece, slot):
        self.distance = screen.distance
        self.coordinate_zero = screen.coordinate_zero
        self.slot = slot
        self.slots = self.set_slots()
        self.info = piece.info


    def set_slots(self):

        slot_zero = [self.coordinate_zero[0] + round(self.distance/3 * .5),
                     round(self.coordinate_zero[1] + self.distance * 1.5)]

        slot_one = [slot_zero[0] + round(self.distance/3), slot_zero[1]]

        slot_two = [slot_one[0] + round(self.distance/3), slot_one[1]]

        return slot_zero, slot_one, slot_two


    def get_raw_cell_location(self, column, row):
        column_spot = self.coordinate_zero[0] + round(self.distance/9) * (column+1) - round(self.distance/18)
        row_spot = self.coordinate_zero[1] + round(self.distance/9) * (row+1) - round(self.distance/18)
        return column_spot, row_spot



    def adjust_for_piece_offset(self, x, y):

        # piece moves north of mouse
        y = y + round(self.distance/9) * 2


        # print("servo attempting to move offsetrs from info")
        # print(f"offset is ... x = {self.info.offset[0]} \n y = {self.info.offset[1]}")

        x = x + self.info.offset[0]
        y = y + self.info.offset[1]

        return x, y

    def place(self, column, row):
        x, y = self.get_raw_cell_location(column, row)
        x, y = self.adjust_for_piece_offset(x, y)
        # pick up from slot
        # drag to adjusted cell
        starting_slot = self.slots[self.slot]
        pag.moveTo(starting_slot[0], starting_slot[1])
        pag.sleep(.2)
        pag.mouseDown(button="left")
        pag.moveTo(x, y)
        pag.sleep(.3)
        pag.mouseUp(button="left")
        pag.sleep(.6)





    def dummy(self, column, row):
        x, y = self.get_raw_cell_location(column, row)
        x, y = self.adjust_for_piece_offset(x, y)
        # pick up from slot
        # drag to adjusted cell
        starting_slot = self.slots[self.slot]
        pag.mouseDown(button="left", x=starting_slot[0], y=starting_slot[1])
        pag.moveTo(x, y, duration=.1)


class PieceInfo:

    def __init__(self, name, slot):
        self.piece_name = name
        self.piece_grid = self.fill_piece_grid(self.info_dict[self.piece_name]["grid"])
        self.offset = [self.info_dict[self.piece_name]["x_offset"],
                       self.info_dict[self.piece_name]["y_offset"]]


    info_dict = \
        {"top_right_big_L":
            {"x_offset": 50,
             "y_offset": 90,
             "grid": ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2))
             },
         "point_down_big_T":
             {"x_offset": 50,
              "y_offset": 90,
              "grid": ((0, 0), (1, 0), (2, 0), (1, 1), (1, 2))
              },
         "S_right_lateral_vert":
             {"x_offset": 27,
              "y_offset": 85,
              "grid": ((0, 0), (0, 1), (1, 1), (1, 2))
              },
         "point_right_big_T":
             {"x_offset": 50,
              "y_offset": 90,
              "grid": ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1))
              },
         "bottom_right_little_L":
             {"x_offset": 25,
              "y_offset": 40,
              "grid": ((1, 0), (0, 1), (1, 1))
              },
         "point_up_big_T":
             {"x_offset": 50,
              "y_offset": 90,
              "grid": ((1, 0), (1, 1), (1, 2), (0, 2), (2, 2))
              },
         "length_five_vert":
             {"x_offset": 0,
              "y_offset": 190,
              "grid": ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4))
              },
         "top_right_little_L":
             {"x_offset": 25,
              "y_offset": 40,
              "grid": ((0, 0), (1, 0), (1, 1))
              },
         "bottom_right_big_L":
             {"x_offset": 50,
              "y_offset": 90,
              "grid": ((2, 0), (2, 1), (2, 2), (0, 2), (1, 2))
              },
         "length_four_vert":
             {"x_offset": 0,
              "y_offset": 140,
              "grid": ((0, 0), (0, 1), (0, 2), (0, 3))
              },
         "bottom_right_right_medium_L":
             {"x_offset": 25,
              "y_offset": 90,
              "grid": ((1, 0), (1, 1), (1, 2), (0, 2))
              },
         "length_three_horz":
             {"x_offset": 50,
              "y_offset": -10,
              "grid": ((0, 0), (1, 0), (2, 0))
              },
         "bottom_bottom_left_medium_L":
             {"x_offset": 50,
              "y_offset": 40,
              "grid": ((0, 0), (0, 1), (1, 1), (2, 1))
              },
         "length_three_vert":
             {"x_offset": 0,
              "y_offset": 90,
              "grid": ((0, 0), (0, 1), (0, 2))
              },
         "cross":
             {"x_offset": 50,
              "y_offset": 90,
              "grid": ((1, 0), (0, 1), (1, 1), (2, 1), (1, 2))
              },
         "length_one":
             {"x_offset": 0,
              "y_offset": -10,
              "grid": ((0, 0), (0, 0))
              },
         "length_two_vert":
             {"x_offset": 0,
              "y_offset": 40,
              "grid": ((0, 0), (0, 1))
              },
         "length_four_horz":
             {"x_offset": 77,
              "y_offset": -10,
              "grid": ((0, 0), (1, 0), (2, 0), (3, 0))
              },
         "length_two_horz":
             {"x_offset": 25,
              "y_offset": -10,
              "grid": ((0, 0), (1, 0))
              },
         "top_left_left_medium_L":
             {"x_offset": 25,
              "y_offset": 87,
              "grid": ((0, 0), (1, 0), (0, 1), (0, 2))
              },
         "point_down_little_T":
             {"x_offset": 50,
              "y_offset": 40,
              "grid": ((0, 0), (1, 0), (2, 0), (1, 1))
              },
         "square":
             {"x_offset": 25,
              "y_offset": 40,
              "grid": ((0, 0), (1, 0), (0, 1), (1, 1))
              },
         "bottom_left_big_L":
             {"x_offset": 50,
              "y_offset": 90,
              "grid": ((0, 0), (0, 1), (0, 2), (1, 2), (2, 2))
              },
         "length_five_horz":
             {"x_offset": 100,
              "y_offset": -10,
              "grid": ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0))
              },
         "S_left_lateral_vert":
             {"x_offset": 25,
              "y_offset": 83,
              "grid": ((1, 0), (0, 1), (1, 1), (0, 2))
              },
         "point_left_little_T":
             {"x_offset": 25,
              "y_offset": 90,
              "grid": ((1, 0), (0, 1), (1, 1), (1, 2))
              },
         "top_left_big_L":
             {"x_offset": 50,
              "y_offset": 90,
              "grid": ((0, 0), (1, 0), (2, 0), (0, 1), (0, 2))
              },
         "bottom_left_little_L":
             {"x_offset": 25,
              "y_offset": 40,
              "grid": ((0, 0), (0, 1), (1, 1))
              },
         "point_right_little_T":
             {"x_offset": 25,
              "y_offset": 90,
              "grid": ((0, 0), (0, 1), (1, 1), (0, 2))
              },
         "S_left_lateral_horz":
             {"x_offset": 50,
              "y_offset": 40,
              "grid": ((0, 0), (1, 0), (1, 1), (2, 1))
              },
         "top_left_little_L":
             {"x_offset": 25,
              "y_offset": 40,
              "grid": ((0, 0), (1, 0), (0, 1))
              },
         "S_right_lateral_horz":
             {"x_offset": 50,
              "y_offset": 41,
              "grid": ((1, 0), (2, 0), (0, 1), (1, 1))
              },
         "top_top_right_medium_L":
             {"x_offset": 50,
              "y_offset": 40,
              "grid": ((0, 0),(1, 0),(2, 0),(2, 1))
              },
         "point_left_big_T":
             {"x_offset": 50,
              "y_offset": 90,
              "grid": ((2, 0), (0, 1), (1, 1), (2, 1), (2, 2))
              },
         "point_up_little_T":
             {"x_offset": 50,
              "y_offset": 40,
              "grid": ((1, 0), (0, 1), (1, 1), (2, 1))
              },
         "diag_two_leans_left":
             {"x_offset": 25,
              "y_offset": 40,
              "grid": ((0, 0), (1, 1))
              },
         "U_down":
             {"x_offset": 50,
              "y_offset": 40,
              "grid": ((0, 0), (2, 0), (0, 1), (1, 1), (2, 1))
              },
         "U_right":
             {"x_offset": 25,
              "y_offset": 95,
              "grid": ((0, 0), (1, 0), (1, 1), (0, 2), (1, 2))
              },
         "U_left":
             {"x_offset": 25,
              "y_offset": 93,
              "grid": ((0, 0), (1, 0), (0, 1), (0, 2), (1, 2))
              },
         "diag_three_left_lean":
             {"x_offset": 53,
              "y_offset": 93,
              "grid": ((0, 0), (1, 1), (2, 2))
              },
         "U_up":
             {"x_offset": 50,
              "y_offset": 40,
              "grid": ((0, 0), (1, 0), (2, 0), (0, 1), (2, 1))
              },
         "diag_three_right_lean":
             {"x_offset": 53,
              "y_offset": 93,
              "grid": ((2, 0), (1, 1), (0, 2))
              },
         "diag_two_right_lean":
             {"x_offset": 25,
              "y_offset": 40,
              "grid": ((1, 0), (0, 1))
              },



         }


    def fill_piece_grid(self, cells):
        grid = np.zeros((5, 5))
        for cell in cells:
            grid[cell[1], cell[0]] = 1

        return grid


class OptimizationLogger:

    def __init__(self, time=0.0, turns=0, is_logged=True):
        self.run_time = round(time)
        self.run_turns = turns
        self.save_path_name = "logger.txt"
        if is_logged:
            self.append_file_with_run()


    def append_file_with_run(self):

        try:
            with open(self.save_path_name, mode="r") as file:
                run_count = file.readlines()

            current_run_count = len(run_count) + 1

        except FileNotFoundError:
            current_run_count = 1


        run = f"{self.run_turns} Turns. {self.run_time} Seconds. {current_run_count} Game Number."

        with open(self.save_path_name, mode="a") as file:
            file.write(run)
            file.write("\n")


    def get_averages_from_file(self):

        with open(self.save_path_name, mode="r") as file:
            all_runs = file.readlines()

        scores = []
        times = []
        for line in all_runs:
            number = line.split("Turns")[0]
            number = number.split(" ")[0]
            number = int(number)
            scores.append(number)

            number = line.split("Turns. ")[1]
            number = number.split(" Seconds")[0]
            number = int(number)
            times.append(number)

            average_score = sum(scores) / len(scores)
            average_time = sum(times) / len(times)

        return average_score, average_time, len(all_runs)

    def print_averages_to_console(self):
        score, time, run_count = self.get_averages_from_file()

        print(f"Average Turn Count: {round(score)}\nAverage Time:       {round(time)}\nOver {run_count} Runs.")


class TurnDecider:

    def __init__(self, screen, board):
        self.screen = screen
        self.board = BoardInstance(grid=board.grid.copy())
        self.eligible_slots = [0, 1, 2]
        self.best_placement_per_slot = [None, None, None]
        self.best_score_per_slot = [None, None, None]


    def find_best_score_for_slot(self, slot):

        try:
            piece = PieceIdentifier(self.screen, slot)
            # print(f"Checking valid spots for {piece.info.piece_name} in slot {slot}")
        except ValueError:
            print(f"----**---- Could not identify piece in slot {slot}, raising ValueError ----**----")
            raise ValueError

        self.board.try_every_spot(piece)
        self.board.analyse_all_instances()

        try:
            keep_grid = self.board.grid.copy()
            placement = self.board.get_best_placement_and_update()


            score = BoardAnalyzer(self.board.grid)
            score, _, _ = score.report()
            self.board.grid = keep_grid

        except IndexError:
            print(f"----**---- No valid placements for {piece.info.piece_name}. Setting score to None ----**----")
            score = None


        return score


    def fill_bests_for_all_eligible_slots(self):
        self.best_placement_per_slot = [None, None, None]
        self.best_score_per_slot = [None, None, None]

        for slot in self.eligible_slots:
            score = self.find_best_score_for_slot(slot)
            # self.best_placement_per_slot[slot] = placement
            self.best_score_per_slot[slot] = score


    def get_best_slot(self):
        print(f"Here are the best scores {self.best_score_per_slot}")

        for index in range(len(self.best_score_per_slot)):

            if not self.best_score_per_slot[index]:
                self.best_score_per_slot[index] = -999999999

        # todo this is flipped
        # sends index of the minimum score in score list which is later used as slot list index
        return self.best_score_per_slot.index(max(self.best_score_per_slot))


class RestartKicker:
    def __init__(self):
        self.game_over_style = ""
        self.new_game_redo_button = "new_game_needle_image.png"
        self.new_game_main_screen = "new_game_main_screen_needle_image.png"




    def find_game_over_style(self):
        print("finding game over style")
        button = pag.locateOnScreen("new_game_needle_image.png", confidence=.75)
        try:
            if button:
                print("game ended naturally, clicking new game")
                self.game_over_style = "redo"
                pag.click(pag.center(button))
                pag.sleep(4)
            else:
                print("hit a dead end piece, starting over manually.")
                self.game_over_style = "fresh"
                self.start_fresh_game()
        except pag.FailSafeException:
            print("game ended by user. press [`] to end.")
            keyboard.wait("`")



    def start_fresh_game(self):
        print("hitting esc to back out.")
        keyboard.send("esc")
        pag.sleep(1)
        print("finding the new game button")
        button = pag.locateOnScreen("new_game_main_screen_needle_image.png")

        if button:
            # fresh game, button is blue
            print(f"found blue button at {button}")
            pag.click(pag.center(button))
            print("ready to start!")
        else:
            # continue game button is blue, new is white.
            print(f"found white button at {button}")
            button = pag.locateOnScreen("new_game_needle_image.png")
            pag.click(pag.center(button))

            pag.sleep(1)
            button = pag.locateOnScreen("new_game_main_screen_needle_image.png", confidence=.8)
            pag.click(pag.center(button))
            print("ready to start!")





        # search for new_game button
        #     if found then click it and wait
        #     if not found then look for little arrow
        # click arrow - > click new game -> click confirm


        pass


class Plotter:
    def __init__(self):
        self.scores = []
        self.good = []
        self.bad = []


    def plot_scores(self, color="b"):
        plt.plot(self.scores, color)
        plt.pause(0.05)
        plt.draw()

    def plot_good(self, color="g"):
        plt.plot(self.good, color)
        plt.pause(0.05)
        plt.draw()

    def plot_bad(self, color="r"):
        plt.plot(self.bad, color)
        plt.pause(0.05)
        plt.draw()




def test_identify_images():
    screen = ScreenLocator(use_defaults=True)

    piece = PieceIdentifier(screen.region_zero, screen.scale)

    servo = PieceMover(screen.distance, screen.coordinate_zero)

    name = piece.identify_from_image()
    print(name)

    piece.region = screen.region_one
    name = piece.identify_from_image()
    print(name)

    piece.region = screen.region_two
    name = piece.identify_from_image()
    print(name)


def dummy_test_offsets():
    screen = ScreenLocator()

    # piece = PieceIdentifier(screen, 0)
    # servo = PieceMover(screen, piece, 0)
    # servo.dummy(0, 0)

    piece = PieceIdentifier(screen, 1)
    servo = PieceMover(screen, piece, 1)
    servo.dummy(0, 0)
    #
    # piece = PieceIdentifier(screen, 2)
    # servo = PieceMover(screen, piece, 2)
    # servo.dummy(0, 0)


def play_game():

    start_time = time.time()
    turn_count = 0


    screen = ScreenLocator(use_defaults=True)

    board = BoardInstance()


    try:
        while True:
            for slot in range(0, 3):

                turn_count = turn_count + 1
                print(f"\t NEXT TURN V number {turn_count}")

                try:
                    piece = PieceIdentifier(screen, slot)
                except ValueError:
                    print("----**---- Could not identify, raising ValueError ----**----")
                    raise ValueError



                board.try_every_spot(piece)
                board.analyse_all_instances()
                servo = PieceMover(screen, piece, slot)

                print(f"currently placing {piece.info.piece_name} as seen")

                try:
                    placement = board.get_best_placement_and_update()
                except IndexError:
                    print("----**---- No valid placements, raising ValueError. ----**----")
                    raise ValueError


                servo.place(placement[1], placement[0])

                # this prints the board analysis onto console
                # BoardAnalyzer(board.grid).report(verbose=True)
                # keyboard.read_key()


                print(f"\t END OF TURN ^ number {turn_count}")

            pag.sleep(.2)
    except ValueError:
        print("Game ended!")
    finally:
        end_time = time.time()
        print(f"full time was {end_time - start_time} seconds")
        print(f"Game ended in {turn_count} turns.")
        log = OptimizationLogger(time=(end_time - start_time), turns=turn_count)
        print(f"added to {log.save_path_name}")
        print("-------------------------------------------")
        print("Current Log Shows...")
        log.print_averages_to_console()



        print("-------------------------------------------")


def play_turner_game():
    """
    Plays the best slot first instead of left to right.
    """

    is_ended_for_good = False


    start_time = time.time()
    turn_count = 0

    screen = ScreenLocator(use_defaults=True)

    board = BoardInstance()

    turner = TurnDecider(screen, board)

    plt.clf()

    plotter = Plotter()
    plotter.plot_scores()
    plotter.plot_bad()
    plotter.plot_good()

    pag.sleep(.5)


    try:
        while True:

            while turner.eligible_slots:
                turn_count = turn_count + 1
                print(f"\t NEXT TURN V number {turn_count}, eligible spots = {turner.eligible_slots}")


                turner.fill_bests_for_all_eligible_slots()

                slot = turner.get_best_slot()

                piece = PieceIdentifier(screen, slot)
                servo = PieceMover(screen, piece, slot)

                print(f"currently placing {piece.info.piece_name} from slot {slot}.")




                board.try_every_spot(piece)
                board.analyse_all_instances()



                try:
                    placement = board.get_best_placement_and_update()
                    turner.board.grid = board.grid.copy()
                    # print(board.grid)
                except IndexError:
                    print("----**---- No valid placements, raising ValueError. ----**----")
                    raise ValueError

                servo.place(placement[1], placement[0])

                current_score, bad_score, good_score = BoardAnalyzer(board.grid).report(verbose=False)
                print(f"adding score to plot {current_score}")
                plotter.scores.append(current_score)
                plotter.bad.append(bad_score)
                plotter.good.append(good_score)


                turner.eligible_slots.remove(slot)

                # this prints the board analysis onto console
                # BoardAnalyzer(board.grid).report(verbose=True)
                # keyboard.read_key()

                print(f"\t END OF TURN ^ number {turn_count}")

                plotter.plot_scores("b")
                plotter.plot_good()
                plotter.plot_bad()

                if not turner.eligible_slots:
                    pag.sleep(.2)
                    turner.eligible_slots = [0, 1, 2]
                pag.sleep(.2)

    except ValueError:
        print("Game ended!")
    except pag.FailSafeException:
        print("Game Ended by using FailSafeException.")
        is_ended_for_good = True
    finally:
        end_time = time.time()
        print(f"full time was {end_time - start_time} seconds")
        print(f"Game ended in {turn_count} turns.")
        log = OptimizationLogger(time=(end_time - start_time), turns=turn_count)
        print(f"added to {log.save_path_name}")
        print("-------------------------------------------")
        print("Current Log Shows...")
        log.print_averages_to_console()
        plotter.plot_scores("r")

        if is_ended_for_good:
            print("game paused until [`] is pressed")
            plt.show()
            keyboard.wait("`")
        print("-------------------------------------------")


def click_new_game():



    print("Looking for New Game button...")
    pag.sleep(1)
    button = pag.locateOnScreen("new_game_needle_image.png", confidence=.75)



    if button:
        print("Clicking New Game button...")
        pag.click(pag.center(button))
        pag.sleep(3)
    else:
        print("Couldn't Find New Game button. Maybe already in a game.")


def start_endless_replay():
    restart = RestartKicker()

    time_log = OptimizationLogger(is_logged=False)
    time_log.save_path_name = "time_log.txt"


    game_start_time = time.time()



    while True:
        # play_game()

        play_turner_game()


        time_log.run_time = round(time.time() - game_start_time)
        time_log.append_file_with_run()


        pag.sleep(4)
        restart.find_game_over_style()
        pag.sleep(1)


def wait_for_keypress_to_start_game(key="`", wait_time=1):

        # todo add timer so i know for video timestamps of best score


        print(f"waiting on input in {wait_time} seconds")
        pag.sleep(wait_time)

        print(f"Press [{key}] to begin game.")
        press_key = keyboard.read_key()


        if press_key == key:
            print(f"{key} was pressed. So the game will start.")
            start_endless_replay()
        else:
            print(f"[{press_key}] was pressed but not [{key}] so program quit.")


if __name__ == "__main__":

    a = 1

    # dummy_test_offsets()

    wait_for_keypress_to_start_game(key="`", wait_time=.5)

    # logger = OptimizationLogger(is_logged=False)
    # logger.print_averages_to_console()

    b = 1