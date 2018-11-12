import numpy
import random
import collections


class BeeClust:
    FREE_SPOT = 0
    BEE_NORTH = 1
    BEE_EAST = 2
    BEE_SOUTH = 3
    BEE_WEST = 4
    WALL = 5
    HEATER = 6
    COOLER = 7

    def __init__(self, map, p_changedir=0.2, p_wall=0.8, p_meet=0.8, k_temp=0.9,
                 k_stay=50, T_ideal=35, T_heater=40, T_cooler=5, T_env=22, min_wait=2):
        if not isinstance(map, numpy.core.multiarray.ndarray):
            raise TypeError('map is not a ndarray of integers')
        self._map = numpy.array(map)
        if (map > self.COOLER).any() or map.ndim != 2:
            raise ValueError('map values can be maximally 7 and dim = 2')
        if not isinstance(p_changedir, float) and not isinstance(p_changedir, int):
            raise TypeError('p_changedir is not float')
        if float(p_changedir) < 0 or float(p_changedir) > 1:
            raise ValueError(f'ValueError: {p_changedir} probability p_changedir must be between 0 and 1 and positive')
        self._p_changedir = float(p_changedir)
        if not isinstance(p_wall, float) and not isinstance(p_wall, int):
            raise TypeError('p_wall is not float')
        if float(p_wall) < 0 or float(p_wall) > 1:
            raise ValueError(f'ValueError: {p_wall} probability p_wall must be between 0 and 1 and positive')
        self._p_wall = float(p_wall)
        if not isinstance(p_meet, float) and not isinstance(p_meet, int):
            raise TypeError('p_meet is not float')
        if float(p_meet) < 0 or float(p_meet) > 1:
            raise ValueError(f'ValueError: {p_meet} probability p_meet must be between 0 and 1 and positive')
        self._p_meet = float(p_meet)
        if not isinstance(k_temp, float):
            raise TypeError('k_temp is not float')
        if float(k_temp) < 0:
            raise ValueError(f'ValueError: {k_temp} k_temp must be positive')
        self._k_temp = float(k_temp)
        if not isinstance(k_stay, int):
            raise TypeError('k_stay is not int')
        if float(k_stay) < 0:
            raise ValueError(f'ValueError: {k_stay} k_stay must be positive')
        self._k_stay = int(k_stay)
        if not isinstance(T_ideal, int):
            raise TypeError('T_ideal is not int')
        self._t_ideal = int(T_ideal)
        if not isinstance(T_heater, int):
            raise TypeError('T_heater is not int')
        self._t_heater = int(T_heater)
        if not isinstance(T_cooler, int):
            raise TypeError('T_cooler is not int')
        self._t_cooler = int(T_cooler)
        if not isinstance(T_env, int):
            raise TypeError('T_env is not int')
        self._t_env = int(T_env)
        if int(T_heater) < int(T_env) or int(T_cooler) > int(T_env):
            raise ValueError('T_heater < T or T_cooler > T_env')
        if not isinstance(min_wait, int):
            raise TypeError('min_wait is not int')
        if int(min_wait) < 0:
            raise ValueError(f'ValueError: {min_wait} min_wait must be positive')
        self._min_wait = int(min_wait)
        self._heatmap = self._count_heatmap()

    def _get_probability(self, probability):
        return random.random() < probability

    def tick(self):
        return self._move_bees()

    def _bee_direction(self, actual_direction):
        return numpy.random.choice(numpy.random.choice([[x for x in ({self.BEE_EAST, self.BEE_NORTH, self.BEE_SOUTH,
                                                                      self.BEE_WEST} - {actual_direction})],
                                                        [actual_direction]], p=[self._p_changedir,
                                                                                1 - self._p_changedir]))

    def _move_bees(self):
        moved = 0
        for bee_pos in self.bees:
            moved += self._move_bee(bee_pos, self._bee_direction(self._map[bee_pos]))
        return moved

    def _wait(self, bee_pos):
        return -max(self._min_wait, int(self._k_stay / (1 + abs(self._t_ideal - self._heatmap[bee_pos]))))

    def _get_new_pos(self, actual_pos, direction):
        if direction == self.BEE_SOUTH:
            if actual_pos[0] < len(self._map) - 1:
                return actual_pos[0] + 1, actual_pos[1]
            return actual_pos
        if direction == self.BEE_WEST:
            if actual_pos[1] > 0:
                return actual_pos[0], actual_pos[1] - 1
            return actual_pos
        if direction == self.BEE_EAST:
            if actual_pos[1] < len(self._map[0]) - 1:
                return actual_pos[0], actual_pos[1] + 1
            return actual_pos
        if actual_pos[0] > 0:
            return actual_pos[0] - 1, actual_pos[1]
        return actual_pos

    def _move_bee(self, actual_bee_position, _bee_direction):
        moved = 0
        if _bee_direction == self.BEE_WEST:
            if actual_bee_position[1] == 0 or self._map[actual_bee_position[0], actual_bee_position[1] - 1] in \
                    [self.WALL, self.HEATER, self.COOLER]:
                self._map[actual_bee_position] = self._change_direction(_bee_direction) if not \
                    self._get_probability(self._p_wall) else self._wait(actual_bee_position)
            elif self._map[actual_bee_position[0], actual_bee_position[1] - 1] in \
                [self.BEE_WEST, self.BEE_NORTH, self.BEE_SOUTH, self.BEE_EAST] or \
                    self._map[actual_bee_position[0], actual_bee_position[1] - 1] < 0:
                if self._get_probability(self._p_meet):
                    self._map[actual_bee_position] = self._wait(actual_bee_position)
            else:
                new_dir = self._bee_direction(_bee_direction)
                self._map[actual_bee_position] = 0
                self._map[self._get_new_pos(actual_bee_position, _bee_direction)] = new_dir
                moved = 1
        elif _bee_direction == self.BEE_EAST:
            if actual_bee_position[1] >= len(self._map[0]) - 1 or self._map[actual_bee_position[0],
                                                                            actual_bee_position[1] + 1] in \
                    [self.WALL, self.HEATER, self.COOLER]:
                self._map[actual_bee_position] = self._change_direction(_bee_direction) if not \
                    self._get_probability(self._p_wall) else self._wait(actual_bee_position)
            elif self._map[actual_bee_position[0], actual_bee_position[1] + 1] in \
                [self.BEE_WEST, self.BEE_NORTH, self.BEE_SOUTH, self.BEE_EAST] or \
                    self._map[actual_bee_position[0], actual_bee_position[1] + 1] < 0:
                if self._get_probability(self._p_meet):
                    self._map[actual_bee_position] = self._wait(actual_bee_position)
            else:
                new_dir = self._bee_direction(_bee_direction)
                self._map[actual_bee_position] = 0
                self._map[self._get_new_pos(actual_bee_position, _bee_direction)] = new_dir
                moved = 1

        elif _bee_direction == self.BEE_NORTH:
            if actual_bee_position[0] == 0 or self._map[actual_bee_position[0] - 1, actual_bee_position[1]] in \
                    [self.WALL, self.HEATER, self.COOLER]:
                self._map[actual_bee_position] = self._change_direction(_bee_direction) if not \
                    self._get_probability(self._p_wall) else self._wait(actual_bee_position)
            elif self._map[actual_bee_position[0] - 1, actual_bee_position[1]] in \
                [self.BEE_WEST, self.BEE_NORTH, self.BEE_SOUTH, self.BEE_EAST] or \
                    self._map[actual_bee_position[0] - 1, actual_bee_position[1]] < 0:
                if self._get_probability(self._p_meet):
                    self._map[actual_bee_position] = self._wait(actual_bee_position)
            else:
                new_dir = self._bee_direction(_bee_direction)
                self._map[actual_bee_position] = 0
                self._map[self._get_new_pos(actual_bee_position, _bee_direction)] = new_dir
                moved = 1

        elif _bee_direction == self.BEE_SOUTH:
            if actual_bee_position[0] >= len(self._map) - 1 or self._map[actual_bee_position[0] + 1,
                                                                         actual_bee_position[1]] in\
                    [self.WALL, self.HEATER, self.COOLER]:
                self._map[actual_bee_position] = self._change_direction(_bee_direction) if not \
                    self._get_probability(self._p_wall) else self._wait(actual_bee_position)
            elif self._map[actual_bee_position[0] + 1, actual_bee_position[1]] in \
                [self.BEE_WEST, self.BEE_NORTH, self.BEE_SOUTH, self.BEE_EAST] or \
                    self._map[actual_bee_position[0] + 1, actual_bee_position[1]] < 0:
                if self._get_probability(self._p_meet):
                    self._map[actual_bee_position] = self._wait(actual_bee_position)
            else:
                new_dir = self._bee_direction(_bee_direction)
                self._map[actual_bee_position] = 0
                self._map[self._get_new_pos(actual_bee_position, _bee_direction)] = new_dir
                moved = 1
        elif self._map[actual_bee_position] == -1:
            self._map[actual_bee_position] = numpy.random.choice([self.BEE_EAST, self.BEE_NORTH,
                                                                  self.BEE_SOUTH, self.BEE_WEST])
        elif self._map[actual_bee_position] < -1:
            self._map[actual_bee_position] += 1
        return moved

    def _change_direction(self, direction):
        d = {
            self.BEE_EAST: self.BEE_WEST,
            self.BEE_NORTH: self.BEE_SOUTH,
            self.BEE_WEST: self.BEE_EAST,
            self.BEE_SOUTH: self.BEE_NORTH
        }
        return d[direction]

    def _count_heatmap(self):
        ret = numpy.where(self._map == self.HEATER, float(self._t_heater), 0) + numpy.where(self._map == self.COOLER,
                                                                                            float(self._t_cooler), 0) +\
            numpy.where(self._map == self.WALL, numpy.NaN, 0) + numpy.where((self._map != self.WALL) & (self._map !=
                                                                                                        self.COOLER) &
                                                                            (self._map != self.HEATER), self._t_env, 0)
        if ((self._map == self.HEATER) | (self._map == self.COOLER)).any():
            for i in [self.HEATER, self.COOLER]:
                for pos in list(zip(*numpy.where(self._map == i))):
                    counted = {pos}
                    self._count_heat((pos[0] - 1, pos[1] - 1), ret, counted) # upleft
                    self._count_heat((pos[0] - 1, pos[1]), ret, counted)  # up
                    self._count_heat((pos[0] - 1, pos[1] + 1), ret, counted)  # upright
                    self._count_heat((pos[0], pos[1] + 1), ret, counted)  # right
                    self._count_heat((pos[0] + 1, pos[1] + 1), ret, counted)  # downright
                    self._count_heat((pos[0] + 1, pos[1]), ret, counted)  # down
                    self._count_heat((pos[0] + 1, pos[1] - 1), ret, counted)  # downleft
                    self._count_heat((pos[0], pos[1] - 1), ret, counted)  # left
        return ret

    def _count_heat(self, pos, ret, counted):
        if pos[0] < 0 or pos[0] >= len(self._map):
            return
        if pos[1] < 0 or pos[1] >= len(self._map[pos[0]]):
            return
        obstacles = [self.WALL, self.HEATER, self.COOLER]
        if self._map[pos] in obstacles:
            return
        if pos in counted:
            return
        dist_heater = self._find_closest(self.HEATER, pos) if (self._map == self.HEATER).any() else 0
        dist_cooler = self._find_closest(self.COOLER, pos) if (self._map == self.COOLER).any() else 0
        heating = (1 / dist_heater) * (self._t_heater - self._t_env) if dist_heater else 0
        cooling = (1 / dist_cooler) * (self._t_env - self._t_cooler) if dist_cooler else 0
        ret[pos] = self._t_env + self._k_temp * (max(heating, 0) - max(cooling, 0))
        counted.add(pos)
        self._count_heat((pos[0] - 1, pos[1] - 1), ret, counted)  # upleft
        self._count_heat((pos[0] - 1, pos[1]), ret, counted)  # up
        self._count_heat((pos[0] - 1, pos[1] + 1), ret, counted)  # upright
        self._count_heat((pos[0], pos[1] + 1), ret, counted)  # right
        self._count_heat((pos[0] + 1, pos[1] + 1), ret, counted)  # downright
        self._count_heat((pos[0] + 1, pos[1]), ret, counted)  # down
        self._count_heat((pos[0] + 1, pos[1] - 1), ret, counted)  # downleft
        self._count_heat((pos[0], pos[1] - 1), ret, counted)  # left

    def _find_closest(self, end_point, start_position):
        q = collections.deque([[start_position]])
        explored = {start_position}
        obstacles = [self.WALL, self.HEATER, self.COOLER]
        obstacles.remove(end_point)
        while q:
            p = q.popleft()
            x, y = p[-1]
            if self._map[x, y] == end_point:
                return len(p) - 1
            for pos_x, pos_y in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1), (x - 1, y - 1), (x - 1, y + 1),
                                 (x + 1, y - 1), (x + 1, y + 1)):
                if 0 <= pos_x < len(self._map) and 0 <= pos_y < len(self._map[0]) and \
                        self._map[pos_x, pos_y] not in obstacles and (pos_x, pos_y) not in explored:
                    q.append(p + [(pos_x, pos_y)])
                    explored.add((pos_x, pos_y))

    def _get_swarms(self, pos, added):
        if pos in added:
            return []
        if pos[0] < 0 or pos[0] >= len(self._map):
            return []
        if pos[1] < 0 or pos[1] >= len(self._map[pos[0]]):
            return []
        if self._map[pos] >= 0 and self._map[pos] not in [self.BEE_EAST, self.BEE_SOUTH, self.BEE_NORTH, self.BEE_WEST]:
            return []
        added.add(pos)
        return [pos] + self._get_swarms((pos[0] + 1, pos[1]), added) + self._get_swarms((pos[0] - 1, pos[1]), added) + \
               self._get_swarms((pos[0], pos[1] - 1), added) + self._get_swarms((pos[0], pos[1] + 1), added)

    @property
    def bees(self):
        return list(zip(*numpy.where((self._map < 0) | ((self._map >= self.BEE_NORTH) & (self._map <= self.BEE_WEST)))))

    @property
    def map(self):
        return self._map

    @property
    def heatmap(self):
        return self._heatmap

    @property
    def swarms(self):
        added = set()
        ret = []
        for pos in self.bees:
            res = self._get_swarms(pos, added)
            if res:
                ret.append(res)
        return ret

    @property
    def score(self):
        mean = 0.0
        bees = self.bees
        for bee_pos in bees:
            mean += self._heatmap[bee_pos]
        return mean / len(bees)

    def forget(self):
        self._map = numpy.where((self._map <= -1) | ((self._map >= 1) & (self._map <= 4)), -1, self._map)

    def recalculate_heat(self):
        self._heatmap = self._count_heatmap()
