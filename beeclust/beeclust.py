import numpy
from . import beeclust_cython


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
        self._map = map.astype(numpy.int8)
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
        self.recalculate_heat()

    def tick(self):
        return beeclust_cython.tick(self.map, self.heatmap, self._p_changedir, self._p_wall, self._p_meet,
                                    self._min_wait, self._k_stay, self._t_ideal)

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
        return beeclust_cython.swarms(self._map)

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
        heatmap = numpy.full(self.map.shape, self._t_env, dtype='float64')
        self._heatmap = beeclust_cython.recalculate_heat(heatmap, self._map, self._t_heater, self._t_cooler,
                                                         self._t_env, self._k_temp)
