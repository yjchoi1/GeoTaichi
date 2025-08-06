import numpy as np
import warnings
import random

from src.nurbs.BasicVolume import PrimitiveVolume


class NurbsVolume(object):
    def __init__(self) -> None:
        self.dim = 3
        self.precision = 18
        self.clamped = True
        self._delta = 1e-8

        self.degree = [0., 0., 0.]
        self.numctrlpts = [0., 0., 0.]
        self.gauss_number = [2, 2, 2]

        self.knotvector_u = []
        self.knotvector_v = []
        self.knotvector_w = []
        self.unique_knotvector_u = []
        self.unique_knotvector_v = []
        self.unique_knotvector_w = []
        self.element_num = []
        self.knot_num = []

        self.ctrlpts = []
        self.weights = []

    def get_order_u(self):
        return self.degree[0] + 1
    
    def get_order_v(self):
        return self.degree[1] + 1
    
    def get_order_w(self):
        return self.degree[2] + 1
    
    def create(self, degree, node_number, gauss_number, shape: PrimitiveVolume):
        if isinstance(degree, (list, np.ndarray, tuple)):
            self.degree = list(degree)
        else:
            raise TypeError("The /degree/ type should be a list, numpy.array or tuple")
        
        if isinstance(node_number, (list, np.ndarray, tuple)):
            self.numctrlpts = list(node_number)
        else:
            raise TypeError("The /node_number/ type should be a list, numpy.array or tuple")

        if self.numctrlpts[0] < 1:
            raise ValueError("Divisions in the x-direction cannot be less than 1")

        if self.numctrlpts[1] < 1:
            raise ValueError("Divisions in the y-direction cannot be less than 1")
        
        if self.numctrlpts[2] < 1:
            raise ValueError("Divisions in the z-direction cannot be less than 1")

        if not isinstance(self.numctrlpts[0], int):
            self.numctrlpts[0] = int(self.numctrlpts[0])
            warnings.warn("%d will be used as the value of num_u" % self.numctrlpts[0], UserWarning)

        if not isinstance(self.numctrlpts[1], int):
            self.numctrlpts[1] = int(self.numctrlpts[1])
            warnings.warn("%d will be used as the value of num_v" % self.numctrlpts[1], UserWarning)

        if not isinstance(self.numctrlpts[2], int):
            self.numctrlpts[2] = int(self.numctrlpts[2])
            warnings.warn("%d will be used as the value of num_v" % self.numctrlpts[2], UserWarning)

        if isinstance(gauss_number, int):
            self.gauss_number = [gauss_number, gauss_number, gauss_number]
        elif isinstance(gauss_number, (np.ndarray, list, tuple)):
            if len(list(gauss_number)) != 3:
                raise RuntimeError("The dimension of GaussNumber must be 3")
            if not (gauss_number[0] == gauss_number[1] and gauss_number[0] == gauss_number[2]):
                raise RuntimeError("The number of gauss point must equal along u-, v-, w-axes in current version")
            self.gauss_number = list(gauss_number)

        self.degree, self.numctrlpts = shape.check_parameters(self.degree, self.numctrlpts)

    def reset_ctrlpts(self):
        if self.ctrlpts:
            self.ctrlpts[:] = []
            self.numctrlpts[0] = 0
            self.numctrlpts[1] = 0
            self.numctrlpts[2] = 0

    def generate_ctrlpts(self, shape: PrimitiveVolume):
        self.reset_ctrlpts()
        self.set_ctrlpts(shape.generate_ctrlpts(self.numctrlpts))

    def generate_weights(self, shape: PrimitiveVolume):
        if not self.ctrlpts:
            raise ValueError("Generate the grid first")
        self.set_weights(shape.generate_weights(self.numctrlpts))

    def set_ctrlpts(self, ctrlpts, numctrlpts=None):
        self.ctrlpts = ctrlpts
        new_numctrlpts = self.numctrlpts if numctrlpts is None else numctrlpts
        
        if len(ctrlpts[0][0]) != new_numctrlpts[0] or len(ctrlpts[0]) != new_numctrlpts[1] or len(ctrlpts) != new_numctrlpts[2]:
            raise ValueError(f"The number of control point {[len(ctrlpts[0][0]), len(ctrlpts[0]), len(ctrlpts)]} is not consistent with the specified {new_numctrlpts}")
        self.numctrlpts = [len(ctrlpts[0][0]), len(ctrlpts[0]), len(ctrlpts)]

    def set_weights(self, weights):
        self.weights = weights
        if len(weights[0][0]) != self.numctrlpts[0] or len(weights[0]) != self.numctrlpts[1] or len(weights) != self.numctrlpts[2]:
            raise RuntimeError(f"The dimension of weights {[len(weights[0][0]), len(weights[0]), len(weights)]} is not consistent with control point {self.numctrlpts}")

    def read_ctrlpts(self, ctrlpts):
        if isinstance(ctrlpts, (list, tuple)):
            if len(ctrlpts[0][0]) != self.numctrlpts[0] or len(ctrlpts[0]) != self.numctrlpts[1] or len(ctrlpts) != self.numctrlpts[2]:
                raise ValueError("Input must be the same size with the grid points")
            self.set_ctrlpts([[[float(point) for point in line] for line in facet] for facet in ctrlpts])
        else:
            raise TypeError("The input should be a list, tuple")

    def read_weights(self, value):
        if not self.ctrlpts:
            raise ValueError("Generate the grid first")
        if isinstance(value, (int, float)):
            if value <= 0:
                raise ValueError("Weight value must be bigger than 0")
            self.set_weights([[[float(value) for _ in range(self.numctrlpts[0])] for _ in range(self.numctrlpts[1])] for _ in range(self.numctrlpts[2])])
        elif isinstance(value, (list, tuple)):
            if len(value[0][0]) != self.numctrlpts[0] or len(value[0]) != self.numctrlpts[1] or len(value) != self.numctrlpts[2]:
                raise ValueError("Input must be the same size with the grid points")
            if all(val <= 0 for val in value):
                raise ValueError("Weight values must be bigger than 0")
            self.set_weights([[[float(point) for point in line] for line in facet] for facet in value])
        else:
            raise TypeError("The input should be a list, tuple or a single int, float value")

    def generate_bumps(self, num_bumps, bump_height=5., base_extent=2, base_adjust=0, max_trials=25):
        if not self.ctrlpts:
            raise RuntimeError("Grid must be generated before calling this function")

        if not isinstance(num_bumps, int):
            num_bumps = int(num_bumps)
            warnings.warn("Number of bumps must be an integer value. Automatically rounding to %d" % num_bumps, UserWarning)

        if isinstance(bump_height, (list, tuple)):
            if len(bump_height) != num_bumps:
                raise ValueError("Number of bump heights must be equal to number of bumps")
            else:
                bump_height_is_array = True
        else:
            bump_height_is_array = False
            bump_height = [float(bump_height)]

        if base_extent < 1:
            raise ValueError("Base size must be bigger than 1 grid point")

        if (2 * base_extent) + base_adjust > self.numctrlpts[0] or (2 * base_extent) + base_adjust > self.numctrlpts[1]:
            raise ValueError("The area of the base must be less than the area of the grid")

        bump_list = []

        len_v = len(self.ctrlpts[0])
        len_u = len(self.ctrlpts[0][0])

        max_trials = int(max_trials)

        for _ in range(0, num_bumps):
            trials = 0
            while trials < max_trials:
                u = random.randint(base_extent, (len_u - 1) - base_extent)
                v = random.randint(base_extent, (len_v - 1) - base_extent)
                temp = [u, v]
                if self.check_bump(bump_list, temp, base_extent, base_adjust):
                    bump_list.append(temp)
                    trials = max_trials + 1 
                    break
                else:
                    trials = trials + 1
            if trials == max_trials:
                raise RuntimeError("Cannot generate %d bumps with a base extent of %d on this grid. "
                                   "You need to generate a grid larger than %dx%d." % (num_bumps, base_extent, self.numctrlpts[0], self.numctrlpts[1]))

        idx = 0
        for u, v in bump_list:
            h_increment = bump_height[idx] / base_extent
            height = h_increment
            for j in range(base_extent - 1, -1, -1):
                self._create_bump(u, v, j, height)
                height += h_increment
            if bump_height_is_array:
                idx += 1

    def check_bump(self, uv_list, to_be_checked_uv, base_extent, padding):
        if not uv_list:
            return True

        for uv in uv_list:
            u = to_be_checked_uv[0]
            v = to_be_checked_uv[1]
            check_list = []
            for ur in range(-(base_extent + 1 + padding), base_extent + 2 + padding):
                for vr in range(-(base_extent + 1 + padding), base_extent + 2 + padding):
                    check_list.append([u + ur, v + vr])
            for check in check_list:
                if abs(uv[0] - check[0]) < self._delta and abs(uv[1] - check[1]) < self._delta:
                    return False
        return True

    def _create_bump(self, u, v, jump, height):
        start_u = u - jump
        stop_u = u + jump + 1
        start_v = v - jump
        stop_v = v + jump + 1

        for j in range(start_v, stop_v):
            for i in range(start_u, stop_u):
                    self.ctrlpts[-1][j][i][2] = height

    def generate_knots(self, shape: PrimitiveVolume):
        self.set_knot(shape.generate_knot_u(self.degree[0], self.numctrlpts[0]),
                      shape.generate_knot_v(self.degree[1], self.numctrlpts[1]),
                      shape.generate_knot_w(self.degree[2], self.numctrlpts[2]))

    def set_knot(self, knotvector_u, knotvector_v, knotvector_w):
        self.knotvector_u = knotvector_u
        self.knotvector_v = knotvector_v
        self.knotvector_w = knotvector_w
        self.unique_knotvector_u = list(np.unique(np.array(knotvector_u)))
        self.unique_knotvector_v = list(np.unique(np.array(knotvector_v)))
        self.unique_knotvector_w = list(np.unique(np.array(knotvector_w)))
        self.element_num = [len(self.unique_knotvector_u) - 1, len(self.unique_knotvector_v) - 1, len(self.unique_knotvector_w) - 1]
        self.knot_num = [len(self.knotvector_u), len(self.knotvector_v), len(self.knotvector_w)]

    def normalize_knot(self, knot_vector, decimals=18):
        try:
            if knot_vector is None or len(knot_vector) == 0:
                raise ValueError("Input knot vector cannot be empty")
        except TypeError as e:
            print("An error occurred: {}".format(e.args[-1]))
            raise TypeError("Knot vector must be a list or tuple")
        except Exception:
            raise

        first_knot = float(knot_vector[0])
        last_knot = float(knot_vector[-1])
        denominator = last_knot - first_knot
        knot_vector_out = [float(("{:." + str(decimals) + "f}").format((float(kv) - first_knot) / denominator)) for kv in knot_vector]
        return np.array(knot_vector_out)

    def check_knot(self, degree, num_ctrlpts, knot_vector):
        try:
            if knot_vector is None or len(knot_vector) == 0:
                raise ValueError("Input knot vector cannot be empty")
        except TypeError as e:
            print("An error occurred: {}".format(e.args[-1]))
            raise TypeError("Knot vector must be a list or tuple")
        except Exception:
            raise

        # Check the formula; m = p + n + 1
        if len(knot_vector) != degree + num_ctrlpts + 1:
            return False

        prev_knot = knot_vector[0]
        for knot in knot_vector:
            if prev_knot > knot:
                return False
            prev_knot = knot
        return True