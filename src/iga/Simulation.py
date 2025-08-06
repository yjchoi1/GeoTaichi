import taichi as ti
import warnings

from src.utils.ObjectIO import DictIO
from src.utils.TypeDefination import vec3i, vec3f
from src.utils.linalg import scalar_sum, vector_max, vector_sum, inner_multiply, vector_subtract, type_convert

class Simulation(object):
    def __init__(self) -> None:
        self.dimension = 3
        self.domain = vec3f([0, 0, 0])
        self.boundary = vec3i([0, 0, 0])
        self.gravity = vec3f([0, 0, 0])
        self.monitor_type = []
        self.gauss_num = []
        self.coupling = False
        self.pbc = False

        self.dt = ti.field(float, shape=())
        self.delta = 0.
        self.current_time = 0.
        self.current_step = 0
        self.current_print = 0
        self.CurrentTime = ti.field(float, shape=())

        self.max_material_num = 0
        self.max_patch_num = 1
        self.max_point_num = []
        self.max_element_num = []
        self.max_degree_num = []
        self.max_knot_num = []
        self.gauss_number_per_element = 0
        self.max_influenced_node = 0
        self.max_total_knot_each_patch = 0
        self.max_total_knot = 0
        self.max_total_point_each_patch = 0
        self.max_total_point = 0
        self.max_total_element_each_patch = 0
        self.max_total_element = 0
        self.max_total_gauss_each_patch = 0
        self.max_total_gauss = 0

        self.nkinematic = 0
        self.ntraction = 0
        self.is_continue = True

        self.time = 0.
        self.CFL = 0.2
        self.isadaptive = False
        self.save_interval = 1e6
        self.path = None

        self.visualize_interval = 0.
        self.window_size = (1024, 1024)
        self.camera_up = (0.0, 1.0, 0.0)
        self.look_at = (0.0, 1.0, 0.0)
        self.look_from = (0.0, 0.0, 0.0)
        self.particle_color = (1, 1, 1)
        self.background_color = (0, 0, 0)
        self.point_light = (0, 0, 0)
        self.view_angle = 45.
        self.move_velocity = 0.

        self.displacement_tolerance = 1e-4
        self.residual_tolerance = 1e-15
        self.relative_residual_tolerance = 1e-6
        self.quasi_static = False
        self.max_iteration = 10000
        self.newmark_gamma = 0.5
        self.newmark_beta = 0.25
        self.iter_max = 100
        self.dof_multiplier = 2
        self.update = "Newmark"
        self.solver_type = "Explicit"

    def get_simulation_domain(self):
        return self.domain

    def set_domain(self, domain):
        self.domain = domain

    def set_boundary(self, boundary):
        BOUNDARY = {
                        'None': -1,
                        "Reflect": 0,
                        "Destroy": 1,
                        "Period": 2
                   }
        self.boundary = vec3i([DictIO.GetEssential(BOUNDARY, b) for b in boundary])
        if self.boundary[0] == 2 or self.boundary[1] == 2 or self.boundary[2] == 2:
            self.activate_period_boundary()

    def activate_period_boundary(self):
        self.pbc = True

    def set_gravity(self, gravity):
        self.gravity = gravity

    def set_iga_coupling(self, coupling):
        self.coupling = coupling

    def set_solver_type(self, solver_type):
        typelist = ["Explicit", "Implicit"]
        if not solver_type in typelist:
            raise RuntimeError(f"KeyWord:: /solver_type: {solver_type}/ is invalid. The valid type are given as follows: {typelist}")
        self.solver_type = solver_type

    def set_is_continue(self, is_continue):
        self.is_continue = is_continue

    def set_timestep(self, timestep):
        self.dt[None] = timestep
        self.delta = timestep

    def set_simulation_time(self, time):
        self.time = time

    def set_CFL(self, CFL):
        self.CFL = CFL

    def set_adaptive_timestep(self, isadaptive):
        self.isadaptive = isadaptive

    def set_save_interval(self, save_interval):
        self.save_interval = save_interval

    def set_visualize_interval(self, visualize_interval):
        self.visualize_interval = visualize_interval

    def set_window_size(self, window_size):
        self.window_size = window_size

    def set_save_path(self, path):
        self.path = path

    def set_material_num(self, material_num):
        if material_num <= 0:
            raise ValueError("Max Material number should be larger than 0!")
        self.max_material_num = int(material_num)

    def set_patch_num(self, patch_num):
        if patch_num <= 0:
            raise ValueError("Max Patch number should be larger than 0!")
        self.max_patch_num = int(patch_num)

    def set_point_num(self, point_num):
        if isinstance(point_num, (list, tuple)):
            for i in point_num:
                self.set_uvw_point_num(i)
        elif isinstance(point_num, int):
            for _ in range(3):
                self.set_uvw_point_num(point_num)
        else:
            raise ValueError("Keyword:: /max_point_num/ must be input as list, tuple or int")

    def set_uvw_point_num(self, point_num):
        if point_num <= 0:
            raise ValueError("Max point number should be larger than 0!")
        self.max_point_num.append(int(point_num))

    def set_gauss_num(self, gauss_num):
        if isinstance(gauss_num, (list, tuple)):
            for i in gauss_num:
                self.set_uvw_gauss_num(i)
        elif isinstance(gauss_num, int):
            for _ in range(3):
                self.set_uvw_gauss_num(gauss_num)
        else:
            raise ValueError("Keyword:: /max_gauss_num/ must be input as list, tuple or int")

    def set_uvw_gauss_num(self, gauss_num):
        if gauss_num <= 0:
            raise ValueError("Max gauss number should be larger than 0!")
        self.gauss_num.append(int(gauss_num))

    def set_degree_num(self, degree_num):
        if isinstance(degree_num, (list, tuple)):
            for i in degree_num:
                self.set_uvw_degree_num(i)
        elif isinstance(degree_num, int):
            for _ in range(3):
                self.set_uvw_degree_num(degree_num)
        else:
            raise ValueError("Keyword:: /max_degree_num/ must be input as list, tuple or int")
        
        if any(self.gauss_num) < any(scalar_sum(self.max_degree_num, 1)):
            self.gauss_num = vector_max(self.gauss_num, scalar_sum(self.max_degree_num, 1))
            warnings.warn(f"Gauss number is automatically set as {self.gauss_num}")
        self.max_knot_num = scalar_sum(vector_sum(self.max_point_num, self.max_degree_num), 1)
        self.max_knot_num = type_convert(self.max_knot_num, int)

    def set_uvw_degree_num(self, degree_num):
        if degree_num <= 0:
            raise ValueError("Max degree number should be larger than 0!")
        if degree_num > 5:
            raise ValueError("Max degree number should be smaller than 5!")
        self.max_degree_num.append(int(degree_num))

    def get_default_element_number(self):
        return vector_subtract(self.max_point_num, self.max_degree_num)

    def set_element_num(self, element_num):
        if isinstance(element_num, (list, tuple)):
            for i in element_num:
                self.set_uvw_element_num(i)
        elif isinstance(element_num, int):
            for _ in range(3):
                self.set_uvw_element_num(element_num)
        else:
            raise ValueError("Keyword:: /max_element_num/ must be input as list, tuple or int")

    def set_uvw_element_num(self, element_num):
        if element_num <= 0:
            raise ValueError("Max element number should be larger than 0!")
        self.max_element_num.append(int(element_num))

    def calculate_essential_parameters(self):
        self.gauss_number_per_element = inner_multiply(self.gauss_num)
        self.max_influenced_node = int(inner_multiply(scalar_sum(self.max_degree_num, 1)))
        self.max_total_knot_each_patch = int(inner_multiply(self.max_knot_num))
        self.max_total_knot = int(self.max_patch_num * inner_multiply(self.max_knot_num))
        self.max_total_point_each_patch = int(inner_multiply(self.max_point_num))
        self.max_total_point = int(self.max_patch_num * inner_multiply(self.max_point_num))
        self.max_total_element_each_patch = int(inner_multiply(self.max_element_num))
        self.max_total_element = int(self.max_patch_num * inner_multiply(self.max_element_num))
        self.max_total_gauss_each_patch = int(self.gauss_number_per_element * self.max_total_element_each_patch)
        self.max_total_gauss = int(self.max_patch_num * self.gauss_number_per_element * self.max_total_element_each_patch)

    def set_window_parameters(self, windows):
        self.visualize_interval = DictIO.GetAlternative(windows, "VisualizeInterval", self.save_interval)
        self.window_size = DictIO.GetAlternative(windows, "WindowSize", self.window_size)
        self.camera_up = DictIO.GetAlternative(windows, "CameraUp", self.camera_up)
        self.look_at = DictIO.GetAlternative(windows, "LookAt", self.look_at)
        self.look_from = DictIO.GetAlternative(windows, "LookFrom", (0.7*self.domain[0], -0.4*self.domain[1], 1.5*self.domain[2]))
        self.particle_color = DictIO.GetAlternative(windows, "ParticleColor", (1, 1, 1))
        self.background_color = DictIO.GetAlternative(windows, "BackgroundColor", (0, 0, 0))
        self.point_light = DictIO.GetAlternative(windows, "PointLight", (0.5*self.domain[0], 0.5*self.domain[1], 1.0*self.domain[2]))
        self.view_angle = DictIO.GetAlternative(windows, "ViewAngle", 45.)
        self.move_velocity = DictIO.GetAlternative(windows, "MoveVelocity", 0.01 * (self.domain[0] + self.domain[1] + self.domain[2]))

    def set_constraint_num(self, constraint):
        self.nkinematic = int(DictIO.GetAlternative(constraint, "max_kinematic_constraint", 0))
        self.ntraction = int(DictIO.GetAlternative(constraint, "max_traction_constraint", 0))

    def set_save_data(self, gauss_point, volume, control_point):
        if gauss_point: self.monitor_type.append('gauss_point')
        if volume: self.monitor_type.append('volume')
        if control_point: self.monitor_type.append('control_point')

    def update_critical_timestep(self, dt):
        print("The time step is corrected as:", dt, '\n')
        self.dt[None] = dt
        self.delta = dt

    def set_implicit_parameters(self, implicit_parameters):
        if self.solver_type != "Implicit":
            raise RuntimeError("KeyError:: /solver_type/ should be set as Implicit")
        
        self.update = DictIO.GetAlternative(implicit_parameters, "update_scheme", "Newmark")
        self.displacement_tolerance = DictIO.GetAlternative(implicit_parameters, "displacement_tolerance", self.displacement_tolerance)
        self.residual_tolerance = DictIO.GetAlternative(implicit_parameters, "residual_tolerance", self.residual_tolerance)
        self.relative_residual_tolerance = DictIO.GetAlternative(implicit_parameters, "relative_residual_tolerance", self.relative_residual_tolerance)
        self.quasi_static = DictIO.GetAlternative(implicit_parameters, "quasi_static", self.quasi_static)
        self.max_iteration = DictIO.GetAlternative(implicit_parameters, "max_iteration", self.max_iteration)
        self.newmark_gamma = DictIO.GetAlternative(implicit_parameters, "newmark_gamma", self.newmark_gamma)
        self.newmark_beta = DictIO.GetAlternative(implicit_parameters, "newmark_beta", self.newmark_beta)
        self.iter_max = DictIO.GetAlternative(implicit_parameters, "max_iteration_number", self.iter_max)