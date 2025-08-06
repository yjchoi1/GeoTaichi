from taichi.lang.impl import current_cfg

from src.iga.engine.ExplicitEngine import ExplicitEngine
from src.iga.engine.ImplicitEngine import ImplicitEngine
from src.iga.generator.BodyGenerator import GenerateManager
from src.iga.Recorder import WriteFile
from src.iga.IGABase import Solver
from src.iga.Simulation import Simulation
from src.iga.SceneManager import myScene
from src.utils.ObjectIO import DictIO
from src.utils.TypeDefination import vec3f


class IGA(object):
    def __init__(self, title='A High Performance Multiscale and Multiphysics Simulator', log=True):  
        if log:
            print('# =================================================================== #')
            print('#', "".center(67), '#')
            print('#', "Welcome to GeoTaichi -- Isogeometric Analysis Engine !".center(67), '#')
            print('#', "".center(67), '#')
            print('#', title.center(67), '#')
            print('#', "".center(67), '#')
            print('# =================================================================== #', '\n')
        self.sims = Simulation()
        self.scene = myScene()
        self.generator = GenerateManager()
        self.enginer = None
        self.recorder = None
        self.solver = None

    def set_configuration(self, **kwargs):
        self.sims.set_domain(DictIO.GetEssential(kwargs, "domain"))
        self.sims.set_boundary(DictIO.GetAlternative(kwargs, "boundary", ['None', 'None', 'None']))
        self.sims.set_gravity(DictIO.GetAlternative(kwargs, "gravity", vec3f([0.,0.,-9.8])))
        self.sims.set_iga_coupling(DictIO.GetAlternative(kwargs, "coupling", False))
        self.sims.set_solver_type(DictIO.GetAlternative(kwargs, "solver_type", "Implicit"))

    def set_solver(self, solver):
        self.sims.set_timestep(DictIO.GetEssential(solver, "Timestep"))
        self.sims.set_simulation_time(DictIO.GetEssential(solver, "SimulationTime"))
        self.sims.set_CFL(DictIO.GetAlternative(solver, "CFL", 0.5))
        self.sims.set_adaptive_timestep(DictIO.GetAlternative(solver, "AdaptiveTimestep", False))
        self.sims.set_save_interval(DictIO.GetAlternative(solver, "SaveInterval", self.sims.time / 20.))
        self.sims.set_save_path(DictIO.GetAlternative(solver, "SavePath", 'OutputData'))

    def memory_allocate(self, memory, log=True):    
        self.sims.set_material_num(DictIO.GetAlternative(memory, "max_material_number", 0))
        self.sims.set_patch_num(DictIO.GetAlternative(memory, "max_patch_number", 1))
        self.sims.set_point_num(DictIO.GetAlternative(memory, "max_point_number", [0, 0, 0]))
        self.sims.set_degree_num(DictIO.GetAlternative(memory, "max_degree_number", [2, 2, 2]))
        self.sims.set_gauss_num(DictIO.GetAlternative(memory, "max_gauss_number", [3, 3, 3]))
        self.sims.set_element_num(DictIO.GetAlternative(memory, "max_element_number", self.sims.get_default_element_number()))
        self.sims.set_constraint_num(DictIO.GetAlternative(memory, "max_constraint_number", {}))
        self.scene.activate_basic_class(self.sims)
        if log: 
            self.print_basic_simulation_info()
            self.print_simulation_info()
            if self.sims.dt[None] > 0.:
                self.print_solver_info()
    
    def print_basic_simulation_info(self):
        print(" Basic Configuration ".center(71,"-"))
        print(("Simulation Type: " + str(current_cfg().arch)).ljust(67))
        print(("Simulation Domain: " + str(self.sims.domain)).ljust(67))
        print(("Boundary Condition: " + str(self.sims.boundary)).ljust(67))
        print(("Gravity: " + str(self.sims.gravity)).ljust(67))

    def print_simulation_info(self):
        print(("Solver Type: " + str(self.sims.solver_type)).ljust(67))

    def print_solver_info(self):
        print(("Initial Simulation Time: " + str(self.sims.current_time)).ljust(67))
        print(("Finial Simulation Time: " + str(self.sims.current_time + self.sims.time)).ljust(67))
        print(("Time Step: " + str(self.sims.dt[None])).ljust(67))
        print(("Save Interval: " + str(self.sims.save_interval)).ljust(67))
        print(("Save Path: " + str(self.sims.path)).ljust(67), '\n')

    def add_material(self, model, material):
        self.scene.activate_material(self.sims, model, material)

    def add_patch(self, patch):
        self.scene.check_materials(self.sims)
        self.generator.add_patch(patch, self.sims, self.scene)

    def read_restart(self, file_number, file_path, is_continue=True):
        self.sims.set_is_continue(is_continue)

    def add_boundary_condition(self, boundary=None):
        self.scene.set_boundary_conditions(self.sims, boundary)

    def clean_boundary_condition(self, boundary):
        self.scene.clear_boundary_constraint(self.sims, boundary)

    def write_boundary_condition(self, output_path='OutputData'):
        self.scene.write_boundary_constraint(output_path)

    def select_save_data(self, gauss_point=True, volume=True, control_point=False):
        if volume is True: gauss_point = True
        self.sims.set_save_data(gauss_point, volume, control_point)

    def modify_parameters(self, **kwargs):
        if len(kwargs) > 0:
            self.sims.set_simulation_time(DictIO.GetEssential(kwargs, "SimulationTime"))
            if "Timestep" in kwargs: 
                self.sims.set_timestep(DictIO.GetEssential(kwargs, "Timestep"))
            if "CFL" in kwargs: self.sims.set_CFL(DictIO.GetEssential(kwargs, "CFL"))
            if "AdaptiveTimestep" in kwargs: self.sims.set_adaptive_timestep(DictIO.GetEssential(kwargs, "AdaptiveTimestep"))
            if "SaveInterval" in kwargs: self.sims.set_save_interval(DictIO.GetEssential(kwargs, "SaveInterval"))
            if "SavePath" in kwargs: self.sims.set_save_path(DictIO.GetEssential(kwargs, "SavePath"))
            if "gravity" in kwargs: self.sims.set_gravity(DictIO.GetEssential(kwargs, "gravity"))

    def add_engine(self):
        if self.enginer is None:
            if self.sims.solver_type == "Explicit":
                self.enginer = ExplicitEngine()
            elif self.sims.solver_type == "Implicit":
                self.enginer = ImplicitEngine()
            else:
                option = ["Explicit", "Implicit"]
                raise RuntimeError(f"Keyword:: /SolverType/ error. Only the following options are valid: {option}")
        self.enginer.choose_engine(self.sims)
        self.enginer.choose_boundary_constraints(self.scene)

    def add_recorder(self):
        if self.recorder is None:
            self.recorder = WriteFile(self.sims)

    def add_solver(self, kwargs):
        if self.solver is None:
            self.solver = Solver(self.sims, self.generator, self.enginer, self.recorder)
        self.solver.set_callback_function(kwargs)

    def set_window(self, window):
        self.sims.set_window_parameters(window)

    def add_essentials(self, kwargs):
        self.add_engine()
        self.add_recorder()
        self.add_solver(kwargs)

    def run(self, visualize=False, **kwargs):
        self.add_essentials(kwargs)
        self.check_critical_timestep()
        if visualize is False:
            self.solver.Solver(self.scene)
        else:
            self.sims.set_visualize_interval(DictIO.GetEssential(kwargs, "visualize_interval"))
            self.sims.set_window_size(DictIO.GetAlternative(kwargs, "WindowSize", self.sims.window_size))
            self.solver.Visualize(self.scene)

    def check_critical_timestep(self):
        if self.sims.solver_type == "Explicit":
            print("#", " Check Timestep ... ...".ljust(67))
            critical_timestep = self.scene.get_critical_timestep()
            if self.sims.CFL * critical_timestep < self.sims.dt[None]:
                self.sims.update_critical_timestep(self.sims.CFL * critical_timestep)
            else:
                print("The prescribed time step is sufficiently small\n")
