from src.iga.engine.Engine import Engine
from src.iga.engine.EngineKernel import *
from src.iga.SceneManager import myScene
from src.iga.Simulation import Simulation


class ImplicitEngine(Engine):
    def __init__(self) -> None:
        super().__init__()
        self.apply_displacement_constraint = None
        
    def choose_engine(self, sims: Simulation):
        if sims.update == "Newmark":
            self.compute = self.newmark_integrate
        else:
            raise ValueError(f"The update scheme {sims.update} is not supported yet")

    def choose_boundary_constraints(self, scene: myScene):
        self.apply_traction_constraint = self.no_operation
        self.apply_displacement_constraint = self.no_operation

        if scene.traction_list[None] > 0:
            self.apply_traction_constraint = self.apply_traction_constraints
        if scene.kinematic_list[None] > 0:
            self.apply_kinematic_constraint = self.apply_kinematic_constraints

    def calculate_shape_derivates(self, sims: Simulation, scene: myScene, patchID):
        assemble_bmatrix(patchID, scene.element.gauss_point.gpcoords, sims.max_total_gauss_each_patch, sims.max_total_point_each_patch, sims.max_total_element_each_patch,
                         scene.patch, scene.ctrlpts, scene.element.connectivity, scene.element.dshapefn, scene.element.bmatrix, scene.element.jdet, scene.element.j2det, scene.element.gauss_point.weight)

    def newmark_integrate(self, sims, scene):
        pass

    def apply_traction_constraints(self, sims, scene):
        pass

    def apply_kinematic_constraints(self, sims, scene):
        pass

    def pre_calculation(self, scene: myScene):
        scene.element.calculate(scene.elementNum[0], scene.patch, scene.knotU, scene.knotV, scene.knotW, scene.ctrlpts)
        scene.material.pre_compute_stiffness(scene.gaussNum[0])

    def compute(self, sims: Simulation, scene: myScene):
        iter_num = 0
        convergence = False
        while not convergence and iter_num < sims.iter_max:
            self.calculate_shape_derivates(sims, scene)
            self.apply_traction_constraint(sims, scene)
            self.apply_displacement_constraint(sims, scene)