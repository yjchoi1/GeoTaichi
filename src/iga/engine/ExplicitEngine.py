from src.iga.engine.Engine import Engine
from src.iga.SceneManager import myScene
from src.iga.Simulation import Simulation

class ExplicitEngine(Engine):
    def __init__(self) -> None:
        super().__init__()
        self.apply_velocity_constraint = None

    def choose_engine(self, sims):
        pass

    def choose_boundary_constraints(self, scene: myScene):
        self.apply_traction_constraint = self.no_operation
        self.apply_velocity_constraint = self.no_operation

        if scene.traction_list[None] > 0:
            self.apply_traction_constraint = self.apply_traction_constraints
            self.apply_velocity_constraint = self.apply_kinematic_constraints

    def apply_traction_constraints(self, sims, scene):
        pass

    def apply_kinematic_constraints(self, sims, scene):
        pass

    def pre_calculation(self, scene: myScene):
        scene.element.calculate(scene.elementNum[0], scene.patch, scene.knotU, scene.knotV, scene.knotW, scene.ctrlpts)

    def compute(self, sims, scene):
        pass