import numpy as np


class Engine(object):
    def __init__(self) -> None:
        self.apply_traction_constraint = None
        self.is_verlet_update = np.zeros(1, dtype=np.int32)
        
    def choose_engine(self, sims):
        raise NotImplementedError

    def choose_boundary_constraints(self, sims, scene):
        raise NotImplementedError

    def no_operation(self, sims, scene):
        pass

    def compute_deformation_gradient(self, sims, scene):
        raise NotImplementedError

    def integration(self, sims, scene):
        raise NotImplementedError

    def apply_kinematic_constraints(self, sims, scene):
        raise NotImplementedError

    def apply_traction_constraints(self, sims, scene):
        raise NotImplementedError
    
    def pre_calculation(self, scene):
        raise NotImplementedError
    
    def compute(self, sims, scene):
        raise NotImplementedError

    