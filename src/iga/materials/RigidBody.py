import numpy as np
import taichi as ti

from src.iga.materials.ConstitutiveModelBase import ConstitutiveModelBase
from src.utils.constants import DELTA
from src.utils.ObjectIO import DictIO
from src.utils.TypeDefination import mat3x3, vec6f


class RigidBody(ConstitutiveModelBase):
    def __init__(self, max_material_num, max_patch_num, max_gauss_num, solver_type):
        self.matProps = Rigid.field(shape=max_material_num)
        self.stateVars = StateVariable.field(shape=max_gauss_num * max_patch_num) 

        if solver_type == "Implicit":
            self.stiffness_matrix = None

    def get_state_vars_dict(self, start_particle, end_particle):
        estress = np.ascontiguousarray(self.stateVars.estress.to_numpy()[start_particle:end_particle])
        return {'estress': estress}

    def reload_state_variables(self, state_vars):
        estress = state_vars.item()['estress']
        kernel_reload_state_variables(estress, self.stateVars)
    
    def model_initialize(self, material):
        materialID = DictIO.GetAlternative(material, 'MaterialID', 0)
        if materialID > 0:
            materialID = 0
            
        if self.matProps[materialID].density > 0.:
            print("Previous Material Property will be overwritten!")
        density = DictIO.GetAlternative(material, 'Density', 2650)
        self.matProps[materialID].add_material(density)
        self.matProps[materialID].print_message(materialID)

    def compute_elasto_plastic_stiffness(self, particleNum, particle):
        pass


@ti.dataclass
class StateVariable:
    patchID: int
    materialID: int
    stress: vec6f

    @ti.func
    def _initialize_vars(self, patchID, materialID, stress, matProps):
        self.patchID = patchID
        self.materialID = materialID
        self.stress = stress

    @ti.func
    def _update_vars(self, stress):
        self.estress = 0.


@ti.dataclass
class TLStateVariable:
    estress: float
    deformation_gradient: mat3x3

    @ti.func
    def _initialize_vars(self, np, particle, matProps):
        self.estress = 0.
        self.deformation_gradient = DELTA

    @ti.func
    def _update_deformation_gradient(self, deformation_gradient_rate, dt):
        self.deformation_gradient += deformation_gradient_rate * dt[None]

    @ti.func
    def _update_vars(self, stress):
        self.estress = 0.

    
@ti.dataclass
class Rigid:
    density: float

    def add_material(self, density):
        self.density = density
        
    def print_message(self, materialID):
        print(" Constitutive Model Information ".center(71, '-'))
        print('Constitutive model: Rigid Body')
        print("Model ID: ", materialID)
        print('Density: ', self.density, '\n')

    @ti.func
    def update_particle_volume(self, np, velocity_gradient, stateVars, particle, dt):
        pass
    
    @ti.func
    def update_particle_volume_bbar(self, np, strain_rate, stateVars, particle, dt):
        pass

    @ti.func
    def ComputeStress(self, np, stateVars, particle, dt):
        pass

    @ti.func
    def _get_sound_speed(self):
        return 1e-14
    
    @ti.func
    def compute_elastic_tensor(self, np, stiffness, stateVars):
        return 0

    @ti.func
    def compute_stiffness_tensor(self, np, stiffness, stateVars):
        return 0

@ti.kernel
def kernel_reload_state_variables(estress: ti.types.ndarray(), state_vars: ti.template()):
    for np in range(estress.shape[0]):
        state_vars[np].estress = estress[np]