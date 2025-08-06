import numpy as np
import taichi as ti

from src.iga.materials.ConstitutiveModelBase import ConstitutiveModelBase
from src.utils.constants import EYE
from src.consititutive_model.MaterialKernel import *
from src.utils.ObjectIO import DictIO
from src.utils.TypeDefination import vec6f, mat6x6


class LinearElastic(ConstitutiveModelBase):
    def __init__(self, max_material_num, max_patch_num, max_gauss_num, solver_type):
        super().__init__()
        self.matProps = LinearElasticModel.field(shape=max_material_num)
        self.stateVars = StateVariable.field(shape=max_gauss_num * max_patch_num) 

        if solver_type == "Implicit":
            self.stiffness_matrix = ti.Vector.field(3, float, shape=max_gauss_num * max_patch_num)

    def get_state_vars_dict(self, start_particle, end_particle):
        estress = np.ascontiguousarray(self.stateVars.estress.to_numpy()[start_particle:end_particle])
        return {'estress': estress}
    
    def reload_state_variables(self, state_vars):
        estress = state_vars.item()['estress']
        kernel_reload_state_variables(estress, self.stateVars)

    def model_initialize(self, material):
        materialID = DictIO.GetEssential(material, 'MaterialID')
        self.check_materialID(materialID, self.matProps.shape[0])

        if self.matProps[materialID].density > 0.:
            print("Previous Material Property will be overwritten!")
        density = DictIO.GetAlternative(material, 'Density', 2650)
        young = DictIO.GetEssential(material, 'YoungModulus')
        poisson = DictIO.GetAlternative(material, 'PoissonRatio', 0.3)

        self.matProps[materialID].add_material(density, young, poisson)
        self.matProps[materialID].print_message(materialID)

    def compute_elasto_plastic_stiffness(self, patchID, patch):
        pass


@ti.dataclass
class StateVariable:
    patchID: int
    materialID: int
    stress: vec6f
    deformation_gradient: vec6f
    stiffness_matrix: mat6x6

    @ti.func
    def _initialize_vars(self, patchID, materialID, stress, matProps):
        self.patchID = patchID
        self.materialID = materialID
        self.stress = stress
        self.deformation_gradient = EYE


@ti.dataclass
class LinearElasticModel:
    density: float
    young: float
    poisson: float
    shear: float
    bulk: float

    def add_material(self, density, young, poisson):
        self.density = density
        self.young = young
        self.poisson = poisson

        self.shear = 0.5 * self.young / (1. + self.poisson)
        self.bulk = self.young / (3. * (1 - 2. * self.poisson))
        
    def print_message(self, materialID):
        print(" Constitutive Model Information ".center(71, '-'))
        print('Constitutive model: Elastic Model')
        print("Model ID: ", materialID)
        print('Density: ', self.density)
        print('Young Modulus: ', self.young)
        print('Poisson Ratio: ', self.poisson, '\n')

    @ti.func
    def _get_sound_speed(self):
        sound_speed = 0.
        if self.density > 0.:
            sound_speed = ti.sqrt(self.young * (1 - self.poisson) / (1 + self.poisson) / (1 - 2 * self.poisson) / self.density)
        return sound_speed
    
    @ti.func
    def ComputeStress(self, np, stateVars, dt):  
        velocity_gradient = stateVars[np].velocity_gradient
        de = calculate_strain_increment(velocity_gradient, dt)
        
        shear_modulus, bulk_modulus = self.shear, self.bulk
        stress = stateVars[np].stress

        dstress = ComputeElasticStiffnessTensor(de, shear_modulus, bulk_modulus)
        
        stress += dstress
        stateVars[np].estress = VonMisesStress(stress)
        stateVars[np].stress = stress

    @ti.func
    def compute_elastic_tensor(self, np, stiffness, stateVars):
        ComputeElasticStiffnessTensor(np, self.bulk, self.shear, stiffness)

    @ti.func
    def compute_stiffness_tensor(self, np, stiffness, stateVars):
        return 0


@ti.kernel
def kernel_reload_state_variables(estress: ti.types.ndarray(), state_vars: ti.template()):
    for np in range(estress.shape[0]):
        state_vars[np].estress = estress[np]