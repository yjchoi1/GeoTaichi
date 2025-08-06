import numpy as np
import taichi as ti

from src.iga.materials.ConstitutiveModelBase import ConstitutiveModelBase
from src.consititutive_model.MaterialKernel import *
from src.utils.constants import DELTA
from src.utils.MatrixFunction import principal_tensor, Diagonal, matrix_form
from src.utils.ObjectIO import DictIO
from src.utils.TypeDefination import mat3x3
from src.utils.VectorFunction import voigt_form


class HencyElastic(ConstitutiveModelBase):
    def __init__(self, max_material_num, max_patch_num, max_gauss_num, solver_type):
        super().__init__()
        self.matProps = HencyElasticModel.field(shape=max_material_num)
        self.stateVars = StateVariable.field(shape=max_gauss_num * max_patch_num)

        if solver_type == "Implicit":
            self.stiffness_matrix = ti.Vector.field(3, float, shape=max_gauss_num * max_patch_num)

    def get_state_vars_dict(self, start_particle, end_particle):
        estress = np.ascontiguousarray(self.stateVars.estress.to_numpy()[start_particle:end_particle])
        deformation_gradient = np.ascontiguousarray(self.stateVars.deformation_gradient.to_numpy()[start_particle:end_particle])
        return {'estress': estress, 'deformation_gradient': deformation_gradient}
    
    def reload_state_variables(self, state_vars):
        estress = state_vars.item()['estress']
        deformation_gradient = state_vars.item()['deformation_gradient']
        kernel_reload_state_variables(estress, deformation_gradient, self.stateVars)

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

    def get_lateral_coefficient(self, materialID):
        mu = self.matProps[materialID].poisson
        return 0.3#mu / (1. - mu)


@ti.dataclass
class StateVariable:
    patchID: int
    materialID: int
    estress: float
    deformation_gradient: mat3x3
    stress: mat3x3
    stress0: mat3x3

    @ti.func
    def _initialize_vars(self, patchID, materialID, stress, matProps):
        self.patchID = patchID
        self.materialID = materialID
        self.estress = VonMisesStress(stress)
        self.deformation_gradient = DELTA
        self.stress0 = matrix_form(stress)

    @ti.func
    def _update_deformation_gradient(self, deformation_gradient_rate, dt):
        self.deformation_gradient += deformation_gradient_rate * dt[None]

    @ti.func
    def _update_vars(self, stress):
        self.estress = VonMisesStress(stress)


@ti.dataclass
class HencyElasticModel:
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
    def update_particle_volume(self, np, velocity_gradient, stateVars, particle, dt):
        particle[np].vol *= (DELTA + velocity_gradient * dt[None]).determinant()
        stateVars[np].deformation_gradient = (DELTA + velocity_gradient * dt[None]) @ stateVars[np].deformation_gradient
    
    @ti.func
    def elastic_part(self):
        a1 = self.bulk + (4./3.) * self.shear
        a2 = self.bulk - (2./3.) * self.shear
        return mat3x3([[a1, a2, a2], [a2, a1, a2], [a2, a2, a1]])
    
    @ti.func
    def ComputeStress(self, np, stateVars, particle, dt):
        deformation_gradient = stateVars[np].deformation_gradient

        ijacobian = 1. / deformation_gradient.determinant()
        left_cauchy_green_tensor = deformation_gradient @ deformation_gradient.transpose()
        principal_left_cauchy_green_strain, directors = principal_tensor(left_cauchy_green_tensor)
        principal_hencky_strain = 0.5 * ti.log(principal_left_cauchy_green_strain)
        principal_kirchhoff_stress = self.elastic_part() @ principal_hencky_strain
        principal_cauchy_stress = ijacobian * Diagonal(principal_kirchhoff_stress)
        cauchy_stress = stateVars[np].stress0 + directors @ principal_cauchy_stress @ directors.transpose()
        particle[np].stress = voigt_form(cauchy_stress)
        stateVars[np].stress = cauchy_stress
        stateVars[np].estress = VonMisesStress(voigt_form(cauchy_stress))

    @ti.func
    def ComputePKStress(self, np, stateVars, particle, dt):  
        deformation_gradient = stateVars[np].deformation_gradient

        left_cauchy_green_tensor = deformation_gradient @ deformation_gradient.transpose()
        principal_left_cauchy_green_strain, directors = principal_tensor(left_cauchy_green_tensor)
        principal_hencky_strain = 0.5 * ti.log(principal_left_cauchy_green_strain)
        principal_kirchhoff_stress = self.elastic_part() @ principal_hencky_strain
        kirchhoff_stress = stateVars[np].stress0 + directors @ Diagonal(principal_kirchhoff_stress) @ directors.transpose()
        voigt_stress = voigt_form(kirchhoff_stress)

        stateVars[np].stress = kirchhoff_stress
        stateVars[np].estress = VonMisesStress(voigt_stress)
        particle[np].stress = voigt_stress

    @ti.func
    def compute_elastic_tensor(self, np, stiffness, stateVars):
        self.compute_stiffness_tensor(np, stiffness, stateVars)

    @ti.func
    def compute_stiffness_tensor(self, np, stiffness, stateVars):
        deformation_gradient = stateVars[np].deformation_gradient
        jacobian = deformation_gradient.determinant()
        lambda_ = 3. * self.bulk * self.poisson / (1 + self.poisson) 
        modified_shear = (self.shear - lambda_ * ti.log(jacobian)) / jacobian
        modified_lambda = lambda_ / jacobian

        a1 = modified_lambda + 2. * modified_shear
        a2 = modified_lambda
        stiffness[np][0] = a1
        stiffness[np][1] = a2
        stiffness[np][2] = modified_shear


@ti.kernel
def kernel_reload_state_variables(estress: ti.types.ndarray(), deformation_gradient: ti.types.ndarray(), state_vars: ti.template()):
    for np in range(estress.shape[0]):
        state_vars[np].estress = estress[np]
        state_vars[np].deformation_gradient = deformation_gradient[np]