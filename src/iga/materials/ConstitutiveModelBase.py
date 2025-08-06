import taichi as ti

from src.consititutive_model.MaterialKernel import find_max_sound_speed_


class ConstitutiveModelBase:
    def __init__(self) -> None:
        self.compute = None
        self.matProps = None
        self.stateVars = None
        self.stiffness_matrix = None

    def check_materialID(self, materialID, max_material_num):
        if materialID <= 0: 
            raise RuntimeError(f"MaterialID {materialID} should be larger than 0")
        if materialID > max_material_num - 1:
            raise RuntimeError(f"Keyword:: /max_material_number/ should be set as {materialID + 1}")

    def model_initialization(self, materials):
        if type(materials) is dict:
            self.model_initialize(materials)
        elif type(materials) is list:
            for material in materials:
                self.model_initialize(material)

    def model_initialize(self, material):
        raise NotImplementedError
    
    def state_vars_initialize(self, start_particle, end_particle, particle):
        raise NotImplementedError
    
    def get_state_vars_dict(self, start_particle, end_particle):
        raise NotImplementedError
    
    def compute_stress(self):
        raise NotImplementedError
    
    def get_lateral_coefficient(self, materialID):
        raise NotImplementedError
    
    def reload_state_variables(self, state_vars):
        raise NotImplementedError
    
    def find_max_sound_speed(self):
        return find_max_sound_speed_(self.matProps)
    
    def pre_compute_stiffness(self, totalGaussNum):
        compute_elastic_stiffness_matrix(int(totalGaussNum), self.stiffness_matrix, self.matProps, self.stateVars)
    
    def compute_elasto_plastic_stiffness(self, totalGaussNum):
        compute_stiffness_matrix(int(totalGaussNum), self.stiffness_matrix, self.matProps, self.stateVars)

    def state_vars_initialize(self, start_particle, end_particle, patchID, materialID, initial_stress):
        kernel_initial_state_variables(start_particle, end_particle, patchID, materialID, initial_stress, self.stateVars, self.matProps)


@ti.dataclass
class StateVariable:
    # TODO: add essential state variable for constitutive model
    estress: float

    @ti.func
    def _initialize_vars(self, stress):
        pass

    @ti.func
    def _update_vars(self, stress, epstrain):
        pass


@ti.kernel
def kernel_initial_state_variables(to_beg: int, to_end: int, patchID: int, materialID: int, initial_stress: ti.types.vector(6, float), stateVars: ti.template(), matProps: ti.template()):
    for np in range(to_beg, to_end):
        stateVars[np]._initialize_vars(patchID, materialID, initial_stress, matProps)


@ti.kernel
def compute_elastic_stiffness_matrix(totalGaussNum: int, stiffness_matrix: ti.template(), matProps: ti.template(), stateVars: ti.template()):
    for ngp in range(totalGaussNum):
        materialID = int(stateVars[ngp].materialID)
        matProps[materialID].compute_elastic_tensor(ngp, stiffness_matrix, stateVars)


@ti.kernel
def compute_stiffness_matrix(totalGaussNum: int, stiffness_matrix: ti.template(), matProps: ti.template(), stateVars: ti.template()):
    for ngp in range(totalGaussNum):
        materialID = int(stateVars[ngp].materialID)
        matProps[materialID].compute_stiffness_tensor(ngp, stiffness_matrix, stateVars)