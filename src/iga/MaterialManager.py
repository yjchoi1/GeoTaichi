from src.iga.materials.RigidBody import RigidBody
from src.iga.materials.HencyElastic import HencyElastic
from src.iga.materials.LinearElastic import LinearElastic
from src.iga.materials.MaterialModel import UserDefined

class ConstitutiveModel:
    @classmethod
    def initialize(cls, constitutive_model, max_material_num, max_patch_num, max_gauss_num, solver_type):
        model_type = ["None", "LinearElastic", "HencyElastic", "ElasticPerfectlyPlastic", "MohrCoulomb", "CohesiveModifiedCamClay", "UserDefined"]
        if constitutive_model == "None":
            return RigidBody(max_material_num, max_patch_num, max_gauss_num, solver_type)
        elif constitutive_model == "HencyElastic":
            return HencyElastic(max_material_num, max_patch_num, max_gauss_num, solver_type)
        elif constitutive_model == "LinearElastic":
            return LinearElastic(max_material_num, max_patch_num, max_gauss_num, solver_type)
        elif constitutive_model == "UserDefined":
            return UserDefined(max_material_num, max_patch_num, max_gauss_num, solver_type)
        else:
            raise ValueError(f'Constitutive Model: {constitutive_model} error! Only the following is aviliable:\n{model_type}')
        