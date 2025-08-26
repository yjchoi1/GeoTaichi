import taichi as ti
import numpy as np

from src.physics_model.consititutive_model.infinitesimal_strain.MaterialKernel import (
    calculate_strain_increment2D, 
    calculate_vorticity_increment2D, 
    calculate_strain_increment, 
    calculate_vorticity_increment)
from src.physics_model.consititutive_model.infinitesimal_strain.MaterialKernalNorsand import *
from src.physics_model.consititutive_model.SoftenModel import *
from src.physics_model.consititutive_model.infinitesimal_strain.ElasPlasticity import PlasticMaterial
from src.physics_model.consititutive_model.infinitesimal_strain.InfinitesimalStrainModel import InfinitesimalStrainModel
from src.utils.constants import FTOL, PI, EPS, STOL, MAXITS, Threshold
from src.utils.ObjectIO import DictIO
from src.utils.VectorFunction import voigt_tensor_trace, voigt_tensor_dot
import src.utils.GlobalVariable as GlobalVariable
STOL = 1.0


@ti.data_oriented
class NorSandModel(InfinitesimalStrainModel):
    def __init__(self, material_type="Solid", configuration="UL", solver_type="Explicit", stress_integration="ReturnMapping"):
        super().__init__(material_type, configuration, solver_type)
        self.Gamma = 0.
        self.Lambda = 0.
        self.M_tc = 0.
        self.N = 0.
        self.H0 = 0.
        self.Hy = 0.
        self.Chi_tc = 0.
        self.Ir = 0.
        self.nu = 0.
        self.Psi_0 = 0.
        self.G_max = 0.
        self.p_ref = 0.
        self.m = 0.

    def model_initialize(self, material):
        density = DictIO.GetAlternative(material, 'Density', 2650)
        poisson = DictIO.GetAlternative(material, 'PoissonRatio', 0.3)
        Gamma = DictIO.GetEssential(material, 'Gamma')
        Lambda = DictIO.GetEssential(material, 'Lambda')
        M_tc = DictIO.GetEssential(material, 'M_tc')
        N = DictIO.GetEssential(material, 'N')
        H0 = DictIO.GetEssential(material, 'H0')
        Hy = DictIO.GetEssential(material, 'Hy')
        Chi_tc = DictIO.GetEssential(material, 'Chi_tc')
        Ir = DictIO.GetEssential(material, 'Ir')
        Psi_0 = DictIO.GetEssential(material, 'Psi_0')
        G_max = DictIO.GetEssential(material, 'G_max')
        p_ref = DictIO.GetAlternative(material, 'p_ref', 100.)
        m = DictIO.GetAlternative(material, 'm', 0.)
        self.soft_function = ExponentialSoft()
        self.add_material(density, poisson, Gamma, Lambda, M_tc, N, H0, Hy, Chi_tc, Ir, Psi_0, G_max, p_ref, m)
        self.add_coupling_material(material)

    def add_material(self, density, poisson, Gamma, Lambda, M_tc, N, H0, Hy, Chi_tc, Ir, Psi_0, G_max, p_ref, m):
        self.density = density
        self.poisson = poisson
        self.Gamma = Gamma
        self.Lambda = Lambda
        self.M_tc = M_tc
        self.N = N
        self.H0 = H0
        self.Hy = Hy
        self.chi_tc = Chi_tc
        self.Ir = Ir
        self.Psi_0 = Psi_0
        self.G_max = G_max
        self.p_ref = p_ref
        self.m = m
        self.chi_i = self.chi_tc / (1 - self.Lambda * self.chi_tc / self.M_tc)
        self.max_sound_speed = self.get_sound_speed(self.density, self.young, self.poisson)
        self.is_soft = True

    def print_message(self, materialID):
        print(" Constitutive Model Information ".center(71, '-'))
        print('Constitutive model = NorSand Model')
        print("Model ID: ", materialID)
        print('Density: ', self.density)
        print('Poisson Ratio: ', self.poisson)
        print('Gamma: ', self.Gamma)
        print('Lambda: ', self.Lambda)
        print('M_tc: ', self.M_tc)
        print('N: ', self.N)
        print('H0: ', self.H0)
        print('Hy: ', self.Hy)
        print('Chi_tc: ', self.chi_tc)
        print('Ir: ', self.Ir)
        print('Psi_0: ', self.Psi_0)
        print('G_max: ', self.G_max)
        print('p_ref: ', self.p_ref)
        print('m: ', self.m)
        print('\n')

    def define_state_vars(self):
        return {
            'p_i': float,  # image mean stress
            'void_ratio': float,  # void ratio
            'M_i': float  # image critical stress ratio
        }
       
    @ti.func
    def _initialize_vars_update_lagrangian(self, np, particle, stateVars):
        # Initialize void ratio from initial state parameter psi_0 and current mean stress
        # Compute current p, q and Lode-dependent M
        cur_stress = particle[np].stress
        # Map external compression-negative convention to internal compression-positive
        s_pos = map_stress_comp_neg_to_pos(cur_stress)
        p, q = getPQ(s_pos)
        theta, M = getLodeM(s_pos, self.M_tc)

        # Compute critical void ratio at current p
        e_c = self.Gamma
        if p > 1.0:
            e_c = self.Gamma - self.Lambda * ti.log(p)

        e = self.Psi_0 + e_c

        # Find initial p_i and M_i
        p_i0, _, M_i0 = findp_ipsi_iM_i(e, self.N, M, self.M_tc, self.chi_tc, self.Gamma, self.Lambda, p, q)
        # Fallbacks if solver returns zeros
        if p_i0 <= 0.0:
            p_i0 = ti.max(p, 0.1)
        if M_i0 <= 0.0:
            M_i0 = M

        stateVars[np].void_ratio = e
        stateVars[np].p_i = p_i0
        stateVars[np].M_i = M_i0
    
    @ti.func
    def ComputeStress2D(self, np, previous_stress, velocity_gradient, stateVars, dt):  
        # Use NorSand-specific strain increment (engineering shear), skip rotational stress
        de = calculate_strain_increment2D_eng(velocity_gradient, dt)
        updated = self.ExplicitIntegration(np, previous_stress, de, ti.Vector.zero(float, 3), stateVars)
        return updated

    @ti.func
    def ComputeStress(self, np, previous_stress, velocity_gradient, stateVars, dt):  
        # Use NorSand-specific strain increment (engineering shear), skip rotational stress
        de = calculate_strain_increment_eng(velocity_gradient, dt)
        updated = self.ExplicitIntegration(np, previous_stress, de, ti.Vector.zero(float, 3), stateVars)
        return updated
        

    # ================ UMAT Integration ================
    @ti.func
    def ExplicitIntegration(self, np, previous_stress, de, dw, stateVars):
        """
        UMAT stress integration
        """
        # Skip rotational stress; integrate using UMAT (substepping inside)
        e = stateVars[np].void_ratio
        p_i = stateVars[np].p_i
        M_i = stateVars[np].M_i

        # Material constants
        N = self.N
        H = self.H0
        Hy = self.Hy
        M_tc = self.M_tc
        chi_tc = self.chi_tc
        Gamma = self.Gamma
        Lambda = self.Lambda
        Gmax = self.G_max
        Gexp = self.m
        nu = self.poisson

        # Call Taichi UMAT to advance stress and state
        stress_new, p_inew, M_inew, e_new = UMAT(e, previous_stress, p_i, N, H, Hy, M_i, M_tc, chi_tc, Gamma, Lambda, de, Gmax, Gexp, nu)

        # Update state variables
        stateVars[np].void_ratio = e_new
        stateVars[np].p_i = p_inew
        stateVars[np].M_i = M_inew

        return stress_new

    @ti.func
    def compute_elastic_tensor(self, np, current_stress, stateVars):
        # Provide elastic tangent used by external routines
        p, q = getPQ(current_stress)
        G = self.G_max * (p / 100.0) ** self.m
        # Bulk modulus from Poisson's ratio
        K = (2.0 * (1.0 + self.poisson)) / (3.0 * (1.0 - 2.0 * self.poisson)) * G
        C_e = getC_e(G, K)
        return C_e