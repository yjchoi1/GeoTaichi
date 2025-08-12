import taichi as ti
import numpy as np

from src.physics_model.consititutive_model.infinitesimal_strain.MaterialKernel import *
from src.physics_model.consititutive_model.SoftenModel import *
from src.physics_model.consititutive_model.infinitesimal_strain.ElasPlasticity import PlasticMaterial
from src.utils.constants import FTOL, PI
from src.utils.ObjectIO import DictIO
from src.utils.VectorFunction import voigt_tensor_trace, voigt_tensor_dot
import src.utils.GlobalVariable as GlobalVariable


@ti.data_oriented
class NorSandModel(PlasticMaterial):
    def __init__(self, material_type="Solid", configuration="UL", solver_type="Explicit", stress_integration="ReturnMapping"):
        super().__init__(material_type, configuration, solver_type, stress_integration)
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
        p_ref = DictIO.GetEssential(material, 'p_ref', 100.)
        m = DictIO.GetEssential(material, 'm', 0.)
        self.soft_function = ExponentialSoft()
        self.set_rate_dependent_model(material)
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
        self.psi_0 = Psi_0
        self.G_max = G_max
        self.p_ref = p_ref
        self.m = m
        self.chi_i = self.chi_tc / (1 - self.Lambda * self.chi_tc / self.M_tc)
        self.max_sound_speed = self.get_sound_speed()
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
        print('Psi_0: ', self.psi_0)
        print('G_max: ', self.G_max)
        print('p_ref: ', self.p_ref)
        print('m: ', self.m)
        print('\n')

    def define_state_vars(self):
        return {'p_i': float, 'e': float}
    
    def get_sound_speed(self):
        return 0.

    def choose_soft_function(self, material):
        soft_type = DictIO.GetAlternative(material, "SoftType", None)
        if soft_type == "Exponential":
            self.is_soft = True
            self.soft_function = ExponentialSoft()
        elif soft_type == "Sinh":
            self.is_soft = True
            self.soft_function = SinhSoft()
        else:
            self.is_soft = False

    '''def get_lateral_coefficient(self, start_index, end_index, materialID, stateVars):
        return np.repeat(0.9, end_index - start_index)
        if GlobalVariable.RANDOMFIELD:
            particle_index = np.ascontiguousarray(materialID.to_numpy()[start_index:end_index])
            m_theta = np.ascontiguousarray(stateVars.m_theta.to_numpy()[particle_index])
            return 1. - (3 * m_theta) / (6. + m_theta)
        else:
            m_theta = self.m_theta
            return np.repeat(1. - (3 * m_theta) / (6. + m_theta), end_index - start_index)'''
       
    @ti.func
    def _initialize_vars_update_lagrangian(self, np, particle, stateVars):
        # NOTE: need to check
        stress = particle[np].stress
        p, q, lode = self.ComputeStressInvariants(stress)
        # Initial void ratio from psi_0 and current mean pressure
        e_c = self.Gamma - self.Lambda * ti.log(p)
        e0 = self.psi_0 + e_c
        # Initialize p_i via quadratic solver; fallback if not found
        M = self.find_M(lode, self.M_tc)
        p_i0, psi_i0, M_i0 = self.find_p_i_psi_i_M_i(self.N, self.chi_i, self.Lambda, self.M_tc, self.psi_0, p, q, M)
        if p_i0 <= 0.0:
            p_i0 = p
        stateVars[np].p_i = p_i0
        stateVars[np].e = e0

    # ==================================================== NorSand Model ==================================================== #
    @ti.func
    def ComputeStressInvariants(self, stress):
        # NOTE: ok
        p = -SphericalTensor(stress)
        q = EquivalentDeviatoricStress(stress)
        lode = ComputeLodeAngleWithMode(stress, True)
        return p, q, lode
        
    @ti.func
    def find_M(self, theta, M_tc):
        # NOTE: ok
        # For triaxial compression, g_theta becomes 1, making M to be the same as M_tc
        g_theta = 1. - (M_tc / (3. + M_tc)) * ti.cos(1.5 * theta + PI / 4.)
        return M_tc * g_theta

    @ti.func
    def find_M_i(self, M, M_tc, chi_i, psi_i, N):
        # NOTE: ok
        return M * (1. - chi_i * N * ti.abs(psi_i) / M_tc)

    @ti.func
    def find_psi_psi_i(self, Gamma, Lambda, p, p_i, e):
        # NOTE: ok
        e_c = Gamma - Lambda * ti.log(p)
        psi = e - e_c
        psi_i = psi + Lambda * ti.log(p_i / p)
        return psi, psi_i

    @ti.func
    def find_F(self, p, q, M_i, p_i):
        # NOTE: ok
        return q / p - M_i * (1. - ti.log(p / p_i))

    @ti.func
    def find_p_i_psi_i_M_i(self, N, chi_i, lamb, M_tc, psi, p, q, M):
        # NOTE: ok, but 1/1000 case did not match with the original implementation
        # Set up terms to be used in four quadratic options (matching NumPy exactly)
        a = N * chi_i * lamb / M_tc
        psi_term = N * chi_i * psi / M_tc
        b1 = a - 1.0 + psi_term
        b2 = a + 1.0 + psi_term
        c1 = psi_term + (q / p) / M - 1.0
        c2 = psi_term - (q / p) / M + 1.0
        
        # Find roots to quadratic equations
        x1a = (-b1 + ti.sqrt(b1**2 - 4.0*a*c1)) / (2.0*a)
        x1b = (-b1 - ti.sqrt(b1**2 - 4.0*a*c1)) / (2.0*a)
        x2a = (-b2 + ti.sqrt(b2**2 - 4.0*a*c2)) / (2.0*a)
        x2b = (-b2 - ti.sqrt(b2**2 - 4.0*a*c2)) / (2.0*a)
        
        # # Find roots to quadratic equations (handle complex discriminants properly)
        # discriminant1 = b1**2 - 4*a*c1
        # discriminant2 = b2**2 - 4*a*c2
        
        # # Calculate roots (with discriminant checks for numerical stability)
        # x1a = 0.0
        # x1b = 0.0
        # x2a = 0.0
        # x2b = 0.0
        
        # if discriminant1 >= 0.0:
        #     sqrt_discriminant1 = ti.sqrt(discriminant1)
        #     x1a = (-b1 + sqrt_discriminant1) / (2*a)
        #     x1b = (-b1 - sqrt_discriminant1) / (2*a)
        # else:
        #     # Set to NaN to indicate invalid roots - these will be caught by isnan check
        #     x1a = float('nan')
        #     x1b = float('nan')
        
        # if discriminant2 >= 0.0:
        #     sqrt_discriminant2 = ti.sqrt(discriminant2)
        #     x2a = (-b2 + sqrt_discriminant2) / (2*a)
        #     x2b = (-b2 - sqrt_discriminant2) / (2*a)
        # else:
        #     # Set to NaN to indicate invalid roots - these will be caught by isnan check
        #     x2a = float('nan')
        #     x2b = float('nan')
        
        # Initialize return values with defaults (matching NumPy)
        p_i = 0.0
        psi_i = 0.0
        M_i = 0.0
        
        # Loop through each root (matching NumPy order exactly)
        x_list = ti.Vector([x1a, x1b, x2a, x2b])
        for i in range(4):
            x_test = x_list[i]
            psii_test = x_test * lamb + psi
            pi_test = p * ti.exp(x_test)
            Mi_test = M * (1.0 - N * chi_i * ti.abs(lamb*x_test + psi) / M_tc)

            # Check against yield function which runs through current stress state
            if psii_test > 0.0:
                F = a * x_test**2 + (a - 1.0 + psi_term) * x_test + (psi_term + (q / p) / M - 1.0)
                # NOTE: check if this is valid for norsand. Why Mi_test > 0.1 * M?
                if ti.abs(F) <= 1e-5 and Mi_test < M and Mi_test > 0.1 * M:
                    psi_i = psii_test
                    p_i = pi_test
                    M_i = Mi_test
            elif psii_test < 0.0:
                F = a * x_test**2 + (a + 1.0 + psi_term) * x_test + (psi_term - (q / p) / M + 1.0)
                # NOTE: check if this is valid for norsand. Why Mi_test > 0.1 * M?
                if ti.abs(F) <= 1e-5 and Mi_test < M and Mi_test > 0.1 * M:
                    psi_i = psii_test
                    p_i = pi_test
                    M_i = Mi_test
                    
        return p_i, psi_i, M_i
        
    @ti.func
    def ComputeElasticModulus(self, stress, material_params):
        # NOTE: ok. but Sean's implementation has 0.1 cutoff for stress[0:3] for `p`
        p = -SphericalTensor(stress)
        G = self.G_max * (p / self.p_ref) ** self.m
        K = (2. * (1. + self.poisson) / (3. * (1. - 2. * self.poisson))) * G
        return K, G
    
    @ti.func
    def ComputeYieldFunction(self, stress, internal_vars, material_params):
        # NOTE: ok, but not sure if the initial MPM step's p_i, psi_i, M_i is computed correctly
        p, q, lode = self.ComputeStressInvariants(stress)
        
        p_i, e = internal_vars[0], internal_vars[1]
        
        M = self.find_M(lode, self.M_tc)
        _, psi_i = self.find_psi_psi_i(self.Gamma, self.Lambda, p, p_i, e)
        M_i = self.find_M_i(M, self.M_tc, self.chi_i, psi_i, self.N)
        
        return self.find_F(p, q, M_i, p_i)

    @ti.func
    def ComputeYieldState(self, stress, internal_vars, material_params):
        # NOTE: ok
        f_function = self.ComputeYieldFunction(stress, internal_vars, material_params)
        return f_function > -FTOL, f_function
        
    @ti.func
    def ComputeDfDsigma(self, yield_state, stress, internal_vars, material_params):
        # NOTE: ok
        p, q, lode = self.ComputeStressInvariants(stress)
        p_i, e = internal_vars[0], internal_vars[1]
        
        M = self.find_M(lode, self.M_tc)
        _, psi_i = self.find_psi_psi_i(self.Gamma, self.Lambda, p, p_i, e)
        M_i = self.find_M_i(M, self.M_tc, self.chi_i, psi_i, self.N)

        # Compute dF/dp
        dfdp = M_i - q / p
        # Compute dF/dtheta (initialize to zero to satisfy Taichi's SSA)
        dM_i_dM = 1. - self.chi_i * self.N * ti.abs(psi_i) / self.M_tc
        dfdtheta = 0.0
        if M_i > Threshold:
            dfdtheta = -(q / M_i) * (3. / 2.) * dM_i_dM * (self.M_tc**2 / (3. + self.M_tc)) * ti.sin(3. * lode / 2. + PI / 4.)
        
        dpdsigma = DpDsigma()
        dqdsigma = DqDsigma(stress)  
        # NOTE: assume this is correcly implemented in Geotaichi
        dthetadsigma = DlodeDsigma(stress)  
        
        # For F = q/p - M_i * (1 - ln(p/p_i)), dF/dq = 1/p
        dfdsigma = dfdp * dpdsigma + 1.0 * dqdsigma + dfdtheta * dthetadsigma
        return dfdsigma
    
    @ti.func
    def ComputeDgDsigma(self, yield_state, stress, internal_vars, material_params):
        '''
        Associative flow rule is used in NorSand model.
        '''
        # NOTE: ok
        return self.ComputeDfDsigma(yield_state, stress, internal_vars, material_params)
    
    @ti.func
    def ComputePlasticModulus(self, yield_state, dgdsigma, stress, internal_vars, state_vars, material_params):
        # NOTE: similar to finddFdepsilon_p in norsand_functions.py
        p, q, lode = self.ComputeStressInvariants(stress)
        p_i, e = internal_vars[0], internal_vars[1]
        # State-dependent quantities
        M = self.find_M(lode, self.M_tc)
        psi, psi_i = self.find_psi_psi_i(self.Gamma, self.Lambda, p, p_i, e)
        M_i = self.find_M_i(M, self.M_tc, self.chi_i, psi_i, self.N)
        M_itc = self.find_M_itc(self.N, self.chi_i, psi_i, self.M_tc)
        p_imax = self.find_p_imax(self.chi_i, psi_i, p, M_itc)
        # Hardening modulus and dpi/deps_d^p
        H = self.H0 - self.Hy * psi
        dpideps_pd = H * (M_i / M_itc) * (p_imax - p_i) * (p / p_i)
        # dF/dpi and deviatoric projection of df/dsigma
        psi_sign = 0.0
        if psi_i > 0.0:
            psi_sign = 1.0
        elif psi_i < 0.0:
            psi_sign = -1.0
        dMidpi = -(M / self.M_tc) * self.N * self.chi_i * psi_sign * self.Lambda / p_i
        dfdsigma = self.ComputeDfDsigma(yield_state, stress, internal_vars, material_params)
        # dF/dpi = partial F/partial p_i + dF/dM_i * dM_i/dp_i
        dfdpi = 0.0
        if M_i > Threshold:
            dfdpi = -(p * M_i) / p_i + (-q / M_i) * dMidpi
        else:
            dfdpi = -(p * M_i) / p_i
        dfdq_proj = self.deviatoric_projection(stress, dfdsigma)
        return dfdpi * dpideps_pd * dfdq_proj
    
    @ti.func
    def ComputeInternalVariables(self, dlambda, dgdsigma, stress, internal_vars, material_params):
        # NOTE: not sure where it comes from
        # Update of p_i and void ratio e
        p, q, lode = self.ComputeStressInvariants(stress)
        p_i, e = internal_vars[0], internal_vars[1]
        M = self.find_M(lode, self.M_tc)
        psi, psi_i = self.find_psi_psi_i(self.Gamma, self.Lambda, p, p_i, e)
        M_i = self.find_M_i(M, self.M_tc, self.chi_i, psi_i, self.N)
        M_itc = self.find_M_itc(self.N, self.chi_i, psi_i, self.M_tc)
        p_imax = self.find_p_imax(self.chi_i, psi_i, p, M_itc)
        # Hardening
        H = self.H0 - self.Hy * psi
        e_q_per_lambda = self.deviatoric_projection(stress, dgdsigma)
        dpideps_pd = H * (M_i / M_itc) * (p_imax - p_i) * (p / p_i)
        dpi = dlambda * e_q_per_lambda * dpideps_pd
        # Volumetric plastic strain leads to void ratio update
        dpvstrain = -ComputeStrainInvariantI1(dlambda * dgdsigma)
        de = -(1.0 + e) * dpvstrain
        return ti.Vector([dpi, de])
    
    @ti.func
    def GetMaterialParameter(self, stress, state_vars):
        # Not used for NorSand elastic modulus; kept for interface compatibility
        return ti.Vector([0.0])
    
    @ti.func
    def GetInternalVariables(self, state_vars):
        return ti.Vector([state_vars.p_i, state_vars.e])
    
    @ti.func
    def UpdateInternalVariables(self, np, internal_vars, stateVars):
        stateVars[np].p_i = internal_vars[0]
        stateVars[np].e = internal_vars[1]

    # ---------------- NorSand helpers ----------------
    @ti.func
    def find_M_itc(self, N, chi_i, psi_i, M_tc):
        # NOTE: ok
        return M_tc - N * chi_i * ti.abs(psi_i)

    @ti.func
    def find_p_imax(self, chi_i, psi_i, p, M_itc):
        # NOTE: ok
        D_min = chi_i * psi_i
        return p * ti.exp(-D_min / M_itc)

    @ti.func
    def deviatoric_projection(self, stress, vec6):
        eq = 0.0
        p = -SphericalTensor(stress)
        q = EquivalentDeviatoricStress(stress)
        if q > Threshold:
            eq += (stress[0] + p) / q * vec6[0]
            eq += (stress[1] + p) / q * vec6[1]
            eq += (stress[2] + p) / q * vec6[2]
            eq += 2.0 * stress[3] / q * vec6[3]
            eq += 2.0 * stress[4] / q * vec6[4]
            eq += 2.0 * stress[5] / q * vec6[5]
        return eq

    @ti.func
    def compute_elastic_tensor(self, np, current_stress, stateVars):
        bulk_modulus, shear_modulus = self.ComputeElasticModulus(current_stress, stateVars[np])
        return ComputeElasticStiffnessTensor(bulk_modulus, shear_modulus)
    
    # @ti.func
    def get_current_material_parameter(self, state_vars):
        raise NotImplementedError