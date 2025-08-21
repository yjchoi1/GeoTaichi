import taichi as ti
import numpy as np

from src.physics_model.consititutive_model.infinitesimal_strain.MaterialKernel import *
from src.physics_model.consititutive_model.SoftenModel import *
from src.physics_model.consititutive_model.infinitesimal_strain.ElasPlasticity import PlasticMaterial
from src.utils.constants import FTOL, PI, EPS, STOL, MAXITS, Threshold
from src.utils.ObjectIO import DictIO
from src.utils.VectorFunction import voigt_tensor_trace, voigt_tensor_dot
import src.utils.GlobalVariable as GlobalVariable
STOL = 1.0


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
        p_ref = DictIO.GetAlternative(material, 'p_ref', 100.)
        m = DictIO.GetAlternative(material, 'm', 0.)
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
        # Store stress, strain, p_i, and e as state variables
        return {
            'p_i': float, 
            'e': float,
            'stress': vec6f,
            'strain': vec6f
        }
    
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
       
    @ti.func
    def _initialize_vars_update_lagrangian(self, np, particle, stateVars):
        # Initialize based on reference code
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
        stateVars[np].stress = stress
        stateVars[np].strain = vec6f(0, 0, 0, 0, 0, 0)

    # ==================================================== NorSand Model ==================================================== #
    @ti.func
    def ComputeStressInvariants(self, stress):
        # Use asin convention from reference code
        p = -SphericalTensor(stress)
        q = EquivalentDeviatoricStress(stress)
        lode = ComputeLodeAngleWithMode(stress, True)
        return p, q, lode
        
    @ti.func
    def find_M(self, theta, M_tc):
        # g_theta function from reference code
        g_theta = 1. - (M_tc / (3. + M_tc)) * ti.cos(1.5 * theta + PI / 4.)
        return M_tc * g_theta

    @ti.func
    def find_M_i(self, M, M_tc, chi_i, psi_i, N):
        return M * (1. - chi_i * N * ti.abs(psi_i) / M_tc)

    @ti.func
    def find_psi_psi_i(self, Gamma, Lambda, p, p_i, e):
        e_c = Gamma - Lambda * ti.log(p)
        psi = e - e_c
        psi_i = psi + Lambda * ti.log(p_i / p)
        return psi, psi_i

    @ti.func
    def find_F(self, p, q, M_i, p_i):
        return q / p - M_i * (1. - ti.log(p / p_i))

    @ti.func
    def find_p_i_psi_i_M_i(self, N, chi_i, lamb, M_tc, psi, p, q, M):
        # Quadratic solver from reference code
        a = N * chi_i * lamb / M_tc
        psi_term = N * chi_i * psi / M_tc
        b1 = a - 1.0 + psi_term
        b2 = a + 1.0 + psi_term
        c1 = psi_term + (q / p) / M - 1.0
        c2 = psi_term - (q / p) / M + 1.0
        
        # Find roots to quadratic equations
        discriminant1 = b1**2 - 4.0*a*c1
        discriminant2 = b2**2 - 4.0*a*c2
        
        x1a = (-b1 + ti.sqrt(discriminant1)) / (2.0*a) if discriminant1 >= 0 else 0.0
        x1b = (-b1 - ti.sqrt(discriminant1)) / (2.0*a) if discriminant1 >= 0 else 0.0
        x2a = (-b2 + ti.sqrt(discriminant2)) / (2.0*a) if discriminant2 >= 0 else 0.0
        x2b = (-b2 - ti.sqrt(discriminant2)) / (2.0*a) if discriminant2 >= 0 else 0.0
        
        # Initialize return values with defaults
        p_i = 0.0
        psi_i = 0.0
        M_i = 0.0
        
        # Loop through each root
        x_list = ti.Vector([x1a, x1b, x2a, x2b])
        for i in range(4):
            x_test = x_list[i]
            psii_test = x_test * lamb + psi
            pi_test = p * ti.exp(x_test)
            Mi_test = M * (1.0 - N * chi_i * ti.abs(lamb*x_test + psi) / M_tc)

            # Check against yield function which runs through current stress state
            if psii_test > 0.0:
                F = a * x_test**2 + (a - 1.0 + psi_term) * x_test + (psi_term + (q / p) / M - 1.0)
                if ti.abs(F) <= 1e-5 and Mi_test < M and Mi_test > 0.1 * M:
                    psi_i = psii_test
                    p_i = pi_test
                    M_i = Mi_test
            elif psii_test < 0.0:
                F = a * x_test**2 + (a + 1.0 + psi_term) * x_test + (psi_term - (q / p) / M + 1.0)
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
        p, q, lode = self.ComputeStressInvariants(stress)
        
        p_i, e = internal_vars[0], internal_vars[1]
        
        M = self.find_M(lode, self.M_tc)
        _, psi_i = self.find_psi_psi_i(self.Gamma, self.Lambda, p, p_i, e)
        M_i = self.find_M_i(M, self.M_tc, self.chi_i, psi_i, self.N)
        
        return self.find_F(p, q, M_i, p_i)

    @ti.func
    def ComputeYieldState(self, stress, internal_vars, material_params):
        f_function = self.ComputeYieldFunction(stress, internal_vars, material_params)
        return f_function > -FTOL, f_function
        
    @ti.func
    def ComputeDfDsigma(self, yield_state, stress, internal_vars, material_params):
        p, q, lode = self.ComputeStressInvariants(stress)
        p_i, e = internal_vars[0], internal_vars[1]
        
        M = self.find_M(lode, self.M_tc)
        _, psi_i = self.find_psi_psi_i(self.Gamma, self.Lambda, p, p_i, e)
        M_i = self.find_M_i(M, self.M_tc, self.chi_i, psi_i, self.N)

        # Compute dF/dp
        dfdp = M_i - q / p
        
        # Compute dF/dtheta
        dM_i_dM = 1. - self.chi_i * self.N * ti.abs(psi_i) / self.M_tc
        dfdtheta = 0.0
        if M_i > Threshold:
            dfdtheta = -(q / M_i) * (3. / 2.) * dM_i_dM * (self.M_tc**2 / (3. + self.M_tc)) * ti.sin(3. * lode / 2. + PI / 4.)
        
        dpdsigma = DpDsigma()
        dqdsigma = DqDsigma(stress)
        dthetadsigma = DlodeDsigma(stress)
        
        # For F = q/p - M_i * (1 - ln(p/p_i)), dF/dq = 1/p
        dfdsigma = dfdp * dpdsigma + 1.0 * dqdsigma + dfdtheta * dthetadsigma
        return dfdsigma
    
    @ti.func
    def ComputeDgDsigma(self, yield_state, stress, internal_vars, material_params):
        # Associative flow rule
        return self.ComputeDfDsigma(yield_state, stress, internal_vars, material_params)
    
    @ti.func
    def ComputePlasticModulus(self, yield_state, dgdsigma, stress, internal_vars, state_vars, material_params):
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
        return ti.Vector([dpi, de, 0.0, 0.0])  # Add zeros for stress and strain updates
    
    @ti.func
    def GetMaterialParameter(self, stress, state_vars):
        # Not used for NorSand elastic modulus; kept for interface compatibility
        return ti.Vector([0.0])
    
    @ti.func
    def GetInternalVariables(self, state_vars):
        # Return p_i, e, and current stress/strain for tracking
        return ti.Vector([state_vars.p_i, state_vars.e, 0.0, 0.0])
    
    @ti.func
    def UpdateInternalVariables(self, np, internal_vars, stateVars):
        stateVars[np].p_i = internal_vars[0]
        stateVars[np].e = internal_vars[1]
        # Stress and strain are updated separately in ExplicitIntegration

    # ---------------- NorSand helpers ----------------
    @ti.func
    def find_M_itc(self, N, chi_i, psi_i, M_tc):
        return M_tc - N * chi_i * ti.abs(psi_i)

    @ti.func
    def find_p_imax(self, chi_i, psi_i, p, M_itc):
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

    # ================ Modified Euler (ME) Integration ================
    @ti.func
    def ExplicitIntegration(self, np, previous_stress, de, dw, stateVars):
        """
        Modified Euler integration following the reference code
        """
        state_vars = stateVars[np]
        
        # Get current state
        p_i = state_vars.p_i
        e = state_vars.e
        stress = previous_stress
        
        # Check elastic factor
        internal_vars = ti.Vector([p_i, e, 0.0, 0.0])
        material_params = self.GetMaterialParameter(stress, state_vars)
        alpha = self.CalculateElasticFactor(np, de, stress, internal_vars, material_params, stateVars)
        
        # Elastic step
        stress = self.ComputeElasticStress(alpha, de, stress, material_params)
        
        # Plastic integration if needed
        if ti.abs(1. - alpha) > Threshold:
            # Use ME driver for plastic integration
            stress_new, p_i_new = self.ME_driver(stress, (1. - alpha) * de, p_i, e, state_vars)
            stress = stress_new
            state_vars.p_i = p_i_new
        
        # Update void ratio based on total strain increment
        deps_v = de[0] + de[1] + de[2]
        de_update = (1.0 + e) * deps_v
        state_vars.e = e - de_update
        
        # Update stress with rotation
        update_stress = stress + self.ComputeSigrotStress(dw, previous_stress)
        
        # Update state variables
        state_vars.stress = update_stress
        # print(update_stress)
        state_vars.strain += de
        
        # Update in global state
        stateVars[np] = state_vars
        
        return update_stress

    @ti.func
    def ME_driver(self, sigma_vec, depsilon, p_i_init, e_init, state_vars):
        """
        Modified Euler with sub-stepping for error control
        Based on ME function from reference code
        """
        # Initialize
        T = 0.0
        dT = 1.0
        sigma0 = sigma_vec
        p_i0 = p_i_init
        e0 = e_init
        
        steps = 0
        max_steps = 100
        converged = 0
        
        # Begin looping until T reaches 1
        while T < 1.0 and steps < max_steps and converged == 0:
            flag_tol = 0
            
            while flag_tol == 0 and steps < max_steps:
                steps += 1
                
                # Compute depsilon for substep increment n
                depsilon_n = dT * depsilon
                
                # Forward Euler (predictor) routine
                dsigma0, dp_i0, _ = self.compute_stress_increment(sigma0, p_i0, e0, depsilon_n, state_vars)
                
                # Compute new sigma1, p_i1 for FE step
                sigma1 = sigma0 + dsigma0
                p_i1 = p_i0 + dp_i0
                
                # Update void ratio
                deps_v = depsilon_n[0] + depsilon_n[1] + depsilon_n[2]
                de = (1 + e0) * deps_v
                e1 = e0 - de
                
                # Modified Euler (predictor-corrector) routine
                dsigma1, dp_i1, _ = self.compute_stress_increment(sigma1, p_i1, e1, depsilon_n, state_vars)
                
                # Compute new sigma2, p_i2 for ME step
                sigma2 = sigma0 + 0.5 * (dsigma0 + dsigma1)
                p_i2 = p_i0 + 0.5 * (dp_i0 + dp_i1)
                e2 = e1
                
                # Compute errors for stress, PIV
                Error_sigma = 0.0
                Error_pi = 0.0
                
                sigma2_norm = self.voigt_norm(sigma2)
                if sigma2_norm > Threshold:
                    Error_sigma = 0.5 * self.voigt_norm(dsigma1 - dsigma0) / sigma2_norm
                
                # Avoid division by zero for Error_pi
                if ti.abs(p_i2) > 1e-10:
                    Error_pi = 0.5 * ti.abs(dp_i1 - dp_i0) / ti.abs(p_i2)
                
                # Compute relative error
                Rn = ti.max(Error_sigma, Error_pi)
                Rn = ti.max(Rn, EPS)
                
                # Update stresses, epsilon_p, psi if successful
                if Rn <= STOL:
                    flag_tol = 1
                    
                    # Check if stresses are on yield surface
                    p2, q2, lode2 = self.ComputeStressInvariants(sigma2)
                    M2 = self.find_M(lode2, self.M_tc)
                    _, psi_i2 = self.find_psi_psi_i(self.Gamma, self.Lambda, p2, p_i2, e2)
                    M_i2 = self.find_M_i(M2, self.M_tc, self.chi_i, psi_i2, self.N)
                    F2 = self.find_F(p2, q2, M_i2, p_i2)
                    
                    # Correct if not on yield surface
                    if ti.abs(F2) > FTOL:
                        sigma2, p_i2 = self.stress_correction(sigma2, p_i2, e2, F2)
                    
                    # Update for next step
                    sigma0 = sigma2
                    p_i0 = p_i2
                    e0 = e0 - de
                    T = T + dT
                    
                    if T == 1.0:
                        converged = 1
                else:
                    # Update dT parameters
                    q = ti.max(0.9 * ti.sqrt(STOL / Rn), 0.1)
                    q = ti.min(q, 1.0)
                    dT = ti.max(q * dT, 1e-6)
                    dT = ti.min(dT, 1.0 - T)
        
        return sigma0, p_i0

    @ti.func
    def compute_stress_increment(self, sigma, p_i, e, depsilon_n, state_vars):
        """
        Compute stress and internal variable increments for a given state
        """
        # Get current invariants
        p, q, lode = self.ComputeStressInvariants(sigma)
        
        # Compute state parameters
        M = self.find_M(lode, self.M_tc)
        psi, psi_i = self.find_psi_psi_i(self.Gamma, self.Lambda, p, p_i, e)
        M_i = self.find_M_i(M, self.M_tc, self.chi_i, psi_i, self.N)
        M_itc = self.find_M_itc(self.N, self.chi_i, psi_i, self.M_tc)
        p_imax = self.find_p_imax(self.chi_i, psi_i, p, M_itc)
        
        # Get derivatives
        internal_vars = ti.Vector([p_i, e, 0.0, 0.0])
        material_params = ti.Vector([0.0])  # Dummy for NorSand
        dfdsigma = self.ComputeDfDsigma(1, sigma, internal_vars, material_params)
        dgdsigma = self.ComputeDgDsigma(1, sigma, internal_vars, material_params)
        
        # Compute elastic properties
        bulk_modulus, shear_modulus = self.ComputeElasticModulus(sigma, material_params)
        
        # Compute C_e @ depsilon_n
        dsig_e = ElasticTensorMultiplyVector(depsilon_n, bulk_modulus, shear_modulus)
        
        # Compute plastic modulus
        plastic_modulus = self.ComputePlasticModulus(1, dgdsigma, sigma, internal_vars, state_vars, material_params)
        
        # Compute denominator
        tempMat = ElasticTensorMultiplyVector(dgdsigma, bulk_modulus, shear_modulus)
        denom = voigt_tensor_dot(dfdsigma, tempMat) - plastic_modulus
        
        # Compute dlambda
        dlambda = 0.0
        if ti.abs(denom) > Threshold:
            dlambda = voigt_tensor_dot(dfdsigma, dsig_e) / denom
            dlambda = ti.max(dlambda, 0.0)
        
        # Compute increments
        dsigma = dsig_e - dlambda * tempMat
        
        # Compute dpi
        H = self.H0 - self.Hy * psi
        e_q = self.deviatoric_projection(sigma, dgdsigma)
        dpideps_pd = H * (M_i / M_itc) * (p_imax - p_i) * (p / p_i)
        depsilon_qp = dlambda * e_q
        dp_i = depsilon_qp * dpideps_pd
        
        return dsigma, dp_i, depsilon_qp

    @ti.func 
    def stress_correction(self, sigma, p_i, e, F):
        """
        Stress correction algorithm based on reference code
        Implements the two-stage correction with internal variable updates
        """
        sigma_0 = sigma
        p_i0 = p_i
        F0 = F
        converged = 0
        
        # Final results to return
        sigma_final = sigma
        p_i_final = p_i
        
        for i in ti.static(range(10)):  # Maximum 10 iterations
            if converged == 0:
                # Get current stress invariants
                p0, q0, lode0 = self.ComputeStressInvariants(sigma_0)
                
                # Compute state-dependent quantities
                M0 = self.find_M(lode0, self.M_tc)
                psi0, psi_i0 = self.find_psi_psi_i(self.Gamma, self.Lambda, p0, p_i0, e)
                M_i0 = self.find_M_i(M0, self.M_tc, self.chi_i, psi_i0, self.N)
                M_itc0 = self.find_M_itc(self.N, self.chi_i, psi_i0, self.M_tc)
                p_imax0 = self.find_p_imax(self.chi_i, psi_i0, p0, M_itc0)
                
                if i == 0:
                    F0 = F  # Use input F for first iteration
                else:
                    F0 = self.find_F(p0, q0, M_i0, p_i0)  # Recompute F for subsequent iterations
                
                # Compute elastic moduli and stiffness
                bulk_modulus, shear_modulus = self.ComputeElasticModulus(sigma_0, ti.Vector([0.0]))
                
                # Get derivatives
                internal_vars = ti.Vector([p_i0, e, 0.0, 0.0])
                material_params = ti.Vector([0.0])
                dfdsigma0 = self.ComputeDfDsigma(1, sigma_0, internal_vars, material_params)
                
                # Compute plastic modulus components
                H = self.H0 - self.Hy * psi0
                e_q = self.deviatoric_projection(sigma_0, dfdsigma0)
                dpideps_pd = H * (M_i0 / M_itc0) * (p_imax0 - p_i0) * (p0 / p_i0)
                dfdepsilon0_dfdsigma0 = self.ComputePlasticModulus(1, dfdsigma0, sigma_0, internal_vars, ti.Vector([0.0]), material_params)
                
                # Compute denominator for full elasto-plastic correction
                tempMat = ElasticTensorMultiplyVector(dfdsigma0, bulk_modulus, shear_modulus)
                dfdsig_C_dfdsig = voigt_tensor_dot(dfdsigma0, tempMat)
                denom = dfdsig_C_dfdsig - dfdepsilon0_dfdsigma0
                
                # Compute plastic multiplier
                del_lambda = 0.0
                if ti.abs(denom) > Threshold:
                    del_lambda = F0 / denom
                
                # Compute dF/dp_i for internal variable update
                # First term: ∂F/∂p_i = -p*M_i/p_i
                parfparpi = -p0 * M_i0 / p_i0
                
                # Second term: ∂F/∂M_i * ∂M_i/∂p_i
                dfdMi = 0.0
                if M_i0 > Threshold:
                    dfdMi = -q0 / M_i0
                
                # Third term: ∂M_i/∂p_i
                psi_sign = 0.0
                if psi_i0 > 0.0:
                    psi_sign = 1.0
                elif psi_i0 < 0.0:
                    psi_sign = -1.0
                dMidpi = -(M0 / self.M_tc) * self.N * self.chi_i * psi_sign * self.Lambda / p_i0
                
                # Combine terms for dF/dp_i
                dfdpi = parfparpi + dfdMi * dMidpi
                
                # Compute B0 ratio for p_i update
                B0 = 0.0
                if ti.abs(dfdpi) > Threshold:
                    B0 = dfdepsilon0_dfdsigma0 / dfdpi
                
                # Update internal variables (p_i correction)
                p_i1 = p_i0 + 20.0 * del_lambda * B0  # Factor of 20 from reference
                
                # Stress remains unchanged in this implementation (sigma_1 = sigma_0)
                sigma_1 = sigma_0
                
                # Check convergence with updated p_i
                F1 = self.find_F(p0, q0, M_i0, p_i1)
                
                if ti.abs(F1) <= FTOL:
                    converged = 1
                    sigma_final = sigma_1
                    p_i_final = p_i1
                
                # Fallback correction if primary method diverges and not yet converged
                if converged == 0 and ti.abs(F1) > ti.abs(F0):
                    # Simple Newton correction as fallback
                    dfdsigmadfdsigma = voigt_tensor_dot(dfdsigma0, dfdsigma0)
                    if ti.abs(dfdsigmadfdsigma) > Threshold:
                        del_lambda_fallback = F0 / dfdsigmadfdsigma
                        sigma_1 = sigma_0 - del_lambda_fallback * dfdsigma0
                        p_i1 = p_i0  # Don't update p_i in fallback
                        
                        # Check fallback convergence
                        p1, q1, lode1 = self.ComputeStressInvariants(sigma_1)
                        M1 = self.find_M(lode1, self.M_tc)
                        _, psi_i1 = self.find_psi_psi_i(self.Gamma, self.Lambda, p1, p_i1, e)
                        M_i1 = self.find_M_i(M1, self.M_tc, self.chi_i, psi_i1, self.N)
                        F1 = self.find_F(p1, q1, M_i1, p_i1)
                        
                        if ti.abs(F1) <= FTOL:
                            converged = 1
                            sigma_final = sigma_1
                            p_i_final = p_i1
                
                # Update for next iteration if not converged
                if converged == 0:
                    sigma_0 = sigma_1
                    p_i0 = p_i1
                    sigma_final = sigma_1
                    p_i_final = p_i1
        
        # Return final values
        return sigma_final, p_i_final

    @ti.func
    def voigt_norm(self, vec):
        """
        Compute L2 norm of Voigt vector with proper scaling
        """
        scaled_vec = vec
        # Shear components scaled by sqrt(2)
        norm_sq = vec[0]**2 + vec[1]**2 + vec[2]**2 + 2.0*(vec[3]**2 + vec[4]**2 + vec[5]**2)
        return ti.sqrt(norm_sq)