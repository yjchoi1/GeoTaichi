import taichi as ti


# NorSand utility functions implemented in Taichi, converted from ref_NorSand/norsand_utils.py


@ti.func
def map_stress_comp_neg_to_pos(stress):
    # Map external convention (compression negative) to internal (compression positive)
    s = ti.Vector.zero(float, 6)
    for i in ti.static(range(3)):
        s[i] = -stress[i]
    for i in ti.static(range(3, 6)):
        s[i] = stress[i]
    return s


@ti.func
def map_stress_comp_pos_to_neg(stress):
    # Map internal convention (compression positive) back to external (compression negative)
    s = ti.Vector.zero(float, 6)
    for i in ti.static(range(3)):
        s[i] = -stress[i]
    for i in ti.static(range(3, 6)):
        s[i] = stress[i]
    return s


@ti.func
def map_strain_comp_neg_to_pos(strain):
    # Map external strain (compression negative volumetric) to internal (compression positive)
    e = ti.Vector.zero(float, 6)
    for i in ti.static(range(3)):
        e[i] = -strain[i]
    for i in ti.static(range(3, 6)):
        e[i] = strain[i]
    return e


@ti.func
def getPQ(stress):
    # Returns p, q for a Voigt stress vector (6,)
    s = ti.Vector.zero(float, 6)
    for i in ti.static(range(6)):
        s[i] = stress[i]
    # Tension cutoff on normal stresses
    for i in ti.static(range(3)):
        if s[i] < 0.1:
            s[i] = 0.1
    p = (s[0] + s[1] + s[2]) / 3.0
    dq = 0.5 * (
        (s[0] - s[1]) * (s[0] - s[1])
        + (s[1] - s[2]) * (s[1] - s[2])
        + (s[2] - s[0]) * (s[2] - s[0])
        + 6.0 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5])
    )
    q = ti.sqrt(ti.max(dq, 0.0))
    return p, q


@ti.func
def getInvariants(stress):
    # Returns J2, J3, dJ2dsigma(6,), dJ3dsigma(6,)
    p, q = getPQ(stress)
    one = ti.Vector([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    s = stress - p * one

    J2 = (
        (1.0 / 6.0)
        * (
            (stress[0] - stress[1]) * (stress[0] - stress[1])
            + (stress[1] - stress[2]) * (stress[1] - stress[2])
            + (stress[2] - stress[0]) * (stress[2] - stress[0])
        )
        + stress[3] * stress[3]
        + stress[4] * stress[4]
        + stress[5] * stress[5]
    )

    J3 = (
        s[0] * s[1] * s[2]
        - s[0] * s[5] * s[5]
        - s[1] * s[4] * s[4]
        - s[2] * s[3] * s[3]
        + 2.0 * s[3] * s[5] * s[4]
    )

    dJ2dsigma = ti.Vector.zero(float, 6)
    dJ2dsigma[0] = s[0]
    dJ2dsigma[1] = s[1]
    dJ2dsigma[2] = s[2]
    dJ2dsigma[3] = 2.0 * stress[3]
    dJ2dsigma[4] = 2.0 * stress[4]
    dJ2dsigma[5] = 2.0 * stress[5]

    dJ3dsigma = ti.Vector.zero(float, 6)
    dJ3dsigma[0] = (
        -1.0 / 3.0 * s[0] * s[1]
        - 1.0 / 3.0 * s[0] * s[2]
        + 2.0 / 3.0 * s[1] * s[2]
        - 2.0 / 3.0 * s[5] * s[5]
        + 1.0 / 3.0 * s[4] * s[4]
        + 1.0 / 3.0 * s[3] * s[3]
    )
    dJ3dsigma[1] = (
        -1.0 / 3.0 * s[0] * s[1]
        + 2.0 / 3.0 * s[0] * s[2]
        - 1.0 / 3.0 * s[1] * s[2]
        + 1.0 / 3.0 * s[5] * s[5]
        - 2.0 / 3.0 * s[4] * s[4]
        + 1.0 / 3.0 * s[3] * s[3]
    )
    dJ3dsigma[2] = (
        2.0 / 3.0 * s[0] * s[1]
        - 1.0 / 3.0 * s[0] * s[2]
        - 1.0 / 3.0 * s[1] * s[2]
        + 1.0 / 3.0 * s[5] * s[5]
        + 1.0 / 3.0 * s[4] * s[4]
        - 2.0 / 3.0 * s[3] * s[3]
    )
    dJ3dsigma[3] = -2.0 * s[2] * s[3] + 2.0 * s[5] * s[4]
    dJ3dsigma[4] = -2.0 * s[1] * s[4] + 2.0 * s[3] * s[5]
    dJ3dsigma[5] = -2.0 * s[0] * s[5] + 2.0 * s[3] * s[4]

    return J2, J3, dJ2dsigma, dJ3dsigma


@ti.func
def getLodeM(stress, M_tc):
    # Returns theta, M(lode)
    J2, J3, dJ2dsigma, dJ3dsigma = getInvariants(stress)
    J2J3_term = 0.0
    if J2 != 0.0:
        J2J3_term = J3 / ti.sqrt(J2 * J2 * J2)
    sin3theta = 3.0 * ti.sqrt(3.0) / 2.0 * J2J3_term
    if sin3theta > 0.99:
        sin3theta = 1.0
    if sin3theta < -0.99:
        sin3theta = -1.0
    theta = (1.0 / 3.0) * ti.asin(sin3theta)
    if theta > ti.math.pi / 6.0:
        theta = ti.math.pi / 6.0
    if theta < -ti.math.pi / 6.0:
        theta = -ti.math.pi / 6.0
    c = M_tc / (3.0 + M_tc)
    g = 1.0 - (c * ti.cos(1.5 * theta + ti.math.pi / 4.0))
    M = M_tc * g
    return theta, M


@ti.func
def getC_e(G, K):
    # Elastic tangent (6x6)
    C_e = ti.Matrix.zero(float, 6, 6)
    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            if i == j:
                C_e[i, j] = K + (4.0 / 3.0) * G
            else:
                C_e[i, j] = K - (2.0 / 3.0) * G
    for i in ti.static(range(3, 6)):
        C_e[i, i] = G
    return C_e


@ti.func
def getDev(vec, stress):
    # Deviatoric projection of any Voigt vector given stress
    p, q = getPQ(stress)
    Dev = 0.0
    if q > 1e-10:
        Dev = ((stress[0] - p) / q) * vec[0]
        Dev += ((stress[1] - p) / q) * vec[1]
        Dev += ((stress[2] - p) / q) * vec[2]
        Dev += (2.0 * stress[3] / q) * vec[3]
        Dev += (2.0 * stress[4] / q) * vec[4]
        Dev += (2.0 * stress[5] / q) * vec[5]
    return Dev


@ti.func
def getF(p, q, p_i, M_i):
    # Yield function value
    return (q / p) - M_i + M_i * ti.log(p / p_i)


@ti.func
def updateM_i(e, N, M, M_tc, chi_tc, Gamma, Lambda, p, p_i):
    # Update M_i based on state
    chi_i = chi_tc / (1.0 - Lambda * chi_tc / M_tc)
    e_c = Gamma
    if p > 1.0:
        e_c = Gamma - Lambda * ti.log(p)
    psi = e - e_c
    psi_i = psi + Lambda * ti.log(p_i)  # default branch
    if p > 1.0:
        psi_i = psi + Lambda * ti.log(p_i / p)
    M_i = M * (1.0 - chi_i * N * ti.abs(psi_i) / M_tc)
    return M_i


@ti.func
def getp_imax(e, N, M, M_tc, chi_tc, Gamma, Lambda, p, p_i):
    # Returns p_imax, psi_i, chi_i, M_itc
    chi_i = chi_tc / (1.0 - Lambda * chi_tc / M_tc)
    e_c = Gamma
    if p > 1.0:
        e_c = Gamma - Lambda * ti.log(p)
    psi = e - e_c
    psi_i = psi + Lambda * ti.log(p)
    if p > 1.0:
        psi_i = psi + Lambda * ti.log(p_i / p)
    M_i = M * (1.0 - chi_i * N * ti.abs(psi_i) / M_tc)
    M_itc = M_tc * (1.0 - chi_i * N * ti.abs(psi_i) / M_tc)
    p_imax = p * ti.exp(-chi_i * psi_i / M_itc)
    return p_imax, psi_i, chi_i, M_itc


@ti.func
def getp_cap(e, N, M, M_tc, chi_tc, Gamma, Lambda, p, p_i):
    # Returns p_cap
    chi_i = chi_tc / (1.0 - Lambda * chi_tc / M_tc)
    e_c = Gamma
    if p > 1.0:
        e_c = Gamma - Lambda * ti.log(p)
    psi = e - e_c
    psi_i = psi + Lambda * ti.log(p)
    if p > 1.0:
        psi_i = psi + Lambda * ti.log(p_i / p)
    M_itc = M_tc * (1.0 - chi_i * N * ti.abs(psi_i) / M_tc)
    p_cap = p_i * ti.exp(chi_i * psi_i / M_itc)
    return p_cap


@ti.func
def findp_ipsi_iM_i(e, N, M, M_tc, chi_tc, Gamma, Lambda, p, q):
    # Solve initial p_i, psi_i, M_i using quadratic roots
    chi_i = chi_tc / (1.0 - Lambda * chi_tc / M_tc)
    e_c = Gamma
    if p > 1.0:
        e_c = Gamma - Lambda * ti.log(p)
    psi = e - e_c

    a = N * chi_i * Lambda / M_tc
    psi_term = N * chi_i * psi / M_tc
    b1 = a - 1.0 + psi_term
    b2 = a + 1.0 + psi_term
    c1 = psi_term + (q / p) / M - 1.0
    c2 = psi_term - (q / p) / M + 1.0

    disc1 = ti.max(b1 * b1 - 4.0 * a * c1, 0.0)
    disc2 = ti.max(b2 * b2 - 4.0 * a * c2, 0.0)
    x1a = (-b1 + ti.sqrt(disc1)) / (2.0 * a)
    x1b = (-b1 - ti.sqrt(disc1)) / (2.0 * a)
    x2a = (-b2 + ti.sqrt(disc2)) / (2.0 * a)
    x2b = (-b2 - ti.sqrt(disc2)) / (2.0 * a)

    # Initialize outputs
    p_i_out = 0.0
    psii_out = 0.0
    Mi_out = 0.0

    for k in ti.static(range(4)):
        x_test = 0.0
        if k == 0:
            x_test = x1a
        elif k == 1:
            x_test = x1b
        elif k == 2:
            x_test = x2a
        else:
            x_test = x2b

        psii_test = x_test * Lambda + psi
        pi_test = p * ti.exp(x_test)
        Mi_test = M * (1.0 - N * chi_i * ti.abs(Lambda * x_test + psi) / M_tc)

        # Evaluate yield function residuals per sign of psii_test
        if psii_test > 0.0:
            F = a * x_test * x_test + (a - 1.0 + psi_term) * x_test + (psi_term + (q / p) / M - 1.0)
            if ti.abs(F) <= 1e-6 and Mi_test < M and Mi_test > 0.1 * M:
                psii_out = psii_test
                p_i_out = pi_test
                Mi_out = Mi_test
        elif psii_test < 0.0:
            F = a * x_test * x_test + (a + 1.0 + psi_term) * x_test + (psi_term - (q / p) / M + 1.0)
            if ti.abs(F) <= 1e-6 and Mi_test < M and Mi_test > 0.1 * M:
                psii_out = psii_test
                p_i_out = pi_test
                Mi_out = Mi_test

    return p_i_out, psii_out, Mi_out


@ti.func
def getdfdsigma(e, stress, p_i, N, M_tc, chi_tc, Gamma, Lambda, M_i):
    # df/dsigma vector (6,)
    p, q = getPQ(stress)
    J2, J3, dJ2dsigma, dJ3dsigma = getInvariants(stress)
    theta, M = getLodeM(stress, M_tc)
    p_imax, psi_i, chi_i, M_itc = getp_imax(e, N, M, M_tc, chi_tc, Gamma, Lambda, p, p_i)

    dpdsig = ti.Vector([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0, 0.0, 0.0])
    dqdsig = ti.Vector.zero(float, 6)
    for i in ti.static(range(3)):
        if q == 0.0:
            dqdsig[i] = 0.0
        else:
            dqdsig[i] = (3.0 / (2.0 * q)) * (stress[i] - p)
    for i in ti.static(range(3, 6)):
        if q < 1e-10:
            dqdsig[i] = 0.0
        else:
            dqdsig[i] = (3.0 / q) * stress[i]

    dthetadsig = ti.Vector.zero(float, 6)
    if ti.sqrt(J2) >= 0.01:
        if ti.abs(ti.cos(3.0 * theta)) >= 1e-4:
            coeff = ti.sqrt(3.0) / (2.0 * ti.cos(3.0 * theta) * ti.sqrt(J2 * J2 * J2))
            for i in ti.static(range(6)):
                dthetadsig[i] = coeff * (dJ3dsigma[i] - (1.5) * (J3 / J2) * dJ2dsigma[i])

    dfdp = M_i - q / p
    dfdq = 1.0
    dfdMi = -q / M_i
    dMidtheta = (1.5 * M_tc * M_tc / (3.0 + M_tc)) * ti.sin(1.5 * theta + ti.math.pi / 4.0)

    dfdsigma = ti.Vector.zero(float, 6)
    for i in ti.static(range(6)):
        dfdsigma[i] = dfdp * dpdsig[i] + dfdq * dqdsig[i] + dfdMi * dMidtheta * dthetadsig[i]
    return dfdsigma


@ti.func
def getdfdepsq(e, stress, p_i, N, H, Hy, M_tc, chi_tc, Gamma, Lambda, M_i):
    # Returns dfdpi, dpidepsq
    p, q = getPQ(stress)
    J2, J3, dJ2dsigma, dJ3dsigma = getInvariants(stress)
    theta, M = getLodeM(stress, M_tc)
    p_imax, psi_i, chi_i, M_itc = getp_imax(e, N, M, M_tc, chi_tc, Gamma, Lambda, p, p_i)

    dfdMi = -q / M_i
    dfdpi = -M_i * p / p_i
    dMidpi = 0.0
    if psi_i > 0.0:
        dMidpi = -M * chi_i * N / M_tc * Lambda / p_i
    else:
        dMidpi = M * chi_i * N / M_tc * Lambda / p_i
    dfdpi += dfdMi * dMidpi

    psi = e - Gamma + Lambda * ti.log(p)
    dpidepsq = (H - (Hy * psi)) * M_i / M_itc * (p_imax - p_i) * (p / p_i)
    return dfdpi, dpidepsq


@ti.func
def getC_p(e, stress, p_i, N, H, Hy, M_i, M_tc, chi_tc, Gamma, Lambda, C_e):
    # Plastic tangent and auxiliary quantities
    dfdsigma = getdfdsigma(e, stress, p_i, N, M_tc, chi_tc, Gamma, Lambda, M_i)
    dfdpi, dpidepsq = getdfdepsq(e, stress, p_i, N, H, Hy, M_tc, chi_tc, Gamma, Lambda, M_i)
    dfdq_ = getDev(dfdsigma, stress)

    # Ce_dfdsigma = (C_e^T) @ dfdsigma (component form as in reference)
    Ce_dfdsigma = ti.Vector.zero(float, 6)
    for i in ti.static(range(6)):
        Ce_dfdsigma[i] = 0.0
        for j in ti.static(range(6)):
            Ce_dfdsigma[i] += dfdsigma[j] * C_e[j, i]

    dfdsigma_Ce_dfdsigma = 0.0
    for i in ti.static(range(6)):
        dfdsigma_Ce_dfdsigma += dfdsigma[i] * Ce_dfdsigma[i]

    denominator = dfdsigma_Ce_dfdsigma - dfdpi * dpidepsq * dfdq_

    # De_dfdsigma = C_e @ dfdsigma
    De_dfdsigma = ti.Vector.zero(float, 6)
    for i in ti.static(range(6)):
        De_dfdsigma[i] = 0.0
        for j in ti.static(range(6)):
            De_dfdsigma[i] += C_e[i, j] * dfdsigma[j]

    # (De_dfdsigma outer dfdsigma) / denominator
    De_dfdsigma_dfdsigma = ti.Matrix.zero(float, 6, 6)
    for i in ti.static(range(6)):
        for j in ti.static(range(6)):
            De_dfdsigma_dfdsigma[i, j] = (De_dfdsigma[i] * dfdsigma[j]) / denominator

    # Dp = (De_dfdsigma_dfdsigma) @ C_e
    Dp = ti.Matrix.zero(float, 6, 6)
    for i in ti.static(range(6)):
        for j in ti.static(range(6)):
            val = 0.0
            for z in ti.static(range(6)):
                val += De_dfdsigma_dfdsigma[i, z] * C_e[z, j]
            Dp[i, j] = val

    C_p = ti.Matrix.zero(float, 6, 6)
    for i in ti.static(range(6)):
        for j in ti.static(range(6)):
            C_p[i, j] = C_e[i, j] - Dp[i, j]
    return C_p, dfdq_, denominator, Ce_dfdsigma


@ti.func
def getElastoPlasticComponents(e, stress, p_i, N, H, Hy, M_i, M_tc, chi_tc, Gamma, Lambda, C_e, subdstrain):
    # Returns dp_i(scalar), deps_p(6,), deps_pD(scalar), dlambda_p(scalar), C_p(6x6)
    dfdsigma = getdfdsigma(e, stress, p_i, N, M_tc, chi_tc, Gamma, Lambda, M_i)
    dfdpi, dpidepsq = getdfdepsq(e, stress, p_i, N, H, Hy, M_tc, chi_tc, Gamma, Lambda, M_i)
    C_p, dfdq, denominator, Ce_dfdsigma = getC_p(e, stress, p_i, N, H, Hy, M_i, M_tc, chi_tc, Gamma, Lambda, C_e)

    # dstress = C_e @ subdstrain
    dstress = ti.Vector.zero(float, 6)
    for i in ti.static(range(6)):
        dstress[i] = 0.0
        for j in ti.static(range(6)):
            dstress[i] += C_e[i, j] * subdstrain[j]

    # dlambda_p = (dfdsigma . dstress) / denominator
    dlambda_p = 0.0
    for i in ti.static(range(6)):
        dlambda_p += dfdsigma[i] * dstress[i]
    dlambda_p = dlambda_p / denominator

    deps_p = dlambda_p * dfdsigma
    deps_pD = getDev(deps_p, stress)
    dp_i_val = deps_pD * dpidepsq

    return dp_i_val, deps_p, deps_pD, dlambda_p, C_p


@ti.func
def StressCorrection(e, stress, p_i, N, H, Hy, M_i, M_tc, chi_tc, Gamma, Lambda, C_e, F):
    # Returns corrected (stress, p_i)
    dfdsigma = getdfdsigma(e, stress, p_i, N, M_tc, chi_tc, Gamma, Lambda, M_i)
    dfdpi, dpidepsq = getdfdepsq(e, stress, p_i, N, H, Hy, M_tc, chi_tc, Gamma, Lambda, M_i)
    C_p, dfdq, denominator, Ce_dfdsigma = getC_p(e, stress, p_i, N, H, Hy, M_i, M_tc, chi_tc, Gamma, Lambda, C_e)

    del_lambda = F / denominator
    B0 = dpidepsq * dfdq

    stress1 = ti.Vector.zero(float, 6)
    for i in ti.static(range(6)):
        stress1[i] = stress[i] - del_lambda * Ce_dfdsigma[i]
    p_i1 = p_i + del_lambda * B0

    p1, q1 = getPQ(stress1)
    theta1, M1 = getLodeM(stress1, M_tc)
    M_i1 = updateM_i(e, N, M1, M_tc, chi_tc, Gamma, Lambda, p1, p_i1)
    F1 = getF(p1, q1, p_i1, M_i1)

    if ti.abs(F1) > ti.abs(F):
        denom = 0.0
        for i in ti.static(range(6)):
            denom += dfdsigma[i] * dfdsigma[i]
        if denom != 0.0:
            del_lambda = F / denom
        else:
            del_lambda = 0.0
        for i in ti.static(range(6)):
            stress1[i] = stress[i] - del_lambda * dfdsigma[i]
        p_i1 = p_i

    return stress1, p_i1


@ti.func
def errorEuler(sig_init, dsig1, dsig2, p_iinit, dpi1, dpi2, dT_old):
    # Error estimation for Modified Euler substep control
    STOL = 1e-4
    dT_min = 1e-6

    sigma = ti.Vector.zero(float, 6)
    E = ti.Vector.zero(float, 6)
    EnorSqrt = 0.0
    SigNorSqrt = 0.0
    for i in ti.static(range(6)):
        E[i] = 0.5 * (-dsig1[i] + dsig2[i])
        EnorSqrt += E[i] * E[i]
        sigma[i] = sig_init[i] + 0.5 * (dsig1[i] + dsig2[i])
        SigNorSqrt += sigma[i] * sigma[i]

    denom_sig = ti.sqrt(SigNorSqrt)
    if denom_sig == 0.0:
        denom_sig = 1.0
    error1 = 0.5 * (ti.sqrt(EnorSqrt) / denom_sig)

    p_i_mid = p_iinit + 0.5 * (dpi1 + dpi2)
    denom_pi = ti.abs(p_i_mid)
    if denom_pi == 0.0:
        denom_pi = 1.0
    error2 = 0.5 * (ti.abs(dpi2 - dpi1) / denom_pi)
    error = ti.max(error1, error2)

    reductionFactor = 1.0
    dT = dT_old
    iflag_sub = 0
    if error > STOL:
        reductionFactor = 0.9 * ti.sqrt(STOL / error)
        if reductionFactor < 0.1:
            reductionFactor = 0.1
        if reductionFactor > 2.0:
            reductionFactor = 2.0
        dT = reductionFactor * dT_old
        if dT < dT_min:
            dT = dT_min
        iflag_sub = 1
    else:
        reductionFactor = ti.min(0.9 * ti.sqrt(STOL / error), 1.1)
        if reductionFactor < 0.1:
            reductionFactor = 0.1
        if reductionFactor > 2.0:
            reductionFactor = 2.0
        dT = reductionFactor * dT_old
        if dT < dT_min:
            dT = dT_min
        iflag_sub = 0

    return sigma, p_i_mid, dT, iflag_sub


@ti.func
def Pegasus(F0, F1, alpha0, alpha1, dsig_tr, p_i, M_i, C_e, subdStrain_old, stress_old, e_old):
    # Recover strains past YS, void ratio using Pegasus algorithm.
    Tol = 1e-6
    iteration = 0

    dElastStress = ti.Vector.zero(float, 6)
    dElastStrain = ti.Vector.zero(float, 6)
    subdStrain = ti.Vector.zero(float, 6)
    stress = ti.Vector.zero(float, 6)

    SigNew = ti.Vector.zero(float, 6)
    alpha = 0.0
    while iteration < 11:
        # Secant/Pegasus update
        alpha = alpha1 - F1 * (alpha1 - alpha0) / (F1 - F0)

        for i in ti.static(range(6)):
            SigNew[i] = stress[i] + alpha * dsig_tr[i]

        p_new, q_new = getPQ(SigNew)
        Fnew = getF(p_new, q_new, p_i, M_i)

        if Fnew <= Tol:
            break

        if Fnew * F0 < 0.0:
            alpha1 = alpha0
            F1 = F0
        else:
            F1 = F1 * F0 / (F0 + Fnew)

        alpha0 = alpha
        F0 = Fnew
        iteration = iteration + 1

    for i in ti.static(range(6)):
        dElastStrain[i] = alpha * subdStrain_old[i]
        subdStrain[i] = (1.0 - alpha) * subdStrain_old[i]

    # dElastStress = C_e @ dElastStrain
    for i in ti.static(range(6)):
        dElastStress[i] = 0.0
        for j in ti.static(range(6)):
            dElastStress[i] += C_e[i, j] * dElastStrain[j]

    for i in ti.static(range(6)):
        stress[i] = stress_old[i] + dElastStress[i]

    VOL = dElastStrain[0] + dElastStrain[1] + dElastStrain[2]
    e = e_old - VOL * (1.0 + e_old)

    return subdStrain, stress, e


@ti.func
def UMAT(e, stress, p_i, N, H, Hy, M_i, M_tc, chi_tc, Gamma, Lambda, dstrain, Gmax, Gexp, nu):
    # Update stress and state variables using the NorSand constitutive model.
    FTOL = 1e-6
    ITER = 10

    # Map external compression-negative convention to internal compression-positive
    stress = map_stress_comp_neg_to_pos(stress)
    dstrain = map_strain_comp_neg_to_pos(dstrain)

    # Cap shear stresses
    s = ti.Vector.zero(float, 6)
    for i in ti.static(range(6)):
        s[i] = stress[i]
    for i in ti.static(range(3, 6)):
        if s[i] < 0.1:
            s[i] = 0.0

    # overwrite local stress
    for i in ti.static(range(6)):
        stress[i] = s[i]

    p, q = getPQ(stress)

    G = Gmax * (p / 100.0) ** Gexp
    K = (2.0 * (1.0 + nu)) / (3.0 * (1.0 - 2.0 * nu)) * G
    C_e = getC_e(G, K)

    # dstressTrial = C_e @ dstrain
    dstressTrial = ti.Vector.zero(float, 6)
    for i in ti.static(range(6)):
        dstressTrial[i] = 0.0
        for j in ti.static(range(6)):
            dstressTrial[i] += C_e[i, j] * dstrain[j]
    stressTrial = stress + dstressTrial
    pTrial, qTrial = getPQ(stressTrial)
    FTrial = getF(pTrial, qTrial, p_i, M_i)
    F = getF(p, q, p_i, M_i)
    
    # print("NorSand UMAT State:")
    # print("  p: ", p, ", q: ", q)
    # print("  stress : ", stress)
    # print("  dstrain: ", dstrain)
    # print("  FTrial: ", FTrial, ", F: ", F)
    # print("f  p_i: ", p_i, ", M_i: ", M_i, ", e: ", e)

    if FTrial <= FTOL:
        depsV = dstrain[0] + dstrain[1] + dstrain[2]
        e = e - depsV * (1.0 + e)
        stress = stressTrial
        iflag_yield = 0
        # depsP (unused downstream here)
        p = pTrial
        q = qTrial
        theta, M = getLodeM(stress, M_tc)
        M_i = updateM_i(e, N, M, M_tc, chi_tc, Gamma, Lambda, p, p_i)
    else:
        if F < -FTOL:
            alpha0 = 0.0
            alpha1 = 1.0
            subdStrain, stress, e = Pegasus(F, FTrial, alpha0, alpha1, dstressTrial, p_i, M_i, C_e, dstrain, stress, e)
            dstrain = subdStrain
        elif ti.abs(F) < FTOL:
            pass  # elastoplastic unloading
        else:
            pass  # illegal stress state

        # Modified Euler Algorithm
        iflag_yield = 1
        T = 0.0
        dT = 1.0
        # substeps counter (unused externally)
        p, q = getPQ(stress)

        while T < 1.0:
            subdstrain = dT * dstrain
            iflag = 1
            # Initialize placeholders for outputs updated inside sub-iteration
            e1 = e
            stress2 = ti.Vector.zero(float, 6)
            for i in ti.static(range(6)):
                stress2[i] = stress[i]
            p_i2 = p_i
            while iflag == 1:
                subdepsV = subdstrain[0] + subdstrain[1] + subdstrain[2]
                e1 = e - subdepsV * (1.0 + e)

                theta, M = getLodeM(stress, M_tc)
                M_i = updateM_i(e1, N, M, M_tc, chi_tc, Gamma, Lambda, p, p_i)

                G = Gmax * (p / 100.0) ** Gexp
                K = (2.0 * (1.0 + nu)) / (3.0 * (1.0 - 2.0 * nu)) * G
                C_e = getC_e(G, K)

                dp_i1, deps_p1, deps_pD1, dlambda_p1, C_p1 = getElastoPlasticComponents(
                    e1, stress, p_i, N, H, Hy, M_i, M_tc, chi_tc, Gamma, Lambda, C_e, subdstrain)

                # dStress1 = C_p1 @ subdstrain
                dStress1 = ti.Vector.zero(float, 6)
                for i in ti.static(range(6)):
                    dStress1[i] = 0.0
                    for j in ti.static(range(6)):
                        dStress1[i] += C_p1[i, j] * subdstrain[j]

                # First Euler update
                stress1 = stress + dStress1
                p_i1 = p_i + dp_i1

                p1, q1 = getPQ(stress1)
                theta1, M1 = getLodeM(stress1, M_tc)
                M_i1 = updateM_i(e1, N, M1, M_tc, chi_tc, Gamma, Lambda, p1, p_i1)
                G = Gmax * (p1 / 100.0) ** Gexp
                K = (2.0 * (1.0 + nu)) / (3.0 * (1.0 - 2.0 * nu)) * G

                C_e = getC_e(G, K)
                dp_i2, deps_p2, deps_pD2, dlambda_p2, C_p2 = getElastoPlasticComponents(
                    e1, stress1, p_i1, N, H, Hy, M_i1, M_tc, chi_tc, Gamma, Lambda, C_e, subdstrain)
                # dStress2 = C_p2 @ subdstrain
                dStress2 = ti.Vector.zero(float, 6)
                for i in ti.static(range(6)):
                    dStress2[i] = 0.0
                    for j in ti.static(range(6)):
                        dStress2[i] += C_p2[i, j] * subdstrain[j]

                stress2, p_i2, dT, iflag = errorEuler(stress, dStress1, dStress2, p_i, dp_i1, dp_i2, dT)
                if iflag == 1:
                    subdstrain = dT * subdstrain
                else:
                    iflag = 0

            stress = stress2
            p_i = p_i2
            e = e1

            # Update stiffness at current p
            G = Gmax * (p / 100.0) ** Gexp
            K = (2.0 * (1.0 + nu)) / (3.0 * (1.0 - 2.0 * nu)) * G
            C_e = getC_e(G, K)

            p, q = getPQ(stress)
            theta, M = getLodeM(stress, M_tc)
            M_i = updateM_i(e, N, M, M_tc, chi_tc, Gamma, Lambda, p, p_i)
            F = getF(p, q, p_i, M_i)

            icount = 0
            while ti.abs(F) > FTOL and icount <= ITER:
                icount += 1
                stress, p_i = StressCorrection(e, stress, p_i, N, H, Hy, M_i, M_tc, chi_tc, Gamma, Lambda, C_e, F)

                p, q = getPQ(stress)
                G = Gmax * (p / 100.0) ** Gexp
                K = (2.0 * (1.0 + nu)) / (3.0 * (1.0 - 2.0 * nu)) * G
                C_e = getC_e(G, K)
                theta, M = getLodeM(stress, M_tc)
                M_i = updateM_i(e, N, M, M_tc, chi_tc, Gamma, Lambda, p, p_i)

                F = getF(p, q, p_i, M_i)

            if T + dT > 1.0:
                dT = 1.0 - T
            T = T + dT

    stress_new = stress
    p_inew = p_i
    M_inew = M_i
    e_new = e

    # Map back to external compression-negative convention
    stress_new = map_stress_comp_pos_to_neg(stress_new)
    return stress_new, p_inew, M_inew, e_new


@ti.func
def calculate_strain_increment_eng(velocity_gradient, dt):
    # Engineering shear convention: gamma_ij = (du_i/dx_j + du_j/dx_i)
    de = ti.Vector.zero(float, 6)
    dts = dt[None]
    de[0] = velocity_gradient[0, 0] * dts
    de[1] = velocity_gradient[1, 1] * dts
    de[2] = velocity_gradient[2, 2] * dts
    de[3] = (velocity_gradient[0, 1] + velocity_gradient[1, 0]) * dts  # gamma12
    de[4] = (velocity_gradient[1, 2] + velocity_gradient[2, 1]) * dts  # gamma23
    de[5] = (velocity_gradient[0, 2] + velocity_gradient[2, 0]) * dts  # gamma13
    return de

@ti.func
def calculate_strain_increment2D_eng(velocity_gradient, dt):
    # Plane strain, engineering shear convention
    de = ti.Vector.zero(float, 6)
    dts = dt[None]
    de[0] = velocity_gradient[0, 0] * dts
    de[1] = velocity_gradient[1, 1] * dts
    de[2] = 0.0
    de[3] = (velocity_gradient[0, 1] + velocity_gradient[1, 0]) * dts  # gamma12
    de[4] = 0.0
    de[5] = 0.0
    return de
