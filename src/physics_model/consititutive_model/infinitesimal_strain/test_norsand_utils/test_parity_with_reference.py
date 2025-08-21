import os
import sys
import numpy as np
import pytest
import taichi as ti

# Ensure project root is in sys.path when running this file directly
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../"))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.physics_model.consititutive_model.infinitesimal_strain.MaterialKernalNorsand import (
    getPQ,
    getInvariants,
    getLodeM,
    getC_e,
    getDev,
    getF,
    updateM_i,
    getp_imax,
    getp_cap,
    getdfdsigma,
    getdfdepsq,
    UMAT,
    Pegasus,
)
from src.physics_model.consititutive_model.infinitesimal_strain.ref_NorSand import norsand_utils as ref


ti.init(arch=ti.cpu, default_fp=ti.f64)


# Stress states for testing different loading conditions
STRESS_STATES = [
    # Pure hydrostatic pressure (no shear)
    ("hydrostatic", np.array([100.0, 100.0, 100.0, 0.0, 0.0, 0.0], dtype=np.float64)),
    
    # Combined pressure and shear stress
    ("deviatoric_shear", np.array([120.0, 130.0, 110.0, 6.0, -3.0, 4.5], dtype=np.float64)),
    
    # Triaxial compression test condition
    ("triaxial_compression", np.array([200.0, 100.0, 50.0, 20.0, 0.0, -10.0], dtype=np.float64)),
    
    # Triaxial extension test condition
    ("triaxial_extension", np.array([50.0, 100.0, 200.0, -20.0, 0.0, 10.0], dtype=np.float64)),
    
    # Plane strain condition (common in geotechnical problems)
    ("plane_strain", np.array([150.0, 140.0, 160.0, -5.0, 7.5, -2.0], dtype=np.float64)),
    
    # Very low stress levels (near surface conditions)
    ("low_stress", np.array([0.05, 0.1, 0.2, 0.0, 0.0, 0.0], dtype=np.float64)),
    
    # High stress levels (deep underground conditions)
    ("high_stress", np.array([500.0, 450.0, 400.0, 30.0, -15.0, 25.0], dtype=np.float64)),
    
    # Pure shear loading (constant mean stress)
    ("pure_shear", np.array([100.0, 100.0, 100.0, 25.0, 0.0, 0.0], dtype=np.float64)),
]

# Vector inputs for deviatoric projections
VECTOR_INPUTS = [
    # Unit-like vectors for testing normalization
    ("unit_vectors", np.array([1.0, 0.0, -1.0, 0.2, 0.0, -0.2], dtype=np.float64)),
    
    # Random small vectors for general testing
    ("random_vectors", np.array([0.1, -0.2, 0.05, 0.3, -0.4, 0.2], dtype=np.float64)),
    
    # Stress-like magnitude vectors
    ("stress_like", np.array([50.0, -30.0, 20.0, 10.0, -5.0, 15.0], dtype=np.float64)),
]

# NorSand material parameters (typical ranges from literature)
NOR_SAND_PARAMS = {
    "N": [0.8, 1.0, 1.2],                    # Dilatancy parameter
    "M_tc": [1.2, 1.5, 1.8],                 # Critical state friction ratio
    "chi_tc": [0.2, 0.3, 0.4],               # Critical state parameter
    "Gamma": [0.9, 1.1, 1.3],                # Reference void ratio
    "Lambda": [0.05, 0.08, 0.12],            # Compression index
    "H": [50.0, 100.0, 150.0],               # Hardening parameter
    "Hy": [10.0, 30.0, 50.0],                # Yield surface parameter
}

# Elastic parameters
ELASTIC_PARAMS = {
    "G": [30.0, 50.0, 80.0],                 # Shear modulus
    "K": [90.0, 150.0, 250.0],               # Bulk modulus
}

# State variables
STATE_VARS = {
    "e": [0.6, 0.7, 0.8],                    # Void ratio
    "p_i": [100.0, 150.0, 200.0],            # Image mean stress
}

# =============================================================================


@ti.kernel
def ti_getPQ(stress: ti.types.ndarray(ndim=1, dtype=ti.f64), out: ti.types.ndarray(ndim=1, dtype=ti.f64)):
    v = ti.Vector.zero(float, 6)
    for i in range(6):
        v[i] = stress[i]
    p, q = getPQ(v)
    out[0] = p
    out[1] = q


@ti.kernel
def ti_getInvariants(stress: ti.types.ndarray(ndim=1, dtype=ti.f64), outJ: ti.types.ndarray(ndim=1, dtype=ti.f64), out_dJ2: ti.types.ndarray(ndim=1, dtype=ti.f64), out_dJ3: ti.types.ndarray(ndim=1, dtype=ti.f64)):
    v = ti.Vector.zero(float, 6)
    for i in range(6):
        v[i] = stress[i]
    J2, J3, dJ2, dJ3 = getInvariants(v)
    outJ[0] = J2
    outJ[1] = J3
    for i in range(6):
        out_dJ2[i] = dJ2[i]
        out_dJ3[i] = dJ3[i]


@ti.kernel
def ti_getLodeM(stress: ti.types.ndarray(ndim=1, dtype=ti.f64), M_tc: ti.f64, out: ti.types.ndarray(ndim=1, dtype=ti.f64)):
    v = ti.Vector.zero(float, 6)
    for i in range(6):
        v[i] = stress[i]
    theta, M = getLodeM(v, M_tc)
    out[0] = theta
    out[1] = M


@ti.kernel
def ti_getC_e(G: ti.f64, K: ti.f64, out: ti.types.ndarray(ndim=2, dtype=ti.f64)):
    C = getC_e(G, K)
    for i in range(6):
        for j in range(6):
            out[i, j] = C[i, j]


@ti.kernel
def ti_getDev(vec: ti.types.ndarray(ndim=1, dtype=ti.f64), stress: ti.types.ndarray(ndim=1, dtype=ti.f64), out: ti.types.ndarray(ndim=1, dtype=ti.f64)):
    v = ti.Vector.zero(float, 6)
    s = ti.Vector.zero(float, 6)
    for i in range(6):
        v[i] = vec[i]
        s[i] = stress[i]
    out[0] = getDev(v, s)


@ti.kernel
def ti_getF(p: ti.f64, q: ti.f64, p_i: ti.f64, M_i: ti.f64, out: ti.types.ndarray(ndim=1, dtype=ti.f64)):
    out[0] = getF(p, q, p_i, M_i)


@ti.kernel
def ti_updateM_i(e: ti.f64, N: ti.f64, M: ti.f64, M_tc: ti.f64, chi_tc: ti.f64, Gamma: ti.f64, Lambda: ti.f64, p: ti.f64, p_i: ti.f64, out: ti.types.ndarray(ndim=1, dtype=ti.f64)):
    out[0] = updateM_i(e, N, M, M_tc, chi_tc, Gamma, Lambda, p, p_i)


@ti.kernel
def ti_getp_imax(e: ti.f64, N: ti.f64, M: ti.f64, M_tc: ti.f64, chi_tc: ti.f64, Gamma: ti.f64, Lambda: ti.f64, p: ti.f64, p_i: ti.f64, out: ti.types.ndarray(ndim=1, dtype=ti.f64)):
    p_imax, psi_i, chi_i, M_itc = getp_imax(e, N, M, M_tc, chi_tc, Gamma, Lambda, p, p_i)
    out[0] = p_imax
    out[1] = psi_i
    out[2] = chi_i
    out[3] = M_itc


@ti.kernel
def ti_getp_cap(e: ti.f64, N: ti.f64, M: ti.f64, M_tc: ti.f64, chi_tc: ti.f64, Gamma: ti.f64, Lambda: ti.f64, p: ti.f64, p_i: ti.f64, out: ti.types.ndarray(ndim=1, dtype=ti.f64)):
    out[0] = getp_cap(e, N, M, M_tc, chi_tc, Gamma, Lambda, p, p_i)


@ti.kernel
def ti_getdfdsigma(e: ti.f64, stress: ti.types.ndarray(ndim=1, dtype=ti.f64), p_i: ti.f64, N: ti.f64, M_tc: ti.f64, chi_tc: ti.f64, Gamma: ti.f64, Lambda: ti.f64, M_i: ti.f64, out: ti.types.ndarray(ndim=1, dtype=ti.f64)):
    s = ti.Vector.zero(float, 6)
    for i in range(6):
        s[i] = stress[i]
    v = getdfdsigma(e, s, p_i, N, M_tc, chi_tc, Gamma, Lambda, M_i)
    for i in range(6):
        out[i] = v[i]


@ti.kernel
def ti_getdfdepsq(e: ti.f64, stress: ti.types.ndarray(ndim=1, dtype=ti.f64), p_i: ti.f64, N: ti.f64, H: ti.f64, Hy: ti.f64, M_tc: ti.f64, chi_tc: ti.f64, Gamma: ti.f64, Lambda: ti.f64, M_i: ti.f64, out: ti.types.ndarray(ndim=1, dtype=ti.f64)):
    s = ti.Vector.zero(float, 6)
    for i in range(6):
        s[i] = stress[i]
    dfdpi, dpidepsq = getdfdepsq(e, s, p_i, N, H, Hy, M_tc, chi_tc, Gamma, Lambda, M_i)
    out[0] = dfdpi
    out[1] = dpidepsq


# ========================= Additional wrappers for Pegasus and UMAT =========================

@ti.kernel
def ti_Pegasus(F0: ti.f64, F1: ti.f64, alpha0: ti.f64, alpha1: ti.f64,
               dsig_tr: ti.types.ndarray(ndim=1, dtype=ti.f64),
               p_i: ti.f64, M_i: ti.f64,
               C_e_in: ti.types.ndarray(ndim=2, dtype=ti.f64),
               subdStrain_old: ti.types.ndarray(ndim=1, dtype=ti.f64),
               stress_old: ti.types.ndarray(ndim=1, dtype=ti.f64),
               e_old: ti.f64,
               out_subdStrain: ti.types.ndarray(ndim=1, dtype=ti.f64),
               out_stress: ti.types.ndarray(ndim=1, dtype=ti.f64),
               out_e: ti.types.ndarray(ndim=1, dtype=ti.f64)):
    v_dsig = ti.Vector.zero(float, 6)
    v_subd_old = ti.Vector.zero(float, 6)
    v_stress_old = ti.Vector.zero(float, 6)
    C_e = ti.Matrix.zero(float, 6, 6)
    for i in range(6):
        v_dsig[i] = dsig_tr[i]
        v_subd_old[i] = subdStrain_old[i]
        v_stress_old[i] = stress_old[i]
        for j in range(6):
            C_e[i, j] = C_e_in[i, j]
    subdStrain, stress, e = Pegasus(F0, F1, alpha0, alpha1, v_dsig, p_i, M_i, C_e, v_subd_old, v_stress_old, e_old)
    for i in range(6):
        out_subdStrain[i] = subdStrain[i]
        out_stress[i] = stress[i]
    out_e[0] = e


@ti.kernel
def ti_UMAT(e: ti.f64,
            stress_in: ti.types.ndarray(ndim=1, dtype=ti.f64),
            p_i: ti.f64, N: ti.f64, H: ti.f64, Hy: ti.f64, M_i: ti.f64,
            M_tc: ti.f64, chi_tc: ti.f64, Gamma: ti.f64, Lambda: ti.f64,
            dstrain_in: ti.types.ndarray(ndim=1, dtype=ti.f64),
            Gmax: ti.f64, Gexp: ti.f64, nu: ti.f64,
            out_stress: ti.types.ndarray(ndim=1, dtype=ti.f64),
            out_scalars: ti.types.ndarray(ndim=1, dtype=ti.f64)):
    s = ti.Vector.zero(float, 6)
    d = ti.Vector.zero(float, 6)
    for i in range(6):
        s[i] = stress_in[i]
        d[i] = dstrain_in[i]
    stress_new, p_inew, M_inew, e_new = UMAT(e, s, p_i, N, H, Hy, M_i, M_tc, chi_tc, Gamma, Lambda, d, Gmax, Gexp, nu)
    for i in range(6):
        out_stress[i] = stress_new[i]
    out_scalars[0] = p_inew
    out_scalars[1] = M_inew
    out_scalars[2] = e_new


@pytest.mark.parametrize(
    "stress_name, stress",
    STRESS_STATES,  # Use all stress states
)
def test_getPQ_matches_reference(stress_name, stress):
    out = np.zeros(2, dtype=np.float64)
    ti_getPQ(stress, out)
    p_ref, q_ref = ref.getPQ(stress.copy())
    
    # Print comparison details
    print(f"\n=== getPQ Test ({stress_name}) ===")
    print(f"Stress: {stress}")
    print(f"Reference p: {p_ref:.8f}, Taichi p: {out[0]:.8f}, Diff: {abs(out[0] - p_ref):.2e}")
    print(f"Reference q: {q_ref:.8f}, Taichi q: {out[1]:.8f}, Diff: {abs(out[1] - q_ref):.2e}")
    
    assert np.allclose(out, [p_ref, q_ref], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "stress_name, stress",
    STRESS_STATES,  # Use all stress states
)
def test_getInvariants_matches_reference(stress_name, stress):
    outJ = np.zeros(2, dtype=np.float64)
    out_dJ2 = np.zeros(6, dtype=np.float64)
    out_dJ3 = np.zeros(6, dtype=np.float64)
    ti_getInvariants(stress, outJ, out_dJ2, out_dJ3)
    J2_ref, J3_ref, dJ2_ref, dJ3_ref = ref.getInvariants(stress.copy())
    
    # Print comparison details
    print(f"\n=== getInvariants Test ({stress_name}) ===")
    print(f"Stress: {stress}")
    print(f"Reference J2: {J2_ref:.8f}, Taichi J2: {outJ[0]:.8f}, Diff: {abs(outJ[0] - J2_ref):.2e}")
    print(f"Reference J3: {J3_ref:.8f}, Taichi J3: {outJ[1]:.8f}, Diff: {abs(outJ[1] - J3_ref):.2e}")
    print(f"Max dJ2 diff: {np.max(np.abs(out_dJ2 - dJ2_ref)):.2e}")
    print(f"Max dJ3 diff: {np.max(np.abs(out_dJ3 - dJ3_ref)):.2e}")
    
    assert np.allclose(outJ[0], J2_ref, rtol=1e-6, atol=1e-6)
    assert np.allclose(outJ[1], J3_ref, rtol=1e-6, atol=1e-6)
    assert np.allclose(out_dJ2, dJ2_ref, rtol=1e-6, atol=1e-6)
    assert np.allclose(out_dJ3, dJ3_ref, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("stress_name, stress", STRESS_STATES)  # Use all stress states
@pytest.mark.parametrize("M_tc", NOR_SAND_PARAMS["M_tc"])
def test_getLodeM_matches_reference(stress_name, stress, M_tc):
    out = np.zeros(2, dtype=np.float64)
    ti_getLodeM(stress, M_tc, out)
    theta_ref, M_ref = ref.getLodeM(stress.copy(), M_tc)
    
    # Print comparison details
    print(f"\n=== getLodeM Test ({stress_name}, M_tc={M_tc}) ===")
    print(f"Stress: {stress}, M_tc: {M_tc}")
    print(f"Reference theta: {theta_ref:.8f}, Taichi theta: {out[0]:.8f}, Diff: {abs(out[0] - theta_ref):.2e}")
    print(f"Reference M: {M_ref:.8f}, Taichi M: {out[1]:.8f}, Diff: {abs(out[1] - M_ref):.2e}")
    
    assert np.allclose(out, [theta_ref, M_ref], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("G", ELASTIC_PARAMS["G"])
@pytest.mark.parametrize("K", ELASTIC_PARAMS["K"])
def test_getC_e_matches_reference(G, K):
    out = np.zeros((6, 6), dtype=np.float64)
    ti_getC_e(G, K, out)
    C_e_ref = ref.getC_e(G, K)
    assert np.allclose(out, C_e_ref, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("stress_name, stress", STRESS_STATES)  # Use all stress states
@pytest.mark.parametrize("vec_name, vec", VECTOR_INPUTS)  # Use all vector inputs
def test_getDev_matches_reference(stress_name, stress, vec_name, vec):
    out = np.zeros(1, dtype=np.float64)
    ti_getDev(vec, stress, out)
    dev_ref = ref.getDev(vec.copy(), stress.copy())
    assert np.allclose(out[0], dev_ref, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("stress_name, stress", STRESS_STATES)  # Use all stress states
@pytest.mark.parametrize("p_i", STATE_VARS["p_i"])
@pytest.mark.parametrize("N", NOR_SAND_PARAMS["N"])
@pytest.mark.parametrize("M_tc", NOR_SAND_PARAMS["M_tc"])  # Use all M_tc values
@pytest.mark.parametrize("chi_tc", NOR_SAND_PARAMS["chi_tc"])  # Use all chi_tc values
@pytest.mark.parametrize("Gamma", [NOR_SAND_PARAMS["Gamma"][1]])  # Use middle value to limit combinations
@pytest.mark.parametrize("Lambda", [NOR_SAND_PARAMS["Lambda"][1]])  # Use middle value to limit combinations
def test_updateM_i_and_getF_match_reference(stress_name, stress, p_i, N, M_tc, chi_tc, Gamma, Lambda):
    # Use M from Lode angle
    theta_ref, M_ref = ref.getLodeM(stress.copy(), M_tc)
    p_ref, q_ref = ref.getPQ(stress.copy())
    M_i_ref = ref.updateM_i(0.7, N, M_ref, M_tc, chi_tc, Gamma, Lambda, p_ref, p_i)
    F_ref = ref.getF(p_ref, q_ref, p_i, M_i_ref)

    out = np.zeros(1, dtype=np.float64)
    ti_updateM_i(0.7, N, M_ref, M_tc, chi_tc, Gamma, Lambda, p_ref, p_i, out)
    
    # Print comparison details for M_i
    print(f"\n=== updateM_i Test ({stress_name}, p_i={p_i}, N={N}) ===")
    print(f"Stress: {stress}, p_i: {p_i}, N: {N}, M_tc: {M_tc}")
    print(f"Reference M_i: {M_i_ref:.8f}, Taichi M_i: {out[0]:.8f}, Diff: {abs(out[0] - M_i_ref):.2e}")
    
    assert np.allclose(out[0], M_i_ref, rtol=1e-6, atol=1e-6)

    outF = np.zeros(1, dtype=np.float64)
    ti_getF(p_ref, q_ref, p_i, M_i_ref, outF)
    
    # Print comparison details for F
    print(f"Reference F: {F_ref:.8f}, Taichi F: {outF[0]:.8f}, Diff: {abs(outF[0] - F_ref):.2e}")
    
    assert np.allclose(outF[0], F_ref, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("stress_name, stress", STRESS_STATES)  # Use all stress states
@pytest.mark.parametrize("e", STATE_VARS["e"])  # Use all void ratios
@pytest.mark.parametrize("N", NOR_SAND_PARAMS["N"])  # Use all N values
@pytest.mark.parametrize("M_tc", [NOR_SAND_PARAMS["M_tc"][1]])  # Use middle value to limit combinations
@pytest.mark.parametrize("chi_tc", [NOR_SAND_PARAMS["chi_tc"][1]])  # Use middle value to limit combinations
@pytest.mark.parametrize("Gamma", [NOR_SAND_PARAMS["Gamma"][1]])  # Use middle value to limit combinations
@pytest.mark.parametrize("Lambda", [NOR_SAND_PARAMS["Lambda"][1]])  # Use middle value to limit combinations
def test_getp_imax_and_getp_cap_match_reference(stress_name, stress, e, N, M_tc, chi_tc, Gamma, Lambda):
    p_ref, q_ref = ref.getPQ(stress.copy())
    theta_ref, M_ref = ref.getLodeM(stress.copy(), M_tc)
    p_i = 120.0
    pimax_ref, psi_i_ref, chi_i_ref, M_itc_ref = ref.getp_imax(e, N, M_ref, M_tc, chi_tc, Gamma, Lambda, p_ref, p_i)
    pcap_ref = ref.getp_cap(e, N, M_ref, M_tc, chi_tc, Gamma, Lambda, p_ref, p_i)

    out4 = np.zeros(4, dtype=np.float64)
    ti_getp_imax(e, N, M_ref, M_tc, chi_tc, Gamma, Lambda, p_ref, p_i, out4)
    assert np.allclose(out4, [pimax_ref, psi_i_ref, chi_i_ref, M_itc_ref], rtol=1e-6, atol=1e-6)

    out1 = np.zeros(1, dtype=np.float64)
    ti_getp_cap(e, N, M_ref, M_tc, chi_tc, Gamma, Lambda, p_ref, p_i, out1)
    assert np.allclose(out1[0], pcap_ref, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("stress_name, stress", STRESS_STATES)  # Use all stress states
@pytest.mark.parametrize("e", [STATE_VARS["e"][1]])  # Use middle void ratio to limit combinations
@pytest.mark.parametrize("p_i", STATE_VARS["p_i"])  # Use all p_i values
@pytest.mark.parametrize("N", NOR_SAND_PARAMS["N"])  # Use all N values
@pytest.mark.parametrize("M_tc", [NOR_SAND_PARAMS["M_tc"][1]])  # Use middle value to limit combinations
@pytest.mark.parametrize("chi_tc", [NOR_SAND_PARAMS["chi_tc"][1]])  # Use middle value to limit combinations
@pytest.mark.parametrize("Gamma", [NOR_SAND_PARAMS["Gamma"][1]])  # Use middle value to limit combinations
@pytest.mark.parametrize("Lambda", [NOR_SAND_PARAMS["Lambda"][1]])  # Use middle value to limit combinations
def test_getdfdsigma_matches_reference(stress_name, stress, e, p_i, N, M_tc, chi_tc, Gamma, Lambda):
    theta_ref, M_ref = ref.getLodeM(stress.copy(), M_tc)
    p_ref, q_ref = ref.getPQ(stress.copy())
    M_i_ref = ref.updateM_i(e, N, M_ref, M_tc, chi_tc, Gamma, Lambda, p_ref, p_i)
    dfdsigma_ref = ref.getdfdsigma(e, stress.copy(), p_i, N, M_tc, chi_tc, Gamma, Lambda, M_i_ref)

    out6 = np.zeros(6, dtype=np.float64)
    ti_getdfdsigma(e, stress, p_i, N, M_tc, chi_tc, Gamma, Lambda, M_i_ref, out6)
    assert np.allclose(out6, dfdsigma_ref, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("stress_name, stress", STRESS_STATES)  # Use all stress states
@pytest.mark.parametrize("e", [STATE_VARS["e"][1]])  # Use middle void ratio to limit combinations
@pytest.mark.parametrize("p_i", [STATE_VARS["p_i"][1]])  # Use middle p_i to limit combinations
@pytest.mark.parametrize("N", NOR_SAND_PARAMS["N"])  # Use all N values
@pytest.mark.parametrize("H", NOR_SAND_PARAMS["H"])  # Use all H values
@pytest.mark.parametrize("Hy", NOR_SAND_PARAMS["Hy"])  # Use all Hy values
@pytest.mark.parametrize("M_tc", [NOR_SAND_PARAMS["M_tc"][1]])  # Use middle value to limit combinations
@pytest.mark.parametrize("chi_tc", [NOR_SAND_PARAMS["chi_tc"][1]])  # Use middle value to limit combinations
@pytest.mark.parametrize("Gamma", [NOR_SAND_PARAMS["Gamma"][1]])  # Use middle value to limit combinations
@pytest.mark.parametrize("Lambda", [NOR_SAND_PARAMS["Lambda"][1]])  # Use middle value to limit combinations
def test_getdfdepsq_matches_reference(stress_name, stress, e, p_i, N, H, Hy, M_tc, chi_tc, Gamma, Lambda):
    theta_ref, M_ref = ref.getLodeM(stress.copy(), M_tc)
    p_ref, q_ref = ref.getPQ(stress.copy())
    M_i_ref = ref.updateM_i(e, N, M_ref, M_tc, chi_tc, Gamma, Lambda, p_ref, p_i)
    dfdpi_ref, dpidepsq_ref = ref.getdfdepsq(e, stress.copy(), p_i, N, H, Hy, M_tc, chi_tc, Gamma, Lambda, M_i_ref)

    out2 = np.zeros(2, dtype=np.float64)
    ti_getdfdepsq(e, stress, p_i, N, H, Hy, M_tc, chi_tc, Gamma, Lambda, M_i_ref, out2)
    assert np.allclose(out2[0], dfdpi_ref, rtol=1e-6, atol=1e-6)
    assert np.allclose(out2[1], dpidepsq_ref, rtol=1e-6, atol=1e-6)


# ========================= Tests for Pegasus and UMAT parity =========================

@pytest.mark.parametrize("stress_name, stress", STRESS_STATES)
def test_Pegasus_matches_reference(stress_name, stress):
    # Fixed parameters to limit combinations
    e = 0.7
    p_i = 120.0
    N = 1.0
    M_tc = NOR_SAND_PARAMS["M_tc"][1]
    chi_tc = NOR_SAND_PARAMS["chi_tc"][1]
    Gamma = NOR_SAND_PARAMS["Gamma"][1]
    Lambda = NOR_SAND_PARAMS["Lambda"][1]

    # Elastic parameters for C_e and trial stress
    Gmax = 80.0
    Gexp = 0.5
    nu = 0.3

    # Small trial strain increment
    dstrain = np.array([1e-5, -2e-5, 0.5e-5, 3e-5, -1e-5, 2e-5], dtype=np.float64)

    # Compute reference quantities
    p_ref, q_ref = ref.getPQ(stress.copy())
    theta_ref, M_ref = ref.getLodeM(stress.copy(), M_tc)
    M_i = ref.updateM_i(e, N, M_ref, M_tc, chi_tc, Gamma, Lambda, p_ref, p_i)
    G = Gmax * (p_ref / 100.0) ** Gexp
    K = (2.0 * (1.0 + nu)) / (3.0 * (1.0 - 2.0 * nu)) * G
    C_e = ref.getC_e(G, K)
    dsig_tr = C_e @ dstrain
    F0 = ref.getF(p_ref, q_ref, p_i, M_i)
    p_trial, q_trial = ref.getPQ((stress + dsig_tr).copy())
    F1 = ref.getF(p_trial, q_trial, p_i, M_i)

    subd_ref, stress_ref, e_ref = ref.Pegasus(F0, F1, 0.0, 1.0, dsig_tr.copy(), p_i, M_i, C_e.copy(), dstrain.copy(), stress.copy(), e)

    # Taichi run
    out_subd = np.zeros(6, dtype=np.float64)
    out_stress = np.zeros(6, dtype=np.float64)
    out_e = np.zeros(1, dtype=np.float64)
    ti_Pegasus(F0, F1, 0.0, 1.0, dsig_tr, p_i, M_i, C_e, dstrain, stress, e, out_subd, out_stress, out_e)

    assert np.allclose(out_subd, subd_ref, rtol=1e-6, atol=1e-6)
    assert np.allclose(out_stress, stress_ref, rtol=1e-6, atol=1e-6)
    assert np.allclose(out_e[0], e_ref, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("stress_name, stress", STRESS_STATES)
def test_UMAT_matches_reference(stress_name, stress):
    # Fixed parameters to limit combinations
    e = 0.7
    p_i = 120.0
    N = 1.0
    H = NOR_SAND_PARAMS["H"][1]
    Hy = NOR_SAND_PARAMS["Hy"][1]
    M_tc = NOR_SAND_PARAMS["M_tc"][1]
    chi_tc = NOR_SAND_PARAMS["chi_tc"][1]
    Gamma = NOR_SAND_PARAMS["Gamma"][1]
    Lambda = NOR_SAND_PARAMS["Lambda"][1]
    # Elastic parameters
    Gmax = 80.0
    Gexp = 0.5
    nu = 0.3

    # Small strain increment
    dstrain = np.array([1e-5, -2e-5, 0.5e-5, 3e-5, -1e-5, 2e-5], dtype=np.float64)

    # Derive M_i at start
    p_ref, q_ref = ref.getPQ(stress.copy())
    theta_ref, M_ref = ref.getLodeM(stress.copy(), M_tc)
    M_i = ref.updateM_i(e, N, M_ref, M_tc, chi_tc, Gamma, Lambda, p_ref, p_i)

    stress_new_ref, p_i_new_ref, M_i_new_ref, e_new_ref = ref.UMAT(e, stress.copy(), p_i, N, H, Hy, M_i, M_tc, chi_tc, Gamma, Lambda, dstrain.copy(), Gmax, Gexp, nu)

    out_stress = np.zeros(6, dtype=np.float64)
    out_scalars = np.zeros(3, dtype=np.float64)
    ti_UMAT(e, stress, p_i, N, H, Hy, M_i, M_tc, chi_tc, Gamma, Lambda, dstrain, Gmax, Gexp, nu, out_stress, out_scalars)

    assert np.allclose(out_stress, stress_new_ref, rtol=1e-6, atol=1e-6)
    assert np.allclose(out_scalars[0], p_i_new_ref, rtol=1e-6, atol=1e-6)
    assert np.allclose(out_scalars[1], M_i_new_ref, rtol=1e-6, atol=1e-6)
    assert np.allclose(out_scalars[2], e_new_ref, rtol=1e-6, atol=1e-6)


