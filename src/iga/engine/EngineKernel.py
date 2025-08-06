import taichi as ti

from src.utils.constants import ZEROMAT3x3
from src.utils.MatrixFunction import determinant3x3, get_jacobian_inverse3
from src.utils.TypeDefination import vec3f


# ======================================== Implicit IGA ======================================== #
@ti.func
def calculate_jacobian(patchID, max_gauss_num, max_point_num, max_element_num, influenced_node, ngp, linear_ele, connectivity, ctrlpts, dshapefn, jacobian):
    for i in range(influenced_node):
        dsdxi = dshapefn[ngp + patchID * max_gauss_num, i].dsdxi
        dsdet = dshapefn[ngp + patchID * max_gauss_num, i].dsdet
        dsdze = dshapefn[ngp + patchID * max_gauss_num, i].dsdze
        pos = ctrlpts[connectivity[linear_ele + patchID * max_element_num, i] + patchID * max_point_num].x

        jacobian[0, 0] += pos[0] * dsdxi
        jacobian[0, 1] += pos[0] * dsdet
        jacobian[0, 2] += pos[0] * dsdze
        jacobian[1, 0] += pos[1] * dsdxi
        jacobian[1, 1] += pos[1] * dsdet
        jacobian[1, 2] += pos[1] * dsdze
        jacobian[2, 0] += pos[2] * dsdxi
        jacobian[2, 1] += pos[2] * dsdet
        jacobian[2, 2] += pos[2] * dsdze


@ti.func
def calculate_bmatrix(patchID, max_gauss_num, influenced_node, ngp, dshapefn, bmatrix, jacobian):
    inv_j = get_jacobian_inverse3(jacobian)
    for i in range(influenced_node):
        dsdxi = dshapefn[ngp + patchID * max_gauss_num, i].dsdxi
        dsdet = dshapefn[ngp + patchID * max_gauss_num, i].dsdet
        dsdze = dshapefn[ngp + patchID * max_gauss_num, i].dsdze

        dsdx = inv_j[0, 0] * dsdxi + inv_j[0, 1] * dsdet + inv_j[0, 2] * dsdze
        dsdy = inv_j[1, 0] * dsdxi + inv_j[1, 1] * dsdet + inv_j[1, 2] * dsdze
        dsdz = inv_j[2, 0] * dsdxi + inv_j[2, 1] * dsdet + inv_j[2, 2] * dsdze

        bmatrix[ngp + patchID * max_gauss_num, i] = vec3f(dsdx, dsdy, dsdz)


@ti.kernel
def assemble_bmatrix(patchID: int, gaussPerEle: int, max_gauss_num: int, max_point_num: int, max_element_num: int, patch: ti.template(), ctrlpts: ti.template(), 
                     connectivity: ti.template(), dshapefn: ti.template(), bmatrix: ti.template(), jdet: ti.template(), j2det: ti.template(), weight: ti.template()):
    influenced_node = patch[patchID].influencedNum
    gauss_num = patch[patchID].gaussNum

    for ngp in range(gauss_num):
        gauss_id = ngp % gaussPerEle
        linear_ele = ngp // gaussPerEle
        jacobian = ZEROMAT3x3
        
        calculate_jacobian(patchID, max_gauss_num, max_point_num, max_element_num, influenced_node, ngp, linear_ele, connectivity, ctrlpts, dshapefn, jacobian)
        calculate_bmatrix(patchID, max_gauss_num, influenced_node, ngp, dshapefn, bmatrix, jacobian)
        jdet[ngp + patchID * max_gauss_num] = determinant3x3(jacobian) * weight[gauss_id] * j2det[linear_ele]


# ========================================================= #
#                   Assemble K matrix                       #
# ========================================================= #
@ti.kernel
def kernel_moment_balance_cg_quasi_static(patchID: int, gaussPerEle: int, max_gauss_num: int, max_point_num: int, max_element_num: int, patch: ti.template(), connectivity: ti.template(), disp_constraint: ti.template(), 
                                          bmatrix: ti.template(), jdet: ti.template(), stiffness_matrix: ti.template(), unknown_vector: ti.template(), m_dot_v: ti.template()):
    m_dot_v.fill(0)

    influenced_node = patch[patchID].influencedNum
    gauss_num = patch[patchID].gaussNum
    for ngp in range(gauss_num):
        offset = ngp + patchID * max_gauss_num
        stiffness = stiffness_matrix[offset]
        determinant_j = jdet[offset] 
        linear_ele = ngp // gaussPerEle
        for i in range(influenced_node):
            idshape = bmatrix[offset, i]
            ilocate = connectivity[linear_ele + patchID * max_element_num, i] + patchID * max_point_num
            for j in range(influenced_node):
                jdshape = bmatrix[offset, j]
                jlocate = connectivity[linear_ele + patchID * max_element_num, j] + patchID * max_point_num
                assemble_stiffness_matrix(ilocate, jlocate, idshape, jdshape, determinant_j, stiffness, m_dot_v, unknown_vector)


@ti.kernel
def kernel_moment_balance_cg_dynamic(patchID: int, gaussPerEle: int, max_gauss_num: int, max_point_num: int, max_element_num: int, beta: float, patch: ti.template(), connectivity: ti.template(), disp_constraint: ti.template(), 
                                     shapefn: ti.template(), bmatrix: ti.template(), jdet: ti.template(), dt: ti.template(), stiffness_matrix: ti.template(), unknown_vector: ti.template(), m_dot_v: ti.template()):
    m_dot_v.fill(0)
    constant = 1. / dt[None] / dt[None] / beta

    influenced_node = patch[patchID].influencedNum
    gauss_num = patch[patchID].gaussNum
    for ngp in range(gauss_num):
        linear_ele = ngp // gaussPerEle
        offset = ngp + patchID * max_gauss_num
        stiffness = stiffness_matrix[offset]
        determinant_j = jdet[offset] 
        
        for i in range(influenced_node):
            ishape = shapefn[offset, i]
            idshape = bmatrix[offset, i]
            ilocate = connectivity[linear_ele + patchID * max_element_num, i] + patchID * max_point_num
            for j in range(influenced_node):
                jshape = shapefn[offset, j]
                jdshape = bmatrix[offset, j]
                jlocate = connectivity[linear_ele + patchID * max_element_num, j] + patchID * max_point_num
                assemble_stiffness_matrix(ilocate, jlocate, idshape, jdshape, determinant_j, stiffness, m_dot_v, unknown_vector)
                assemble_mass_matrix(ilocate, jlocate, ishape, jshape, m_dot_v, unknown_vector)


@ti.func
def assemble_stiffness_matrix(ilocate, jlocate, idshape, jdshape, determinant_j, stiffness_matrix, m_dot_v, vector):
    m_dot_v[ilocate]    += determinant_j * ((idshape[0] * jdshape[0] * stiffness_matrix[0] + (idshape[2] * jdshape[2] + idshape[1] * jdshape[1]) * stiffness_matrix[2])* vector[jlocate] + \
                                            (idshape[0] * jdshape[1] * stiffness_matrix[1] + idshape[1] * jdshape[0] * stiffness_matrix[2]) * vector[jlocate+1] + \
                                            (idshape[0] * jdshape[2] * stiffness_matrix[1] + idshape[2] * jdshape[0] * stiffness_matrix[2]) * vector[jlocate+2]) 
    
    m_dot_v[ilocate+1]  += determinant_j * ((idshape[1] * jdshape[0] * stiffness_matrix[1] + idshape[0] * jdshape[1] * stiffness_matrix[2]) * vector[jlocate] + \
                                            (idshape[1] * jdshape[1] * stiffness_matrix[0] + (idshape[2] * jdshape[2] + idshape[0] * jdshape[0]) * stiffness_matrix[2]) * vector[jlocate+1] + \
                                            (idshape[1] * jdshape[2] * stiffness_matrix[1] + idshape[2] * jdshape[1] * stiffness_matrix[2]) * vector[jlocate+2])
    
    m_dot_v[ilocate+2]  += determinant_j * ((idshape[2] * jdshape[0] * stiffness_matrix[1] + idshape[0] * jdshape[2] * stiffness_matrix[2]) * vector[jlocate] + \
                                            (idshape[2] * jdshape[1] * stiffness_matrix[1] + idshape[1] * jdshape[2] * stiffness_matrix[2]) * vector[jlocate+1] + \
                                            (idshape[2] * jdshape[2] * stiffness_matrix[0] + (idshape[1] * jdshape[1] + idshape[0] * jdshape[0]) * stiffness_matrix[2]) * vector[jlocate+2])
    

@ti.func
def assemble_mass_matrix(ilocate, jlocate, ishape, jshape, m_dot_v, unknown_vector):
    pass


@ti.func
def jacobian_mapping_1d(ele_u, element_u):
    return 0.5 * (element_u[ele_u, 1] - element_u[ele_u, 0])

@ti.func
def jacobian_mapping_2d(ele_u, element_u, ele_v, element_v):
    return jacobian_mapping_1d(ele_u, element_u) * jacobian_mapping_1d(ele_v, element_v)

@ti.func
def jacobian_mapping_3d(ele_u, element_u, ele_v, element_v, ele_w, element_w):
    return jacobian_mapping_1d(ele_u, element_u) * jacobian_mapping_1d(ele_v, element_v) * jacobian_mapping_1d(ele_w, element_w) 

@ti.func
def master2parameteric(ele_u, element_u, local_pos):
    return 0.5 * (element_u[ele_u, 1] - element_u[ele_u, 0]) * local_pos + element_u[ele_u, 1] + element_u[ele_u, 0]

# ======================================== Explicit IGA ======================================== #