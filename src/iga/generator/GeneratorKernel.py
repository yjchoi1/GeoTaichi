import taichi as ti

from src.utils.TypeDefination import vec3f
from src.utils.ScalarFunction import vectorize_id
from src.utils.VectorFunction import inner_multiply

@ti.kernel
def kernel_initialize_patch(patchID: int, bodyID: int, ctrlpts_num: ti.types.vector(3, int), knot_num: ti.types.vector(3, int), element_num: ti.types.vector(3, int),
                            degree: ti.types.vector(3, int), gauss_number: ti.types.vector(3, int), patch: ti.template()):
    influencedNum = inner_multiply(degree + 1)
    gaussNumPerCell = inner_multiply(gauss_number)
    patch[patchID]._set_patch(bodyID,  gaussNumPerCell, influencedNum, degree, element_num, knot_num, ctrlpts_num, gauss_number)
    
    
@ti.kernel
def kernel_initialize_range(patchID: int, patch: ti.template(), initial_point_num: int, initial_gauss_num: int, initial_ele_num: ti.types.vector(3, int), initial_element_num: int, initial_knot_num: ti.types.vector(3, int)):    
    patch[patchID]._set_range(initial_point_num, initial_gauss_num, initial_ele_num, initial_element_num, initial_knot_num)


@ti.kernel
def kernel_add_point_position(patchID: int, start_locate: int, node_number: int, numctrlpts: ti.types.vector(3, int), ctrlpts: ti.types.ndarray(), weights: ti.types.ndarray(), point: ti.template()):
    for np in range(start_locate, node_number):
        i, j, k = vectorize_id(np, numctrlpts)
        point[np]._set_patch_id(patchID)
        point[np]._set_ctrlpts_position(vec3f(ctrlpts[k, j, i, 0], ctrlpts[k, j, i, 1], ctrlpts[k, j, i, 2]))
        point[np]._set_ctrlpts_weight(weights[k, j, i])


@ti.kernel
def kernel_add_knots(start_locate: int, knotNum: int, iknot: ti.types.ndarray(), knot: ti.template()):
    ti.loop_config(serialize=True)
    for nk in range(start_locate, knotNum):
        knot[nk] = iknot[nk]

    