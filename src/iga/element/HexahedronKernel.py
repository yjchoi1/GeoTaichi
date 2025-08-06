import taichi as ti

from src.nurbs.taichi.NurbsGeometry import *
from src.utils.ScalarFunction import linearize, vectorize_id
from src.utils.TypeDefination import vec2f, vec2i


@ti.kernel
def find_neighbor_knot_per_element_(patchID: int, iknot: ti.types.ndarray(), element: ti.template()):
    offset = 0
    previousKnotVal = 0.

    ti.loop_config(serialize=True)
    for i in range(iknot.shape[0]):
        currentKnotVal = iknot[i]
        if currentKnotVal != previousKnotVal:
            element[offset]._set_patch_id(patchID)
            element[offset]._set_element_range(vec2f(previousKnotVal, currentKnotVal))
            element[offset]._set_knot_indices(vec2i(i - 1, i))
            offset += 1
        previousKnotVal = currentKnotVal


@ti.kernel
def find_connectivity_per_element_(eleNum: int, element_num: int, degree: int, element: ti.template()):
    ti.loop_config(block_dim=32, parallelize=16)
    for ne in range(eleNum, eleNum + element_num):
        knot_indices = element[ne].knot_indices
        p = 0
        for i in range(knot_indices[0] - degree, knot_indices[0] + 1):
            element[ne].connectivity[p] = i
            p += 1
        element[ne]._set_connectivity_number(p)


@ti.kernel
def build_global_connectivity(eleNumU: int, eleNumV: int, eleNumW: int, elementNum: int, numctrlpts: ti.types.vector(3, int), element_num: ti.types.vector(3, int), 
                              connectivity: ti.template(), elementU: ti.template(), elementV: ti.template(), elementW: ti.template()):
    for new in range(element_num[2]):
        woffset = new + eleNumW
        wConn = elementW[woffset].connectivity
        for nev in range(element_num[1]):
            voffset = nev + eleNumV
            vConn = elementV[voffset].connectivity
            for neu in range(element_num[0]):
                uoffset = neu + eleNumU
                uConn = elementU[uoffset].connectivity
                
                uConnNum = elementU[uoffset].connNum
                vConnNum = elementV[voffset].connNum
                wConnNum = elementW[woffset].connNum
                linear_element = linearize(neu, nev, new, element_num)
                for i in range(wConnNum):
                    for j in range(vConnNum):
                        for k in range(uConnNum):
                            linear_conn = k + j * uConnNum + i * uConnNum * vConnNum
                            linear_ctrlpts = uConn[k] + vConn[j] * numctrlpts[0] + wConn[i] * numctrlpts[0] * numctrlpts[1]
                            connectivity[linear_element + elementNum, linear_conn] = linear_ctrlpts


@ti.func
def parent_to_parametric_space(xi, point):
    return 0.5 * ((xi[1] - xi[0]) * point + xi[0] + xi[1])


@ti.func
def jacobian_paramertic_parent_mapping(xi, eta, zeta):
    J2xi = 0.5 * (xi[1] - xi[0])
    J2eta = 0.5 * (eta[1] - eta[0])
    J2zeta = 0.5 * (zeta[1] - zeta[0])
    return J2xi * J2eta * J2zeta


@ti.kernel
def update(totalElementNum: int, gpcoord: ti.template(), elementU: ti.template(), elementV: ti.template(), elementW: ti.template(), shapefn: ti.template(), dshapefn: ti.template(), 
           j2det: ti.template(), patch: ti.template(), knotU: ti.template(), knotV: ti.template(), knotW: ti.template(), ctrlpts: ti.template()):
    for ne in range(totalElementNum):
        patchID = elementU[ne].patchID
        degree = patch[patchID].degree
        element_number = patch[patchID].element_num
        knot_num = patch[patchID].knot_num

        local_ne = ne - patch[patchID]._get_element_start()
        start_ele = patch[patchID]._get_ele_start()
        start_point = patch[patchID]._get_point_start()
        start_gauss = patch[patchID]._get_gauss_start()
        start_knot = patch[patchID]._get_knot_start()
        ei, ej, ek = vectorize_id(local_ne, element_number)
        xiRange = elementU[ei + start_ele[0]].element_range
        etaRange = elementV[ej + start_ele[1]].element_range
        zetaRange = elementW[ek + start_ele[2]].element_range
        j2det[ne] = jacobian_paramertic_parent_mapping(xiRange, etaRange, zetaRange)

        for ngp in range(gpcoord.shape[0]):
            xi = parent_to_parametric_space(xiRange, gpcoord[ngp][0])
            eta = parent_to_parametric_space(etaRange, gpcoord[ngp][1])
            zeta = parent_to_parametric_space(zetaRange, gpcoord[ngp][2])
            
            gaussID = ne * gpcoord.shape[0] + ngp
            NurbsBasisDers(start_point, start_gauss, start_knot, gaussID, xi, eta, zeta, degree[0], degree[1], degree[2], knot_num, knotU, knotV, knotW, ctrlpts.weight, shapefn, dshapefn)


@ti.kernel
def update2nd(totalElementNum: int, gpcoord: ti.template(), elementU: ti.template(), elementV: ti.template(), elementW: ti.template(), shapefn: ti.template(), dshapefn: ti.template(), 
              j2det: ti.template(), patch: ti.template(), knotU: ti.template(), knotV: ti.template(), knotW: ti.template(), ctrlpts: ti.template()):
    for ne in range(totalElementNum):
        patchID = elementU[ne].patchID
        degree = patch[patchID].degree
        element_number = patch[patchID].element_num
        knot_num = patch[patchID].knot_num

        local_ne = ne - patch[patchID]._get_element_start()
        start_ele = patch[patchID]._get_ele_start()
        start_point = patch[patchID]._get_point_start()
        start_gauss = patch[patchID]._get_gauss_start()
        start_knot = patch[patchID]._get_knot_start()
        ei, ej, ek = vectorize_id(local_ne, element_number)

        xiRange = elementU[ei + patchID * start_ele[0]].element_range
        etaRange = elementV[ej + patchID * start_ele[1]].element_range
        zetaRange = elementW[ek + patchID * start_ele[2]].element_range
        j2det[ne] = jacobian_paramertic_parent_mapping(xiRange, etaRange, zetaRange)

        for ngp in range(gpcoord.shape[0]):
            xi = parent_to_parametric_space(xiRange, gpcoord[ngp][0])
            eta = parent_to_parametric_space(etaRange, gpcoord[ngp][1])
            zeta = parent_to_parametric_space(zetaRange, gpcoord[ngp][2])
            
            gaussID = ne * gpcoord.shape[0] + ngp
            NurbsBasis2ndDers(start_point, start_gauss, start_knot, gaussID, xi, eta, zeta, degree[0], degree[1], degree[2], knot_num, knotU, knotV, knotW, ctrlpts.weight, shapefn, dshapefn)



            