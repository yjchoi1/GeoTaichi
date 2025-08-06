import taichi as ti

from src.utils.constants import Threshold, MThreshold
from src.utils.TypeDefination import vec3f
from src.utils.ScalarFunction import linearize, vectorize_id
from src.utils.VectorFunction import SquaredLength


@ti.func
def find_pre_location(start_index, patchID, inodes, constraint, locate: ti.template()):
    for pre in range(start_index):
        if inodes == constraint[pre].pointID and patchID == constraint[pre].patchID:
            locate = pre
            break


@ti.kernel
def kernel_initialize_boundary(constraint: ti.template()):
    for i in constraint:
        constraint[i].patchID = -1
        constraint[i].pointID = -1


@ti.kernel
def set_kinematic_constraint(patchNum: int, lists: ti.types.ndarray(), constraint: ti.template(), active_direction: ti.types.vector(3, int), 
                             value: ti.types.vector(3, float), is_in_region: ti.template(), point: ti.template(), patch: ti.template()):
    count = 0
    start_index = lists[0]
    for ncp in range(point.shape[0]):
        for np in range(patchNum):
            if ncp > patch[np].pointNum: continue

            if is_in_region(point[ncp, np].x, 0.):
                locate = ti.atomic_add(count, 1)
                find_pre_location(start_index, ncp, constraint, locate)
                constraint[locate].set_boundary_condition(np, ncp, active_direction, value)
    lists[0] += count


@ti.kernel
def set_traction_contraint(patchNum: int, lists: ti.types.ndarray(), constraint: ti.template(), value: ti.types.vector(3, float), 
                           is_in_region: ti.template(), point: ti.template(), patch: ti.template()):
    count = 0
    start_index = lists[0]
    for ncp in range(point.shape[0]):
        for np in range(patchNum):
            if ncp > patch[np].pointNum: continue

            if is_in_region(point[ncp, np].x, 0.):
                locate =  ti.atomic_add(count, 1)
                find_pre_location(start_index, ncp, constraint, locate)
                constraint[locate].set_boundary_condition(np, ncp, value)
    lists[0] += count


@ti.kernel
def clear_kinematic_constraint(lists: ti.types.ndarray(), constraint: ti.template(), is_in_region: ti.template(), point: ti.template()):   
    remain_num = 0
    ti.loop_config(serialize=True)
    for nboundary in range(lists[0]):
        patchID = constraint[nboundary].patchID
        ncp = constraint[nboundary].node 
        if is_in_region(point[ncp, patchID].x, 0.):
            constraint[remain_num].node = constraint[nboundary].node
            constraint[remain_num].fix_v = constraint[nboundary].fix_v
            constraint[remain_num].unfix_v = constraint[nboundary].unfix_v
            constraint[remain_num].velocity = constraint[nboundary].velocity

            if remain_num < nboundary:
                constraint[nboundary].node = -1
                constraint[nboundary].velocity = vec3f(0, 0, 0)
            remain_num += 1
    lists[0] = remain_num


@ti.kernel
def clear_traction_constraint(lists: ti.types.ndarray(), constraint: ti.template(), is_in_region: ti.template(), point: ti.template()):    
    remain_num = 0
    ti.loop_config(serialize=True)
    for nboundary in range(lists[0]):
        patchID = constraint[nboundary].patchID
        ncp = constraint[nboundary].node 
        if is_in_region(point[ncp, patchID].x, 0.):
            constraint[remain_num].node = constraint[nboundary].node
            constraint[remain_num].level = constraint[nboundary].level
            constraint[remain_num].traction = constraint[nboundary].traction
            
            if remain_num < nboundary:
                constraint[nboundary].node = -1
                constraint[nboundary].level = ti.u8(255)
                constraint[nboundary].traction = vec3f(0, 0, 0)
            remain_num += 1
    lists[0] = remain_num


@ti.kernel
def apply_kinematic_constraint(cut_off: float, lists: int, constraints: ti.template(), node: ti.template()):
    for nboundary in range(lists):
        nodeID = constraints[nboundary].node
        grid_level = int(constraints[nboundary].level)
        if node[nodeID, grid_level].m > cut_off:
            fix_v = int(constraints[nboundary].fix_v)
            unfix_v = int(constraints[nboundary].unfix_v)
            prescribed_velocity = constraints[nboundary].velocity
            node[nodeID, grid_level].velocity_constraint(fix_v, unfix_v, prescribed_velocity)


@ti.kernel
def apply_traction_constraint(lists: int, constraints: ti.template(), node: ti.template()):
    for nboundary in range(lists):
        nodeID = constraints[nboundary].node
        grid_level = int(constraints[nboundary].level)
        if node[nodeID, grid_level].m > Threshold:
            node[nodeID, grid_level].force += constraints[nboundary].traction


@ti.kernel
def find_min_element_size(maxPointNum: int, patchID: int, patch: ti.template(), ctrlpt: ti.template()) -> float:
    min_element_size = MThreshold
    for np in range(patch[patchID].pointNum):
        ctrlpts_num = patch[patchID].ctrlpts_num
        i, j, k = vectorize_id(np, ctrlpts_num)

        if i == ctrlpts_num[0] - 1 or j == ctrlpts_num[1] - 1 or k == ctrlpts_num[2] - 1: continue

        point = ctrlpt[np + patchID * maxPointNum].x
        point1 = ctrlpt[linearize(i + 1, j, k) + patchID * maxPointNum].x
        point2 = ctrlpt[linearize(i, j + 1, k) + patchID * maxPointNum].x
        point3 = ctrlpt[linearize(i, j, k + 1) + patchID * maxPointNum].x
        min_dist = ti.min(SquaredLength(point, point1), SquaredLength(point, point2), SquaredLength(point, point3))
        ti.atomic_min(min_element_size, min_dist)
    return ti.sqrt(min_element_size)
