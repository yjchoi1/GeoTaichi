import taichi as ti

from src.utils.TypeDefination import vec3i, vec3f, vec2f, vec2i, vec5i, vec3u8, vec6f
from src.utils.VectorFunction import clamp
from src.utils.BitFunction import Zero2OneVector


@ti.dataclass
class Patch:
    bodyID: int
    gaussNumPerCell: int
    influencedNum: int
    point_start: int
    gauss_start: int
    element_start: int
    ele_start: vec3i
    knot_start: vec3i
    degree: vec3i
    element_num: vec3i
    knot_num: vec3i
    ctrlpts_num: vec3i
    gauss_num: vec3i

    @ti.func
    def _get_point_start(self):
        return self.point_start
    
    @ti.func
    def _get_gauss_start(self):
        return self.gauss_start
    
    @ti.func
    def _get_ele_start(self):
        return self.ele_start
    
    @ti.func
    def _get_ele_ustart(self):
        return self.ele_start[0]
    
    @ti.func
    def _get_ele_vstart(self):
        return self.ele_start[1]
    
    @ti.func
    def _get_ele_wstart(self):
        return self.ele_start[2]
    
    @ti.func
    def _get_element_start(self):
        return self.element_start
    
    @ti.func
    def _get_knot_start(self):
        return self.knot_start
    
    @ti.func
    def _get_knot_ustart(self):
        return self.knot_start[0]
    
    @ti.func
    def _get_knot_vstart(self):
        return self.knot_start[1]
    
    @ti.func
    def _get_knot_wstart(self):
        return self.knot_start[2]

    @ti.func
    def _get_degree(self):
        return self.degree

    @ti.func
    def _get_knot_num(self):
        return self.knot_num

    @ti.func
    def _get_point_number(self):
        return self.ctrlpts_num
    
    @ti.func
    def _get_element_number(self):
        return self.element_num
    
    @ti.func
    def _get_gauss_number(self):
        return self.gauss_num

    @ti.func
    def _set_patch(self, bodyID, gaussNumPerCell, influencedNum, degree, element_num, knot_num, ctrlpts_num, gauss_num):
        self.bodyID = int(bodyID)
        self.gaussNumPerCell = int(gaussNumPerCell)
        self.influencedNum = int(influencedNum)
        self.degree = degree
        self.element_num = element_num
        self.knot_num = knot_num
        self.ctrlpts_num = ctrlpts_num
        self.gauss_num = gauss_num

    @ti.func
    def _set_range(self, initial_point_num, initial_gauss_num, initial_ele_num, initial_element_num, initial_knot_num):
        self.point_start = initial_point_num
        self.gauss_start = initial_gauss_num
        self.ele_start = initial_ele_num
        self.element_start = initial_element_num
        self.knot_start = initial_knot_num


@ti.dataclass
class ControlPoint:
    patchID: int
    x: vec3f
    weight: float
    displacement: vec3f

    @ti.func
    def _set_patch_id(self, patchID):
        self.patchID = patchID

    @ti.func
    def _set_ctrlpts_position(self, position):
        self.x = position

    @ti.func
    def _set_ctrlpts_weight(self, weight):
        self.weight = weight


@ti.dataclass
class Element:
    patchID: int
    element_range: vec2f
    knot_indices: vec2i
    connNum: int
    connectivity: vec5i

    @ti.func
    def _set_patch_id(self, patchID):
        self.patchID = patchID

    @ti.func
    def _set_element_range(self, element_range):
        self.element_range = element_range

    @ti.func
    def _set_knot_indices(self, knot_indices):
        self.knot_indices = knot_indices

    @ti.func
    def _set_connectivity(self, connectivity):
        self.connectivity = connectivity

    @ti.func
    def _set_connectivity_number(self, number):
        self.connNum = number
        

@ti.dataclass
class FirstDerivateShapeFunction:
    dsdxi: float
    dsdet: float
    dsdze: float


@ti.dataclass
class SecondDerivateShapeFunction:
    d2sdxi2: float
    d2sdet2: float
    d2sdze2: float
    d2sdxe: float
    d2sdez: float
    d2sdxz: float


@ti.dataclass
class TractionConstraint:
    patchID: int
    pointID: int
    traction: vec3f

    @ti.func
    def set_boundary_condition(self, patchID, pointID, traction):
        self.patchID = patchID
        self.pointID = pointID
        self.traction += traction


@ti.dataclass
class KinematicConstraint:
    patchID: int
    pointID: int
    fix: vec3u8
    unfix: vec3u8
    value: vec3f

    @ti.func
    def set_boundary_condition(self, patchID, pointID, direction, value):
        self.patchID = patchID
        self.pointID = pointID
        self.fix_v += ti.cast(direction, ti.u8)
        self.fix_v = ti.cast(clamp(0, 1, ti.cast(self.fix_v, int)), ti.u8)
        self.unfix_v = ti.cast(Zero2OneVector(self.fix_v), ti.u8)
        self.value += value

