import numpy as np
import taichi as ti

from src.iga.BaseStruct import *
from src.iga.Simulation import Simulation
from src.iga.generator.NurbsVolume import NurbsVolume
from src.iga.element.HexahedronKernel import *
from src.utils.linalg import vector_max
from src.utils.TypeDefination import vec3i
from src.mesh.GaussPoint import GaussPointInRectangle


class HexahedronElement8Nodes(object):
    def __init__(self) -> None:
        self.grid_nodes = 8
        self.influenced_node = 8
        self.maxKnotNum = 0
        self.maxElementNum = 0
        
        self.enum = None
        self.gnum = None

        self.eleU = None
        self.eleV = None
        self.eleW = None
        self.connectivity = None

        self.gauss_point = None
        self.shapefn = None
        self.dshapefn = None
        self.ddshapefn = None
        self.bmatrix = None
        self.jdet = None

        self.calculate = None

    def element_initialize(self, sims: Simulation):
        self.maxKnotNum = max(sims.max_knot_num)
        self.maxElementNum = max(sims.max_element_num)

        self.enum = np.zeros((sims.max_patch_num, 3), dtype=np.int32)
        self.gnum = np.zeros((sims.max_patch_num, 3), dtype=np.int32)

        self.eleU = Element.field(shape=(self.maxElementNum * sims.max_patch_num))
        self.eleV = Element.field(shape=(self.maxElementNum * sims.max_patch_num))
        self.eleW = Element.field(shape=(self.maxElementNum * sims.max_patch_num))

        self.connectivity = ti.field(int, shape=(sims.max_total_element_each_patch * sims.max_patch_num, sims.max_influenced_node))

        self.gauss_point = GaussPointInRectangle(sims.gauss_num)
        self.gauss_point.create_gauss_point()
 
        self.shapefn = ti.field(float)
        self.dshapefn = FirstDerivateShapeFunction.field()
        self.bmatrix = ti.Vector.field(3, float)
        self.jdet = ti.field(float)
        if sims.coupling:
            self.ddshapefn = SecondDerivateShapeFunction.field()
            ti.root.dense(ti.ij, (sims.max_total_gauss_each_patch * sims.max_patch_num, sims.max_influenced_node)).place(self.shapefn, self.dshapefn, self.ddshapefn, self.bmatrix, self.jdet)
            self.calculate = self.calculate_nurbs2nd
        else:
            ti.root.dense(ti.ij, (sims.max_total_gauss_each_patch * sims.max_patch_num, sims.max_influenced_node)).place(self.shapefn, self.dshapefn, self.bmatrix, self.jdet)
            self.calculate = self.calculate_nurbs
    
        self.j2det = ti.field(float, shape=sims.max_total_element_each_patch * sims.max_patch_num)
        
    def get_total_element_number(self, patchID):
        return self.enum[patchID, 0] * self.enum[patchID, 1] * self.enum[patchID, 2]
    
    def get_total_grid_number(self, patchID):
        return self.gnum[patchID, 0] * self.gnum[patchID, 1] * self.gnum[patchID, 2]

    def find_connectivity(self, sims: Simulation, objects: NurbsVolume, patchID, eleNum, elementNum):
        numctrlpts = vec3i(objects.numctrlpts)
        degree = vec3i(objects.degree)
        
        element_num = np.array(objects.element_num)
        KnotU = np.array(objects.knotvector_u)
        KnotV = np.array(objects.knotvector_v) 
        KnotW = np.array(objects.knotvector_w)

        self.enum[patchID, :] = element_num
        self.gnum[patchID, :] = element_num + 1

        find_neighbor_knot_per_element_(int(patchID), KnotU, self.eleU)
        find_neighbor_knot_per_element_(int(patchID), KnotV, self.eleV)
        find_neighbor_knot_per_element_(int(patchID), KnotW, self.eleW)

        find_connectivity_per_element_(int(eleNum[0]), int(element_num[0]), int(degree[0]), self.eleU)
        find_connectivity_per_element_(int(eleNum[1]), int(element_num[1]), int(degree[1]), self.eleV)
        find_connectivity_per_element_(int(eleNum[2]), int(element_num[2]), int(degree[2]), self.eleW)

        build_global_connectivity(int(eleNum[0]), int(eleNum[1]), int(eleNum[2]), int(elementNum[0]), numctrlpts, 
                                  vec3i(objects.element_num), self.connectivity, self.eleU, self.eleV, self.eleW)

    def check_element_num(self, sims: Simulation, element_number):
        if any(element_number) > any(sims.max_element_num):
            raise ValueError ("The maximum element number should be set as: ", vector_max(element_number, sims.max_element_num))

    def calculate_nurbs(self, elementNum, patch, knotU, knotV, knotW, ctrlpts):
        update(int(elementNum), self.gauss_point.gpcoords, self.eleU, self.eleV, self.eleW, self.shapefn, self.dshapefn, self.j2det, patch, knotU, knotV, knotW, ctrlpts)
    
    def calculate_nurbs2nd(self, elementNum, patch, knotU, knotV, knotW, ctrlpts):
        update2nd(int(elementNum), self.gauss_point.gpcoords, self.eleU, self.eleV, self.eleW, self.shapefn, self.dshapefn, self.j2det, patch, knotU, knotV, knotW, ctrlpts)

    