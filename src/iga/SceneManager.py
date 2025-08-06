import os, warnings

import taichi as ti
import numpy as np

from src.iga.BaseKernel import *
from src.iga.BaseStruct import *
from src.iga.MaterialManager import ConstitutiveModel
from src.iga.meshing.HexahedronElement import HexahedronElement8Nodes
from src.iga.Simulation import Simulation
from src.utils.linalg import vector_max
from src.utils.ObjectIO import DictIO
from src.utils.RegionFunction import RegionFunction
from src.utils.TypeDefination import vec3f 

class myScene(object):
    def  __init__(self) -> None:
        self.element_type = "R8N3D"

        self.patch = None
        self.ctrlpts = None
        self.material = None
        self.element = None
        self.knotU = None
        self.knotV = None
        self.knotW = None
        self.visualize_knot = None
        self.kinematic_boundary = None
        self.traction_boundary = None

        self.kinematic_list = np.zeros(1, dtype=np.int32)
        self.traction_list = np.zeros(1, dtype=np.int32)
        self.patchNum = np.zeros(1, dtype=np.int32)
        self.knotNum = np.zeros(3, dtype=np.int32)
        self.eleNum = np.zeros(3, dtype=np.int32)
        self.pointNum = np.zeros(1, dtype=np.int32)
        self.elementNum = np.zeros(1, dtype=np.int32)
        self.gaussNum = np.zeros(1, dtype=np.int32)

    def activate_basic_class(self, sims: Simulation):
        sims.calculate_essential_parameters()
        self.activate_patch(sims)
        self.activate_knot(sims)
        self.activate_point(sims)
        self.activate_element(sims)
        self.activate_boundary_constraints(sims)

    def activate_patch(self, sims: Simulation):
        if self.patch is None:
            self.patch = Patch.field(shape=sims.max_patch_num)
    
    def activate_knot(self, sims: Simulation):
        if self.knotU is None:
            self.knotU = ti.field(float)
        if self.knotV is None:
            self.knotV = ti.field(float)
        if self.knotW is None:
            self.knotW = ti.field(float)
        ti.root.dense(ti.i, max(sims.max_knot_num) * sims.max_patch_num).place(self.knotU, self.knotV, self.knotW)
    
    def activate_point(self, sims: Simulation):
        if self.ctrlpts is None and min(sims.max_point_num) > 0:
            if sims.solver_type == "Explicit":
                self.ctrlpts = ControlPoint.field()
            elif sims.solver_type == "Implicit":
                self.ctrlpts = ControlPoint.field()
            ti.root.dense(ti.i, sims.max_total_point_each_patch * sims.max_patch_num).place(self.ctrlpts)

    def activate_material(self, sims: Simulation, model, materials):
        self.material = ConstitutiveModel.initialize(model, sims.max_material_num, sims.max_patch_num, sims.max_total_gauss_each_patch, sims.solver_type)
        self.material.model_initialization(materials)

    def check_materials(self, sims):
        if self.material is None:
            self.activate_material(sims, "None", materials={})

    def activate_element(self, sims: Simulation):
        if not self.element is None:
            warnings.warn("Warning: Previous elements will be override!")
        self.element = HexahedronElement8Nodes()
        self.element.element_initialize(sims)
    
    def activate_boundary_constraints(self, sims: Simulation):
        self.region = RegionFunction(sims.dimension, types="IGA")
        if self.traction_boundary is None and sims.ntraction > 0.:
            self.traction_boundary = TractionConstraint.field(shape=sims.ntraction)
        if self.kinematic_boundary is None and sims.nkinematic > 0.:
            self.kinematic_boundary = KinematicConstraint.field(shape=sims.nkinematic)

    def set_boundary_conditions(self, sims: Simulation, boundary):
        boundary_type = DictIO.GetEssential(boundary, "BoundaryType")
        self.region.set_region(DictIO.GetEssential(boundary, "Region"), override=True, printf=False)
        self.region.check_in_domain()
        
        if boundary_type == "KinematicConstraint":
            if self.kinematic_boundary is None:
                raise RuntimeError("Error:: /max_velocity_constraint/ is set as zero!")
            
            xvelocity = DictIO.GetAlternative(boundary, "VelocityX", None)
            yvelocity = DictIO.GetAlternative(boundary, "VelocityY", None)
            zvelocity = DictIO.GetAlternative(boundary, "VelocityZ", None)
            velocity = DictIO.GetAlternative(boundary, "Velocity", [None, None, None])

            if not velocity[0] is None or not velocity[1] is None or not velocity[2] is None:
                xvelocity = velocity[0]
                yvelocity = velocity[1]
                zvelocity = velocity[2]

            if xvelocity is None and yvelocity is None and zvelocity is None:
                raise KeyError("The prescribed velocity has not been set")
            
            fix_v, velocity = vec3i(0, 0, 0), vec3f(0, 0, 0)
            if not xvelocity is None:
                fix_v[0] = 1
                velocity[0] = xvelocity
            if not yvelocity is None:
                fix_v[1] = 1
                velocity[1] = yvelocity
            if not zvelocity is None:
                fix_v[2] = 1
                velocity[2] = zvelocity

            set_kinematic_constraint(self.patchNum[0], self.kinematic_list, self.kinematic_boundary, fix_v, velocity, self.region.function, self.ctrlpts, self.patch)
            self.check_kinematic_constraint_num(sims)
            print("Boundary Type: Kinematic Constraint")
            self.region.print_info()
            if not xvelocity is None:
                print("Prescribed Kinematic along X axis = ", float(xvelocity))
            if not yvelocity is None:
                print("Prescribed Kinematic along Y axis = ", float(yvelocity))
            if not zvelocity is None:
                print("Prescribed Kinematic along Z axis = ", float(zvelocity))
            print('\n')

        elif boundary_type == "TractionConstraint":
            if self.traction_boundary is None:
                raise RuntimeError("Error:: /max_traciton_constraint/ is set as zero!")

            fex = DictIO.GetEssential(boundary, "ExternalForce")
            set_traction_contraint(self.patchNum[0], self.traction_list, self.traction_boundary, fex, self.region.function, self.ctrlpts, self.patch) 
            self.check_traction_constraint_num(sims)
            print("Boundary Type: Traction Constraint")
            self.region.print_info()
            print("Grid Force = ", fex, '\n')

    def clear_boundary_constraint(self, sims: Simulation, boundary):
        boundary_type = DictIO.GetEssential(boundary, "BoundaryType")
        self.region.set_region(DictIO.GetEssential(boundary, "Region"), override=True, printf=False)
        self.region.check_in_domain()

        if boundary_type == "KinematicConstraint":
            initial_count = self.kinematic_list[0]
            clear_kinematic_constraint(self.kinematic_list, self.kinematic_boundary, self.region.function, self.ctrlpts)
            final_count = self.kinematic_list[0]
            print("Clear boundary Type: Kinematic Constraint")
            self.region.print_info()
            print("Total clear points: ", final_count - initial_count)
        elif boundary_type == "TractionConstraint":
            initial_count = self.kinematic_list[0]
            clear_traction_constraint(self.traction_list, self.traction_boundary, self.region.function, self.ctrlpts)
            final_count = self.kinematic_list[0]
            print("Clear boundary Type: Kinematic Constraint")
            self.region.print_info()
            print("Total clear points: ", final_count - initial_count)

    def write_boundary_constraint(self, output_path):
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        if self.kinematic_list[None] > 0:
            pass
        if self.traction_list[None] > 0:
            pass

    def read_boundary_constraint(self, sims: Simulation, boundary_constraint):
        print(" Read Boundary Information ".center(71,"-"))
        if not os.path.exists(boundary_constraint):
            raise EOFError("Invaild path")

        boundary_constraints = open(boundary_constraint, 'r')
        while True:
            line = str.split(boundary_constraints.readline())
            if not line: break

            elif line[0] == '#': continue

            elif line[0] == "KinematicConstraint":
                if self.kinematic_boundary is None:
                    raise RuntimeError("Error:: /max_kinematic_constraint/ is set as zero!")
                boundary_size = int(line[1])
                self.check_kinematic_constraint_num(sims, boundary_size)
                for _ in range(boundary_size):
                    boundary = str.split(boundary_constraints.readline())
                    self.kinematic_boundary[self.kinematic_list[0]].set_boundary_condition(int(boundary[0]), int(boundary[1]),
                                                                                            vec3f(float(boundary[2]), float(boundary[3]), float(boundary[4])),
                                                                                            vec3f(float(boundary[5]), float(boundary[6]), float(boundary[7])))
                    self.kinematic_list[0] += 1

            elif line[0] == "TractionConstraint":
                if self.traction_boundary is None:
                    raise RuntimeError("Error:: /max_traction_constraint/ is set as zero!")

                boundary_size = int(line[1])
                self.check_traction_constraint_num(sims, boundary_size)
                for _ in range(boundary_size):
                    boundary = str.split(boundary_constraints.readline())
                    self.traction_boundary[self.traction_list[None]].set_boundary_condition(int(boundary[0]), int(boundary[1]),
                                                                                            vec3f(float(boundary[2]), float(boundary[3]), float(boundary[4])))
                    self.traction_list[None] += 1


    def check_kinematic_constraint_num(self, sims: Simulation, constraint_num=0):
        if self.kinematic_list[0] + constraint_num > sims.nkinematic:
            raise ValueError ("The number of kinematic constraints should be set as: ", self.kinematic_list[0] + constraint_num)
        
    def check_traction_constraint_num(self, sims: Simulation, constraint_num=0):
        if self.traction_list[0] + constraint_num > sims.ntraction:
            raise ValueError ("The number of traction constraints should be set as: ", self.traction_list[0] + constraint_num)
        
    def check_patch_num(self, sims: Simulation, patch_number):
        if self.patchNum[0] + patch_number > sims.max_patch_num:
            raise ValueError ("The maximum patch number should be set as: ", self.patchNum[0] + patch_number)
        
    def check_point_num(self, sims: Simulation, point_number):
        if any(point_number) > any(sims.max_point_num):
            raise ValueError ("The maximum control ctrlpts should be set as: ", vector_max(point_number, sims.max_point_num))
        
    def check_gauss_num(self, sims: Simulation, gauss_number):
        if any(gauss_number) > any(sims.gauss_num):
            raise ValueError ("The maximum gauss points should be set as: ", vector_max(gauss_number, sims.gauss_num))

    def get_critical_timestep(self):
        max_vel = self.material.find_max_sound_speed()
        return self.calc_critical_timestep(max_vel)
    
    def calc_critical_timestep(self, sims: Simulation, velocity):
        min_element_size = MThreshold
        for np in range(self.patchNum[0]):
            min_element_size = min(min_element_size, find_min_element_size(sims.max_total_point_each_patch, np, self.patch, self.ctrlpts))
        return min_element_size / velocity
    
    def find_min_density(self):
        mindensity = 1e15
        for nm in range(self.material.matProps.shape[0]):
            if self.material.matProps[nm].density > 0:
                mindensity = ti.min(mindensity, self.material.matProps[nm].density)
        return mindensity
    
    