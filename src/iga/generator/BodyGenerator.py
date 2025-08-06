import numpy as np
import os


from src.iga.generator.GeneratorKernel import *
from src.iga.generator.NurbsVolume import NurbsVolume
from src.iga.SceneManager import myScene
from src.iga.Simulation import Simulation
from src.utils.linalg import scalar_sum, vector_sum, vector_inverse, inner_multiply
from src.utils.ObjectIO import DictIO
from src.utils.TypeDefination import vec3i, vec6f
from src.nurbs.BasicVolume import *
from third_party.pyevtk.hl import gridToVTK



class GenerateManager(object):
    def __init__(self):
        self.myGenerator = []

    def add_patch(self, body_dict, sims, scene):
        generator = BodyGenerator()
        self.myGenerator.append(generator)
        generator.set_system_strcuture(sims, body_dict)
        generator.begin(sims, scene)
        generator.finalize()


class BodyGenerator(object):
    objects: NurbsVolume
    shape_factor: PrimitiveVolume

    def __init__(self) -> None:
        self.bodyID = 0
        self.patchID = 0
        self.materialID = 0
        self.type = None
        self.shape = None
        self.write_file = False
        self.visualize = True
        self.resolution = [50, 50, 50]
        self.zdirection = [0., 0., 1.]
        self.file = None
        self.refinement = 0

        self.shape_factor = None
        self.node_number = []
        self.objects = None
        self.definition = {}

    def set_system_strcuture(self, sims, body_dict):
        self.bodyID = DictIO.GetEssential(body_dict, "BodyID")
        self.materialID = DictIO.GetEssential(body_dict, "MaterialID")
        self.type = DictIO.GetAlternative(body_dict, "GenerateType", "Basic")
        self.shape = DictIO.GetAlternative(body_dict, "BasicShape", None)
        self.write_file = DictIO.GetAlternative(body_dict, "WriteFile", False)
        self.visualize = DictIO.GetAlternative(body_dict, "Visualize", True)
        self.zdirection = DictIO.GetAlternative(body_dict, "zdirection", [0, 0, 1])
        self.file = DictIO.GetAlternative(body_dict, "File", "Object.txt")
        self.definition = DictIO.GetAlternative(body_dict, "Definition", {})
        self.check_parameters(body_dict)
        self.create_shape_object(body_dict)
        self.create_nurbs_object(sims, body_dict)

    def check_parameters(self, body_dict):
        refinement = DictIO.GetAlternative(body_dict, "Refinement", [0, 0, 0])
        resolution = DictIO.GetAlternative(body_dict, "Resolution", self.resolution)

        if isinstance(resolution, (list, np.ndarray, tuple)):
            self.resolution = list(resolution)
        elif isinstance(resolution, float):
            self.resolution = [resolution, resolution, resolution]

        if isinstance(refinement, (list, np.ndarray, tuple)):
            self.refinement = list(refinement)
        elif isinstance(refinement, float):
            self.refinement = [refinement, refinement, refinement]

        if self.definition is False and self.type == "UserDefined":
            raise RuntimeError("Keyword:: /Definition/ has not been set")
        
    def create_shape_object(self, body_dict):
        if self.shape == "Rectangle":
            self.shape_factor = Rectangle()
        elif self.shape == "Sperical":
            self.shape_factor = Sperical()
        elif self.shape == "Tube":
            self.shape_factor = Tube()
        elif self.shape == "Cylinder":
            self.shape_factor = Cylinder()
        else:
            if self.type == "Basic":
                raise RuntimeError("Keyword:: /BasicShape/ has not been specified") 
            else: 
                self.shape_factor = NoneShape()
        self.shape_factor.set_parameters(DictIO.GetEssential(body_dict, "Dimensions"))

    def create_nurbs_object(self, sims: Simulation, body_dict):
        degree = DictIO.GetAlternative(body_dict, "Degree", sims.max_degree_num)
        node_number = DictIO.GetEssential(body_dict, "NodeNumber")
        gauss_number = DictIO.GetAlternative(body_dict, "GaussNumber", scalar_sum(degree, 1))
        self.check_degree(sims, degree)

        self.objects = NurbsVolume()
        self.objects.create(degree, node_number, gauss_number, self.shape_factor)

    def check_degree(self, sims: Simulation, degree):
        if any(degree) != max(sims.max_degree_num):
            raise RuntimeError("Currently the degree of patch must be equal to /max_degree_num/")

    def begin(self, sims, scene: myScene):
        print('#', "Start adding patch(es) ......")
        if self.type == "Basic":
            self.MakeMesh()
        elif self.type == "OBJ":
            self.MakeMeshFromOBJ(scene)
        elif self.type == "TXT":
            self.MakeMeshFromTXT(scene)
        elif self.type == "UserDefined":
            self.MakeUserDefined(scene)
        self.knotRefinement()
        self.GenerateMesh(sims, scene)
        self.print_info()
        self.scene_visualization()

    def finalize(self):
        pass

    def print_info(self):
        print("body ID = ", self.bodyID)
        print("patch ID = ", self.patchID)
        print("Material ID = ", self.materialID)
        print("The number of knot vectors = ", [int(i) for i in scalar_sum(vector_sum(self.objects.numctrlpts, self.objects.degree), 1)])
        print("The number of control points = ", self.objects.numctrlpts)
        print("The number of elements = ", self.objects.element_num)
        print("The number of guass points = ", self.objects.gauss_number)
        print('\n')

    def MakeMesh(self):
        self.objects.generate_ctrlpts(self.shape_factor)
        self.objects.generate_weights(self.shape_factor)
        self.objects.generate_knots(self.shape_factor)

    def GenerateMesh(self, sims: Simulation, scene: myScene):
        self.patchID = scene.patchNum[0]
        node_number = self.shape_factor.get_ctrlpts_num(self.objects.numctrlpts)
        scene.check_point_num(sims, self.objects.numctrlpts)
        scene.check_gauss_num(sims, self.objects.gauss_number)
        scene.element.check_element_num(sims, self.objects.element_num)
        scene.check_patch_num(sims, 1)

        degree = vec3i(self.objects.degree)
        numctrlpts = vec3i(self.objects.numctrlpts)
        ctrlpts = np.array(self.objects.ctrlpts)
        weights = np.array(self.objects.weights)
        knotvector_u = np.array(self.objects.knotvector_u)
        knotvector_v = np.array(self.objects.knotvector_v)
        knotvector_w = np.array(self.objects.knotvector_w)
        numknot = vec3i(self.objects.knot_num)
        numelement = vec3i(self.objects.element_num)
        numgauss = vec3i(self.objects.gauss_number)
        
        start_point = int(scene.pointNum[0])
        end_point = int(scene.pointNum[0]) + int(inner_multiply(self.objects.numctrlpts))
        initial_stress = DictIO.GetAlternative(self.definition, "InitialStress", vec6f(0., 0., 0., 0., 0., 0.))
        scene.material.state_vars_initialize(start_point, end_point, self.patchID, self.materialID, initial_stress)
        
        kernel_add_point_position(int(scene.patchNum[0]), int(scene.pointNum[0]), node_number, numctrlpts, ctrlpts, weights, scene.ctrlpts)
        kernel_add_knots(int(scene.knotNum[0]), knotvector_u.shape[0], knotvector_u, scene.knotU)
        kernel_add_knots(int(scene.knotNum[1]), knotvector_v.shape[0], knotvector_v, scene.knotV)
        kernel_add_knots(int(scene.knotNum[2]), knotvector_w.shape[0], knotvector_w, scene.knotW)
        kernel_initialize_patch(int(self.patchID), self.bodyID, numctrlpts, numknot, numelement, degree, numgauss, scene.patch)
        scene.element.find_connectivity(sims, self.objects, scene.patchNum[0], scene.eleNum, scene.elementNum)
        self.UpdateEssentialNumber(scene)

    def UpdateEssentialNumber(self, scene: myScene):
        initial_point_num = int(scene.pointNum[0])
        initial_gauss_num = int(scene.gaussNum[0])
        initial_ele_num = vec3i(scene.eleNum)
        initial_element_num = int(scene.elementNum[0])
        initial_knot_num = vec3i(scene.knotNum)

        scene.knotNum += np.array(self.objects.knot_num)
        scene.eleNum += np.array(self.objects.element_num)

        scene.patchNum[0] += 1
        scene.elementNum[0] += inner_multiply(self.objects.element_num)
        scene.gaussNum[0] += inner_multiply(self.objects.gauss_number) * inner_multiply(self.objects.element_num)
        scene.pointNum[0] += inner_multiply(self.objects.numctrlpts)

        kernel_initialize_range(int(self.patchID), scene.patch, initial_point_num, initial_gauss_num, initial_ele_num, initial_element_num, initial_knot_num)

    def MakeMeshFromOBJ(self, scene):
        volume = exchange.import_obj(self.file)

    def MakeMeshFromTXT(self, scene):
        if not os.path.exists(self.file):
            raise EOFError("Invaild particle path")
        objects = np.loadtxt(self.file, unpack=True, comments='#').transpose()

    def MakeUserDefined(self):
        self.objects.ctrlpts = DictIO.GetEssential(self.definition, "ControlPoint")
        self.objects.knotvector_u = DictIO.GetEssential(self.definition, "KnotU")
        self.objects.knotvector_v = DictIO.GetEssential(self.definition, "KnotV")
        self.objects.knotvector_w = DictIO.GetEssential(self.definition, "KnotW")

    def set_general_volume_from_third_party(self):
        numCtrlPts = self.objects.numctrlpts[0] * self.objects.numctrlpts[1] * self.objects.numctrlpts[2]
        p_ctrlpts = list(np.array(self.objects.ctrlpts).reshape((numCtrlPts, 3)))
        p_weights = list(np.array(self.objects.weights).reshape(numCtrlPts))
        t_ctrlptsw = compatibility.combine_ctrlpts_weights(p_ctrlpts, p_weights)
        n_ctrlptsw = compatibility.flip_ctrlpts_u(t_ctrlptsw, *self.objects.numctrlpts)

        volume = NURBS.Volume()
        volume.degree = self.objects.degree
        volume.set_ctrlpts(n_ctrlptsw, *self.objects.numctrlpts)
        volume.knotvector_u = self.objects.knotvector_u
        volume.knotvector_v = self.objects.knotvector_v
        volume.knotvector_w = self.objects.knotvector_w
        return volume

    def knotRefinement(self):
        if any(self.refinement) > 0:
            volume = self.set_general_volume_from_third_party()
            operations.refine_knotvector(volume, self.refinement)

            t_ctrlptsw = compatibility.flip_ctrlpts(volume._control_points, *volume.cpsize)
            self.objects.set_ctrlpts(list(np.array(t_ctrlptsw)[:, 0:3].reshape((*vector_inverse(volume._control_points_size), 3))), numctrlpts=volume._control_points_size)
            self.objects.set_weights(list(np.array(t_ctrlptsw)[:, 3].reshape(*vector_inverse(volume._control_points_size))))
            self.objects.set_knot(volume.knotvector_u, volume.knotvector_v, volume.knotvector_w)

    def compute_insert_num(self, refine_num, knot):
        number = np.unique(knot).shape[0]
        for _ in range(refine_num):
            number = 2 * number - 1
        return number - np.unique(knot).shape[0]

    def knotRefinementC(self):
        if any(self.refinement) > 0:
            numCtrlPts = self.objects.numctrlpts[0] * self.objects.numctrlpts[1] * self.objects.numctrlpts[2]
            p_ctrlpts = np.array(self.objects.ctrlpts).reshape((numCtrlPts, 3))
            p_weights = np.array(self.objects.weights).reshape(numCtrlPts)

            insert_num_x = self.compute_insert_num(self.refinement[0], self.objects.knotvector_u)
            insert_num_y = self.compute_insert_num(self.refinement[1], self.objects.knotvector_v)
            insert_num_z = self.compute_insert_num(self.refinement[2], self.objects.knotvector_w)

            knot_num_u = len(self.objects.knotvector_u) + insert_num_x
            knot_num_v = len(self.objects.knotvector_v) + insert_num_y
            knot_num_w = len(self.objects.knotvector_w) + insert_num_z
            
            total_points_x = self.objects.numctrlpts[0] + insert_num_x
            total_points_y = self.objects.numctrlpts[1] + insert_num_y
            total_points_z = self.objects.numctrlpts[2] + insert_num_z
            total_points = total_points_x * total_points_y * total_points_z

            results = hrefine_nurbs.hrefine_nurbs3d(self.refinement[0], self.refinement[1], self.refinement[2], self.objects.degree[0], self.objects.degree[0], self.objects.degree[0], 
                                                    self.objects.numctrlpts[0], self.objects.numctrlpts[1], self.objects.numctrlpts[2], 
                                                    np.ascontiguousarray(p_ctrlpts[:, 0]), np.ascontiguousarray(p_ctrlpts[:, 1]), np.ascontiguousarray(p_ctrlpts[:, 2]), np.ascontiguousarray(p_weights), 
                                                    np.ascontiguousarray(np.array(self.objects.knotvector_u)), np.ascontiguousarray(np.array(self.objects.knotvector_v)), np.ascontiguousarray(np.array(self.objects.knotvector_w)))
            
            if len(results) != 4 * total_points + knot_num_u + knot_num_v + knot_num_w:
                raise RuntimeError("Wrong output: control points and weights")
            
            self.objects.set_ctrlpts(list(results[0 : 3 * total_points].reshape((total_points_z, total_points_y, total_points_x, 3))), numctrlpts=[total_points_x, total_points_y, total_points_z])
            self.objects.set_weights(list(results[3 * total_points : 4 * total_points].reshape((total_points_z, total_points_y, total_points_x))))
            self.objects.set_knot(list(results[4 * total_points : 4 * total_points + knot_num_u]),
                                list(results[4 * total_points + knot_num_u : 4 * total_points + knot_num_u + knot_num_v]),
                                list(results[4 * total_points + knot_num_u + knot_num_v : -1]))
        
    def scene_visualization(self):
        knotU = np.array(self.objects.knotvector_u)
        knotV = np.array(self.objects.knotvector_v)
        knotW = np.array(self.objects.knotvector_w)
        spaceU = (np.max(knotU) - np.min(knotU)) / self.resolution[0]
        spaceV = (np.max(knotV) - np.min(knotV)) / self.resolution[1]
        spaceW = (np.max(knotW) - np.min(knotW)) / self.resolution[2]
        numCtrlPts = self.objects.numctrlpts[0] * self.objects.numctrlpts[1] * self.objects.numctrlpts[2]
        points = np.array(self.objects.ctrlpts).reshape((numCtrlPts, 3))
        weights = np.array(self.objects.weights).reshape(numCtrlPts)
        p, q, r = self.objects.degree[0], self.objects.degree[1], self.objects.degree[2]

        xi = np.arange(np.min(knotU), np.max(knotU) + spaceU, spaceU)
        eta = np.arange(np.min(knotV), np.max(knotV) + spaceV, spaceV)
        zeta = np.arange(np.min(knotW), np.max(knotW) + spaceW, spaceW)
        numPtsU = xi.shape[0]
        numPtsV = eta.shape[0] 
        numPtsW = zeta.shape[0]
        numPtsUV = numPtsU * numPtsV
        numPts = numPtsUV * numPtsW

        interpolatedPoints = np.zeros((numPts, 3))
        for c in range(numPts):
            i = (c % (numPtsUV)) % numPtsU
            j = (c % (numPtsUV)) // numPtsU
            k = c // (numPtsUV)
            u = nurbs_interpolation.nurbs_interpolation3d(xi[i], eta[j], zeta[k], p, q, r, 
                                                          np.ascontiguousarray(knotU), np.ascontiguousarray(knotV), np.ascontiguousarray(knotW), 
                                                          np.ascontiguousarray(points[:, 0]), np.ascontiguousarray(weights))
            v = nurbs_interpolation.nurbs_interpolation3d(xi[i], eta[j], zeta[k], p, q, r, 
                                                          np.ascontiguousarray(knotU), np.ascontiguousarray(knotV), np.ascontiguousarray(knotW), 
                                                          np.ascontiguousarray(points[:, 1]), np.ascontiguousarray(weights))
            w = nurbs_interpolation.nurbs_interpolation3d(xi[i], eta[j], zeta[k], p, q, r, 
                                                          np.ascontiguousarray(knotU), np.ascontiguousarray(knotV), np.ascontiguousarray(knotW), 
                                                          np.ascontiguousarray(points[:, 2]), np.ascontiguousarray(weights))
            interpolatedPoints[c] = [u[0], v[0], w[0]]

        if self.visualize and self.write_file:
            interpolatedPointsU = np.ascontiguousarray(interpolatedPoints[:, 0].reshape((numPtsW, numPtsV, numPtsU)))
            interpolatedPointsV = np.ascontiguousarray(interpolatedPoints[:, 1].reshape((numPtsW, numPtsV, numPtsU)))
            interpolatedPointsW = np.ascontiguousarray(interpolatedPoints[:, 2].reshape((numPtsW, numPtsV, numPtsU)))
            gridToVTK(f"NurbsVolume", interpolatedPointsU, interpolatedPointsV, interpolatedPointsW, cellData={}, pointData={})
        elif self.visualize and not self.write_file:
            from src.nurbs.NurbsPlot import NurbsPlot
            plot = NurbsPlot(xi.shape[0] * eta.shape[0] * zeta.shape[0], dims=3, evaplts=interpolatedPoints)
            plot.initialize()
            plot.PlotVolume()

    def WriteFile(self):
        volume = self.set_general_volume_from_third_party()
        exchange.export_json(volume, "volume.json")