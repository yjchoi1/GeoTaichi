import os

import numpy as np
import nurbs_interpolation

from src.iga.generator.BodyGenerator import BodyGenerator, GenerateManager
from src.iga.Simulation import Simulation
from src.iga.SceneManager import myScene
from src.utils.linalg import inner_multiply
from third_party.pyevtk.hl import pointsToVTK, gridToVTK


class WriteFile(object):
    def __init__(self, sims):
        self.vtk_path = None
        self.point_path  = None
        self.basic_path = None

        self.save_point = self.no_operation

        self.mkdir(sims)
        self.manage_function(sims)

    def no_operation(self, sims, scene):
        pass

    def manage_function(self, sims: Simulation):
        if 'gauss_point' in sims.monitor_type:
            self.save_point = self.monitor_gauss_point
        if 'control_point' in sims.monitor_type:
            self.save_point = self.monitor_point
        if 'volume' in sims.monitor_type:
            self.save_point = self.monitor_volume

    def output(self, sims, scene):
        self.save_point(sims, scene)

    def mkdir(self, sims: Simulation):
        if not os.path.exists(sims.path):
            os.makedirs(sims.path)

        self.vtk_path = None
        self.point_path  = None
        self.basic_path = None

        self.basic_path = sims.path 
        self.point_path = sims.path + '/points'
        self.vtk_path = sims.path + '/vtks'
        if not os.path.exists(self.basic_path):
            os.makedirs(self.basic_path)
        if not os.path.exists(self.point_path):
            os.makedirs(self.point_path)
        if not os.path.exists(self.vtk_path):
            os.makedirs(self.vtk_path)

    def monitor_basic(self, sims: Simulation, scene: myScene, generator: GenerateManager):
        for patchID in range(scene.patchNum[0]):
            mygenerator: BodyGenerator = generator.myGenerator[patchID]
            knotU = np.ascontiguousarray(np.array(mygenerator.objects.knotvector_u))
            knotV = np.ascontiguousarray(np.array(mygenerator.objects.knotvector_v))
            knotW = np.ascontiguousarray(np.array(mygenerator.objects.knotvector_w))
            degree = np.ascontiguousarray(np.array(mygenerator.objects.degree))
            points = np.ascontiguousarray(np.array(mygenerator.objects.ctrlpts))
            weights = np.ascontiguousarray(np.array(mygenerator.objects.weights))
            knot_number = np.ascontiguousarray(np.array([knotU.shape[0], knotV.shape[0], knotW.shape[0]]))
            numctrlpts = np.ascontiguousarray(np.array(mygenerator.objects.numctrlpts))
            np.savez(self.basic_path+f'/Patch{patchID}Basic{sims.current_print:06d}', knotvector_u=knotU, knotvector_v=knotV, knotvector_w=knotW, degree=degree,
                                                                                      points=points, weights=weights, knot_number=knot_number, control_point_number=numctrlpts)

    def monitor_point(self, sims: Simulation, scene: myScene, generator: GenerateManager):
        for patchID in range(scene.patchNum[0]):
            mygenerator: BodyGenerator = generator.myGenerator[patchID]
            numCtrlPts = mygenerator.objects.numctrlpts[0] * mygenerator.objects.numctrlpts[1] * mygenerator.objects.numctrlpts[2]
            points = np.ascontiguousarray(scene.ctrlpts.x.to_numpy()[0:numCtrlPts, patchID])
            weights = np.ascontiguousarray(scene.ctrlpts.weight.to_numpy()[0:numCtrlPts, patchID])
            displacement = np.ascontiguousarray(scene.ctrlpts.displacement.to_numpy()[0:numCtrlPts, patchID])
            state_vars = {}
            self.visualize_ctrlpts(points, state_vars)
            np.savez(self.basic_path+f'/Patch{patchID}Basic{sims.current_print:06d}', points=points, weights=weights, displacement=displacement)

    def visualize_ctrlpts(self, sims: Simulation, points, state_vars):
        pointsToVTK(self.vtk_path+f"ControlPoint{sims.current_print:06d}", points[:, 0], points[:, 1], points[:, 2], data=state_vars)

    def monitor_gauss_point(self, sims: Simulation, scene: myScene, generator: GenerateManager):
        for patchID in range(scene.patchNum[0]):
            mygenerator: BodyGenerator = generator.myGenerator[patchID]
            gauss_number = mygenerator.objects.gauss_number
            element_number = scene.element.enum[patchID, :]
            total_gauss_number = inner_multiply(gauss_number) * inner_multiply(list(element_number))
            stress = np.ascontiguousarray(scene.material.stateVars.stress.to_numpy()[0:total_gauss_number, patchID])
            deformation_gradient = np.ascontiguousarray(scene.material.stateVars.deformation_gradient.to_numpy()[0:total_gauss_number, patchID])
            stiffness_matrix = np.ascontiguousarray(scene.material.stateVars.stiffness_matrix.to_numpy()[0:total_gauss_number, patchID])
            np.savez(self.basic_path+f'/Patch{patchID}GaussPoint{sims.current_print:06d}', stress=stress, deformation_gradient=deformation_gradient, stiffness_matrix=stiffness_matrix)

    def monitor_volume(self, sims: Simulation, scene: myScene, generator: GenerateManager):
        pass
            
    def visualize_volume(self, sims: Simulation, scene: myScene, mygenerator: BodyGenerator, state_vars):
        knotU = np.array(mygenerator.objects.knotvector_u)
        knotV = np.array(mygenerator.objects.knotvector_v)
        knotW = np.array(mygenerator.objects.knotvector_w)
        spaceU = (np.max(knotU) - np.min(knotU)) / mygenerator.resolution[0]
        spaceV = (np.max(knotV) - np.min(knotV)) / mygenerator.resolution[1]
        spaceW = (np.max(knotW) - np.min(knotW)) / mygenerator.resolution[2]
        numCtrlPts = mygenerator.objects.numctrlpts[0] * mygenerator.objects.numctrlpts[1] * mygenerator.objects.numctrlpts[2]
        points = scene.ctrlpts.x.to_numpy().reshape((numCtrlPts, 3))
        weights = scene.ctrlpts.weight.to_numpy().reshape(numCtrlPts)
        p, q, r = mygenerator.objects.degree[0], mygenerator.objects.degree[1], mygenerator.objects.degree[2]

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

        interpolatedPointsU = np.ascontiguousarray(interpolatedPoints[:, 0].reshape((numPtsW, numPtsV, numPtsU)))
        interpolatedPointsV = np.ascontiguousarray(interpolatedPoints[:, 1].reshape((numPtsW, numPtsV, numPtsU)))
        interpolatedPointsW = np.ascontiguousarray(interpolatedPoints[:, 2].reshape((numPtsW, numPtsV, numPtsU)))
        gridToVTK(self.vtk_path+f"NurbsVolume{sims.current_print:06d}", interpolatedPointsU, interpolatedPointsV, interpolatedPointsW, cellData={}, pointData=state_vars)