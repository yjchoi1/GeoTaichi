import taichi as ti

from src.mpm.boundaries.BoundaryCore import apply_reflection_constraint, apply_velocity_constraint
from src.mpm.Simulation import Simulation
from src.mpm.SceneManager import myScene
from src.mpm.SpatialHashGrid import SpatialHashGrid
from src.mpm.engines.Engine import Engine
from src.mpm.engines.EngineKernel import *
from src.mpm.engines.AssembleMatrixKernel import *
from src.mpm.engines.FreeSurfaceDetection import *
from src.utils.linalg import no_operation


class IncompressibleEngine(Engine):
    def __init__(self, sims) -> None:
        super().__init__(sims)
        self.poisson_solver = None
        self.operator = None

    def choose_engine(self, sims: Simulation):
        if sims.discretization == "FDM":
            if sims.linear_solver == "PCG":
                self.compute = self.fdm_discretization_matrix_free
        elif sims.discretization == "FEM":
            self.compute = self.fem_discretization

    def manage_function(self, sims: Simulation):
        self.is_verlet_update = self.is_need_update_verlet_table
        self.bulid_neighbor_list = no_operation
        if sims.neighbor_detection:
            self.compute_nodal_kinematic = no_operation
            self.execute_board_serach = self.update_verlet_table
            self.system_resolve = self.compute_nodal_kinematic
            self.bulid_neighbor_list = self.board_search

            self.free_surface_by_geometry = no_operation
            if sims.free_surface_detection:
                self.free_surface_by_geometry = self.detection_free_surface

            self.compute_boundary_direction = no_operation
            if sims.boundary_direction_detection:
                self.compute_boundary_direction = self.detection_boundary_direction

    def choose_boundary_constraints(self, sims: Simulation, scene: myScene):
        pass

    def reset_grid_message(self, scene: myScene):
        scene.node.grid_reset(scene.mass_cut_off)

    def compute_nodal_kinematics(self, sims: Simulation, scene: myScene):
        kernel_mass_momentum_mac_cell_p2g(scene.element.grid_nodes, int(scene.particleNum[0]), scene.element.grid_size, scene.element.igrid_size, scene.node, scene.particle, scene.element.calLength, scene.element.boundary_type)

    def update_velocity_gradient_2D(self, sims: Simulation, scene: myScene):
        for materialID in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[materialID]
            end_index = scene.material.mapping[materialID + 1]
            kernel_update_velocity_gradient_2D(scene.element.grid_nodes, start_index, end_index, sims.dt, scene.node, scene.particle, scene.material.materialID, scene.material.matProps[materialID + 1], scene.material.stateVars,
                                               scene.element.LnID, scene.element.dshape_fn, scene.element.node_size)
    
    def update_velocity_gradient(self, sims: Simulation, scene: myScene):
        for materialID in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[materialID]
            end_index = scene.material.mapping[materialID + 1]
            kernel_update_velocity_gradient(scene.element.grid_nodes, start_index, end_index, sims.dt, scene.node, scene.particle, scene.material.materialID, scene.material.matProps[materialID + 1], scene.material.stateVars,
                                            scene.element.LnID, scene.element.dshape_fn, scene.element.node_size)

    def compute_force(self, sims: Simulation, scene: myScene):
        for materialID in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[materialID]
            end_index = scene.material.mapping[materialID + 1]
            kernel_viscous_force_p2g(scene.element.grid_nodes, start_index, end_index, sims.dt, scene.node, scene.particle, scene.element.LnID, scene.element.dshape_fn, scene.element.node_size, scene.material.matProps[materialID + 1])

    def compute_force_2D(self, sims: Simulation, scene: myScene):
        for materialID in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[materialID]
            end_index = scene.material.mapping[materialID + 1]
            kernel_viscous_force_p2g_2D(scene.element.grid_nodes, start_index, end_index, sims.dt, scene.node, scene.particle, scene.element.LnID, scene.element.dshape_fn, scene.element.node_size, scene.material.matProps[materialID + 1])

    def reflection_constraints(self, sims: Simulation, scene: myScene):
        apply_reflection_constraint(scene.mass_cut_off, int(scene.boundary.reflection_list[0]), scene.boundary.reflection_boundary, scene.is_rigid, scene.node)

    def velocity_constraints(self, sims: Simulation, scene: myScene):
        apply_velocity_constraint(scene.mass_cut_off, int(scene.boundary.velocity_list[0]), scene.boundary.velocity_boundary, scene.is_rigid, scene.node)

    def apply_boundary_constraints(self, sims: Simulation, scene: myScene):
        self.apply_reflection_constraints(sims, scene)
        self.apply_velocity_constraints(sims, scene)

    def compute_grid_velcity(self, sims: Simulation, scene: myScene):
        kernel_compute_grid_velocity_gravity(scene.mass_cut_off, sims.gravity, sims.dt, scene.node)

    def identify_fluid_domain(self, scene: myScene):
        kernel_find_fluid_domain(int(scene.particleNum[0]), scene.element.igrid_size, scene.element.cell.type, scene.particle)

    def prepare_poisson_equations(self, sims: Simulation, scene: myScene):
        self.unknow_vector.fill(0)
        for materialID in range(scene.material.mapping.shape[0] - 1):
            kernel_assemble_poisson_equation_dynamic(scene.element.ghost_cell, scene.element.cnum, scene.element.gnum, scene.element.grid_size, scene.element.igrid_size, sims.dt, scene.node, scene.element.flag, scene.element.cell.surface_tension, scene.element.cell.type, scene.material.matProps[materialID + 1], self.right_hand_vector)
        kernel_preconditioning_poisson_equation_matrix(scene.element.ghost_cell, scene.element.cnum, scene.element.grid_size, scene.element.igrid_size, scene.element.flag, scene.element.cell.type, self.diag_A)

    def apply_pressures(self, sims: Simulation, scene: myScene):
        kernel_update_cell_pressure(scene.element.ghost_cell, scene.element.cnum, scene.element.cell.pressure, scene.element.flag, scene.element.cell.type, self.unknow_vector)
        for materialID in range(scene.material.mapping.shape[0] - 1):
            kernel_correct_velocity(scene.element.ghost_cell, scene.element.gnum, scene.element.grid_size, sims.dt, scene.material.matProps[materialID + 1], scene.element.cell.pressure, scene.element.cell.surface_tension, scene.element.cell.type, scene.node)
        kernel_compute_grid_acceleration(scene.mass_cut_off, scene.node)

    def compute_particle_kinematics(self, sims: Simulation, scene: myScene):
        kernel_kinemaitc_mac_cell_g2p(scene.element.grid_nodes, sims.alphaPIC, sims.dt, int(scene.particleNum[0]), scene.element.grid_size, scene.element.igrid_size, scene.node, scene.particle, scene.element.calLength, scene.element.boundary_type)
    
    def find_active_nodes(self, scene: myScene):
        find_active_fdm_cell(scene.element.ghost_cell, scene.element.cellSum, scene.element.cnum, scene.element.cell.type, scene.element.flag)
        scene.element.pse.run(scene.element.flag)
        return set_active_fdm_cell_dofs(scene.element.ghost_cell, scene.element.cellSum, scene.element.cnum, scene.element.cell.type, scene.element.flag)

    def pre_calculation(self, sims: Simulation, scene: myScene, neighbor: SpatialHashGrid):
        scene.element.calculate_characteristic_length(sims, int(scene.particleNum[0]), scene.particle, scene.psize)
        if sims.neighbor_detection:
            grid_mass_reset(scene.mass_cut_off, scene.node)
            scene.check_in_domain(sims)
            self.find_free_surface_by_density(sims, scene)
            neighbor.place_particles(scene)
            self.compute_boundary_direction(scene, neighbor)
            self.free_surface_by_geometry(scene, neighbor)
            grid_mass_reset(scene.mass_cut_off, scene.node)
        if sims.discretization == "FDM": 
            if sims.linear_solver == "PCG":
                scene.element.cell.set_ptr(pressure=None, cell_type=None)
                init_boundary(scene.element.ghost_cell, scene.element.cnum, scene.element.cell.type)
                self.identify_fluid_domain(scene)
                total_dofs = estimate_active_fdm_cell_dofs(scene.element.ghost_cell, scene.element.cnum, scene.element.cell.type)

                from src.linear_solver.MatrixFreePCG import MatrixFreePCG
                from src.mpm.engines.Operator import PoissonEquationOperator
                self.possion_solver = MatrixFreePCG(int(sims.dof_multiplier * total_dofs))
                self.operator = PoissonEquationOperator(sims.dimension)
                self.unknow_vector = ti.field(dtype=float)                               
                self.right_hand_vector = ti.field(dtype=float)                               
                self.diag_A = ti.field(dtype=float)   
                ti.root.dense(ti.i, int(sims.dof_multiplier * total_dofs)).place(self.unknow_vector, self.right_hand_vector, self.diag_A)
                self.operator.link_ptrs(scene)
                self.operator.update_active_dofs(int(sims.dof_multiplier * total_dofs))
        elif sims.discretization == "FEM":
            from src.linear_solver.CompressedSparseRow import CompressedSparseRow
            self.poisson_solver = CompressedSparseRow()

    def fdm_discretization_matrix_free(self, sims: Simulation, scene: myScene, neighbor=None):
        self.identify_fluid_domain(scene)
        self.compute_nodal_kinematics(sims, scene)
        self.compute_grid_velcity(sims, scene)
        #self.apply_boundary_constraints(sims, scene)
        enforce_boundary(scene.element.ghost_cell, scene.element.cnum, scene.element.cell.type, scene.node)
        total_dofs = self.find_active_nodes(scene)
        self.prepare_poisson_equations(sims, scene)
        #print(self.right_hand_vector)
        self.possion_solver.solve(self.operator, self.right_hand_vector, self.unknow_vector, self.diag_A, total_dofs, maxiter=total_dofs, tol=sims.residual_tolerance)
        #print(self.unknow_vector)
        self.apply_pressures(sims, scene)
        self.compute_particle_kinematics(sims, scene)

    def fem_discretization(self, sims, scene: myScene, neighbor=None):
        pass