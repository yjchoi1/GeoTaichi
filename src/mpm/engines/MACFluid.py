import taichi as ti

from src.mpm.Simulation import Simulation
from src.mpm.SceneManager import myScene
from src.mpm.SpatialHashGrid import SpatialHashGrid
from src.mpm.engines.ULExplicitEngine import Engine
from src.mpm.engines.EngineKernel import *
from src.mpm.engines.FreeSurfaceDetection import *
from src.linear_solver.MultiGridPCG import MGPCGPoissonSolver
from src.utils.linalg import no_operation


@ti.data_oriented
class MACFluid(Engine):
    def __init__(self, sims: Simulation) -> None:
        super().__init__(sims)
        self.dimension = sims.dimension
        self.n_mg_levels = 4
        self.pre_and_post_smoothing = 2
        self.bottom_smoothing = 10
        self.iterations = 50
        self.gravity = ti.Vector([0., -9.8])
        self.p0 = 0.01
        
    def manage_function(self, sims):
        self.bulid_neighbor_list = no_operation
        self.compute = self.substep

    def choose_engine(self, sims: Simulation):
        pass

    def choose_boundary_constraints(self, sims: Simulation, scene: myScene):
        pass

    def valid_contact(self, sims: Simulation, scene: myScene):
        pass

    def pre_calculation(self, sims: Simulation, scene: myScene, neighbor: SpatialHashGrid):
        self.particle = scene.particle
        self.grid = scene.node
        self.element = scene.element
        self.material = scene.material.matProps
        self.scale_A = sims.delta / (self.material[1].density * self.element.grid_size[0] * self.element.grid_size[0])
        self.scale_b = 1. / self.element.grid_size[0]
        self.poisson_solver = MGPCGPoissonSolver(sims.dimension, self.element.cnum, self.n_mg_levels, self.pre_and_post_smoothing, self.bottom_smoothing)
        self.init_boundary()
        self.pressure_p2g(int(scene.particleNum[0]))

    def reset_grid_message(self, scene):
        pass

    @ti.kernel
    def init_boundary(self):         
        for I in ti.grouped(self.grid.cell_type):
            if any(I <= 2) or any(I >= self.element.gnum - 3):
                self.grid.cell_type[I] = 2    

    @ti.func
    def is_valid(self, I):
        return all(I >= 0) and all(I < self.element.gnum)

    @ti.func
    def is_fluid(self, I):
        return self.is_valid(I) and int(self.grid.cell_type[I]) == 1

    @ti.func
    def is_solid(self, I):
        return not self.is_valid(I) or int(self.grid.cell_type[I]) == 2

    @ti.func
    def is_air(self, I):
        return self.is_valid(I) and int(self.grid.cell_type[I]) == 0
    
    @ti.kernel
    def _reset(self):
        for k in ti.static(range(self.dimension)):
            self.grid.velocity[k].fill(0)
            self.grid.grid_m[k].fill(0)

    @ti.kernel
    def _reset_tension(self):
        for I in ti.grouped(self.grid.tension):
            self.grid.tension[I] = ti.Matrix.zero(float, self.dimension, 1)

    @ti.kernel
    def calculate_grid_velocity(self):
        for k in ti.static(range(self.dimension)):
            for I in ti.grouped(self.grid.grid_m[k]):
                if self.grid.grid_m[k][I] > 0:
                    self.grid.velocity[k][I] /= self.grid.grid_m[k][I]

    @ti.kernel
    def add_gravity(self, dt: ti.template()):
        for k in ti.static(range(self.dimension)):
            if ti.static(self.gravity[k] != 0):
                g = self.gravity[k]
                for I in ti.grouped(self.grid.velocity[k]):
                    self.grid.velocity[k][I] += g * dt[None]

    @ti.kernel
    def enforce_boundary(self):
        for I in ti.grouped(self.grid.cell_type):
            if self.grid.cell_type[I] == 2:
                if I[0] <= 2 or I[0] >= self.element.gnum[0] - 3:
                    self.grid.velocity[0][I] = 0
                    self.grid.velocity[0][I + ti.Vector.unit(self.dimension, 0)] = 0
                elif I[1] <= 2 or I[1] >= self.element.gnum[1] - 3:
                    self.grid.velocity[1][I] = 0
                    self.grid.velocity[1][I + ti.Vector.unit(self.dimension, 1)] = 0
                else:
                    self.grid.velocity[0][I] = 0
                    self.grid.velocity[0][I + ti.Vector.unit(self.dimension, 0)] = 0
                    self.grid.velocity[1][I] = 0
                    self.grid.velocity[1][I + ti.Vector.unit(self.dimension, 1)] = 0

    @ti.kernel
    def apply_pressure(self, dt: ti.template()):
        dx = self.element.cal_min_grid_size()
        scale = dt[None] / (self.material[1].density * dx)
        for k in ti.static(range(self.dimension)):
            for I in ti.grouped(self.grid.cell_type):
                I_1 = I - ti.Vector.unit(self.dimension, k)
                if self.is_fluid(I_1) or self.is_fluid(I):
                    if self.is_solid(I_1) or self.is_solid(I): self.grid.velocity[k][I] = 0
                    elif self.is_air(I):
                        self.grid.velocity[k][I] -= scale * (self.p0 - self.grid.pressure[I_1])
                    elif self.is_air(I_1):
                        self.grid.velocity[k][I] -= scale * (self.grid.pressure[I] - self.p0)
                    else: self.grid.velocity[k][I] -= scale * (self.grid.pressure[I] - self.grid.pressure[I_1])

    @ti.func
    def splat_vp_apic(self, data, weights, pos, v, c, stagger, mass):
        base = (pos / self.element.grid_size - (stagger + 0.5)).cast(int)
        fx = pos / self.element.grid_size - (base.cast(float) + stagger)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                dpos = (offset.cast(float) - fx) * self.element.grid_size
                weight = w[i][0] * w[j][1]
                data[base + offset] += weight * mass * (v + c.dot(dpos))
                weights[base + offset] += weight * mass

    @ti.func
    def sample_vp_apic(self, data, pos, stagger):
        base = (pos / self.element.grid_size - (stagger + 0.5)).cast(int)
        fx = pos / self.element.grid_size - (base.cast(float) + stagger)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        v_pic = 0.0
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            weight = w[i][0] * w[j][1]
            v_pic += weight * data[base + offset]
        return v_pic

    @ti.func
    def sample_cp_apic(self, grid_v, xp, stagger):
        base = (xp / self.element.grid_size - (stagger + 0.5)).cast(int)
        fx = xp / self.element.grid_size - (base.cast(float) + stagger)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        w_grad = [fx - 1.5, -2 * (fx - 1), fx - 0.5]
        cp = ti.Vector([0.0, 0.0])

        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            weight_grad = ti.Vector([w_grad[i][0] * w[j][1]/self.element.grid_size[0], w[i][0] * w_grad[j][1]/self.element.grid_size[1]])
            cp += weight_grad * grid_v[base + offset]
        return cp
    
    @ti.func
    def splat_pressure(self, data, pos, pressure):
        base = (pos / self.element.grid_size - 1).cast(int)
        fx = pos / self.element.grid_size - (base.cast(float) + 0.5)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                weight = w[i][0] * w[j][1]
                data[base + offset] += weight * pressure

    @ti.func
    def sample_pressure(self, data, pos):
        base = (pos / self.element.grid_size - 1).cast(int)
        fx = pos / self.element.grid_size - (base.cast(float) + 0.5)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        pressure = 0.0
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            weight = w[i][0] * w[j][1]
            pressure += weight * data[base + offset]
        return pressure

    @ti.kernel
    def mass_momentum_p2g(self, particleNum: int):
        for k in ti.static(range(self.dimension)):
            self.grid.velocity[k].fill(0)
            self.grid.grid_m[k].fill(0)

        for p in range(particleNum):
            stagger = 0.5 * (1 - ti.Vector.unit(self.dimension, 0))
            self.splat_vp_apic(self.grid.velocity[0], self.grid.grid_m[0], self.particle.x[p], self.particle[p].v[0], self.particle[p].xvelocity_gradient, stagger, self.particle[p].m)
            stagger = 0.5 * (1 - ti.Vector.unit(self.dimension, 1))
            self.splat_vp_apic(self.grid.velocity[1], self.grid.grid_m[1], self.particle.x[p], self.particle[p].v[1], self.particle[p].yvelocity_gradient, stagger, self.particle[p].m)
            if ti.static(self.dimension == 3):
                stagger = 0.5 * (1 - ti.Vector.unit(self.dimension, 2))
                self.splat_vp_apic(self.grid.velocity[2], self.grid.grid_m[2], self.particle.x[p], self.particle[p].v[2], self.particle[p].zvelocity_gradient, stagger, self.particle[p].m)

        for k in ti.static(range(self.dimension)):
            for I in ti.grouped(self.grid.grid_m[k]):
                if self.grid.grid_m[k][I] > 0:
                    self.grid.velocity[k][I] /= self.grid.grid_m[k][I]

    @ti.kernel
    def identify_fluid_p(self, particleNum: int):
        for I in ti.grouped(self.grid.cell_type):
            if not self.is_solid(I):
                self.grid.cell_type[I] = 0
        for p in range(particleNum):
            pos = self.particle.x[p]
            idx = ti.cast(ti.floor(pos / self.element.grid_size), int)
            if not self.is_solid(idx):
                self.grid.cell_type[idx] = 1

    @ti.kernel
    def mark_valid(self, k : ti.template()):
        for I in ti.grouped(self.grid.velocity[k]):
            I_1 = I - ti.Vector.unit(self.dimension, k)
            if self.is_fluid(I_1) or self.is_fluid(I):
                self.grid.valid[I] = 1
            else:
                self.grid.valid[I] = 0

    @ti.kernel
    def kinemaitc_g2p(self, particleNum: int, dt: ti.template()):
        for p in range(particleNum):
            stagger = 0.5 * (1 - ti.Vector.unit(self.dimension, 0))
            self.particle[p].v[0] = self.sample_vp_apic(self.grid.velocity[0], self.particle[p].x, stagger)
            self.particle[p].xvelocity_gradient = self.sample_cp_apic(self.grid.velocity[0], self.particle[p].x, stagger)
            stagger = 0.5 * (1 - ti.Vector.unit(self.dimension, 1))
            self.particle[p].v[1] = self.sample_vp_apic(self.grid.velocity[1], self.particle[p].x, stagger)
            self.particle[p].yvelocity_gradient = self.sample_cp_apic(self.grid.velocity[1], self.particle[p].x, stagger)
            if ti.static(self.dimension == 3):
                stagger = 0.5 * (1 - ti.Vector.unit(self.dimension, 2))
                self.particle[p].v[2] = self.sample_vp_apic(self.grid.velocity[2], self.particle[p].x, stagger)
                self.particle[p].zvelocity_gradient = self.sample_cp_apic(self.grid.velocity[2], self.particle[p].x, stagger)
            self.particle[p].x += self.particle[p].v * dt[None]

    @ti.kernel
    def pressure_p2g(self, particleNum: int):
        for np in range(particleNum):
            self.splat_pressure(self.grid.pressure, self.particle[np].x, self.particle[np].pressure)
    
    @ti.kernel
    def pressure_g2p(self, particleNum: int):
        for np in range(particleNum):
            self.particle[np].pressure = self.sample_pressure(self.grid.pressure, self.particle[np].x)

    @ti.kernel
    def build_b_kernel(self, cell_type: ti.template(), b: ti.template()):
        for I in ti.grouped(cell_type):
            if cell_type[I] == 1:
                for k in ti.static(range(self.dimension)):
                    offset = ti.Vector.unit(self.dimension, k)
                    b[I] += (self.grid.velocity[k][I] - self.grid.velocity[k][I + offset])
                b[I] *= self.scale_b

        for I in ti.grouped(cell_type):
            if cell_type[I] == 1:
                for k in ti.static(range(self.dimension)):
                    for s in ti.static((-1, 1)):
                        offset = ti.Vector.unit(self.dimension, k) * s
                        if cell_type[I + offset] == 2:
                            if s < 0: b[I] -= self.scale_b * (self.grid.velocity[k][I] - 0)
                            else: b[I] += self.scale_b * (self.grid.velocity[k][I + offset] - 0)
                        elif cell_type[I + offset] == 0:
                            b[I] += self.scale_A * self.p0
                                
    def build_b(self):
        self.build_b_kernel(self.poisson_solver.grid_type[0], self.poisson_solver.b)

    @ti.kernel
    def build_A_kernel(self, grid_type : ti.template(), Adiag : ti.template(), Ax : ti.template()):
        for I in ti.grouped(grid_type):
            if grid_type[I] == 1:
                for k in ti.static(range(self.dimension)):
                    for s in ti.static((-1, 1)):
                        offset = ti.Vector.unit(self.dimension, k) * s
                        if grid_type[I + offset] == 1:
                            Adiag[I] += self.scale_A
                            if ti.static(s > 0):
                                Ax[I][k] = -self.scale_A
                        elif grid_type[I + offset] == 0:
                            Adiag[I] += self.scale_A

    def build_A(self, level):
        self.build_A_kernel(self.poisson_solver.grid_type[level], self.poisson_solver.Adiag[level], self.poisson_solver.Ax[level])
        
    def solve_pressure(self):
        self.poisson_solver.reinitialize(self.grid.cell_type)
        self.build_b()
        self.build_A(0)

        for l in range(1, self.n_mg_levels):
            self.poisson_solver.init_gridtype(self.poisson_solver.grid_type[l - 1], self.poisson_solver.grid_type[l])
            self.build_A(l)
        self.poisson_solver.solve(self.iterations)
        self.grid.pressure.copy_from(self.poisson_solver.x)

    def substep(self, sims: Simulation, scene: myScene):
        self.mass_momentum_p2g(int(scene.particleNum[0]))
        self.identify_fluid_p(int(scene.particleNum[0]))
        self.add_gravity(sims.dt)
        self.enforce_boundary()
        self.solve_pressure()
        self.apply_pressure(sims.dt)
        self.kinemaitc_g2p(int(scene.particleNum[0]), sims.dt)
        #self.pressure_g2p(int(scene.particleNum[0]))