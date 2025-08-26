import sys
sys.path.append("/home/yj/works/GeoTaichi_Yihao_v1")
import pdb
from geotaichi import *

# ti.init(arch=ti.cpu, cpu_max_num_threads=1)
init(arch='cpu', debug=False)

pressure = 100  # in kPa
save_path = f"examples/ElementTest/DrainedNorSand/{int(pressure)}kpa"

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([3., 3., 3.]), 
                      background_damping=0., 
                      gravity=ti.Vector([0., 0., 0.]),
                      alphaPIC=0.0, 
                      mapping="USL", 
                      shape_function="Linear",
                      stress_integration="SubStepping",
                      gauss_number=2)

mpm.set_solver(solver={
                           "Timestep":                   1e-4,
                           "SimulationTime":             30,
                           "SaveInterval":               0.2,
                           "SavePath":                   save_path
                      })

mpm.memory_allocate(memory={
                                "max_material_number":    1,
                                "max_particle_number":    80,
                                "max_constraint_number":  {
                                                               "max_velocity_constraint":   24,
                                                               "max_traction_constraint":   24
                                                          }
                            })

mpm.add_material(model="NorSand",
                 material={
                               "MaterialID": 1,
                               "Density": 1650,  # typical sand density (kg/m3)
                               "PoissonRatio": 0.15,  # typical for sands
                               "Lambda": 0.02,  # slope of CSL defined in e-ln(p) space (0.01~0.07)
                               "Gamma": 0.9,  # CSL mean effective stress at p=1kPa (0.9~1.4)
                               "M_tc": 1.20,  # critical friction ratio, with triaxial compression as a reference condition (1.2~1.5)
                               "N": 0.3,  # Volumetric coupling parameter (0.2~0.5)
                               "H0": 300.0,  # hardening modulus, such that H = H_0 + H_y * psi (75~500)
                               "Hy": 50.0,  # hardening modulus (200~500)
                               "chi_tc": 3.5,  # Relates minimum dilatancy to corresponding psi_tc
                               "Ir": 300.0,  # elasticity (100~600)
                               "psi_0": 0.05,  # initial state parameter (-0.25~0.25)
                               "G_max": 4000,  # max shear modulus (kPa) (30~100 MPa)
                               "p_ref": 100,  # reference pressure (kPa) (50~1000 kPa)
                               "m": 0.0,  # small-strain modulus exponent (0.01~0.05)
                 })

mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([1., 1., 1.])
                        })


mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([1., 1., 1.]),
                            "BoundingBoxSize": ti.Vector([1., 1., 1.]),
                            "zdirection": ti.Vector([0., 0., 1.])
                      }])

mpm.add_body(body={
                       "Template": [{
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "ParticleStress": {
                                                              "GravityField":     False,
                                                              "InternalStress":   ti.Vector([-pressure, -pressure, -pressure, 0., 0., 0.])
                                                         },
                                       "InitialVelocity":ti.Vector([0, 0, 0]),
                                       "FixVelocity":    ["Free", "Free", "Free"]    
                                       
                                   }]
                   })


# Top: push down the top surface with contant velocity
velocity = 0.002
mpm.scene.boundary.velocity_boundary[0].node = 37
mpm.scene.boundary.velocity_boundary[0].level = 0
mpm.scene.boundary.velocity_boundary[0].dirs = 2
mpm.scene.boundary.velocity_boundary[0].velocity = -velocity
mpm.scene.boundary.velocity_boundary[1].node = 38
mpm.scene.boundary.velocity_boundary[1].level = 0
mpm.scene.boundary.velocity_boundary[1].dirs = 2
mpm.scene.boundary.velocity_boundary[1].velocity = -velocity
mpm.scene.boundary.velocity_boundary[2].node = 41
mpm.scene.boundary.velocity_boundary[2].level = 0
mpm.scene.boundary.velocity_boundary[2].dirs = 2
mpm.scene.boundary.velocity_boundary[2].velocity = -velocity
mpm.scene.boundary.velocity_boundary[3].node = 42
mpm.scene.boundary.velocity_boundary[3].level = 0
mpm.scene.boundary.velocity_boundary[3].dirs = 2
mpm.scene.boundary.velocity_boundary[3].velocity = -velocity

# Bottom: push up the bottom surface with velocity 0.005
mpm.scene.boundary.velocity_boundary[4].node = 21
mpm.scene.boundary.velocity_boundary[4].level = 0
mpm.scene.boundary.velocity_boundary[4].dirs = 2
mpm.scene.boundary.velocity_boundary[4].velocity = velocity
mpm.scene.boundary.velocity_boundary[5].node = 22
mpm.scene.boundary.velocity_boundary[5].level = 0
mpm.scene.boundary.velocity_boundary[5].dirs = 2
mpm.scene.boundary.velocity_boundary[5].velocity = velocity
mpm.scene.boundary.velocity_boundary[6].node = 25
mpm.scene.boundary.velocity_boundary[6].level = 0
mpm.scene.boundary.velocity_boundary[6].dirs = 2
mpm.scene.boundary.velocity_boundary[6].velocity = velocity
mpm.scene.boundary.velocity_boundary[7].node = 26
mpm.scene.boundary.velocity_boundary[7].level = 0
mpm.scene.boundary.velocity_boundary[7].dirs = 2
mpm.scene.boundary.velocity_boundary[7].velocity = velocity
mpm.scene.boundary.velocity_list[0] = 8

# Top. Possitive pressure means compression (negative stress)
p = pressure/4.
mpm.scene.boundary.traction_boundary[0].node = 37
mpm.scene.boundary.traction_boundary[0].level = 0
mpm.scene.boundary.traction_boundary[0].dirs = 0
mpm.scene.boundary.traction_boundary[0].traction = p
mpm.scene.boundary.traction_boundary[1].node = 37
mpm.scene.boundary.traction_boundary[1].level = 0
mpm.scene.boundary.traction_boundary[1].dirs = 1
mpm.scene.boundary.traction_boundary[1].traction = p
mpm.scene.boundary.traction_boundary[2].node = 37
mpm.scene.boundary.traction_boundary[2].level = 0
mpm.scene.boundary.traction_boundary[2].dirs = 2
mpm.scene.boundary.traction_boundary[2].traction = -p

mpm.scene.boundary.traction_boundary[3].node = 38
mpm.scene.boundary.traction_boundary[3].level = 0
mpm.scene.boundary.traction_boundary[3].dirs = 0
mpm.scene.boundary.traction_boundary[3].traction = -p
mpm.scene.boundary.traction_boundary[4].node = 38
mpm.scene.boundary.traction_boundary[4].level = 0
mpm.scene.boundary.traction_boundary[4].dirs = 1
mpm.scene.boundary.traction_boundary[4].traction = p
mpm.scene.boundary.traction_boundary[5].node = 38
mpm.scene.boundary.traction_boundary[5].level = 0
mpm.scene.boundary.traction_boundary[5].dirs = 2
mpm.scene.boundary.traction_boundary[5].traction = -p

mpm.scene.boundary.traction_boundary[6].node = 41
mpm.scene.boundary.traction_boundary[6].level = 0
mpm.scene.boundary.traction_boundary[6].dirs = 0
mpm.scene.boundary.traction_boundary[6].traction = p
mpm.scene.boundary.traction_boundary[7].node = 41
mpm.scene.boundary.traction_boundary[7].level = 0
mpm.scene.boundary.traction_boundary[7].dirs = 1
mpm.scene.boundary.traction_boundary[7].traction = -p
mpm.scene.boundary.traction_boundary[8].node = 41
mpm.scene.boundary.traction_boundary[8].level = 0
mpm.scene.boundary.traction_boundary[8].dirs = 2
mpm.scene.boundary.traction_boundary[8].traction = -p

mpm.scene.boundary.traction_boundary[9].node = 42
mpm.scene.boundary.traction_boundary[9].level = 0
mpm.scene.boundary.traction_boundary[9].dirs = 0
mpm.scene.boundary.traction_boundary[9].traction = -p
mpm.scene.boundary.traction_boundary[10].node = 42
mpm.scene.boundary.traction_boundary[10].level = 0
mpm.scene.boundary.traction_boundary[10].dirs = 1
mpm.scene.boundary.traction_boundary[10].traction = -p
mpm.scene.boundary.traction_boundary[11].node = 42
mpm.scene.boundary.traction_boundary[11].level = 0
mpm.scene.boundary.traction_boundary[11].dirs = 2
mpm.scene.boundary.traction_boundary[11].traction = -p

# Bottom
mpm.scene.boundary.traction_boundary[12].node = 21
mpm.scene.boundary.traction_boundary[12].level = 0
mpm.scene.boundary.traction_boundary[12].dirs = 0
mpm.scene.boundary.traction_boundary[12].traction = p
mpm.scene.boundary.traction_boundary[13].node = 21
mpm.scene.boundary.traction_boundary[13].level = 0
mpm.scene.boundary.traction_boundary[13].dirs = 1
mpm.scene.boundary.traction_boundary[13].traction = p
mpm.scene.boundary.traction_boundary[14].node = 21
mpm.scene.boundary.traction_boundary[14].level = 0
mpm.scene.boundary.traction_boundary[14].dirs = 2
mpm.scene.boundary.traction_boundary[14].traction = p

mpm.scene.boundary.traction_boundary[15].node = 22
mpm.scene.boundary.traction_boundary[15].level = 0
mpm.scene.boundary.traction_boundary[15].dirs = 0
mpm.scene.boundary.traction_boundary[15].traction = -p
mpm.scene.boundary.traction_boundary[16].node = 22
mpm.scene.boundary.traction_boundary[16].level = 0
mpm.scene.boundary.traction_boundary[16].dirs = 1
mpm.scene.boundary.traction_boundary[16].traction = p
mpm.scene.boundary.traction_boundary[17].node = 22
mpm.scene.boundary.traction_boundary[17].level = 0
mpm.scene.boundary.traction_boundary[17].dirs = 2
mpm.scene.boundary.traction_boundary[17].traction = p

mpm.scene.boundary.traction_boundary[18].node = 25
mpm.scene.boundary.traction_boundary[18].level = 0
mpm.scene.boundary.traction_boundary[18].dirs = 0
mpm.scene.boundary.traction_boundary[18].traction = p
mpm.scene.boundary.traction_boundary[19].node = 25
mpm.scene.boundary.traction_boundary[19].level = 0
mpm.scene.boundary.traction_boundary[19].dirs = 1
mpm.scene.boundary.traction_boundary[19].traction = -p
mpm.scene.boundary.traction_boundary[20].node = 25
mpm.scene.boundary.traction_boundary[20].level = 0
mpm.scene.boundary.traction_boundary[20].dirs = 2
mpm.scene.boundary.traction_boundary[20].traction = p

mpm.scene.boundary.traction_boundary[21].node = 26
mpm.scene.boundary.traction_boundary[21].level = 0
mpm.scene.boundary.traction_boundary[21].dirs = 0
mpm.scene.boundary.traction_boundary[21].traction = -p
mpm.scene.boundary.traction_boundary[22].node = 26
mpm.scene.boundary.traction_boundary[22].level = 0
mpm.scene.boundary.traction_boundary[22].dirs = 1
mpm.scene.boundary.traction_boundary[22].traction = -p
mpm.scene.boundary.traction_boundary[23].node = 26
mpm.scene.boundary.traction_boundary[23].level = 0
mpm.scene.boundary.traction_boundary[23].dirs = 2
mpm.scene.boundary.traction_boundary[23].traction = p
mpm.scene.boundary.traction_list[0] = 24

mpm.select_save_data(grid=True, object=True)

mpm.run()

mpm.postprocessing(
    read_path= save_path,
    write_strain_component=True,
    write_background_grid=True)
