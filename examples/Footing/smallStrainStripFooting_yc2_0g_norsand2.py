import sys
sys.path.append('/home/yj/works/GeoTaichi_Yihao_v1')

from geotaichi import *

init(device_memory_GB=7.0, dim=2, arch='cpu')

output_path = 'smallStrainStripFooting_yc2_0g_norsand2'

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([4.5, 7.0]),
                      background_damping=0.1,
                      gravity=ti.Vector([0., -9.81]),  # Independent var
                      alphaPIC=0.,  # i.e., FLIP
                      mapping="USL",
                      shape_function="CubicBSpline")  # Independent var

mpm.set_solver(solver={
    "Timestep": 1e-6,  # 2e-5
    "SimulationTime": 1.0,  #  7.0
    "SaveInterval": 0.01,
    "SavePath": output_path
})

mpm.memory_allocate(memory={
    "max_material_number": 1,
    "max_particle_number": 514000,
    "max_constraint_number": {
        "max_velocity_constraint": 12680,
        "max_particle_traction_constraint": 178540,
        "max_reflection_constraint": 178540
    }
})

mpm.add_contact(contact_type="MPMContact", friction=0.)

mpm.add_material(model="NorSand",
                 material={
                     "MaterialID": 1,
                     "Density": 1650,  # typical sand density (kg/m3)
                     "PoissonRatio": 0.15,  # typical for sands
                     "Lambda": 0.02,  # slope of CSL defined in e-ln(p) space (0.01~0.07)
                     "Gamma": 0.9,  # CSL mean effective stress at p=1kPa (0.9~1.4)
                     "M_tc": 1.20,
                     # critical friction ratio, with triaxial compression as a reference condition (1.2~1.5)
                     "N": 0.3,  # Volumetric coupling parameter (0.2~0.5)
                     "H0": 300.0,  # hardening modulus, such that H = H_0 + H_y * psi (75~500)
                     "Hy": 50.0,  # hardening modulus (200~500)
                     "chi_tc": 3.5,  # Relates minimum dilatancy to corresponding psi_tc
                     "Ir": 100.0,  # elasticity (100~600)
                     "psi_0": 0.05,  # initial state parameter (-0.25~0.25)
                     "G_max": 100,  # max shear modulus (kPa) (30~100 MPa)
                     "p_ref": 100,  # reference pressure (kPa) (50~1000 kPa)
                     "m": 0.0,  # small-strain modulus exponent (0.01~0.05)
                 })

mpm.add_material(model="MohrCoulomb",
                 material={
                     "MaterialID": 2,
                     "Density": 1800.,
                     "YoungModulus": 10e6,
                     "PossionRatio": 0.33,
                     "Cohesion": 1000,  # 1000
                     "Friction": 30,
                     "Dilation": 0.
                 })

mpm.add_element(element={
    "ElementType": "Q4N2D",
    "ElementSize": ti.Vector([0.05, 0.05]),
    # "Contact": {
    #     "ContactDetection": "MPMContact",
    #     "Friction": 0.395,
    # }
})

mpm.add_region(region=[
    {
        "Name": "region1",
        "Type": "Rectangle2D",
        "BoundingBoxPoint": ti.Vector([0, 0]),
        "BoundingBoxSize": ti.Vector([4.5, 6.0]),
        "ydirection": ti.Vector([0., 1.])
    },
    {
        "Name": "region2",
        "Type": "Rectangle2D",
        "BoundingBoxPoint": ti.Vector([0, 6.0]),
        "BoundingBoxSize": ti.Vector([0.45, 1.0]),
        "ydirection": ti.Vector([0., 1.])
    },
    {
        "Name": "region3",
        "Type": "Rectangle2D",
        "BoundingBoxPoint": ti.Vector([0.45, 6.0]),
        "BoundingBoxSize": ti.Vector([4.5 - 0.45, 1.0]),
        # "BoundingBoxPoint": ti.Vector([0.45, 6.0 - 0.05]),
        # "BoundingBoxSize": ti.Vector([4.5 - 0.45, 0.05]),
        "ydirection": ti.Vector([0., 1.])
    }
]
)

mpm.add_body(body={
    "Template": [
        {
            "RegionName": "region1",
            "nParticlesPerCell": 2,
            "BodyID": 0,
            "MaterialID": 1,
            "ParticleStress": {
                "GravityField": True,
                # "InternalStress": ti.Vector([-50e3, -50e3, -50e3, 0., 0., 0.])
            },
            "InitialVelocity": ti.Vector([0., 0.]),
            "FixVelocity": ["Free", "Free"]
        },
        {
            "RegionName": "region3",
            "nParticlesPerCell": 2,
            "BodyID": 2,
            "MaterialID": 2,
            "ParticleStress": {
                "GravityField": True,
                # "InternalStress": ti.Vector([-50e3, -50e3, -50e3, 0., 0., 0.])
            },
            "InitialVelocity": ti.Vector([0., 0.]),
            "FixVelocity": ["Free", "Free"]
        },
        {
            "RegionName": "region2",
            "nParticlesPerCell": 2,
            "BodyID": 1,
            "RigidBody": True,
            "Density": 1800,
            # "Density": 1000,
            # "ParticleStress": {},
            # "Traction": {},
            "InitialVelocity": ti.Vector([0., -0.1]),
            "FixVelocity": ["Fix", "Fix"]
        }
    ]
})

mpm.add_boundary_condition(boundary=[
    {
        "BoundaryType": "VelocityConstraint",
        "Velocity": [0., 0.],
        "StartPoint": [0., 0.],
        "EndPoint": [4.5, 0.],  # Bottom boundary - fix both x and y
    },
    {
        "BoundaryType": "VelocityConstraint",
        "Velocity": [0., None],
        "StartPoint": [0., 0.],
        "EndPoint": [0., 7.0]  # Left boundary - fix only x
    },
    {
        "BoundaryType": "VelocityConstraint",
        "Velocity": [0., None],
        "StartPoint": [4.5, 0.0],
        "EndPoint": [4.5, 7.0]  # Right boundary - fix only x
    }
])

mpm.select_save_data(grid=True, object=True)

mpm.run()

mpm.postprocessing(
    write_background_grid=True,
    write_strain_component=True,
    read_path=output_path)
