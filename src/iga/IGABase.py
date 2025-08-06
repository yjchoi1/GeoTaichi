import time


from src.iga.engine.Engine import Engine
from src.iga.generator.BodyGenerator import GenerateManager
from src.iga.Recorder import WriteFile
from src.iga.SceneManager import myScene
from src.iga.Simulation import Simulation
from src.utils.constants import Threshold
from src.utils.ObjectIO import DictIO
from src.utils.TypeDefination import vec3f


class Solver:
    sims: Simulation
    generator: GenerateManager
    engine: Engine
    recorder: WriteFile

    def __init__(self, sims, generator, engine, recorder):
        self.sims = sims
        self.generator = generator
        self.engine = engine
        self.recorder = recorder

        self.last_save_time = 0.
        self.last_print_time = 0.

    def set_callback_function(self, kwargs):
        functions = DictIO.GetAlternative(kwargs, "function", None)
        if functions is None:
            self.function = self.no_operation
        else:
            self.function = functions

    def no_operation(self):
        pass
        
    def save_file(self, scene):
        print('# Step =', self.sims.current_step, '   ', 'Save Number =', self.sims.current_print, '   ', 'Simulation time =', self.sims.current_time, '\n')
        self.recorder.output(self.sims, scene)

    def Solver(self, scene):
        print("#", " Start Simulation ".center(67,"="), "#")
       
        self.recorder.monitor_basic(self.sims, scene, self.generator)
        self.engine.pre_calculation(scene)
        if self.sims.current_time < Threshold:
            self.save_file(scene)
            self.sims.current_step += 1
            self.sims.current_print += 1
            self.last_save_time = -1. * self.sims.delta

        start_time = time.time()
        while self.sims.current_time <= self.sims.time:
            self.core(scene)

            if self.sims.current_time - self.last_save_time + 0.1 * self.sims.delta > self.sims.save_interval:
                self.save_file(scene)
                self.last_save_time = 1. * self.sims.current_time
                self.sims.current_print += 1

            self.sims.current_time += self.sims.delta
            self.sims.current_step += 1

        end_time = time.time()

        if abs(self.sims.current_time - self.last_save_time) > self.sims.save_interval:
            self.save_file(scene)
            self.last_save_time = 1. * self.sims.current_time
            self.sims.current_print += 1

        print('Physical time = ', end_time - start_time)
        print("#", " End Simulation ".center(67,"="), "#", '\n')

    def core(self, scene: myScene):
        self.engine.compute(self.sims, scene)
        self.function()