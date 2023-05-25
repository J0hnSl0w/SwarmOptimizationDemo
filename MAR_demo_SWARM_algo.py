import argparse
import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation


class SwarmAlgoDemo:
    def __init__(self, population: int, position_min: float, position_max: float, generation: int,
                 fitness_criterion: float):
        """
        Demó szkript a SWARM algorimtus implementálásával.

        :param population: A populáció mérete
        :param position_min: Minimum pozíció
        :param position_max: Maximum pozíció
        :param generation: Generűciók maximális száma, ami alatt meg kell találni az optimumot
        :param fitness_criterion: Célfüggvény kritéruma
        """

        self.population = population
        self.min_pos = position_min
        self.max_pos = position_max
        self.num_of_gens = generation
        self.fit_crit = fitness_criterion

        self.particles = None
        self.pbest_position = None
        self.pbest_fitness = None
        self.gbest_position = None
        self.velocity = None

        self.prev_i = 0  # segédváltozó a szkript állapotjelző printhez
        self.figure = None
        self.axis = None
        self.images = []

        self.__init_particles()
        self.__init_plot()

    def __init_particles(self):
        """
        Populáció és a hozzá tartozó kezdő értékek felépítése
        """

        # Populáció definiálása
        self.particles = [
            [random.uniform(self.min_pos, self.max_pos) for j in range(2)] for i in range(self.population)
        ]

        # Részecskék kezdő pozíciója, később legjobb pozíció
        self.pbest_position = self.particles

        # Célfüggvény ellenőrzése
        self.pbest_fitness = [self._fitness_function(p[0], p[1]) for p in self.particles]

        # A legjobban illeszkedő részecske indexének kiválasztása
        gbest_index = np.argmin(self.pbest_fitness)

        # A globálisan legjobb részecske kiválasztása
        self.gbest_position = self.pbest_position[gbest_index]

        # A részecskék sebessége (0-tól kezdve)
        self.velocity = [[0.0 for j in range(2)] for i in range(self.population)]

    def __init_plot(self):
        """
        Szemléltető ábra inicializálása.
        """

        self.figure = plt.figure(figsize=(10, 10))

        self.axis = self.figure.add_subplot(111, projection='3d')
        self.axis.set_xlabel('x')
        self.axis.set_ylabel('y')
        self.axis.set_zlabel('z')

        x0 = np.linspace(self.min_pos, self.max_pos, 80)
        y0 = np.linspace(self.min_pos, self.max_pos, 80)
        x, y = np.meshgrid(x0, y0)
        z = self._fitness_function(x, y)
        self.axis.plot_wireframe(x, y, z, color='r', linewidth=0.2)

    @staticmethod
    def _fitness_function(x1, x2):
        """
        Célfüggvény:
        Tegyük fel, hogy az optimálandó probléma az alábbi függvénnyel közelíthető:
            f(x1,x2)=(x1+2*-x2+3)^2 + (2*x1+x2-8)^2

        Az optimálás célja, a függvény minimumának megtalálása, ami 0.
        """

        f1 = x1 + 2 * -x2 + 3
        f2 = 3 * x1 + x2 - 8
        return f1 ** 2 + f2 ** 2

    @staticmethod
    def _update_velocity(particle, velocity, pbest, gbest, w_min=0.5, max=1.0, c1=0.1, c2=0.1):
        """
        A részecskék sebességének frissítése.

        :param particle: Sron következő részecske
        :param velocity: A részecske sebessége
        :param pbest: A részecske eddigi legjobb pozíciója
        :param gbest: A generációk által elért legjobb pozíció
        :param w_min: Minimum inercia súly
        :param max: Állandó az inerciák maximumának meghatározásához
        :param c1:
        :param c2:
        :return:Új sebesség mátrix
        """

        # Új sebességmátrix összeállítása
        num_particle = len(particle)
        new_velocity = np.array([0.0 for i in range(num_particle)])

        # Normál előszlású, random generált inerciák
        r1 = random.uniform(0, max)
        r2 = random.uniform(0, max)
        w = random.uniform(w_min, max)

        # Új sebesség számítása
        for i in range(num_particle):
            new_velocity[i] = w * velocity[i] + c1 * r1 * (pbest[i] - particle[i]) + c2 * r2 * (gbest[i] - particle[i])

        return new_velocity

    @staticmethod
    def _update_position(particle, velocity):
        """
        Aktuális részecske pozíciójának frissítése.

        :param particle: Aktuális részecske
        :param velocity: A részecske sebessége
        :return: Új részecske
        """

        new_particle = particle + velocity
        return new_particle

    def _calc_fitness(self):
        # Legjobb illeszkedés számítása
        self.pbest_fitness = [self._fitness_function(p[0], p[1]) for p in self.particles]

        # A legjobban illeszkedő adatok indexének kikeresése
        self.gbest_index = np.argmin(self.pbest_fitness)

        # A legjobban illeszkedő részecske pozíciója
        self.gbest_position = self.pbest_position[self.gbest_index]

    def _print_results(self, loop_back_index):
        """
        (Rész)eredmények megjelenítése konzolon
        """

        if loop_back_index % 10 == 0 and self.prev_i < loop_back_index:
            print('Globális legjobb pozíció: ', self.gbest_position)
            print('Legjobb közelítések minimuma: ', min(self.pbest_fitness))
            print('Legjobb közelítések átlaga: ', np.average(self.pbest_fitness))
            print('Generációk száma: ', loop_back_index)
            print('~' * 50, '\n')
            self.prev_i = loop_back_index

    def _add_to_plot(self):
        """
        Minden generáció hozzáadása a szimuláció ábrájára.
        """

        image = self.axis.scatter3D([self.particles[n][0] for n in range(self.population)],
                                    [self.particles[n][1] for n in range(self.population)],
                                    [self._fitness_function(self.particles[n][0],
                                                            self.particles[n][1]) for n in range(self.population)],
                                    c='b')
        self.images.append([image])

    def generate_animation(self, file_name: str, save_anim: bool = True, repeat_delay: int = 0, fps: int = 10):
        """
        Bemutató animáció generálása.
        """

        if self.figure is None:
            assert ValueError('Nem található ábra!')

        animated_image = animation.ArtistAnimation(self.figure, self.images, repeat_delay=repeat_delay)
        plt.show()

        if save_anim:
            animated_image.save(filename=file_name, fps=fps, writer='pillow')

    def run_optimization(self):
        # Iteráció a generációkon
        for t in range(self.num_of_gens):

            # Ha az átlagos közelítés elérte az előre definiált kritériumot, az iteráció leáll.
            if np.average(self.pbest_fitness) <= self.fit_crit:
                break

            else:
                for n in range(self.population):
                    # A részecskék szebességének frissítése
                    self.velocity[n] = self._update_velocity(self.particles[n], self.velocity[n], self.pbest_position[n],
                                                             self.gbest_position)

                    # A részecskék elmozgatása új pozícióba
                    self.particles[n] = self._update_position(self.particles[n], self.velocity[n])

                    # (Rész)eredmények megjelenítése konzolon
                    self._print_results(t)

            self._add_to_plot()  # Részeredmények hozzáadasa az ábrához.
            self._calc_fitness()  # A kritérium ellenőrzése.


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--population', required=False, type=int, default=80)
    parser.add_argument('-i', '--position_min', required=False, type=float, default=-100.0)
    parser.add_argument('-a', '--position_max', required=False, type=float, default=100.0)
    parser.add_argument('-g', '--generation', required=False, type=int, default=100)
    parser.add_argument('-c', '--fitness_criterion', required=False, type=float, default=0.0001)
    parser.add_argument('-f', '--file_name', required=False, type=str, default='demo_animation.gif')
    parser.add_argument('-s', '--save_anim', required=False, type=bool, default=True)
    parser.add_argument('-r', '--frame_rate', required=False, type=int, default=10)

    args = parser.parse_args()

    demo = SwarmAlgoDemo(args.population, args.position_min, args.position_max, args.generation,
                         args.fitness_criterion)

    demo.run_optimization()
    demo.generate_animation('demo_animation.gif')
