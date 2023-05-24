# SwarmOptimizationDemo

**Első futtatás előtt:**
1. Miniconda: https://docs.conda.io/en/latest/miniconda.html
2. Virtuális környezet létrehozása: `conda create --name <environment_name> python=3.8`
3. requirements.txt telepítése: `pip install -r requirements.txt`

**Használat**
~~~
$[conda activate <env_name>]
$python3 MAR_demo_SWARM_algo.py [-h] [-p POPULATION] [-i POSITION_MIN] [-a POSITION_MAX] [-g GENERATION] 
                                [-c FITNESS_CRITERION] [-f FILE_NAME] [-s SAVE_ANIM] [-r FRAME_RATE]

Opcionális argumentumok:
    -p/--population          int, a populáció mértete
    -i/--position_min        float, a részecskék pozíciójának maximuma
    -a/--position_max        float, a részecskék pozíciójának minimuma
    -g/--generation          int, a generációk maximum darabszáma
    -c/--fitness_criterion   float, a célfüggvény kritériuma
    -f/--file_name           str, ha mentjük az animációt, ez lesz a neve
    -s/--save_anim           bool, True, ha akarjuk menteni az animációt, egyébkén False
    -r/--frame_rate          int, az animáció képfrissítése
~~~
Default values:
~~~
population:          80
position_min:        -100.0
position_max:        100.0
generation:          100
fitness_criterion:   0.0001
file_name:           'demo_animation.gif'
save_anim:           True
frame_rate:          10
~~~
