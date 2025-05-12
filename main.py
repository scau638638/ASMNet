# from evolve.evolve import evolve
from evolve.bpso import bpso
import argparse

# evolve()
# bpso()

if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, nargs='*', default=[0])
    args =parser.parse_args()

    bpso(args)