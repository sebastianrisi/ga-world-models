import argparse
import sys
from os.path import join, exists
from os import mkdir, unlink, listdir, getpid
import torch
import numpy as np
import train
from train import GAIndividual
from ga import GA
import random

import multiprocessing

#To run on headless:
#xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python3 main.py

torch.set_num_threads(1)

def main(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument('--pop-size', type=int, help='Population size.', default = 200)

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')

    parser.add_argument('--generations', type=int, default=1000, metavar='N',
                            help='number of generations to train (default: 1000)')

    parser.add_argument('--threads', type=int, default=10, metavar='N',
                            help='threads')

    parser.add_argument('--setting', type=int, default=1, metavar='N',
                            help='0 = standard deep NE, 1 = mutate one component at a time')

    parser.add_argument('--test', type=str, default='', metavar='N',
                            help='0 = no protection, 1 = protection')

    parser.add_argument('--folder', type=str, default='results', metavar='N',
                            help='folder to store results')

    parser.add_argument('--top', type=int, default=3, metavar='N',
                            help='numer of top elites that should be re-evaluated')
    
    parser.add_argument('--elite_evals', type=int, default=20, metavar='N',
                            help='how many times should the elite be evaluated')                        

    parser.add_argument('--timelimit', type=int, default=1000, metavar='N',
                            help='time limit per evaluation')    

    parser.add_argument('--discrete', type=int, default=0, metavar='N',
                            help='discrete VAE (default 0)')

    """Settings
    0 = mutate all components: controller, VAE, MDRNN  (deep NE)
    1 = mutate only one component at a time
    """

    args = parser.parse_args()

    if (args.pop_size % 2 != 0):
        print("Error: Population size needs to be an even number.")
        exit()

    #The track generation process might use another random seed, so even with same random seed here results can be different
    random.seed(args.seed)

    device = 'cpu'

    if args.test!='':       #Testing a specific genome
        to_evaluate = []
        #E.g. "python3 main.py --test best_1_2_G581.p --threads 1"

        max_rollouts = 3 #

        print ("Evaluate individual")
        t1 = GAIndividual('cpu', sys.maxsize, 1, multi=False)
        t1.load_solution(args.test)
        to_evaluate.append(t1)

        log_file = open("log.txt", 'a')

        for ind in to_evaluate:
            if (args.threads == 1):
                average = []
                print("Evaluting individual ",args.test)
                for i in range(max_rollouts):
                    f = ind.r_gen.do_rollout(True, False)[0]
                    print("Fitness ",f)
                    average += [f]

                print("Average ", np.average(average), " sd ",np.std(average) )

            else:
                print("Evaluating individual ",args.test)
                pool = multiprocessing.Pool(args.threads)
                ind.multi = True
                ind.run_solution(pool, max_rollouts, early_termination=False, force_eval = True)
                avg_f,sd = ind.evaluate_solution(max_rollouts)
                print("Average ",avg_f, " sd ",sd)
                log_file.write("%f\t%f" % (avg_f, sd))
                log_file.flush()

        log_file.close()
        exit()

    if not exists(args.folder):
        mkdir(args.folder)

    ga = GA(args.elite_evals, args.top, args.threads, args.timelimit, args.pop_size, args.setting, args.discrete==1) #mutation rate, crossover rate

    ga.run(args.generations, "{0}_{1}_".format(args.setting, args.seed), args.folder ) #pop_size, num_gens

if __name__ == '__main__':

    main(sys.argv)
