import sys, random
import numpy as np
import pickle
import torch
import time
import math
from os.path import join, exists
import multiprocessing
import gc
import copy

from multiprocessing import set_start_method
set_start_method('forkserver', force=True)


class GA:
    def __init__(self, elite_evals, top, threads, timelimit, pop_size, setting, discrete_VAE):
        '''
        Constructor. 
        '''
        self.top  = top  #Number of top individuals that should be reevaluated
        self.elite_evals = elite_evals  #Number of times should the top individuals be evaluated

        self.pop_size = pop_size

        self.threads = threads
        multi_process = threads>1

        self.truncation_threshold = int(pop_size/2)  #Should be dividable by two

        from train import GAIndividual

        self.P = []
        for i in range(pop_size):
            self.P.append(GAIndividual('cpu', timelimit, setting, multi= multi_process, discrete_VAE=discrete_VAE ) )

        
    def run(self, max_generations, filename, folder):

        Q = []
        
        max_fitness = -sys.maxsize

        fitness_file = open(folder+"/fitness_"+filename+".txt", 'a')

        ind_fitness_file = open(folder+"/individual_fitness_"+filename+".txt", 'a')
        
        i = 0
        P = self.P

        pop_name = folder+"/pop_"+filename+".p"

        #Load previously saved population
        if exists( pop_name ):
            pop_tmp = torch.load(pop_name)

            print("Loading existing population ",pop_name, len(pop_tmp))

            idx = 0
            for s in pop_tmp:
                 P[idx].r_gen.vae.load_state_dict ( s['vae'].copy() )
                 P[idx].r_gen.controller.load_state_dict ( s['controller'].copy() )
                 P[idx].r_gen.mdrnn.load_state_dict ( s['mdrnn'].copy() )

                 i = s['generation'] + 1
                 idx+=1
                 

        while (True): 
            pool = multiprocessing.Pool(self.threads)

            start_time = time.time()

            print("Generation ", i)
            sys.stdout.flush()

            print("Evaluating individuals: ",len(P) )
            for s in P:  
                s.run_solution(pool, 1, force_eval=True)

            fitness = []

            for s in P:
                s.is_elite = False
                f, _ = s.evaluate_solution(1)
                fitness += [f]

            self.sort_objective(P)

            max_fitness_gen = -sys.maxsize #keep track of highest fitness this generation

            print("Evaluating elites: ", self.top)

            for k in range(self.top):      
                P[k].run_solution(pool, self.elite_evals)
            
            for k in range(self.top):

                f, _ = P[k].evaluate_solution(self.elite_evals) 

                if f>max_fitness_gen:
                    max_fitness_gen = f
                    elite = P[k]

                if f > max_fitness: #best fitness ever found
                    max_fitness = f
                    print("\tFound new champion ", max_fitness )

                    best_ever = P[k]
                    sys.stdout.flush()
                    
                    torch.save({'vae': elite.r_gen.vae.state_dict(), 'controller': elite.r_gen.controller.state_dict(), 'mdrnn':elite.r_gen.mdrnn.state_dict(), 'fitness':f}, "{0}/best_{1}G{2}.p".format(folder, filename, i))

            elite.is_elite = True  #The best 

            sys.stdout.flush()

            pool.close()

            Q = []

            if len(P) > self.truncation_threshold-1:
                del P[self.truncation_threshold-1:]

            P.append(elite) #Maybe it's in there twice now but that's okay

            save_pop = []

            for s in P:
                 ind_fitness_file.write( "Gen\t%d\tFitness\t%f\n" % (i, -s.fitness )  )  
                 ind_fitness_file.flush()

                 save_pop += [{'vae': s.r_gen.vae.state_dict(), 'controller': s.r_gen.controller.state_dict(), 'mdrnn':s.r_gen.mdrnn.state_dict(), 'fitness':fitness, 'generation':i}]
                 

            if (i % 25 == 0):
                print("saving population")
                torch.save(save_pop, folder+"/pop_"+filename+".p")
                print("done")

            print("Creating new population ...", len(P))
            Q = self.make_new_pop(P)

            P.extend(Q)

            elapsed_time = time.time() - start_time

            print( "%d\tAverage\t%f\tMax\t%f\tMax ever\t%f\tTime\t%f\n" % (i, np.mean(fitness), max_fitness_gen, max_fitness, elapsed_time) )  # python will convert \n to os.linesep

            fitness_file.write( "%d\tAverage\t%f\tMax\t%f\tMax ever\t%f\tTime\t%f\n" % (i, np.mean(fitness), max_fitness_gen, max_fitness,  elapsed_time) )  # python will convert \n to os.linesep
            fitness_file.flush()

            if (i > max_generations):
                break

            gc.collect()

            i += 1

        print("Testing best ever: ")
        pool = multiprocessing.Pool(self.threads)

        best_ever.run_solution(pool, 100, early_termination=False, force_eval = True)
        avg_f, sd = best_ever.evaluate_solution(100)
        print(avg_f, sd)
        
        fitness_file.write( "Test\t%f\t%f\n" % (avg_f, sd) ) 

        fitness_file.close()

        ind_fitness_file.close()

                                
    def sort_objective(self, P):
        for i in range(len(P) - 1, -1, -1):
            for j in range(1, i + 1):
                s1 = P[j - 1]
                s2 = P[j]
                
                if s1.fitness > s2.fitness:
                    P[j - 1] = s2
                    P[j] = s1
                    

    def make_new_pop(self, P):
        '''
        Make new population Q, offspring of P. 
        '''
        Q = []
        
        while len(Q) < self.truncation_threshold:
            selected_solution = None
            
            s1 = random.choice(P)
            s2 = s1
            while s1 == s2:
                s2 = random.choice(P)

            if s1.fitness < s2.fitness: #Lower is better
                selected_solution = s1
            else:
                selected_solution  = s2

            if s1.is_elite:  #If they are the elite they definitely win
                selected_solution = s1
            elif s2.is_elite:  
                selected_solution = s2

            child_solution = selected_solution.clone_individual() 
            child_solution.mutate()

            if (not child_solution in Q):    
                Q.append(child_solution)
        
        return Q
        
