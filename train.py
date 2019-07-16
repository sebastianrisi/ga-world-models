import math 
import torch
from torchvision import transforms
from multiprocessing import Lock
import numpy as np
from models import MDRNNCell, VAE, Controller
import gym
import copy
import random

# Hardcoded for now. Note: Size of latent vector (LSIZE) is increased to 128 for DISCRETE representation
ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE =\
    3, 32, 256, 64, 64

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor()
])


class RolloutGenerator(object):
    """ Utility to generate rollouts.
    :attr device: device used to run VAE, MDRNN and Controller
    :attr time_limit: rollouts have a maximum of time_limit timesteps
    """
    def __init__(self, device, time_limit, discrete_VAE):
        """ Build vae, rnn, controller and environment. """

        self.env = gym.make('CarRacing-v0')
        
        self.device = device

        self.time_limit = time_limit

        self.discrete_VAE = discrete_VAE

        #Because the represenation is discrete, we increase the size of the latent vector
        if (self.discrete_VAE):
            LSIZE = 128

        self.vae = VAE(3, LSIZE, 1024)
        self.mdrnn = MDRNNCell(LSIZE, ASIZE, RSIZE, 5)
        self.controller = Controller(LSIZE, RSIZE, ASIZE)


    def get_action_and_transition(self, obs, hidden):
        """ Get action and transition.

        Encode obs to latent using the VAE, then obtain estimation for next
        latent and next hidden state using the MDRNN and compute the controller
        corresponding action.

        :args obs: current observation (1 x 3 x 64 x 64) torch tensor
        :args hidden: current hidden state (1 x 256) torch tensor

        :returns: (action, next_hidden)
            - action: 1D np array
            - next_hidden (1 x 256) torch tensor
        """
        _, latent_mu, _ = self.vae(obs)

        if (self.discrete_VAE):  

            bins=np.array([-1.0,0.0,1.0])

            latent_mu = torch.tanh(latent_mu)
            newdata=bins[np.digitize(latent_mu,bins[1:])]+1

            latent_mu = torch.from_numpy(newdata).float()    

        action = self.controller(latent_mu, hidden[0] )

        mus, sigmas, logpi, rs, d, next_hidden = self.mdrnn(action, latent_mu, hidden)

        return action.squeeze().cpu().numpy(), next_hidden


    def do_rollout(self, render=False,  early_termination=True):

        with torch.no_grad():
            
            self.env = gym.make('CarRacing-v0')

            obs = self.env.reset()

            self.env.render('rgb_array') 

            hidden = [
                torch.zeros(1, RSIZE)#.to(self.device)
                for _ in range(2)]

            neg_count = 0

            cumulative = 0
            i = 0
            while True:
                obs = transform(obs).unsqueeze(0)#.to(self.device)
                
                action, hidden = self.get_action_and_transition(obs, hidden)
                #Steering: Real valued in [-1, 1] 
                #Gas: Real valued in [0, 1]
                #Break: Real valued in [0, 1]

                obs, reward, done, _ = self.env.step(action)
                
                #Count how many times the car did not get a reward (e.g. was outside track)
                neg_count = neg_count+1 if reward < 0.0 else 0   

                if render:
                    o = self.env.render("human")
                
                #To speed up training, determinte evaluations that are outside of track too many times
                if (neg_count>20 and early_termination):  
                    done = True
                
                cumulative += reward
                
                if done or (early_termination and i > self.time_limit):
                    self.env.close()
                    return cumulative, None

                i += 1


def fitness_eval_parallel(pool, r_gen, early_termination=True):#, controller_parameters):
    return pool.apply_async(r_gen.do_rollout, args=(False, early_termination) )


class GAIndividual():
    '''
    GA Individual

    multi = flag to switch multiprocessing on or off
    '''
    def __init__(self, device, time_limit, setting, multi=True, discrete_VAE = False):
        '''
        Constructor.
        '''

        self.device = device
        self.time_limit = time_limit
        self.multi = multi
        self.discrete_VAE = discrete_VAE

        self.mutation_power = 0.01 
            
        self.setting = setting

        self.r_gen = RolloutGenerator(device, time_limit, discrete_VAE)
        #self.r_gen.discrete_VAE = self.discrete_VAE

        self.async_results = []
        self.calculated_results = {}

    def run_solution(self, pool, evals=5, early_termination=True, force_eval=False):

        if force_eval:
            self.calculated_results.pop(evals, None)

        if (evals in self.calculated_results.keys()): #Already caculated results
            return

        self.async_results = []

        for i in range(evals):

            if self.multi:
                self.async_results.append (fitness_eval_parallel(pool, self.r_gen, early_termination))#, self.controller_parameters) )
            else:
                self.async_results.append (self.r_gen.do_rollout (False, early_termination) ) 


    def evaluate_solution(self, evals):

        if (evals in self.calculated_results.keys()): #Already calculated?
            mean_fitness, std_fitness = self.calculated_results[evals]

        else:
            if self.multi:
                results = [t.get()[0] for t in self.async_results]
            else:
                results = [t[0] for t in self.async_results]

            mean_fitness = np.mean ( results )
            std_fitness = np.std( results )

            self.calculated_results[evals] = (mean_fitness, std_fitness)

        self.fitness = -mean_fitness

        return mean_fitness, std_fitness


    def load_solution(self, filename):

        s = torch.load(filename)

        self.r_gen.vae.load_state_dict( s['vae'])
        self.r_gen.controller.load_state_dict( s['controller'])
        self.r_gen.mdrnn.load_state_dict( s['mdrnn'])

    
    def clone_individual(self):
        child_solution = GAIndividual(self.device, self.time_limit, self.setting, multi=True, discrete_VAE = self.discrete_VAE)
        child_solution.multi = self.multi

        child_solution.fitness = self.fitness

        child_solution.r_gen.controller = copy.deepcopy (self.r_gen.controller)
        child_solution.r_gen.vae = copy.deepcopy (self.r_gen.vae)
        child_solution.r_gen.mdrnn = copy.deepcopy (self.r_gen.mdrnn)
        
        return child_solution
    
    def mutate_params(self, params):
        for key in params: 
               params[key] += torch.from_numpy( np.random.normal(0, 1, params[key].size()) * self.mutation_power).float()

    def mutate(self):
        
        if self.setting == 0: #Mutate controller, VAE and MDRNN. Normal deep neuroevolution

            self.mutate_params(self.r_gen.controller.state_dict())
            self.mutate_params(self.r_gen.vae.state_dict())
            self.mutate_params(self.r_gen.mdrnn.state_dict())

        if self.setting == 1: #Mutate controller, VAE or mdrnn
            c = np.random.randint(0,3)

            if c==0:
                self.mutate_params(self.r_gen.vae.state_dict())
            elif c==1:
                self.mutate_params(self.r_gen.mdrnn.state_dict() )
            else:
                self.mutate_params(self.r_gen.controller.state_dict())
