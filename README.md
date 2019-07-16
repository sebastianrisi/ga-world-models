# Pytorch implementation of Deep Neuroevolution of World Models

Paper: Risi and Stanley, "Deep Neuroevolution of Recurrent and Discrete World Models"
To appear in: Proceedings of the Conference on Genetic and Evolutionary Computation (GECCO 2019). New York, NY: ACM.

https://arxiv.org/abs/1906.08857


## Prerequisites

The code is partly based on the PyTorch implementation of "World Models" (https://github.com/ctallec/world-models).

Code requieres Python3 and PyTorch (https://pytorch.org). The rest of the requirements are included in the [requirements file](requirements.txt), to install them:
```bash
pip3 install -r requirements.txt
```

## Running the program

The world model is composed of three different components: 

  1. A Variational Auto-Encoder (VAE)
  2. A Mixture-Density Recurrent Network (MDN-RNN)
  3. A linear Controller (C), which takes both the latent encoding and the hidden state of the MDN-RNN as input and outputs the agents action

In contrast to the original world model, all three components are trained end-to-end through evolution. To run training:

```bash
python main.py
```

To test a specific genome:

```bash
python main.py --test best_1_1_G2.p
```

Additional arguments for the training script are:
* **--folder** : The directory to store the training results. 
* **--pop-size** : The population size.
* **--threads** : The number of threads used for training or testing.
* **--discrete** : Switching a discrete version of the VAE on or off.
* **--generations** : The number of generations used for training.
* **--setting** : The setting determining the mutation operator. 0 = Mutate all three modules (VAE, MDN-RNN, C). 1 = Randomly mutate one of those three modules. 


### Notes
When running on a headless server, you will need to use `xvfb-run` to launch the controller training script. For instance,
```bash
xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python main.py
```

When running with a discrete VAE, the size of the latent vector is increased to 128 from the 32-dimensional version used for the standard VAE.

## Authors

* **Sebastian Risi**


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
