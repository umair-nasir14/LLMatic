#| This file has a major part from the pymap_elites framework.
#| Copyright 2019, INRIA
#| Main contributor(s):
#| Jean-Baptiste Mouret, jean-baptiste.mouret@inria.fr
#| Eloise Dalin , eloise.dalin@inria.fr
#| Pierre Desreumaux , pierre.desreumaux@inria.fr
#|
#|
#| **Main paper**: Mouret JB, Clune J. Illuminating search spaces by
#| mapping elites. arXiv preprint arXiv:1504.04909. 2015 Apr 20.
#|
#| This software is governed by the CeCILL license under French law
#| and abiding by the rules of distribution of free software.  You
#| can use, modify and/ or redistribute the software under the terms
#| of the CeCILL license as circulated by CEA, CNRS and INRIA at the
#| following URL "http://www.cecill.info".
#|
#| As a counterpart to the access to the source code and rights to
#| copy, modify and redistribute granted by the license, users are
#| provided only with a limited warranty and the software's author,
#| the holder of the economic rights, and the successive licensors
#| have only limited liability.
#|
#| In this respect, the user's attention is drawn to the risks
#| associated with loading, using, modifying and/or developing or
#| reproducing the software by the user in light of its specific
#| status of free software, that may mean that it is complicated to
#| manipulate, and that also therefore means that it is reserved for
#| developers and experienced professionals having in-depth computer
#| knowledge. Users are therefore encouraged to load and test the
#| software's suitability as regards their requirements in conditions
#| enabling the security of their systems and/or data to be ensured
#| and, more generally, to use and operate it in the same conditions
#| as regards security.
#|
#| The fact that you are presently reading this means that you have
#| had knowledge of the CeCILL license and that you accept its terms.

import math
import numpy as np
import multiprocessing

# from scipy.spatial import cKDTree : TODO -- faster?
from sklearn.neighbors import KDTree

from map_elites import common as cm

from flopth import flopth

from datasets import get_datasets
from utils import read_python_file, extract_code, get_net_name, extract_code_section, get_class, csv_writer, get_max_curiosity, get_max_fitness, count_unique_components, count_parameters, get_network_width_depth_ratio
from utils import Net
from mutation_models import codegen_mutate, codex_mutate, replace_word_mutation
from evaluations import is_trainable
from train import train_net_on_cpu, train_net_on_gpu, forward_pass_on_cpu,forward_pass_on_gpu, detect_layers, transfer_weights

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import numpy as np
import os
import random
import hydra
import ray
from fvcore.nn import FlopCountAnalysis

from conf.config import Config




def __add_to_archive(s, centroid, archive, kdt, type_ind):
    niche_index = kdt.query([centroid], k=1)[1][0][0]
    niche = kdt.data[niche_index]
    n = cm.make_hashable(niche)
    s.centroid = n
    if n in archive:
        if type_ind == "network": 
            if s.fitness < archive[n].fitness:
                archive[n] = s
                return True
        elif type_ind == "prompt":
            if s.fitness > archive[n].fitness:
                archive[n] = s
                return True
        return False
    else:
        archive[n] = s
        return True



def to_specie(net,fit,desc,net_path):
    return cm.Species(net, desc, fit, net_path)

# map-elites algorithm (CVT variant)

@hydra.main(version_base="1.3.0", config_path="conf", config_name="config")    
def main(cfg: Config):
    """CVT MAP-Elites
       Vassiliades V, Chatzilygeroudis K, Mouret JB. Using centroidal voronoi tessellations to scale up the multidimensional archive of phenotypic elites algorithm. IEEE Transactions on Evolutionary Computation. 2017 Aug 3;22(4):623-30.

       Format of the logfile: evals archive_size max mean median 5%_percentile, 95%_percentile

    """
    if cfg.DEVICE == "cuda":
        ray.init(num_gpus=cfg.NUM_GPUS_TOTAL) 
    else:
        ray.init()
    mutation_fn = codex_mutate if cfg.MUTATION == "codex" else codegen_mutate

    prompts = ['"""Add a layer to improve the above network"""',
               '"""Delete a layer to improve the above network"""',
               '"""improve the above network"""',
               '"""improve the above network by reducing the size drastically"""',
               '"""improve the above network by increasing the size drastically"""',
               # Specific prompts below
               '"""Add fully connected layer to improve the above network"""',
               '"""Add convolutional layer to improve the above network"""',
               '"""Add pooling layer to improve the above network"""',
               '"""Add residual connection to improve the above network"""',
               '"""Add multiple residual connections to improve the above network"""',
               '"""Add dropout layer to improve the above network"""',
               '"""Add normalization layer to improve the above network"""',
               '"""Add recurrent layer to improve the above network"""',
                ]
    probabilities = [1.0 / len(prompts)] * len(prompts)

    prompt_score = {}
    for prompt in prompts:
        prompt_score[prompt] = 0
    prompt_score["None"] = 0

    prompt_to_int = {}
    int_to_prompt = {}
    for i,prompt in enumerate(prompts):
        prompt_to_int[prompt] = i
        int_to_prompt[i] = prompt
    
    if cfg.START_FROM_CHECKPOINT:
        best_net = read_python_file("initial_net.py")
        init_gen = 4
        print(f"Starting from the following network:\n{best_net}")
    else:
        init_net = read_python_file(os.path.normpath(f"{cfg.SAVE_DIR}/initial_net.py"))
        init_gen = 0
        print(f"Starting from the following network:\n{init_net}")
    
    
    # create the CVT
    params = cm.default_params
    c = cm.cvt(cfg.N_NICHES,cfg.DIM_MAP,
              params['cvt_samples'], params['cvt_use_cache'])
    kdt = KDTree(c, leaf_size=30, metric='euclidean')
    cm.__write_centroids(c)

    prompt_archive = {} # init archive (empty)
    nets_archive = {}
    n_evals = 0 # number of evaluations since the beginning
    b_evals = 0 # number evaluation since the last dump

    curios_net_path = os.path.normpath(f"{cfg.SAVE_DIR}/initial_net.py")

    c_net_str = read_python_file(curios_net_path)
    Net = get_class(curios_net_path)
    curios_net = Net()
    curios_prompt = prompts[0]
    # main loop

    temperature_start = 0.6
    prev_best_score = np.inf
    prev_best_net_path = None
    prev_best_prompt = None

    exp_name = f"gen-nets_{cfg.MUTATION}_networks-{cfg.NUM_NETS}_temp-{cfg.TEMPERATURE}_net-training-epochs-{cfg.NET_TRAINING_EPOCHS}_niches-{cfg.N_NICHES}_infer-and-flops-as-bd"
    path_nets = f"{cfg.SAVE_DIR}/logs/{exp_name}"
    os.makedirs(path_nets,exist_ok=True)
    log_file = open(os.path.normpath(f"{path_nets}/cvt.dat"),"w")
    out_file = os.path.normpath(f"{path_nets}/exp_results.csv")        
    csv_writer(["generations", "best_loss", "used_prompt"],out_file)
            

    for gen_i in range(init_gen, cfg.GENERATIONS):

        generated_nets = []
        
        # random initialization
        (print(f"[Generation: {gen_i}]"))

        if (len(nets_archive.keys()) < cfg.RANDOM_INIT_NETS) and (len(prompt_archive.keys()) < (int(cfg.RANDOM_INIT_NETS*2))):#params['random_init'] * n_niches:
            
            
            for _i in range(cfg.ROLL_OUTS):
                selected_prompts = []
                generated_net_results = []
                selected_prompts = random.choices(prompts, weights=probabilities,k=cfg.INIT_NUM_NETS)
                for i in range(0, cfg.INIT_NUM_NETS):
                    print(f"Selected prompt for generations: {selected_prompts[i]}")

                for prompt in selected_prompts:
                    generated_net_results.append(mutation_fn.remote(cfg=cfg, prompt=init_net + "\n" + prompt, temperature=temperature_start))


                for i,generated_net in enumerate(generated_net_results):
                    generated_nets.append((ray.get(generated_net),selected_prompts[i],temperature_start))
                
        else:  # variation/selection loop
            
            evo_operator = random.choices(["mutation","crossover"], weights=[0.85,0.15])[0]
            
            print(f"Performing {evo_operator}")
            for _i in range(cfg.ROLL_OUTS):
                generated_net_results = []
                if evo_operator == "mutation":
            
                    if len(nets_archive.keys()) < 3:
                        n_nets = 1
                        selection = 0
                    else:
                        n_nets = 3
                        selection = random.randint(0, 2)

                    curios_nets = []
                    curios_prompts = []
                    curios_temps = []
                    curios_net_paths = []

                    n_curios_nets = get_max_fitness(nets_archive.values(),n_nets)
                    for n in n_curios_nets:
                        curios_nets.append(n.x)
                        curios_net_paths.append(n.net_path)

                    n_prompts_temps = get_max_curiosity(prompt_archive.values(),n_nets)
                    for pt in n_prompts_temps:
                        curios_prompts.append(pt.desc[0])
                        curios_temps.append(pt.desc[1])

                    curios_net = curios_nets[selection]
                    curios_temp = curios_temps[selection]
                    curios_net_path = curios_net_paths[selection]

                    curios_temp_ray = []
                    curios_prompt_ray = []


                    for i in range(cfg.NUM_NETS):
                        # Temperature mutation

                        if i == 1:
                            curios_temp = curios_temps[selection]
                        else:
                            curios_temp += random.uniform(-0.1,0.1)
                            if curios_temp > 1.0:
                                curios_temp = 1.0
                            elif curios_temp < 0.1:
                                curios_temp = 0.1

                        curios_prompt = curios_prompts[selection]

                        curios_temp_ray.append(curios_temp)
                        curios_prompt_ray.append(curios_prompt)

                        print(f"prompt in mutation is {curios_prompt}")

                    for i in range(cfg.NUM_NETS):
                        generated_net_results.append(mutation_fn.remote(cfg=cfg, prompt=read_python_file(curios_net_path) + "\n" + int_to_prompt[int(curios_prompt_ray[i])], temperature = curios_temp_ray[i]))

                    for i,generated_net in enumerate(generated_net_results):
                        generated_nets.append((ray.get(generated_net),int_to_prompt[int(curios_prompt_ray[i])],curios_temp_ray[i]))

                elif evo_operator == "crossover":

                    if len(nets_archive.keys()) < 1:
                        print("No crossover performed")
                    else:
                        curios_nets_ray = []
                        curios_temps_ray = []

                        curios_nets = []
                        curios_prompts = []
                        curios_temps = []
                        curios_net_paths = []


                        if len(nets_archive.keys()) < 2:
                            selection = 0
                            n_nets = 1
                        else:
                            selection = random.randint(1, len(nets_archive.keys())-1)# change it to 1
                            n_nets = 2

                        n_curios_nets = get_max_fitness(nets_archive.values(),n_nets)
                        for n in n_curios_nets:
                            curios_nets.append(n.x)
                            curios_net_paths.append(n.net_path)

                        n_prompts_temps = get_max_curiosity(prompt_archive.values(),n_nets)
                        for pt in n_prompts_temps:
                            curios_prompts.append(pt.desc[0])
                            curios_temps.append(pt.desc[1])




                        for i in range(0, cfg.NUM_NETS):

                            curios_net_str = read_python_file(curios_net_paths[0])
                            curios_net = curios_nets[0]
                            
                            curios_temp = curios_temps[0]
                            curios_prompt = curios_prompts[0]


                            curios_nets_ray.append(read_python_file(curios_net_paths[selection]))


                        crossover_prompt = '"""Combine the above two neural networks and create a third neural network class that also inherits from nn.Module"""'

                        for curios_2nd_net in curios_nets_ray:
                            generated_net_results.append(mutation_fn.remote(cfg=cfg, prompt=curios_net_str + "\n" + curios_2nd_net + "\n" + crossover_prompt, temperature = curios_temp))

                        for i,generated_net in enumerate(generated_net_results):
                            generated_nets.append((ray.get(generated_net),int_to_prompt[int(curios_prompt)],curios_temp))

 
                           
        net_class_name = None
        net_paths=[]
        training_prompts=[]
        training_nets=[]

        for i, k in enumerate(generated_nets):
            
            invalid_net = False
            generation, prompt, temperature = k

            
            net_path = os.path.normpath(f"{path_nets}/network_{gen_i}_{i}_{prompt[3:-3]}_{temperature}.py")
            

            if cfg.MUTATION == "codex":
                    extract_code_section(generation['choices'][0]["text"], prompt, file_path=net_path)
            elif cfg.MUTATION.startswith("codegen"):
                extract_code_section(generation, prompt, file_path=net_path)  
            main_net_focus = read_python_file(net_path)
            print(f"Net in focus:\n {main_net_focus}")

            if not invalid_net:
                try:
                    Net = get_class(net_path)             
                    net = Net()
                except Exception as e:
                    net = curios_net
                    net_path = curios_net_path
                    if isinstance(curios_prompt,str):
                        prompt = curios_prompt
                    elif isinstance(curios_prompt,float) or isinstance(curios_prompt,int):
                        prompt = int_to_prompt[int(curios_prompt)]
                    
                #breakpoint()
                try:
                    is_t = is_trainable(net)
                except Exception:
                    is_t = False
                if is_t:
                    net_paths.append(net_path)
                    training_prompts.append(prompt)
                    training_nets.append(net)
                else:
                    invalid_net = True

                    

            if invalid_net:
                print(f"The network at {net_path} is not trainable")
        

        if training_nets == []:
            continue
    
        if (len(nets_archive.keys()) < cfg.RANDOM_INIT_NETS) and (len(prompt_archive.keys()) < (int(cfg.RANDOM_INIT_NETS*2))):

            inter_results = []
            losses = []
            layers_c_net = detect_layers(curios_net)
            if cfg.DEVICE == "cpu":
                
                for net in training_nets:
                    for layer in layers_c_net:
                        try:
                            net = transfer_weights(curios_net,net,layer)
                            print(f"Weights transferred successfully")
                        except Exception:
                            print("Weights can not transfer")
                for net in training_nets:
                    inter_results.append(forward_pass_on_gpu.remote(net))
            elif cfg.DEVICE == "cuda":
                    for net in training_nets:
                        for layer in layers_c_net:
                            try:
                                net = transfer_weights(curios_net,net,layer)
                                print(f"Weights transferred successfully")
                            except Exception:
                                print("Weights can not transfer")
                    for net in training_nets:
                        inter_results.append(forward_pass_on_gpu.remote(net))
            else:
                raise ValueError(f"{cfg.DEVICE} is not a valid device")
            fitness = []
            for i,result in enumerate(inter_results):
                try:
                    res = ray.get(result)
                    fitness.append([res[0],training_prompts[i],temperature,net_paths[i],res[1]])

                    if fitness[i][0] <= prev_best_score:
                        prev_best_net_path = net_paths[i]
                        prev_best_prompt = training_prompts[i]
                        prev_best_score = fitness[i][0]
                        temperature += 0.05

                    else:
                        temperature -= 0.05
                    if temperature > 1.0:
                        temperature = 1.0
                    elif temperature < 0.1:
                        temperature = 0.1

                        fitness[i][2] = temperature
                except Exception:
                    print("not trainable due to fitness 1")
        
        else:

            if cfg.RAY:
                inter_results = []
                losses = []
                layers_c_net = detect_layers(curios_net)
                if cfg.DEVICE == "cpu":
                    for net in training_nets:
                        for layer in layers_c_net:
                            try:
                                net = transfer_weights(curios_net,net,layer)
                                print(f"Weights transferred successfully")
                            except Exception:
                                print("Weights can not transfer")
                    for net in training_nets:
                        inter_results.append(train_net_on_cpu.remote(net,cfg.NET_TRAINING_EPOCHS))

                elif cfg.DEVICE == "both":
                    for i, net in enumerate(training_nets):
                        if i % 2 == 0:
                            inter_results.append(train_net_on_cpu.remote(net,cfg.NET_TRAINING_EPOCHS))
                        else:
                            inter_results.append(train_net_on_gpu.remote(net,cfg.NET_TRAINING_EPOCHS))

                elif cfg.DEVICE == "cuda":
                    for net in training_nets:
                        for layer in layers_c_net:
                            try:
                                net = transfer_weights(curios_net,net,layer)
                                print(f"Weights transferred successfully")
                            except Exception:
                                print("Weights can not transfer")
                    for net in training_nets:
                        inter_results.append(train_net_on_gpu.remote(net,cfg.NET_TRAINING_EPOCHS))

                else:
                    raise ValueError(f"{cfg.DEVICE} is not a valid device")
                fitness = []
                for i,result in enumerate(inter_results):
                    try:
                        fitness.append([ray.get(result),training_prompts[i],temperature,net_paths[i],0.0])

                        if fitness[i][0] <= prev_best_score:
                            prev_best_net_path = net_paths[i]
                            prev_best_prompt = training_prompts[i]
                            prev_best_score = fitness[i][0]
                            temperature += 0.05
                        else:
                            temperature -= 0.05

                        if temperature > 1.0:
                            temperature = 1.0
                        elif temperature < 0.1:
                            temperature = 0.1

                        fitness[i][2] = temperature
                    except Exception:
                        print("not trainable due to fitness 2")
                
                try:
                    infer_results = []
                    for net in training_nets:
                        infer_results.append(forward_pass_on_gpu.remote(net))
                    
                    
                    for i,result in enumerate(infer_results):
                        try:
                            res = ray.get(result)
                            fitness[i][4] = res[1]
                        except Exception:
                            print("not trainable due to inference speed 1")

                    
                except Exception:
                    print("not trainable due to inference speed 2")
                        
        dummy_inputs = torch.zeros((1, 3, 32, 32))
        net_beh_list = []
        prompt_beh_list = []
        for loss_x,prompt_x,temp_x,net_p,infer_speed in fitness:
            _nn = read_python_file(net_p) #Avoiding sytax errors
            Net = get_class(net_p)             
            net = Net()
            try:
                flps = FlopCountAnalysis(net, dummy_inputs)
                flops = flps.total()
                depth_width_ratio = get_network_width_depth_ratio(net)

            except Exception:
                try:
                    flps = FlopCountAnalysis(curios_net, dummy_inputs)
                    flops = flps.total()
                    depth_width_ratio = get_network_width_depth_ratio(curios_net)
                except Exception:
                    flops = 0
                    depth_width_ratio = 1

            print(f"flops, infer_speed, depth_width_ratio:  {flops}, {infer_speed}, {depth_width_ratio}")
            s_net = to_specie(net,loss_x,np.array([depth_width_ratio, flops]),net_p)
            net_added = __add_to_archive(s_net, s_net.desc, nets_archive, kdt, type_ind = "network")

            if net_added:
                prompt_fit = 1.0 
            else:
                prompt_fit = 0.0

            s_prompt = to_specie(net,prompt_fit,np.array([prompt_to_int[prompt_x], temp_x]),net_p)
            prompt_added = __add_to_archive(s_prompt, s_prompt.desc, prompt_archive, kdt, type_ind = "prompt")
            if not net_added:
                s_prompt.curiosity = s_prompt.curiosity - 0.5
            elif net_added:
                s_prompt.curiosity = s_prompt.curiosity + 1.0 
        
        if gen_i % 1 == 0:
            
            cm.__save_archive(nets_archive, gen_i,name="net")
            cm.__save_archive(prompt_archive, gen_i,name="prompt")
        # write log
        if log_file != None:
            fit_list = np.array([x.fitness for x in nets_archive.values()])
            log_file.write("{} {} {} {} {} {} {}\n".format(gen_i, len(nets_archive.keys()), len(prompt_archive.keys()),
                    fit_list.min(), np.mean(fit_list), np.median(fit_list),
                    np.percentile(fit_list, 5), np.percentile(fit_list, 95)))
            log_file.flush()
    cm.__save_archive(nets_archive, gen_i,name="net")
    cm.__save_archive(prompt_archive, gen_i,name="prompt")
    return nets_archive,prompt_archive

if __name__ == "__main__":
    main()

