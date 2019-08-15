import config

config.init()
import itertools


def main():
    gpu_ids = ['0', '1', '2', '3']
    script_name = [['train_model.py']]
    model_names = [['basic']]
    init_seeds = [[0]]
    control_names = [['lstm'], ['200'], ['200'], ['5'], ['0','0.1','0.3','0.5','0.7','0.9','0.95','1'], ['0'], ['1']]
    control_names = list(itertools.product(*control_names))
    control_names = [['_'.join(control_names[i]) for i in range(len(control_names))]]
    controls = script_name + model_names + init_seeds + control_names
    controls = list(itertools.product(*controls))
    s = '#!/bin/bash\n'
    for i in range(len(controls)):
        s = s + 'CUDA_VISIBLE_DEVICES=\"{}\" python {} --model_name \\\'{}\\\' --init_seed {} --control_name \\\'{}\\\' &\n'.format(
            gpu_ids[i % len(gpu_ids)], *controls[i])
    print(s)
    run_file = open("./run.sh", "w")
    run_file.write(s)
    run_file.close()


if __name__ == '__main__':
    main()