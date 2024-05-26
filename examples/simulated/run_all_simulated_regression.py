import subprocess
import os 
import glob 
import numpy as np

data_path_prefix = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../data/simulated1")
) + '/'

experiments_names = list(set([os.path.dirname(f) for f in glob.glob(data_path_prefix+'**/*.mat', recursive=True)]))
methods = ['ss', 'cosso', 'lgbayescos']#, 


for experiment_name in np.sort(experiments_names): 
    for method in methods:
        if method == 'lgbayescos':
            mixings = ['Exp', 'Dirichlet', 'Horshoe', 'Student']    # 'Exp', 'Dirichlet', 'Horshoe'
        else:
            mixings = ['not defined']
        
        for mixing in mixings:
            print(f"\n ***** Runing {method} for dataset {experiment_name},  mixing {mixing} *****")
            # Run the method 
            #os.system('python examples/uci/run_uci_regression.py --dataset_name dataset_name --model_name method --adaptive adaptive --mixing mixing')
            #print('--dataset_name', dataset_name, '--model_name', method, '--adaptive',  adaptive, '--mixing', mixing )
            args = ['python', 'run_simulated_regression.py',
                        '--dataset_name', experiment_name,
                        '--model_name', method,
                        '--mixing', mixing,
                        #'--num_samples_inf', nb_sample,
                        ]
            subprocess.run(args)