import subprocess

dataset_names = [ "housing", "demand", "energy", "yacht", "autompg", "servo", "stock",  "pumadyn", "concrete"] # "forest", "energy", "pumadyn",  "container",  "concrete",  "stock", "housing", "demand", "servo", "yacht", "stock2"
methods = ['lgbayescos_opt'] # 
#dataset_names = [ "housing", "demand", "energy",]

for dataset_name in dataset_names:
    for method in methods:
        if method == 'lgbayescos_opt':
            mixings = ['Exp', 'Dirichlet']    # 'Exp', 'Dirichlet', 'Horshoe'
        else:
            mixings = ['not defined']
        
        if method == 'ss':
            adaptives = ['NonAdaptive']
        else:
            adaptives = ['NonAdaptive', 'Adaptive'] # 'Adaptive'
        
        for mixing in mixings:
            for adaptive in adaptives:
                print(f"\n ***** Runing {method} for dataset {dataset_name}, adaptive {adaptive}, mixing {mixing} *****")
                # Run the method 
                #os.system('python examples/uci/run_uci_regression.py --dataset_name dataset_name --model_name method --adaptive adaptive --mixing mixing')
                #print('--dataset_name', dataset_name, '--model_name', method, '--adaptive',  adaptive, '--mixing', mixing )
                args = ['python', 'run_uci_regression_opt.py',
                        '--dataset_name', dataset_name,
                        '--model_name', method,
                        '--adaptive', adaptive,
                        '--mixing', mixing,
                        #'--num_samples_inf', nb_sample,
                        ]
                subprocess.run(args)