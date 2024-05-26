import subprocess

dataset_names = [ "yacht", "autompg", "servo", "stock",  "stock2", "pumadyn", "concrete"] # "forest", "energy", "pumadyn",  "container",  "concrete",  "stock", "housing", "demand", "servo", "yacht", "stock2"
dataset_names = [ "pumadyn", "autompg", "servo", "stock"]
methods = ['lgbayescos'] # , 'lgbayescos'] 'ss', , 'cosso'
#dataset_names = [ "housing", "demand", "energy",]
#dataset_names = ["demand"]

for dataset_name in dataset_names:
    for method in methods:
        if method == 'lgbayescos':
            mixings = ['Exp', 'Dirichlet']    # 'Exp', 'Dirichlet', 'Horshoe', 'Dirichlet'
        else:
            mixings = ['not defined']
        
        if method == 'ss':
            adaptives = ['NonAdaptive']
        else:
            adaptives = ['NonAdaptive',  'Adaptive'] # 'Adaptive', 'Adaptive'
        
        for mixing in mixings:
            for adaptive in adaptives:
                print(f"\n ***** Runing {method} for dataset {dataset_name}, adaptive {adaptive}, mixing {mixing} *****")
                # Run the method 
                #os.system('python examples/uci/run_uci_regression.py --dataset_name dataset_name --model_name method --adaptive adaptive --mixing mixing')
                #print('--dataset_name', dataset_name, '--model_name', method, '--adaptive',  adaptive, '--mixing', mixing )
                args = ['python', 'run_uci_regression.py',
                        '--dataset_name', dataset_name,
                        '--model_name', method,
                        '--adaptive', adaptive,
                        '--mixing', mixing,
                        #'--num_samples_inf', nb_sample,
                        ]
                subprocess.run(args)