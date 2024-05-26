# Script adapted from https://github.com/amzn/orthogonal-additive-gaussian-processes

# download UCI datasets from https://github.com/duvenaud/additive-gps/ or https://github.com/hughsalimbeni/bayesian_benchmarks/ and save to ./data directory

import os
import urllib.request
import pandas as pd 
from scipy import io
import numpy as np 
import pickle
from scipy.io import loadmat

pd.options.display.max_columns = None
pd.options.display.width = 250

data_path_prefix = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../data/uci")
) + '/'


# Create if data does not exist
if not os.path.exists(data_path_prefix):
        os.makedirs(data_path_prefix)
        
uci_base_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'

regression_filenames = [
    "autompg", # N = 372, D = 7  
    "housing", # N = 506, D = 13 
    "r_concrete_1030", # N = 1000, D = 8
    "pumadyn8nh", # N = 1000, D = 8
    "energy", # N = 768, D = 8
    "yacht", # N = 308, D = 6
    "forest", # N = 517,  D = 12
    "stock", # N = 536, D = 11
    #"fertility", # N = 100, D = 10
    #"machine", # N = 209, D = 7
    #"pendulum", # N = 630, D = 9
    "servo", # N = 166, D = 4
    #"wine" # N = 178, D = 14
    "container", # N = 15, D = 2
    "demand", # N = 60, D = 12
    #"slump", # N = 103, D = 7
    "stock2", # N = 315, D = 11
    #"computer", # N = 209, D = 9
]



for filename in regression_filenames:
    if not os.path.isfile(f'{data_path_prefix}/{filename}.mat'):
        if filename == "autompg":
            url = f'https://github.com/duvenaud/additive-gps/raw/master/data/regression/autompg/{filename}.mat'
            print(f"Downloading {filename}")
            urllib.request.urlretrieve(url, f'{data_path_prefix}/{filename}.mat')
            d = io.loadmat(f'{data_path_prefix}/{filename}.mat')  
            d["X"] = np.float32(d["X"])
            X, y = d["X"][:, 1:], d["X"][:, :1]
            io.savemat(f'{data_path_prefix}/{filename}.mat', {'X': X, 'y':y}) 
        
        elif filename in ["housing", "r_concrete_1030", "pumadyn8nh"]:
            url = f'https://github.com/duvenaud/additive-gps/raw/master/data/regression/{filename}.mat'
            print(f"Downloading {filename}")
            urllib.request.urlretrieve(url, f'{data_path_prefix}/{filename}.mat')
            d = io.loadmat(f'{data_path_prefix}/{filename}.mat')  
            X, y = d["X"][:1000, :], d["y"][:1000:]
            X, y = np.float32(X), np.float32(y)
            io.savemat(f'{data_path_prefix}/{filename}.mat', {'X': X, 'y':y}) 
        
        elif filename == "energy":
            print(f"Downloading {filename}")
            url = uci_base_url + '00242/ENB2012_data.xlsx'
            df = pd.read_excel(url)
            X = df[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']].values
            y = df[['Y1']].values
            X, y = np.float32(X), np.float32(y)
            io.savemat(f'{data_path_prefix}/{filename}.mat', {'X': X, 'y':y}) 
            
        elif filename == "yacht":
            print(f"Downloading {filename}")
            url = uci_base_url + '/00243/yacht_hydrodynamics.data'
            data = pd.read_fwf(url).dropna().values[:-1, :]
            X = data[:, :-1]
            y = data[:, -1].reshape(-1, 1)            
            X, y = np.float32(X), np.float32(y)
            io.savemat(f'{data_path_prefix}/{filename}.mat', {'X': X, 'y':y}) 
            
        elif filename == "forest":
            print(f"Downloading {filename}")
            url = uci_base_url + '/forest-fires/forestfires.csv'
            df = pd.read_csv(url).dropna()
            df = df.replace(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep',
                            'oct', 'nov', 'dec'], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
            df = df.replace(['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'],
                            [1, 2, 3, 4, 5, 6, 7])
            X = df.values[:, :-1]
            y = df.values[:, -1].reshape(-1, 1)            
            X, y = np.float32(X), np.float32(y)
            io.savemat(f'{data_path_prefix}/{filename}.mat', {'X': X, 'y':y}) 
            
        elif filename == "stock":
            print(f"Downloading {filename}")
            url = uci_base_url + '/00247/data_akbilgic.xlsx'
            data = pd.read_excel(url).dropna().values
            X = data[1:, 1:-1]
            y = data[1:, -1].reshape(-1, 1)
            X, y = np.float32(X), np.float32(y)
            io.savemat(f'{data_path_prefix}/{filename}.mat', {'X': X, 'y':y}) 
            
        #elif filename == "fertility":
        #    print(f"Downloading {filename}")
        #    url = uci_base_url + '/00243/yacht_hydrodynamics.data'
        #    data = pd.read_csv(url).values[:-1, :]
        #    X = data[:, :-1]
        #    y = data[:, -1].reshape(-1, 1)
        #    io.savemat(data_path_prefix +filename+'.mat', {'X': X, 'y':y}) 
        #    
        #elif filename == "machine":
        #    print(f"Downloading {filename}")
        #    url = uci_base_url + '/00243/yacht_hydrodynamics.data'
        #    data = pd.read_csv(url).values[:-1, :]
        #    X = data[:, :-1]
        #    y = data[:, -1].reshape(-1, 1)
        #    io.savemat(data_path_prefix +filename+'.mat', {'X': X, 'y':y}) 
        #    
        #elif filename == "pendulum":
        #    print(f"Downloading {filename}")
        #    url = uci_base_url + '/00243/yacht_hydrodynamics.data'
        #    data = pd.read_csv(url).values[:-1, :]
        #    X = data[:, :-1]
        #    y = data[:, -1].reshape(-1, 1)
        #    io.savemat(data_path_prefix +filename+'.mat', {'X': X, 'y':y}) 
            
        elif filename == "servo":
            print(f"Downloading {filename}")
            url = uci_base_url + 'servo/servo.data'
            data = pd.read_fwf(url, delimiter=',').replace(['A', 
                'B', 'C', 'D', 'E'], [1, 2, 3, 4, 5]).values
            X = data[:, :-1]
            y = data[:, -1].reshape(-1, 1)
            X, y = np.float32(X), np.float32(y)
            print(X.shape, y.shape)
            io.savemat(data_path_prefix +filename+'.mat', {'X': X, 'y':y}) 

            
        #elif filename == "wine":
        #    print(f"Downloading {filename}")
        #    url = uci_base_url + '/00243/yacht_hydrodynamics.data'
        #    data = pd.read_csv(url).values[:-1, :]
        #    X = data[:, :-1]
        #    y = data[:, -1].reshape(-1, 1)
        #    io.savemat(data_path_prefix +filename+'.mat', {'X': X, 'y':y})  
            
        elif filename == "container":
            print(f"Downloading {filename}")
            url = uci_base_url + '/00436/Container_Crane_Controller_Data_Set.csv'
            data = pd.read_csv(url, delimiter=';').replace(',', '.', regex=True).dropna().values       
            X = data[:, :-1]
            y = data[:, -1].reshape(-1, 1)            
            X, y = np.float32(X), np.float32(y)
            io.savemat(f'{data_path_prefix}/{filename}.mat', {'X': X, 'y':y}) 
            
        elif filename == "demand":
            print(f"Downloading {filename}")
            url = uci_base_url + '/00409/Daily_Demand_Forecasting_Orders.csv'
            df = pd.read_csv(url, delimiter=';')        
            X = df.values[:, :-1]
            y = df.values[:, -1].reshape(-1, 1)            
            X, y = np.float32(X), np.float32(y)
            io.savemat(f'{data_path_prefix}/{filename}.mat', {'X': X, 'y':y}) 
        
        #elif filename == "slump":
        #    print(f"Downloading {filename}")
        #    url = uci_base_url + '/concrete/slump/slump_test.data'
        #    data = pd.read_fwf(url, delimiter=',')
        #    print(data.shape)
        #    
        #    print(data.head(3))
        #    
        #    data = pd.read_fwf(url)
        #    print(data.shape)
        #    #.values
        #    #X = data[:, :-1]
        #    #y = data[:, -1].reshape(-1, 1)            
        #    #X, y = np.float32(X), np.float32(y)
        #    #io.savemat(f'{data_path_prefix}/{filename}.mat', {'X': X, 'y':y}) 
            
        elif filename == "stock2":
            print(f"Downloading {filename}")
            url = uci_base_url + '/00390/stock%20portfolio%20performance%20data%20set.xlsx'
            data = pd.read_excel(url).dropna().values
            X = data[1:, 1:7]
            y = data[1:, 7].reshape(-1, 1)
            X, y = np.float32(X), np.float32(y)
            io.savemat(f'{data_path_prefix}/{filename}.mat', {'X': X, 'y':y})           
        
        #elif filename == "computer":
        #    print(f"Downloading {filename}")
        #    url = uci_base_url + '/cpu-performance/machine.data'
        #    df = pd.read_fwf(url, delimiter=';').dropna()
        #    print(df.shape)
        #    print(df.dtypes)
        #    print(df.head(3))
        #    
        #    data = pd.read_fwf(url)
        #    print(data.shape)
        #    #.values[:-1, :]
        #    #X = data[:, :-1]
        #    #y = data[:, -1].reshape(-1, 1)            
        #    #X, y = np.float32(X), np.float32(y)
        #    #io.savemat(f'{data_path_prefix}/{filename}.mat', {'X': X, 'y':y}) 
            
            
        
        else : 
            pass

    else:
        d = io.loadmat(f'{data_path_prefix}/{filename}.mat')          
        X, y = d["X"], d["y"]
        print(filename, X.shape, y.shape, X.dtype, y.dtype)
        
        

# Data info 

covariate_names = {}
output_names = {}

covariate_names["autompg"] = [
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "year",
    "origin",
]

covariate_names["housing"] = [
    "crime",
    "zoned",
    "industrial",
    "river",
    "NOX",
    "rooms",
    "age",
    "empl. dist.",
    "highway acc.",
    "tax",
    "pupil ratio",
    "black pct",
    "low status pct",
]

covariate_names["concrete"] = [
    "Cement",
    "Blast Furnace Slag",
    "Fly Ash",
    "Water",
    "Superplasticizer",
    "Coarse Aggregate",
    "Fine Aggregate",
    "Age",
]

# no covariate name found for pumadyn
covariate_names["pumadyn"] = [f"input {i}" for i in range(8)]



covariate_names["energy"] = [
    "Relative Compactness", 
    "Surface Area",
    "Wall Area",
    "Roof Area",
    "Overall Height",
    "Orientation",
    "Glazing Area",
    "Glazing Area Distr.",
]
output_names["energy"] = "Heating Load"

covariate_names["yacht"] = [
    "Longitud. pos.",
    "Prismatic coeff",
    "Length-displacemt ratio",
    "Beam-draught ratio",
    "Length-beam ratio",
    "Froude number",
]
output_names["yacht"] = "Residuary resistance"

covariate_names["forest"] = [
    "X",
    "Y",
    "month",
    "day",
    "FFMC",
    "DMC",
    "DC",
    "ISI",
    "temp",
    "RH",
    "wind",
    "rain",
]
output_names["forest"] = "burned area"

covariate_names["stock"] = [
    'ISE TL',
    'ISE USD',
    'SP',
    'DAX',
    'FTSE',
    'NIKKEI',
    'BOVESPA',
    'EU',    
    ]
output_names["stock"] = "EM"

#covariate_names["fertility"] = [
#]
#output_names["fertility"] = ""

#ovariate_names["machine"] = [
#
#utput_names["machine"] = ""

#covariate_names["pendulum"] = [
#]
#output_names["pendulum"] = ""

covariate_names["servo"] = [
    "motor",
    "screw",
    "pgain",
    "vgain",
]
output_names["servo"] = "class"

#covariate_names["wine"] = [
#]
#output_names["wine"] = ""


covariate_names["container"] = [
    "speed",
    "angle"
]
output_names["container"] = "power"

covariate_names["demand"] = [
    "Week",
    "Day",
    "Non_urgent",
    "Urgent",
    "type_A", 
    "type_B", 
    "type_C",
    "Fiscal",
    "traffic_",
    "Banking_1",
    "Banking_2",
    "Banking_3",
]
output_names["demand"] = "total orders"

covariate_names["slump"] = [
    "Cement",
    "Slag",
    "Fly",
    "Water",
    "SP",
    "Coarse", 
    "Fine",
]
output_names["slump"] = "Strength" #-1

covariate_names["stock2"] = [
    "B/P",
    "ROE",
    "S/P",
    "return rate",
    "market value",
    "systematic risk,"
]
output_names["stock2"] = "annual return" #-6

covariate_names["computer"] = [
    "vendor name",
    "model name",
    "MYCT",
    "MMIN", 
    "MMAX",
    "CACH", 
    "CHMIN", 
    "CHMAX", 
    "PRP",
]
output_names["computer"] = "ERP"



with open(data_path_prefix +'data_info_inputs.pkl', 'wb') as f:
    pickle.dump(covariate_names, f)
    
with open(data_path_prefix +'data_info_outputs.pkl', 'wb') as f:
    pickle.dump(output_names, f)




