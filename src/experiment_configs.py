

### datasets
datasets = [
    {"dataset_name": "synthetic", "dtype": "S", "input_dims": 72},
    {"dataset_name": "amazon", "dtype": "A", "input_dims": 72},
    {"dataset_name": "brightkite", "dtype": "B", "input_dims": 500}
]

### settings

settings=[
    {"ds": datasets[0], "ds_train": 25, "ds_valid": 22, "ds_test": 24, "name": "unsup syn orig", "comment": "", "scenario": "unsup", "sco":"U"},         #0
    {"ds": datasets[0], "ds_train": 23, "ds_valid": 22, "ds_test": 24, "name": "semsiup syn orig", "comment": "", "scenario": "semsiup", "sco":"E"},

    {"ds": datasets[1], "ds_train": 38, "ds_valid": 39, "ds_test": 40, "name": "unsup amazon orig", "comment": "95% normal in train data", "scenario": "unsup", "sco":"U"},
    {"ds": datasets[1], "ds_train": 35, "ds_valid": 36, "ds_test": 37, "name": "semsiup amazon orig", "comment": "", "scenario": "semsiup", "sco":"E"},

    {"ds": datasets[2], "ds_train": 43, "ds_valid": 44, "ds_test": 41, "name": "unsup brightkite orig", "comment": "", "scenario": "unsup", "sco":"U"},
    {"ds": datasets[2], "ds_train": 42, "ds_valid": 44, "ds_test": 41, "name": "semisup brightkite orig", "comment": "", "scenario": "semsiup", "sco":"E"}, #5

    {"ds": datasets[0], "ds_train": 25, "ds_valid": 22, "ds_test": 6, "name": "unsup syn noise .0", "comment": "", "scenario": "unsup", "sco":"U"},        
    {"ds": datasets[0], "ds_train": 23, "ds_valid": 22, "ds_test": 6, "name": "semsiup syn noise .0", "comment": "", "scenario": "semsiup", "sco":"E"},

    {"ds": datasets[0], "ds_train": 25, "ds_valid": 22, "ds_test": 7, "name": "unsup syn noise .2", "comment": "", "scenario": "unsup", "sco":"U"},
    {"ds": datasets[0], "ds_train": 23, "ds_valid": 22, "ds_test": 7, "name": "semsiup syn noise .2", "comment": "", "scenario": "semsiup", "sco":"E"},

    {"ds": datasets[0], "ds_train": 25, "ds_valid": 22, "ds_test": 8, "name": "unsup syn noise .2", "comment": "", "scenario": "unsup", "sco":"U"},         #10
    {"ds": datasets[0], "ds_train": 23, "ds_valid": 22, "ds_test": 8, "name": "semsiup syn noise .2", "comment": "", "scenario": "semsiup", "sco":"E"},

    {"ds": datasets[0], "ds_train": 25, "ds_valid": 22, "ds_test": 10, "name": "unsup syn novelty .0", "comment": "", "scenario": "unsup", "sco":"U"},
    {"ds": datasets[0], "ds_train": 23, "ds_valid": 22, "ds_test": 10, "name": "semsiup syn novelty .0", "comment": "", "scenario": "semsiup", "sco":"E"},

    {"ds": datasets[0], "ds_train": 25, "ds_valid": 22, "ds_test": 11, "name": "unsup syn novelty .01", "comment": "", "scenario": "unsup", "sco":"U"},
    {"ds": datasets[0], "ds_train": 23, "ds_valid": 22, "ds_test": 11, "name": "semsiup syn novelty .01", "comment": "", "scenario": "semsiup", "sco":"E"},  #15

    {"ds": datasets[0], "ds_train": 25, "ds_valid": 22, "ds_test": 12, "name": "unsup syn novelty .05", "comment": "", "scenario": "unsup", "sco":"U"},
    {"ds": datasets[0], "ds_train": 23, "ds_valid": 22, "ds_test": 12, "name": "semsiup syn novelty .05", "comment": "", "scenario": "semsiup", "sco":"E"},

]

### model
models=[
{ "model_type": "GADFormer", "seg_len": 1, "loss_func": "bce", "heads":8 },
{ "model_type": "GRU", "seg_len": 1, "loss_func": "bce" },

global_seeds=[34,38,30]

#experiments
experiments_unsup_orig=[
    {"setting": settings[0], "model": models[0], "etype":"orig", "scaler":"robust", "seeds":global_seeds},
    {"setting": settings[0], "model": models[1], "etype":"orig", "scaler":"robust", "seeds":global_seeds},
    {"setting": settings[2], "model": models[0], "etype":"orig", "scaler":"robust", "seeds":global_seeds},
    {"setting": settings[2], "model": models[1], "etype":"orig", "scaler":"robust", "seeds":global_seeds},
    {"setting": settings[4], "model": models[0], "etype":"orig", "scaler":"robust", "seeds":global_seeds},
    {"setting": settings[4], "model": models[1], "etype":"orig", "scaler":"robust", "seeds":global_seeds},
]

experiments_semisup_orig=[
    {"setting": settings[1], "model": models[0], "etype":"orig", "scaler":"robust", "seeds":global_seeds},
    {"setting": settings[1], "model": models[1], "etype":"orig", "scaler":"robust", "seeds":global_seeds},
    {"setting": settings[3], "model": models[0], "etype":"orig", "scaler":"robust", "seeds":global_seeds},
    {"setting": settings[3], "model": models[1], "etype":"orig", "scaler":"robust", "seeds":global_seeds},
    {"setting": settings[5], "model": models[0], "etype":"orig", "scaler":"robust", "seeds":global_seeds},
    {"setting": settings[5], "model": models[1], "etype":"orig", "scaler":"robust", "seeds":global_seeds},
]

experiments_unsup_noise=[
    {"setting": settings[6], "model": models[0], "etype":"noise .0", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[6], "model": models[1], "etype":"noise .0", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[8], "model": models[0], "etype":"noise .2", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[8], "model": models[1], "etype":"noise .2", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[10], "model": models[0],"etype":"noise .5", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[10], "model": models[1],"etype":"noise .5", "scaler":"standard", "seeds":global_seeds},
]

experiments_semisup_noise=[
    {"setting": settings[7], "model": models[0],"etype":"noise .0", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[7], "model": models[1],"etype":"noise .0", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[9], "model": models[0],"etype":"noise .2", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[9], "model": models[1],"etype":"noise .2", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[11], "model": models[0],"etype":"noise .5", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[11], "model": models[1],"etype":"noise .5", "scaler":"standard", "seeds":global_seeds},
]

experiments_unsup_novelty=[
    {"setting": settings[12], "model": models[0],"etype":"noise .0", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[12], "model": models[1],"etype":"noise .0", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[14], "model": models[0],"etype":"noise .01", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[14], "model": models[1],"etype":"noise .01", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[16], "model": models[0],"etype":"noise .05", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[16], "model": models[1],"etype":"noise .05", "scaler":"standard", "seeds":global_seeds},
]

experiments_semisup_novelty=[
    {"setting": settings[13], "model": models[0],"etype":"noise .0", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[13], "model": models[1],"etype":"noise .0", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[15], "model": models[0],"etype":"noise .01", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[15], "model": models[1],"etype":"noise .01", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[17], "model": models[0],"etype":"noise .05", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[17], "model": models[1],"etype":"noise .05", "scaler":"standard", "seeds":global_seeds},
]

''' 
files=[
'Trajectorys2_100_100_anom.csv' ,
'Trajectorys2_100_50_anom.csv'   ,
'Trajectorys2_100_75_anom.csv',
'Trajectorys2_100_55_norm.csv',
'Trajectorys2_100_85_norm.csv',
'Trajectorys2_100_400_anom.csv',                 #5
'Trajectorys2_72_400_anom0.0noise.csv',
'Trajectorys2_72_400_anom0.2noise.csv',
'Trajectorys2_72_400_anom0.5noise.csv',
'Trajectorys2_72_400_anom1.0noise.csv',

'Trajectorys2_72_400_anom0.0novelty.csv',         #10
'Trajectorys2_72_400_anom0.01novelty.csv',
'Trajectorys2_72_400_anom0.05novelty.csv',
'Trajectorys2_100_150_anom0.1novelty.csv',
'real_drivers_dataset_thresh_40.csv',
'Trajectorys2_72_200_anom.csv',                                 #15
'Trajectorys2_72_400_anom0.00.csv',   # unsup anom levels
'Trajectorys2_72_400_anom0.01.csv',     # unsup anom levels
'Trajectorys2_72_400_anom0.05.csv',     # unsup anom levels
'Trajectorys2_72_400_anom0.1.csv',      # unsup anom levels
'Trajectorys2_72_400_anom0.2.csv',      # unsup anom levels   #20
'Trajectorys2_72_400_anom0.3.csv',      # unsup anom levels

'Trajectorys2_72_1000_anom.csv',
'Trajectorys2_72_2000_norm.csv',
'Trajectorys2_72_400_anom.csv',
'Trajectorys2_72_2000_anom.csv',   #25

'real_drivers_dataset_thresh_40_unsuptest_108_120.csv',

'real_drivers_dataset_thresh_40_unsuptrain_463_565.csv',

'real_drivers_dataset_thresh_40_unsupval_101_120.csv',

'Trajectorys2_100_10000_norm.csv',
'Trajectorys2_100_5000_anom.csv', #30

'Trajectorys2_100_1000_anom.csv',
'Trajectorys2_100_2000_norm.csv',
'Trajectorys2_100_2000_anom.csv',
'Trajectorys2_100_320_anom.csv',   #34


'real_drivers_dataset_thresh_40_semisuptrain_401_401.csv', #35

'real_drivers_dataset_thresh_40_semisupval_123_187.csv',
'real_drivers_dataset_thresh_40_semisuptest_126_187.csv',
'real_drivers_dataset_thresh_71_unsuptrain_533_565.csv',

'real_drivers_dataset_thresh_71_unsupval_110_120.csv',
'real_drivers_dataset_thresh_71_unsuptest_114_120.csv', #40
'brightkite_500_0336_2.1_test.csv',
'brightkite_500_1446_2.1_train_semi.csv',
'brightkite_500_1569_2.1_train_unsup.csv',
'brightkite_500_0336_2.1_val.csv',   #44   
]
'''
