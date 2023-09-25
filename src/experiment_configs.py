### datasets
datasets = [
    {"dataset_name": "synthetic",  "dtype": "S", "input_dims": 72,  "seg_len": 2, "bs": 256, "lr": 1e-3, "wd": 0.00, "drp": 0.00, "epochs": 150, "hds": 12, "trj_step_feat": ['X_Coord','Y_Coord']},
    {"dataset_name": "amazon",     "dtype": "A", "input_dims": 72,  "seg_len": 2, "bs": 256, "lr": 1e-3, "wd": 0.00, "drp": 0.00, "epochs": 150, "hds": 12, "trj_step_feat": ['X_Coord','Y_Coord']},
    {"dataset_name": "brightkite", "dtype": "B", "input_dims": 500, "seg_len": 2, "bs": 256, "lr": 1e-4, "wd": 0.00, "drp": 0.00, "epochs": 150, "hds": 8, "trj_step_feat": ['X_Coord','Y_Coord']},
    {"dataset_name": "dbcargo",    "dtype": "D", "input_dims": 72,  "seg_len": 2, "bs": 256, "lr": 1e-2, "wd": 0.00, "drp": 0.00, "epochs": 150, "hds": 12, "trj_step_feat": ['X_Coord','Y_Coord']}
]

# ## settings

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

    {"ds": datasets[0], "ds_train": 25, "ds_valid": 22, "ds_test": 8, "name": "unsup syn noise .5", "comment": "", "scenario": "unsup", "sco":"U"},         #10
    {"ds": datasets[0], "ds_train": 23, "ds_valid": 22, "ds_test": 8, "name": "semsiup syn noise .5", "comment": "", "scenario": "semsiup", "sco":"E"},

    {"ds": datasets[0], "ds_train": 25, "ds_valid": 22, "ds_test": 10, "name": "unsup syn novelty .0", "comment": "", "scenario": "unsup", "sco":"U"},
    {"ds": datasets[0], "ds_train": 23, "ds_valid": 22, "ds_test": 10, "name": "semsiup syn novelty .0", "comment": "", "scenario": "semsiup", "sco":"E"},

    {"ds": datasets[0], "ds_train": 25, "ds_valid": 22, "ds_test": 11, "name": "unsup syn novelty .01", "comment": "", "scenario": "unsup", "sco":"U"},
    {"ds": datasets[0], "ds_train": 23, "ds_valid": 22, "ds_test": 11, "name": "semsiup syn novelty .01", "comment": "", "scenario": "semsiup", "sco":"E"},  #15

    {"ds": datasets[0], "ds_train": 25, "ds_valid": 22, "ds_test": 12, "name": "unsup syn novelty .05", "comment": "", "scenario": "unsup", "sco":"U"},
    {"ds": datasets[0], "ds_train": 23, "ds_valid": 22, "ds_test": 12, "name": "semsiup syn novelty .05", "comment": "", "scenario": "semsiup", "sco":"E"},
    
    {"ds": datasets[3], "ds_train": 45, "ds_valid": 47, "ds_test": 48, "name": "unsup dbcargo orig", "comment": "", "scenario": "unsup", "sco":"U"},
    {"ds": datasets[3], "ds_train": 46, "ds_valid": 47, "ds_test": 48, "name": "semsiup dbcargo orig", "comment": "", "scenario": "semsiup", "sco":"E"},
    
    {"ds": datasets[1], "ds_train": 49, "ds_valid": 51, "ds_test": 52, "name": "unsup amazon orig", "comment": "", "scenario": "unsup", "sco":"U"},          #20
    {"ds": datasets[1], "ds_train": 50, "ds_valid": 51, "ds_test": 52, "name": "semsiup amazon orig", "comment": "", "scenario": "semsiup", "sco":"E"},
    
    {"ds": datasets[2], "ds_train": 53, "ds_valid": 55, "ds_test": 56, "name": "unsup bright orig", "comment": "", "scenario": "unsup", "sco":"U"},          
    {"ds": datasets[2], "ds_train": 54, "ds_valid": 55, "ds_test": 56, "name": "semsiup bright orig", "comment": "", "scenario": "semsiup", "sco":"E"}
]

### model
models=[
    { "model_type": "GADFormer", "loss_func": "bce", "heads":12 }, #GADFormer
    { "model_type": "GRU", "loss_func": "bce" },
    { "model_type": "TrajBERT", "heads":8 }, #todo: add params from comparision,
    { "model_type": "MainTulGAD"} 
]

global_seeds=list(range(10)) #[34,38,30]

#experiments
experiments_unsup_orig=[
    {"setting": settings[0], "model": models[0], "etype":"orig", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[0], "model": models[1], "etype":"orig", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[0], "model": models[3], "etype":"orig", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[20], "model": models[0], "etype":"orig", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[20], "model": models[1], "etype":"orig", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[20], "model": models[3], "etype":"orig", "scaler":"standard", "seeds":global_seeds},
    #{"setting": settings[4], "model": models[0], "etype":"orig", "scaler":"standard", "seeds":global_seeds},
    #{"setting": settings[4], "model": models[1], "etype":"orig", "scaler":"standard", "seeds":global_seeds},
    #{"setting": settings[4], "model": models[3], "etype":"orig", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[18], "model": models[0], "etype":"orig", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[18], "model": models[1], "etype":"orig", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[18], "model": models[3], "etype":"orig", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[22], "model": models[0], "etype":"orig", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[22], "model": models[1], "etype":"orig", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[22], "model": models[3], "etype":"orig", "scaler":"standard", "seeds":global_seeds},
]

experiments_semisup_orig=[
    {"setting": settings[1], "model": models[0], "etype":"orig", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[1], "model": models[1], "etype":"orig", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[1], "model": models[3], "etype":"orig", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[21], "model": models[0], "etype":"orig", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[21], "model": models[1], "etype":"orig", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[21], "model": models[3], "etype":"orig", "scaler":"standard", "seeds":global_seeds},
    #{"setting": settings[5], "model": models[0], "etype":"orig", "scaler":"standard", "seeds":global_seeds},
    #{"setting": settings[5], "model": models[1], "etype":"orig", "scaler":"standard", "seeds":global_seeds},
    #{"setting": settings[5], "model": models[3], "etype":"orig", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[19], "model": models[0], "etype":"orig", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[19], "model": models[1], "etype":"orig", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[19], "model": models[3], "etype":"orig", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[23], "model": models[0], "etype":"orig", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[23], "model": models[1], "etype":"orig", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[23], "model": models[3], "etype":"orig", "scaler":"standard", "seeds":global_seeds},
]

experiments_unsup_noise=[
    {"setting": settings[6], "model": models[0], "etype":"noise .0", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[6], "model": models[1], "etype":"noise .0", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[6], "model": models[3], "etype":"noise .0", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[8], "model": models[0], "etype":"noise .2", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[8], "model": models[1], "etype":"noise .2", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[8], "model": models[3], "etype":"noise .2", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[10], "model": models[0],"etype":"noise .5", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[10], "model": models[1],"etype":"noise .5", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[10], "model": models[3],"etype":"noise .5", "scaler":"standard", "seeds":global_seeds}
]

experiments_semisup_noise=[
    {"setting": settings[7], "model": models[0],"etype":"noise .0", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[7], "model": models[1],"etype":"noise .0", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[7], "model": models[3],"etype":"noise .0", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[9], "model": models[0],"etype":"noise .2", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[9], "model": models[1],"etype":"noise .2", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[9], "model": models[3],"etype":"noise .2", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[11], "model": models[0],"etype":"noise .5", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[11], "model": models[1],"etype":"noise .5", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[11], "model": models[3],"etype":"noise .5", "scaler":"standard", "seeds":global_seeds}
]

experiments_unsup_novelty=[
    {"setting": settings[12], "model": models[0],"etype":"novelty .0", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[12], "model": models[1],"etype":"novelty .0", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[12], "model": models[3],"etype":"novelty .0", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[14], "model": models[0],"etype":"novelty .01", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[14], "model": models[1],"etype":"novelty .01", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[14], "model": models[3],"etype":"novelty .01", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[16], "model": models[0],"etype":"novelty .05", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[16], "model": models[1],"etype":"novelty .05", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[16], "model": models[3],"etype":"novelty .05", "scaler":"standard", "seeds":global_seeds}
]

experiments_semisup_novelty=[
    {"setting": settings[13], "model": models[0],"etype":"novelty .0", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[13], "model": models[1],"etype":"novelty .0", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[13], "model": models[3],"etype":"novelty .0", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[15], "model": models[0],"etype":"novelty .01", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[15], "model": models[1],"etype":"novelty .01", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[15], "model": models[3],"etype":"novelty .01", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[17], "model": models[0],"etype":"novelty .05", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[17], "model": models[1],"etype":"novelty .05", "scaler":"standard", "seeds":global_seeds},
    {"setting": settings[17], "model": models[3],"etype":"novelty .05", "scaler":"standard", "seeds":global_seeds}
]

experiments_unsup  =[experiments_unsup_orig  , experiments_unsup_noise  , experiments_unsup_novelty]
experiments_semisup=[experiments_semisup_orig, experiments_semisup_noise, experiments_semisup_novelty]
experiments_all    =[experiments_unsup_orig, experiments_semisup_orig, experiments_unsup_noise, experiments_semisup_noise, experiments_unsup_novelty, experiments_semisup_novelty]


''' #----------------------AVAILABLE DATASETS-----------------------#
'''
files=['Trajectorys2_100_100_anom.csv' ,
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

       'amazon_drivers_dataset_thresh_40_unsuptest_108_120.csv',

       'amazon_drivers_dataset_thresh_40_unsuptrain_463_565.csv',

       'amazon_drivers_dataset_thresh_40_unsupval_101_120.csv',

       'Trajectorys2_100_10000_norm.csv',
       'Trajectorys2_100_5000_anom.csv', #30

       'Trajectorys2_100_1000_anom.csv',
       'Trajectorys2_100_2000_norm.csv',
       'Trajectorys2_100_2000_anom.csv',
       'Trajectorys2_100_320_anom.csv',   #34
        

       'amazon_drivers_dataset_thresh_40_semisuptrain_401_401.csv', #35

       'amazon_drivers_dataset_thresh_40_semisupval_123_187.csv',
       'amazon_drivers_dataset_thresh_40_semisuptest_126_187.csv',
       'amazon_drivers_dataset_thresh_71_unsuptrain_533_565.csv',

       'amazon_drivers_dataset_thresh_71_unsupval_110_120.csv',
       'amazon_drivers_dataset_thresh_71_unsuptest_114_120.csv', #40
       'brightkite_500_0336_2.1_test.csv',
       'brightkite_500_1446_2.1_train_semi.csv',
       'brightkite_500_1569_2.1_train_unsup.csv',
       'brightkite_500_0336_2.1_val.csv',   #44   
       
       'Trajectories_72_173_dbcargo-trn-unsup.csv',   #45   
       'Trajectories_72_146_dbcargo-trn-semisup.csv',      
       'Trajectories_72_55_dbcargo-vld.csv',      
       'Trajectories_72_44_dbcargo-tst.csv',  
       
       'Trajectories_72_515_amazon-trn-unsup.csv',  # 49
       'Trajectories_72_486_amazon-trn-semisup.csv',  # 50
       'Trajectories_72_161_amazon-vld.csv',  
       'Trajectories_72_129_amazon-tst.csv',
       
       'Trajectories_500_1433_brightkite-trn-unsup.csv',
       'Trajectories_500_1300_brightkite-trn-semisup.csv',
       'Trajectories_500_449_brightkite-vld.csv', # 55
       'Trajectories_500_359_brightkite-tst.csv'
]

