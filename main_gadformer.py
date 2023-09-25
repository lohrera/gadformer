

import argparse
from src.run_experiments import run

def main():
    parser = argparse.ArgumentParser(description='Run GADFormer Experiments.')
    parser.add_argument('--root_dir', type=str, default='./datasets/files_valid/', help="dataset directory")

    args = parser.parse_args()
    print(args.root_dir)
    
    run(args.root_dir)

if __name__ == '__main__':
    main()
   