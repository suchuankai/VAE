import argparse
import train_net
import time

def init_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs"  , type=int , default=100 , help = "epochs of training")
    parser.add_argument("--batch_size", type=int , default=32  , help = "size of batch")
    parser.add_argument("--plot"      , type=bool, default=True, help = "decide plot the picture or not")
    parser.add_argument("--train_path", type=str , default="./data/dataset4-1.csv", help = "decide training path")
    parser.add_argument("--test_path" , type=str , default="./data/dataset4-1.csv", help = "decide testing path")
    return parser.parse_args()


def main():
    start = time.time()
    opt = init_parameters()
    train_net.Train_VAE(opt)
    print(f"Training use {(time.time()-start) / 60} minutes.")
    

if __name__ == '__main__':
    main()   