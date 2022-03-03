import numpy as np
from trainer import Trainer
from hyperparams import Hyperparams as hp

def main():

    modeltrainer = Trainer()

    modeltrainer.update(hp.__dict__)

    params = "{0:25} | {1:25}"
    for k,v in modeltrainer.__dict__.items():
        if isinstance(v,(str,int,float,np.ndarray)):
            print(params.format(k,v))

    modeltrainer.build()

    modeltrainer.train()

if __name__ == '__main__':
    main()