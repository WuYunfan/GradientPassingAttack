## Environment

- Python >= 3.11
- PyTorch >= 2.1.0
- DGL >= 1.1.2

## Dataset

The processed data for Gowalla, Yelp, and Tenrec can be downloaded from [Baidu Wangpan](https://pan.baidu.com/s/153rQx1a4-EtKma6bTpn5mw?pwd=we9f) or [Google Drive](https://drive.google.com/file/d/1IeGXFJ7aEiQJ8fQ5yAJO9aHsHE1THs7h/view?usp=share_link).

Please organize the processed data as follows:

```
├── GradientPassingAttack-master
│   ├── data
│   │   ├── Gowalla
│   │   │   └── time
│   │   ├── Yelp
│   │   └── Tenrec
│   ├── run
│   ├── attacker
│   ├── model.py
│   └── ...
```

For each dataset, the **time** folder contains the dataset split by time, with train and validation sets to train and tune the hyperparameters for different victim recommenders.

## Quick Start

To launch the experiment, use:

```sh
python -u -m run.run
```

We provide the implemented codes of all baseline methods in **a unified framework**.

### Testing on Different Datasets

To test on a different dataset, change the import statements in `run/run.py`:

```python
from config import get_gowalla_config as get_config
from config import get_gowalla_attacker_config as get_attacker_config
```

Replace `'gowalla'` with the desired dataset.

### Running Baseline Attackers

To run baseline attackers, modify the following line in `run/run.py`:

```python
attacker_config = get_attacker_config()[0]
```

Change the index `0` to the index of the specific attacker defined in `config.py`.

### Attacking Different Victim Recommenders

To attack different victim recommenders, update the code in `run/run.py`:

```python
configs = get_config(device)[:-1]
```

Change the index `[:-1]` to the index of the specific victim recommender defined in `config.py`.

### Changing Hyperparameters

The hyperparameters of all methods can be easily changed in `config.py`.