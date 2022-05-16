# Bayesian Soft Actor Critic (BSAC)

## Experiments Setup

This implementation requires Anaconda / OpenAI Gym / Mujoco / PyTorch / rl-plotter.

## Getting Started

  1. Install [OpenAI Gym](https://www.gymlibrary.ml/):
  ```
  pip install gym
  ```

  2. Install [Mujoco](https://mujoco.org/):
 
   - [Download Mujoco 200](https://www.roboti.us/download.html):  
  ```
  mkdir -p ~/.mujoco && cd ~/.mujoco
  wget -P . https://www.roboti.us/download/mjpro200_linux.zip
  unzip mjpro200_linux.zip
  ```
   - Copy your Mujoco license key (mjkey.txt) to the path:
  ```
  cp mjkey.txt ~/.mujoco
  cp mjkey.txt ~/.mujoco/mujoco200_linux/bin
  ```
   - Add environment variables:
  ```
  export LD_LIBRARY_PATH=~/.mujoco/mujoco200/bin${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} 
  export MUJOCO_KEY_PATH=~/.mujoco${MUJOCO_KEY_PATH}
  ```
   - Download [mujoco-py](https://github.com/openai/mujoco-py) and create conda environment:
  ```
  mkdir ~/mujoco_py
  cd ~/mujoco-py
  git clone https://github.com/openai/mujoco-py.git
  conda create -n myenv python=3.6
  source activate myenv
  sudo apt-get install build-essential
  ```
   - Install dependence:
   ```
   cd ~/mujoco-py
   pip install -r requirements.txt
   pip install -r requirements.dev.txt
   python setup.py install
   ```

  3. Install reinforcement learning (RL) plotter -- [rl-plotter](https://github.com/gxywy/rl-plotter):
  ```
  pip install rl_plotter
  ```

## Examples for Training Agent

1. Hopper-V2 with 3 factors BSAC:
```
cd ~/hopper-v2_3bsac
pyhton3 main_bsac.py 
```
2. Walker2d-V2 with 5 factors BSAC:
```
cd ~/walker2d-v2_5bsac
pyhton3 main_bsac.py
```
3. Humanoid-V2:
- 3 factors BSAC:
```
cd ~/humanoid-v2/humanoid-v2_3bsac
pyhton3 main_bsac.py
```
- 5 factors BSAC:
```
cd ~/humanoid-v2/humanoid-v2_5bsac
pyhton3 main_bsac.py
```
- 9 factors BSAC:
```
cd ~/humanoid-v2/humanoid-v2_9bsac
pyhton3 main_bsac.py
```
