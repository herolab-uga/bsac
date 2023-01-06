# Bayesian Soft Actor Critic (BSAC)

 Adopting reasonable strategies is challenging but crucial for an intelligent agent with limited resources working in hazardous, unstructured, and dynamic changing environments to improve the system utility, decrease the overall cost, and increase mission success probability. Deep Reinforcement Learning (DRL) helps organize agents' behaviors and actions based on their state and represents complex strategies (composition of actions). This project proposes a novel hierarchical strategy decomposition approach based on Bayesian chaining to separate an intricate policy into several simple sub-policies and organize their relationships as Bayesian strategy networks (BSN). We integrate this approach into the state-of-the-art DRL method, soft actor-critic (SAC), and build the corresponding Bayesian soft actor-critic (BSAC) model by organizing each sub-policies as a joint policy.

<!-- ## Paper: [Bayesian Strategy Network Based Soft Actor-Critic in Deep Reinforcement Learning](https://arxiv.org/pdf/2208.06033.pdf) -->

## Example

<div align = center>
<img src="https://github.com/RickYang2016/Bayesian-Soft-Actor-Critic/blob/main/figures/walker2d.png" height="243" alt="Hopper-V2 3SABC"><img src="https://github.com/RickYang2016/Bayesian-Soft-Actor-Critic/blob/main/figures/biped_robot.gif" height="243" width="350" alt="Hopper-V2 3SABC Video"/>
</div>

![image](https://github.com/RickYang2016/Bayesian-Soft-Actor-Critic/blob/main/figures/policy_network.png)

## Experiments Setup

This implementation requires [Anaconda](https://www.anaconda.com/) / OpenAI Gym / Mujoco / [PyTorch](https://pytorch.org/) / rl-plotter.

### Getting Started

  1. Install [OpenAI Gym](https://www.gymlibrary.ml/):
  ```
  pip install gym
  ```

  2. Install [Mujoco](https://mujoco.org/):
 
   - [Download Mujoco 200 linux](https://www.roboti.us/download.html):  
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

### Examples for Training Agent

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
cd ~/humanoid-v2_3bsac
pyhton3 main_bsac.py
```
- 5 factors BSAC:
```
cd ~/humanoid-v2_5bsac
pyhton3 main_bsac.py
```
- 9 factors BSAC:
```
cd ~/humanoid-v2_9bsac
pyhton3 main_bsac.py
```

> Note: Before running the code, please set the specific directory in files `main_bsac.py` and `networks.py` for the data updating.

## Evaluation

<div align = center>
<img src="https://github.com/RickYang2016/Bayesian-Soft-Actor-Critic/blob/main/figures/hopper-v2_3bsac.png" height="250" alt="Hopper-V2 3SABC"><img src="https://github.com/RickYang2016/Bayesian-Soft-Actor-Critic/blob/main/figures/hopper-v2_3bsac.gif" height="230" width="156" alt="Hopper-V2 3SABC Video"><img src="https://github.com/RickYang2016/Bayesian-Soft-Actor-Critic/blob/main/figures/hopper-v2.png" height="250" alt="Hopper-V2 3SABC Video"/>
</div>
  
<div align = center>
<img src="https://github.com/RickYang2016/Bayesian-Soft-Actor-Critic/blob/main/figures/walker2d_v2_5bsac.png" height="215" alt="Hopper-V2 3SABC"><img src="https://github.com/RickYang2016/Bayesian-Soft-Actor-Critic/blob/main/figures/walker2d-v2_5bsac.gif" height="230" width="156" alt="Hopper-V2 3SABC Video"><img src="https://github.com/RickYang2016/Bayesian-Soft-Actor-Critic/blob/main/figures/walker2d-v2.png" height="250" alt="Hopper-V2 3SABC Video"/>
</div>

<div align = center>
<img src="https://github.com/RickYang2016/Bayesian-Soft-Actor-Critic/blob/main/figures/humanoid-v2_5bsac.png" height="177" alt="Hopper-V2 3SABC"><img src="https://github.com/RickYang2016/Bayesian-Soft-Actor-Critic/blob/main/figures/humanoid-v2_3bsac.gif" height="230" width="156" alt="Hopper-V2 3SABC Video"><img src="https://github.com/RickYang2016/Bayesian-Soft-Actor-Critic/blob/main/figures/humanoid-v2-compare.png" height="250" alt="Hopper-V2 3SABC Video"/>
</div>

<div align = center>
<img src="https://github.com/RickYang2016/Bayesian-Soft-Actor-Critic/blob/main/figures/humanoid-v2-3%269bsac.png" height="230" alt="Hopper-V2 3SABC"><img src="https://github.com/RickYang2016/Bayesian-Soft-Actor-Critic/blob/main/figures/bsac_compare.png" height="250" alt="Hopper-V2 3SABC Video"/>
</div>

## Conclusion

From theoretical derivation, we formulate the training process of the BSAC and implement it in OpenAI's MuJoCo standard continuous control benchmark domains such as the Hopper, Walker, and the Humanoid. The results illustrated the effectiveness of the proposed architecture in enabling the application domains with high-dimensional action spaces and can achieve higher performance against the state-of-the-art RL methods. Furthermore, we believe that the potential generality and practicability of the BSAC evoke further theoretical and empirical investigations. Especially, implementing the BSAC on real robots is not only a challenging problem but will also help us develop robust computation models for multi-agent/robot systems, such as robot locomotion control, multi-robot planning and navigation, and robot-aided search and rescue missions.
