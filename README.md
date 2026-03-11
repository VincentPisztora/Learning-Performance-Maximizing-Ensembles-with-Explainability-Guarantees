# Learning-Performance-Maximizing-Ensembles-with-Explainability-Guarantees

This repository contains the code referenced in the AAAI24 paper "Learning Performance Maximizing Ensembles with Explainability Guarantees" ([arxiv](https://arxiv.org/abs/2312.12715)) ([AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/29378/30602)).

Please find below a step-by-step guide for training the models described in the paper.

Setup:

0. Create the python environment
   - The packages needed to run all code are specified in the "env_eeg.yml" file. The environment can be built using this yml file as described here: [link](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html)

1. Set directories:
   - "utils.py": Set the 'superfix' variable in the 'get_superfix_prefix_and_make_dir' and 'get_superfix' functions

3. Train underlying task glass box and black box models:
 - Run the pipeline_step_1.py script. This script constructs the underlying task (i.e. regression and classification) datasets.
    - python3 pipeline_step_1.py --run_machine='local' --n_splits=4 --split_type='kfold' > pipeline_step_1_outputs.txt
 - Run the pipeline_step_2.py script

<img src="https://www.google-analytics.com/collect?v=1&tid=G-QST3V3PB55&cid=555&t=event&ec=repo&ea=view&el=Learning-Performance-Maximizing-Ensembles-with-Explainability-Guarantees" style="display:none">

