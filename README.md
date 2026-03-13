# Learning-Performance-Maximizing-Ensembles-with-Explainability-Guarantees

<div align="center">
  <img src="/assets/ToyExample_4Scatter_vE.png" alt="image" width="600">
  <br>
  <footnotesize><em>This figure shows a two-class classification task in which the areas of expertise (the diamond pattern for the glass box and the spiral pattern for the black box model) are complementary. The glass box achieves a 92.7% accuracy, the black box reaches 95.0% accuracy, and the allocated ensemble of the two exceeds both with a 95.8% accuracy. Thus, the resulting EEG allocation improves performance over both component models while also providing explainability (for 20% of observations in this case).</em></footnotesize>
</div>
<br>

This repository contains the code referenced in the AAAI24 paper "Learning Performance Maximizing Ensembles with Explainability Guarantees" ([arxiv](https://arxiv.org/abs/2312.12715)) ([AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/29378/30602)).

In this paper we propose a method for the optimal allocation of observations between an intrinsically explainable glass box model and a black box model. An optimal allocation being defined as one which, for any given explainability level (i.e. the proportion of observations for which the explainable model is the prediction function), maximizes the performance of the ensemble on the underlying task, and maximizes performance of the explainable model on the observations allocated to it, subject to the maximal ensemble performance condition. The proposed method is shown to produce such explainability optimal allocations on a benchmark suite of tabular datasets across a variety of explainable and black box model types. These learned allocations are found to consistently maintain ensemble performance at very high explainability levels (explaining 74% of observations on average), and in some cases even outperforming both the component explainable and black box models while improving explainability.

Please find below a step-by-step guide for training the models described in the paper.

Setup:

0. Create the python environment:
   - The packages needed to run all code are specified in the env_eeg.yml file.
   - The environment can be built using this yml file as described here: [link](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html)

1. Set directories:
   - utils.py: Set the `superfix` variable in the `get_superfix_prefix_and_make_dir` and `get_superfix` functions

3. Train underlying task glass box and black box models:
   - Run the pipeline_step_1.py script. This script constructs the underlying task (i.e. regression and classification) datasets.
     - `python3 pipeline_step_1.py --run_machine='local' --n_splits=4 --split_type='kfold' > pipeline_step_1_outputs.txt`
   - Run the pipeline_step_2.py script. This script fits the ensemble member models on the underlying task datasets.
     - `python3 pipeline_step_2.py --run_machine='local' --model_type='logistic_regression' --n_splits=4 --split_type='kfold' --fit_all_splits=False > pipeline_step_2_outputs.txt`

4. Train the allocator models:
   - Run the pipeline_step_3.py script. This script constructs the allocator training dataset used to learn how to optimally allocate between the glass box and black box models.
     - `python3 pipeline_step_3.py --run_machine='local' > pipeline_step_3_outputs.txt`
   - Run the pipeline_step_4.py script. This script fits all allocator models for each dataset described in the paper. For each (dataset, glass box type, black box type, allocator type) 4-tuple, an allocator (of the specified type) is learned to optimally allocate prediction between the best glass box and black box of the specified type available for the given dataset.
     - `python3 pipeline_step_4.py --run_machine='local' --r_cutoff_type='val' --hyper_set='large' --rank_type='bb_minus_gb' --feat_type='all' --dist_type='cemse' --rep=0 > pipeline_step_4_outputs.txt`

More detailed descriptions of each script and input args are available within each .py file. 


<img src="https://www.google-analytics.com/collect?v=1&tid=G-QST3V3PB55&cid=555&t=event&ec=repo&ea=view&el=Learning-Performance-Maximizing-Ensembles-with-Explainability-Guarantees" style="display:none">

