## RealWebAssist: A Benchmark for Long-Horizon Web Assistance with Real-World Users
### [Paper](https://arxiv.org/abs/2504.10445) | [Project Page](https://scai.cs.jhu.edu/projects/RealWebAssist/)

![sequential_instruction](images/sequential_instruction.png)

This repo features code for the paper [**RealWebAssist: A Benchmark for Long-Horizon Web Assistance with Real-World Users**](https://arxiv.org/abs/2504.10445).

It contains:
* Instructions for using the benchmark for evaluation
* Implementations of the baseline experiments

### Usage:
To use our benchmark, first clone the repository:
```bash
https://github.com/SCAI-JHU/RealWebAssist.git
```
Then please access our dataset with huggingface link https://huggingface.co/datasets/stonehj05/RealWebAssist

Our benchmark evaluates correctness by determining if the model's output coordinates is within one of the correct bounding boxes.

### File Structure

After downloading the dataset, you would see folders 1-10 (corresponding to 10 human participants). Make a new folder **full_human_dataset** and put these folders under that directory

The final file structure should be as follows: 

RealWebAssist/  
├── **model_scripts/** # 🔧 Inference and baseline model implementations  
├── **output_files/** # 📦 Precomputed outputs and results  
│ └── **reasoning_results/** # Reasoning results from baseline models (e.g., GPT-4o, Claude)  
├── **full_human_dataset/** # 👥 Human interaction data (10 participants)  
│ ├── **1/** # Participant 1 data  
│ ├── **2/** # Participant 2 data  
│ ├── ... # More participants (3–10)  
├── **evaluate.py** # 🚀 Entry point for running all evaluations  
├── **environment.yaml** # 🧪 Conda environment specification (Not recommended to use, please follow our instructions for setting up the environment)  
└── **README.md** # 📖 Project overview and usage instructions  

Each folder for human data should have the following structure:
1/  
├── **answer_images/** # Images showing the ground-truth bounding boxes  
├── **audio/** # Audio clips of participant speech  
├── **images/** # Screenshot of the webpages  
├── **extracted_actions.json** # GPT-4o extracted actions (given as history for baseline evaluations)  
├── **extracted_actions_gt.json** # GPT-4o extracted actions with groud truth captions  
├── **questions_gt.json** # Different versions of questions data. Can just use the second one, which also has the task goal labels  
├── **questions_with_task_updated.json**    
├── **questions_with_task.json**  
├── **questions_Wlarge.json**    
├── **transcriptions_GT.json** # Ground-truth transcriptions of speech and different audio models  
├── **transcriptions_Wlarge.json**   
├── **transcriptions_Wturbo.json**   


### Benchmark Implementation:
To reproduce the baseline results, first set up a conda environment and install the necessary packages:
```bash
conda create -n realwebassist python=3.9
pip install transformers
pip install qwen-vl-utils
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install accelerate
```
For reproducibility and to save cost, we provide the reasoning results for GPT-4o, Claude 3.7 Sonnet and OpenAI O1 (O1 costs around $200 USD for the whole benchmark).

To run the baseline experiments without a reasoning model:

For OS-Atlas:
```bash
python evaluate.py --model_to_run os_atlas
```

For UGround:
```bash
python evaluate.py --model_to_run uground
```

For Aria-UI:
```bash
python evaluate.py --model_to_run aria_ui
```

To run the baseline experiments with the saved reasoning results:

For GPT-4o + OS-Atlas:
```bash
python evaluate.py --model_to_run os_atlas_gpt
```

For GPT-4o + UGround:
```bash
python evaluate.py --model_to_run uground_gpt
```

For GPT-4o + Aria-UI:
```bash
python evaluate.py --model_to_run aria_ui_gpt
```

For OpenAI o1 and Claude 3.7 Sonnet we only provide evaluation with UGround.

For OpenAI o1 + UGround:
```bash
python evaluate.py --model_to_run o1
```

For Claude 3.7 Sonnet + UGround:
```bash
python evaluate.py --model_to_run claude
```

If you want to run the reasoning instead of using the existing results, first install these additional packages for OpenAI API:
```bash
pip install openai
pip install python-dotenv
```

And install these additional packages for Claude API:
```bash
pip install anthropic
```

Then, run the scripts:

For gpt-4o:
```bash
python evaluate.py --model_to_run gpt_reasoning
```

For o1:
```bash
python evaluate.py --model_to_run o1_reasoning
```

For claude:
```bash
python evaluate.py --model_to_run claude_reasoning
```

The script will save the reasoning results to the location needed for evaluation scripts

### Evaluating your own model:
To evaluate your own model on our benchmark, follow these steps:
1. Add your model file (i.e. mymodel.py) under /model_scripts
2. Create a get_coordinate(config_data, history, base_dir, output_dir) function that returns the coordinate that the model outputs
3. Add import to evaluate.py (i.e. from model_scripts import mymodel.py)
4. Change the line that calls the get_coordinate function to match the model file (i.e. x, y = my_model.get_coordinate(config_data, history_string, base_dir, image_output_dir)
5. Run evaluate.py

## 🔎 Citations:
Please cite the paper and star this repo if you find it interesting/useful, thanks!

```bibtex
@article{ye2025realwebassist,
  title={Realwebassist: A benchmark for long-horizon web assistance with real-world users},
  author={Ye, Suyu and Shi, Haojun and Shih, Darren and Yun, Hyokun and Roosta, Tanya and Shu, Tianmin},
  journal={arXiv preprint arXiv:2504.10445},
  year={2025}
}
```
