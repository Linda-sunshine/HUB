# HUB
This is the implementation for the paper titled "When Sentiment Analysis Meets Social Network: A Holistic User Behavior Modeling in Opinionated Data". We provide all the source codes for the algorithm and related baselines.
## Quick Start (For Linux and Mac)
* Download the [HUB repo](https://github.com/Linda-sunshine/HUB.git) to your local machine.
* Download the [data]() to the directory that ./src lies in.
* Compile the whole project with [complie file](https://github.com/Linda-sunshine/HUB/blob/master/compile).
```
./compile
```
* Run the algorithm with default setting with [run file](https://github.com/Linda-sunshine/HUB/blob/master/run).
```
./run
```
## Questions regarding running HUB and Baselines
### Q1: What's inside the ./data folder?
**./data** folder has all the data needed for the experiments reported in the paper, including both Amazon data (./data/CoLinAdapt/Amazon/) and Yelp data (./data/CoLinAdapt/YelpNew/). For example, **./data/CoLinAdapt/Amazon** contains the following files which are needed for running experiments with Amazon dataset:
```
CrossGroups_800.txt
Friends.txt
fv_lm_DF_1000.txt
GlobalWeights.txt
SelectedVocab.csv
./Users
```
* **CrossGroups_800.txt** contains the feature indexes for 800 feature groups.
* **Friends.txt** contains the friendship information. In each line, the first string is the user ID and the following strings are his/her friends' IDs.
* **fv_lm_DF_1000.txt** contains the 1000 textual features selected for training language models.
* **GlobalWeights.txt** contains the weights for sentiment features trained on a separate data, which serves as a base model.
* **SelectedVocab.csv** contains the 5000 sentiment features used for training sentiment models.
* **./Users** folder contains 9760 users.

### Q2: How to run the algorithm HUB with different parameters?
We use **-model** to select different algorithms and the default one is HUB.
The following table lists all the parameters for HUB:


### Q2: How to run the algorithm HUB with different parameters?
### Q3: How to run baselines?
### Q4: What does the output mean?

## Citing HUB
We encourage you to cite our work if you have referred it in your work. You can use the following BibTeX citation:
```
@inproceedings{gong2018sentiment,
  title={When Sentiment Analysis Meets Social Network: A Holistic User Behavior Modeling in Opinionated Data},
  author={Gong, Lin and Wang, Hongning},
  booktitle={Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={1455--1464},
  year={2018},
  organization={ACM}
}
```
