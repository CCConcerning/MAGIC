

# Goal Consistency: An Effective Multi-Agent Cooperative Method for Multistage Tasks (MAGIC)

This is the code for the paper:
[Goal Consistency: An Effective Multi-Agent Cooperative Method for Multistage Tasks](https://www.ijcai.org/proceedings/2022/0025.pdf).

## Requirements
* [OpenAI baselines](https://github.com/openai/baselines)
* [Multi-Agent Particle Environments (MPE)](https://github.com/openai/multiagent-particle-envs)
* Known dependencies: Python (3.6.9), OpenAI gym (0.10.5), tensorflow (1.12.0), numpy (1.16.4)

The versions are just what I used and not necessarily strict requirements.

## How to Run
The "Resource Collection" environment from our paper is referred to as `expand_simple_spread` in this repo, 
"Multi-Point Transportation" is referred to as `simple_push_ball`, and "Endangered Wildlife Rescue" is referred to as `rescue`.

```
python train.py --scenario expand_simple_spread --max-episode-len 30 

python train.py --scenario simple_push_ball --max-episode-len 200

python train_rescue.py --scenario rescue --max-episode-len 60
```

## Paper citation

If you used this code for your experiments or found it helpful, consider citing the following paper:

<pre>
@inproceedings{DBLP:conf/ijcai/ChenLZDL22,
  author       = {Xinning Chen and Xuan Liu and Shigeng Zhang and Bo Ding and Kenli Li},
  editor       = {Luc De Raedt},
  title        = {Goal Consistency: An Effective Multi-Agent Cooperative Method for Multistage Tasks},
  booktitle    = {Proceedings of the Thirty-First International Joint Conference on
                  Artificial Intelligence, {IJCAI} 2022, Vienna, Austria, 23-29 July 2022},
  pages        = {172--178},
  publisher    = {ijcai.org},
  year         = {2022},
  url          = {https://doi.org/10.24963/ijcai.2022/25},
  doi          = {10.24963/IJCAI.2022/25},
  timestamp    = {Mon, 13 Mar 2023 11:20:33 +0100},
  biburl       = {https://dblp.org/rec/conf/ijcai/ChenLZDL22.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
</pre>
