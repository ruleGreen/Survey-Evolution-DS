# Survey-Evolution-DS
This is the repo which records the evolution of LM-based dialogue system. We list works in each stage, and will constantly update it, welcome to raise a issue to add new works!!

- Task-oriented Dialogue System (TOD)
    - Natural Language Understanding (NLU)
    - Dialogue State Tracking (DST)
    - Dialogue Policy Learning (DST)
    - Natural Language Generation (NLG)
- Open-domain Dialogue System (ODD)
- Unified Dialogue System (UniDS)

## Survey Paper


- [A Survey of Language Model-based Dialogue System] :fire::fire::fire::fire::fire:
- [End-to-end Task-oriented Dialogue: A Survey of Tasks, Methods, and Future Directions](https://github.com/ruleGreen/Survey-Evolution-DS.git) [End-to-end TOD][EMNLP 2023] :fire::fire::fire:
- [Recent advances in deep learning based dialogue systems: a systematic survey](https://sentic.net/dialogue-systems-survey.pdf) [Artificial Intelligence Review 2023] :fire::fire::fire:
- [A Survey on Recent Advances and Challenges in Reinforcement Learning Methods for Task-oriented Dialogue Policy Learning](https://link.springer.com/article/10.1007/s11633-022-1347-y)[DPL][Machine Intelligence Research 2023] :fire:
- [A Survey on Proactive Dialogue Systems: Problems, Methods, and Prospects](https://arxiv.org/abs/2305.02750) [ODD][IJCAI 2023] :fire::fire:
- [Let's Negotiate! A Survey of Negotiation Dialogue Systems](https://arxiv.org/pdf/2212.09072.pdf) [ODD][Arxiv 2022]
- [Recent advances and challenges in task-oriented dialog systems](https://link.springer.com/article/10.1007/s11431-020-1692-3)[TOD][SCTC 2020]
- [Challenges in Building Intelligent Open-domain Dialog Systems](https://dl.acm.org/doi/abs/10.1145/3383123) [ODD][TOIS 2020]
- [A Survey on Dialogue Systems: Recent Advances and New Frontiers](https://dl.acm.org/doi/10.1145/3166054.3166058) [TOD][ODD][SIGKDD 2017]



## 1st Stage -- SLM: Early Stage

- Eliza, Alice, GUS


## 2nd Stage -- NLM: Independent Development

- [End-to-End Learning of Task-Oriented Dialogs](https://aclanthology.org/N18-4010/)[End-to-end TOD][NAACL 2018] first E2E TOD




## 3rd Stage -- PLM: Fusion Starts!

- [Improving Factual Consistency for Knowledge-Grounded Dialogue Systems via Knowledge Enhancement and Alignment](https://arxiv.org/abs/2310.08372.pdf) [ODD][EMNLP 2023]
- [JoTR: A Joint Transformer and Reinforcement Learning Framework for Dialog Policy Learning](https://arxiv.org/abs/2309.00230.pdf) [DPL][TOD][Arxiv 2023]
- [Retrieval-free Knowledge Injection through Multi-Document Traversal for Dialogue Models](https://aclanthology.org/2023.acl-long.364/) [ODD][ACL 2023]
- [Integrating Pretrained Language Model for Dialogue Policy Evaluation](https://ieeexplore.ieee.org/abstract/document/9747593.pdf) :fire::fire::fire: first work of RLAIF in DPL

### Fusions within TOD

- [Soloist: Building Task Bots at Scale with Transfer Learning and Machine Teaching](https://aclanthology.org/2021.tacl-1.49/) [TACL 2021]

- **PPTOD**: [Multi-Task Pre-Training for Plug-and-Play Task-Oriented Dialogue System](https://aclanthology.org/2022.acl-long.319/) [ACL 2022]

### Fusion between TOD with ODD

- [Q-TOD: A Query-driven Task-oriented Dialogue System](https://aclanthology.org/2022.emnlp-main.489.pdf) [TOD -> ODD][EMNLP 2022]

- [UniDS: A Unified Dialogue System for Chit-Chat and Task-oriented Dialogues](https://aclanthology.org/2022.dialdoc-1.2/) [ODD -> TOD][DialDoc 2022]

- [GODEL: Large-Scale Pre-Training for Goal-Directed Dialog](https://arxiv.org/abs/2206.11309) [TOD -> ODD][Arxiv 2022][[Code](https://github.com/microsoft/GODEL)]


### Fusion between DM and LLM

- **LLaMA2-Chat** Llama 2: Open Foundation and Fine-Tuned Chat Models
- **Parrot**: Enhancing Multi-Turn Chat Models by Learning to Ask Questions
- Enhancing Chat Language Models by Scaling High-quality Instructional Conversations
- **BlenderBot 3**: a deployed conversational agent that continually learns to responsibly engage


## 4nd Stage -- LLM-based Dialogue System

- [TPE: Towards Better Compositional Reasoning over Conceptual Tools with Multi-persona Collaboration](https://arxiv.org/abs/2309.16090.pdf) [ODD][Arxiv 2023] :fire::fire::fire::fire::fire: language agent, tool learning

- [Large Language Models as Source Planner for Personalized Knowledge-grounded Dialogues](https://arxiv.org/pdf/2310.08840.pdf) [ODD][EMNLP 2023]

- [Cue-CoT: Chain-of-thought Prompting for Responding to In-depth Dialogue Questions with LLMs](https://arxiv.org/pdf/2305.11792.pdf) [ODD][EMNLP 2023] :fire::fire::fire: linguistic cues

- [Mirages: On Anthropomorphism in Dialogue Systems](https://arxiv.org/pdf/2305.09800v1.pdf) [ODD] linguistic cues

- [Target-oriented Proactive Dialogue Systems with Personalization: Problem Formulation and Dataset Curation](https://arxiv.org/pdf/2310.07397.pdf) [TOD][EMNLP 2023]
- [Prompt-Based Monte-Carlo Tree Search for Goal-Oriented Dialogue Policy Planning](https://arxiv.org/abs/2305.13660.pdf) [TOD][DPL][EMNLP 2023]

- [Prompting and Evaluating Large Language Models for Proactive Dialogues: Clarification, Target-guided, and Non-collaboration](https://arxiv.org/abs/2305.13626.pdf) [ODD][Proactive][EMNLP 2023]

- [MemoChat: Tuning LLMs to Use Memos for Consistent Long-Range Open-Domain Conversation](https://arxiv.org/pdf/2308.08239.pdf) [ODD][Memory][Arxiv 2023][[Code](https://github.com/LuJunru/MemoChat)]

- [Commonsense-Aware Prompting for Controllable Empathetic Dialogue Generation](https://arxiv.org/abs/2302.01441)  [ODD][Empathetic]

-  [Are LLMs All You Need for Task-Oriented Dialogue?](https://aclanthology.org/2023.sigdial-1.21/) [TOD][SIGDIAL 2023] all sub tasks

- [Prompted LLMs as Chatbot Modules for Long Open-domain Conversation](https://aclanthology.org/2023.findings-acl.277/) [ODD][Memory][ACL 2023]

#### Persona/Character/Profile/Role

- [CharacterChat: Supporting the Creation of Fictional Characters through Conversation and Progressive Manifestation with a Chatbot](https://arxiv.org/abs/2106.12314.pdf)


## What's the future?



## Other Useful Resourecs

1. https://www.promptingguide.ai/papers [prompting engineering papers]
2. https://github.com/iwangjian/Paper-Reading#knowledge-grounded-dialogue


welcome to cite our survey paper.

```
@misc{wang2023survey,
      title={A Survey of the Evolution of Language Model-Based Dialogue Systems}, 
      author={Hongru Wang and Lingzhi Wang and Yiming Du and Liang Chen and Jingyan Zhou and Yufei Wang and Kam-Fai Wong},
      year={2023},
      eprint={2311.16789},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

