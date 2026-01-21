Project Requirement: AI-Generated Video Comment Analysis (v2)
1. Project Overview

Research Goal: Evaluate public acceptance and engagement with AI-generated short videos through text-based analysis.


Core Metrics: Focus on Sentiment (Positive/Negative) and Engagement Depth (Linguistic Complexity).




Design Principle: Use efficient, open-source command-line tools for data collection to avoid API limitations.


2. Technical Stack
Collection: yt-dlp, youtube-comment-downloader.

Processing: Python 3.10+, pandas, json.


NLP & Analysis: transformers (DistilBERT), textstat (Lexical density).

Visualization: seaborn, matplotlib.

3. Directory Structure
Plaintext

/ai-discourse-analysis
├── /data                
│   ├── /raw_json        # Individual JSON files per video
│   └── merged_data.csv  # Final cleaned dataset
├── /scripts
│   ├── collect.sh       # The loop script for data collection
│   ├── processor.py     # JSON to CSV conversion and cleaning
│   ├── analyzer.py      # Sentiment and linguistic scoring
│   └── visualizer.py    # Plotting results
├── requirements.txt
└── main.py              
4. Key Task Modules
T1: Data Collection Logic (collect.sh)

Objective: Bulk download comments from an entire YouTube channel.


Implementation:

使用 yt-dlp --get-id --flat-playlist 获取指定频道的所有视频 ID 。


编写 Bash 或 Python 循环，针对每个 ID 调用 youtube-comment-downloader 。


命令行参考： youtube-comment-downloader --youtubeid $id --output data/raw_json/${id}.json

T2: Data Consolidation (processor.py)
Task: 将 /data/raw_json/ 下的所有 .json 文件读取并合并为一个 Pandas DataFrame。

Cleaning:

提取核心字段：text, votes, replies, time 。


过滤非英语评论（可选，视研究范围而定） 。

去除重复评论和纯表情符号。

T3: Linguistic & Sentiment Analyzer (analyzer.py)

Sentiment Labeling: 使用 DistilBERT 模型为每条评论生成情感分值 。


Complexity Scoring: 使用 textstat 计算词汇密度（Lexical Density）作为衡量“投入程度”的指标 。



Output: 生成 merged_data.csv，包含情感标签、复杂度得分、回复数等字段。

T4: Visualization (visualizer.py)
Chart 1: 情感分布饼图（正面 vs. 负面）。

Chart 2: 散点图 —— 横轴：词汇密度（Engagement），纵轴：情感得分（Acceptance）。

Chart 3: 负面评论词云图，识别用户反感 AI 内容的具体关键词（如 "uncanny", "fake"）。

5. Constraints & Methodology

Evidence Source: 使用 YouTube 评论作为主要文本证据 。



Validation: 针对极高或极低情感分的评论进行人工细读（Close Reading），以校验模型准确性 。

Efficiency: 优先保证抓取脚本的健壮性（断点续传或错误处理）。