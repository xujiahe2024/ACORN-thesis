Multimodal Fact-Checking Core Codes
This repository contains the core analytical scripts for the thesis project investigating the decision-making processes of LLMs Vs Human Annotators in multimodal fact-checking. The framework evaluates how images and text influence check-worthiness, logic extraction, and emotional bias.
Project Structure
The project is divided into four primary modules:
1. AI Automated Annotation (model_reply.py)
This module automates the labeling process using the DSPy framework to prompt various LLMs (e.g., GPT, Gemini, Claude) with a structured survey.
* Survey Design: Includes 16 questions (Q1–Q16) covering understandability, factuality, novelty, domain classification, and check-worthiness.
* Multimodal Input: Processes claims alongside image URLs using vision-capable models.
* Model Benchmarking: Supports batch testing across multiple providers via the OpenRouter API.
2. Logic & Keyword Extraction (key_words.py)
This script performs a qualitative analysis of the reasoning provided by models and humans.
* Codebook Generation: Uses GPT-4o to dynamically generate a "Natural Language Codebook" that summarizes 10–12 core reasoning patterns (e.g., "Could mislead the public" or "Purely personal opinion").
* Stable Extraction: Maps raw reasoning text to logic categories and extracts verbatim "Evidence Keywords".
* Comparative Visualization: Generates bar plots comparing the frequency of logic categories between humans and models across different performance groups.
3. Human Demographic Analysis (demographics.py)
This module analyzes the participant data from human annotators (recruited via Prolific).
* Data Overview: Calculates approval rates, return rates, and timeouts across five different tasks.
* Demographic Profiling: Analyzes distributions for Sex, Age, Ethnicity, Country of Birth, and Employment status.
* Quality Metrics: Tracks task completion time and participation history (Total Approvals) to assess data reliability.
4. Emotion & Bias Analysis (emtion.py)
This script examines the correlation between emotional language and check-worthiness judgments.
* Emotion Lexicon: Uses a proxy NRC-style lexicon to identify words related to Fear, Anger, Sadness, and Joy.
* Binary Mapping: Links the presence of emotion words in reasoning (Q6/Q7) to the final check-worthiness label (Q5).
* Statistical Output: Calculates the "Checkworthy Ratio" for samples with vs. without emotional language.
Requirements
1.Environment Setup
Install the required Python libraries:
pip install dspy-ai openai pandas matplotlib seaborn tqdm openpyxl

2.API Configuration
You must provide an OpenRouter API Key to run the model-based scripts.
* In model_reply.py, set: os.environ["OPENROUTER_API_KEY"].
* In key_words.py, set: OPENROUTER_API_KEY.
Data 
The scripts expect the following file inputs (localized to specific directory paths in the code):
* Gold label: Gold standard labels of HintsOfTruth dataset.
* Annotator Results: human.xlsx and model.xlsx containing responses to the 16-question survey.
* Demographics: CSV files titled demographics_task1.csv through task5.csv.
 Usage
1. Generate AI Responses: Run python model_reply.py to collect structured labels from various LLMs.
2. Analyze Logic: Run python key_words.py to build the logic codebook and compare human vs. AI reasoning.
3. Evaluate Humans: Run python demographics.py to generate a comprehensive report on participant quality and diversity.
4. Check Emotional Bias: Run python emtion.py to see if emotional triggers influence the decision to fact-check a claim.