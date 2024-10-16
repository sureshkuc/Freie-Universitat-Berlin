<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BERT Fine-Tuning for Question Answering on SQuAD</title>
</head>
<body>

<h1>BERT Fine-Tuning for Question Answering on SQuAD</h1>

<p>This repository contains code for fine-tuning BERT models for the task of question answering using the SQuAD dataset. The fine-tuning is performed using various techniques such as LoRA, LoHA, IA3, Prefix-Tuning, and Prompt Encoding.</p>

<h2>Project Overview</h2>

<p>The project provides six Python scripts to fine-tune BERT for question-answering using different methods:</p>

<ol>
    <li><strong>bert_fine_tuning.py</strong>: Standard fine-tuning of BERT.</li>
    <li><strong>bert_fine_tuning_IA3.py</strong>: Fine-tuning using IA3 (Insertion Attention and Adaptation).</li>
    <li><strong>bert_fine_tuning_LoRA.py</strong>: Fine-tuning using LoRA (Low-Rank Adaptation).</li>
    <li><strong>bert_fine_tuning_LoHA.py</strong>: Fine-tuning using LoHA (Low-rank Hardware-Aware Adaptation).</li>
    <li><strong>bert_fine_tuning_prefix-tuning.py</strong>: Prefix-tuning for BERT.</li>
    <li><strong>bert_fine_tuning_prompt_encoder.py</strong>: Fine-tuning using a prompt encoder.</li>
</ol>

<p>Each of these techniques offers different strategies for adapting the pre-trained BERT model to perform well on question-answering tasks, particularly on the SQuAD dataset.</p>

<h2>Table of Contents</h2>
<ul>
    <li><a href="#project-overview">Project Overview</a></li>
    <li><a href="#setup">Setup</a></li>
    <li><a href="#scripts-and-fine-tuning-techniques">Scripts and Fine-Tuning Techniques</a></li>
    <li><a href="#training-process">Training Process</a></li>
    <li><a href="#evaluation">Evaluation</a></li>
    <li><a href="#logging-and-monitoring">Logging and Monitoring</a></li>
    <li><a href="#results">Results</a></li>
</ul>

<h2 id="setup">Setup</h2>

<h3>Prerequisites</h3>
<p>To run the scripts, you'll need the following packages installed:</p>

<ul>
    <li><code>torch</code></li>
    <li><code>transformers</code></li>
    <li><code>datasets</code></li>
    <li><code>mlflow</code></li>
    <li><code>evaluate</code></li>
</ul>

<p>You can install them using pip:</p>

<pre><code>pip install torch transformers datasets mlflow evaluate</code></pre>

<h3>Clone the Repository</h3>
<p>Clone the repository to your local machine:</p>

<pre><code>git clone https://github.com/your-username/bert-fine-tuning-qa.git
cd bert-fine-tuning-qa
</code></pre>

<h3>Dataset</h3>
<p>Ensure you have the SQuAD dataset saved in your local directory. The scripts assume the dataset has been loaded from a previously saved directory using <code>load_from_disk</code>.</p>

<p>For example, you can save the SQuAD dataset using:</p>

<pre><code>from datasets import load_dataset
squad_dataset = load_dataset("squad")
squad_dataset.save_to_disk("./squad_data")
</code></pre>

<h2 id="scripts-and-fine-tuning-techniques">Scripts and Fine-Tuning Techniques</h2>

<h3>1. <code>bert_fine_tuning.py</code></h3>
<p>This script fine-tunes BERT on the SQuAD dataset using the standard method provided by Hugging Face's <code>Trainer</code> API. It logs training metrics and saves the model using MLflow.</p>

<h3>2. <code>bert_fine_tuning_IA3.py</code></h3>
<p>Implements IA3 (Insertion Attention and Adaptation) fine-tuning, which introduces additional attention layers to improve model adaptability with minimal additional parameters.</p>

<h3>3. <code>bert_fine_tuning_LoRA.py</code></h3>
<p>Uses LoRA (Low-Rank Adaptation) to fine-tune BERT. This approach adjusts the model with fewer parameters, resulting in more efficient fine-tuning while maintaining performance.</p>

<h3>4. <code>bert_fine_tuning_LoHA.py</code></h3>
<p>LoHA (Low-rank Hardware-Aware Adaptation) is another fine-tuning method that reduces the number of trainable parameters, optimized for hardware efficiency without sacrificing accuracy.</p>

<h3>5. <code>bert_fine_tuning_prefix-tuning.py</code></h3>
<p>Prefix-tuning adapts BERT by tuning a small number of continuous prompts (prefixes) for each task while keeping the rest of the model frozen. This method is useful for memory-efficient fine-tuning.</p>

<h3>6. <code>bert_fine_tuning_prompt_encoder.py</code></h3>
<p>This script implements fine-tuning via a prompt encoder. It leverages task-specific prompts that are learned during training to adapt the model without requiring changes to the underlying BERT architecture.</p>

<h2 id="training-process">Training Process</h2>

<ol>
    <li><strong>Set Model Checkpoints</strong>: In each script, you can set the model checkpoint (e.g., <code>bert-base-uncased</code>) to use different variants of the BERT model.</li>
    <li><strong>Tokenization and Dataset Preprocessing</strong>: The scripts handle tokenization and preprocessing of the SQuAD dataset. You can adjust parameters like <code>max_length</code> and <code>doc_stride</code> as needed.</li>
    <li><strong>Training</strong>: The models are trained using the Hugging Face <code>Trainer</code> API with the specified learning rates, batch sizes, and number of epochs. Training arguments are customizable within the script.</li>
    <li><strong>Fine-Tuning Specifics</strong>: Depending on the method (IA3, LoRA, etc.), different modules within the BERT architecture are adapted to the task.</li>
</ol>

<h2 id="evaluation">Evaluation</h2>

<p>Each script evaluates the fine-tuned model on the validation set of the SQuAD dataset. The evaluation process includes:</p>

<ul>
    <li>Making predictions using the validation dataset.</li>
    <li>Post-processing of the predictions to match the SQuAD format.</li>
    <li>Computing metrics such as Exact Match (EM) and F1 score using Hugging Face's <code>evaluate</code> package.</li>
</ul>

<h2 id="logging-and-monitoring">Logging and Monitoring</h2>

<p>MLflow is used to log all training parameters, metrics, and models. During training:</p>

<ul>
    <li>Parameters like <code>model_checkpoint</code>, <code>batch_size</code>, <code>max_length</code>, and <code>num_train_epochs</code> are logged.</li>
    <li>Training metrics such as <code>training_time</code> and evaluation results are also logged.</li>
    <li>The fine-tuned model is saved and versioned with MLflow.</li>
</ul>

<h2 id="results">Results</h2>

<p>After training, you can view the results in the MLflow UI, which will include:</p>

<ul>
    <li>Final metrics (EM, F1 score) for the validation set.</li>
    <li>The fine-tuned model artifact for further inference or deployment.</li>
</ul>

<h2>Example Usage</h2>

<p>To run the standard fine-tuning script, for example:</p>

<pre><code>python bert_fine_tuning.py
</code></pre>

<p>Replace with any other script to test specific fine-tuning methods:</p>

<pre><code>python bert_fine_tuning_LoRA.py
</code></pre>

<h2>License</h2>

<p>This project is licensed under the MIT License - see the <a href="LICENSE">LICENSE</a> file for details.</p>

</body>
</html>
