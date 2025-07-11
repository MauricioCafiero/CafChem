import os
from google.colab import userdata
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import json
from huggingface_hub import hf_hub_download
from IPython.display import display, Markdown
import re
from datasets import load_dataset,Dataset
import torch
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from peft import prepare_model_for_kbit_training, get_peft_model
from rdkit.Chem import AllChem, Draw, QED
from rdkit import Chem

def setup_txgemma(which_model: int):
  '''
    Sets up the txgemma model.
    Args:
      which_model: 0 = 2b-predict, 1 = 9b-chat, 2 = 9b-predict, 3 = 27b-chat, 4 = 27b-predict
    Returns:
      model: the chosen TxGemma model
      tokenizer: the tokenizer for the chosen TxGemma model
      pipe: the pipeline for the chosen TxGemma model
      tdc_prompts_json: the prompts for the chosen TxGemma model
  '''  
  MODEL_VARIANTS = ["2b-predict", "9b-chat", "9b-predict", "27b-chat", "27b-predict"]
  MODEL_VARIANT = MODEL_VARIANTS[which_model]

  model_id = f"google/txgemma-{MODEL_VARIANT}"

  if MODEL_VARIANT == "2b-predict":
      additional_args = {}
  else:
      additional_args = {
          "quantization_config": BitsAndBytesConfig(load_in_8bit=False)
      }

  tokenizer = AutoTokenizer.from_pretrained(model_id)
  model = AutoModelForCausalLM.from_pretrained(
      model_id,
      device_map="auto",
      **additional_args,
  )

  pipe = pipeline(
      "text-generation",
      model=model,
      tokenizer=tokenizer,
  )

  tdc_prompts_filepath = hf_hub_download(
      repo_id=model_id,
      filename="tdc_prompts.json",
  )
  global tdc_prompts_json
  with open(tdc_prompts_filepath, "r") as f:
      tdc_prompts_json = json.load(f)
  
  return model, tokenizer, pipe


def get_some_tdc_tasks():
  '''
    Gets some TDC tasks.
    Returns:
      None: prints a list of TDC tasks
  '''
  tasks = ['LD50_Zhu', 'logP_Morgan','BindingDB_kd','BindingDB_ic50','BindingDB_ki',
           'Lipophilicity_AstraZeneca','Solubility_AqSolDB','Bioavailability_Ma',
           'BBB_Martins','Skin_Reaction','Carcinogens_Lagunin','SARSCoV2_Vitro_Touret',
           'SARSCOV2_3CLPro_Diamond','HIV','ClinTox']
  for task in tasks:
    print(task)

def view_task_prompt(task_name: str):
  '''
    Gets the task prompt
    Args:
      task_name: the name of the task
      tdc_prompts_json: the prompts for the chosen TxGemma model
    Returns:
      None: prints the task parameters
  '''
  global tdc_prompts_json
  print(tdc_prompts_json[task_name])
  print("============================================================")
  print("Parameters:")

  search_for = '\{.*\}'
  parameters = re.findall(search_for, tdc_prompts_json[task_name])
  for parameter in parameters:
    parameter = parameter[1:-1]
    print(parameter)

  return parameters

def make_prompt(task_name: str, parameters: list, values: list):
  '''
    Makes a prompt for the chosen task.
    Args:
      task_name: the name of the task
      parameters: the parameters for the task
      values: the values for the parameters
      tdc_prompts_json: the prompts for the chosen TxGemma model
    Returns:
      prompt: the prompt for the chosen task
  '''
  global tdc_prompts_json
  prompt = tdc_prompts_json[task_name]
  for param, val in zip(parameters,values):
    prompt = prompt.replace(param,val)
  return prompt

def generate_text(prompt: str, model, tokenizer):
  '''
    Generates text from the model using the prompt
    Args:
      prompt: the prompt for the model
      model: the model to use
      tokenizer: the tokenizer for the model
    Returns:
      just_answer: the answer to the prompt
      response: the entire response from the model
  '''
  input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
  outputs = model.generate(**input_ids, max_new_tokens=8)
  response = tokenizer.decode(outputs[0], skip_special_tokens=True)

  just_answer = response[len(prompt):]
  return just_answer, response

def generate_chat(prompt: str, model, tokenizer):
  '''
    Generates a chat from the model using the prompt
    Args:
      prompt: the prompt for the model
      model: the model to use
      tokenizer: the tokenizer for the model
    Returns:
      None: prints the chat
  '''
  questions = [
    prompt,
    "Explain your reasoning based on the molecule structure."
  ]

  messages = []

  display(Markdown("\n\n---\n\n"))
  for question in questions:
      display(Markdown(f"**User:**\n\n{question}\n\n---\n\n"))
      messages.append(
          { "role": "user", "content": question },
      )
      # Apply the tokenizer's built-in chat template
      inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
      outputs = model.generate(input_ids=inputs.to("cuda"), max_new_tokens=512)
      response = tokenizer.decode(outputs[0, len(inputs[0]):], skip_special_tokens=True)
      display(Markdown(f"**TxGemma:**\n\n{response}\n\n---\n\n"))
      messages.append(
          { "role": "assistant", "content": response},
      )

class prepare_dataset():
  '''
    reads a CSV file and prepares it for fine-tuning
  '''
  def __init__(self, filename: str, column_names = ['text','labels'], valid_percent = 0.15, push = False, setname = None):
    '''
    Reads information for data preparation.
    Args:
      filename: the name of the CSV file to read
      column_names: the names of the columns in the CSV file
      valid_percent: the percentage of the dataset to use for validation
      push: whether to push the dataset to the HuggingFace Hub
      setname: the name of the dataset in the HuggingFace Hub
    Returns:
      None
    '''
    self.filename = filename
    self.column_names = column_names
    self.valid_percent = valid_percent
    self.push = push
    self.setname = setname

    print("prepare dataset class initiated!")

  def load_dataset(self):
    '''
      Loads a dataset from a CSV and returns a HF dataset.
      Args:
        None
      Returns:
        user_data: the dataset
    '''
    user_data = load_dataset('csv', data_files=self.filename, column_names=self.column_names)

    print("Dataset loaded!")
    return user_data
    
  def define_prompt_template(self, prompt_text: str):
    '''
    The prompt text should contain two parameters: DRUG_SMILES and MOLECULAR_PROPERTIES.
    DRUG_SMILES will be populated from the uploaded dataset, and the 
    MOLECULAR_PROPERTIES will be populated from RDKit. 

    The answer choices, which should also be contained in the prompt text, should 
    correspond to the classes in the dataset.

      Args:
        prompt_text: the prompt text
      Returns:
        prompt_template: the prompt template
    '''
    prompt_template = {"input_text": prompt_text, "output_text": "ANSWER_TEXT"}
    #print(prompt_template)
    return prompt_template

  def fill_prompt_template(self, prompt_template, cat_hash: list):
    '''
      Fills the prompt template with the dataset.
      Args:
        prompt_template: the prompt template to fill
        cat_hash: a list of the answer choices
      Returns:
        training_prompts: the dataset with the prompt template filled
    '''
    dataset = self.load_dataset()
    
    prompts = []
    for smile, class_num in zip(dataset["train"]["text"], dataset["train"]["labels"]):
      molob = Chem.MolFromSmiles(smile)
      qed_descriptor = Chem.QED.default(molob)
      p = Chem.QED.properties(molob)
      prop_string = f"Molecular weight {p[0]:.2f}, partition coefficient: {p[1]:.2f}, Hydrgen-bond acceptors: {p[2]}, \n\
Hydrgen-bond donors: {p[3]}, Polariable Surface Area: {p[4]:.2f}, Rotatable bonds: {p[5]},  Aromatic rings: {p[6]}"
      prompt = prompt_template.copy()
      prompt["input_text"] = prompt["input_text"].replace("DRUG_SMILES",smile)
      prompt["output_text"] = prompt["output_text"].replace("ANSWER_TEXT",cat_hash[class_num])
      prompt["input_text"] = prompt["input_text"].replace("MOLECULAR_PROPERTIES",prop_string)
      prompts.append(prompt)

    training_prompts = Dataset.from_list(prompts)
    
    training_prompts = training_prompts.train_test_split(test_size=self.valid_percent)

    if self.push == True and self.setname != None:
      training_prompts.push_to_hub(setname)
      print("Dataset pushed to HuggingFace Hub")

    return training_prompts

class train_TxF2BPredict():
  '''
    Trains the TxGemma 2B predict model.
  '''
  def __init__(self, batch_size = 16, epochs = 10, trained_model_name = "ft_model", push = False):
    '''
      Reads in the training parameters for the TxGemma 2B predict model.
      Args:
        batch_size: the batch size for the training
        epochs: the number of epochs for the training
        trained_model_name: the name of the trained model
        push: whether to push the trained model to the HuggingFace Hub
      Returns:
        None
    '''
    self.batch_size = batch_size
    self.epochs = epochs
    self.trained_model_name = trained_model_name
    self.push = push
    self.model_id = "google/txgemma-2b-predict"
    print("train_TxF2BPredict class initiated!")

  def formatting_func(self,example):
    '''
      Formats the training examples for the TxGemma 2B predict model.
      Args:
        example: the training example
      Returns:
        text: the formatted training example
    '''
    text = f"{example['input_text']} {example['output_text']}<eos>"
    return text

  def setup_model(self):
    '''
      Sets up the TxGemma 2B predict model for finetuning.
      Args:
        None
      Returns:
        model: the quantized TxGemma model
        tokenizer: the tokenizer for the chosen TxGemma model
    '''
    

    # Use 4-bit quantization to reduce memory usage
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(self.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        self.model_id,
        quantization_config=quantization_config,
        device_map={"":0},
        torch_dtype="auto",
        attn_implementation="eager")
    
    return model, tokenizer
  
  def set_up_peft(self, model):
    '''
      Sets up the PeftModel for finetuning.
      Args:
        model: the quantized TxGemma model
      Returns:
        model: the PeftModel for finetuning
        lora_config: the LoRA configuration for finetuning
    '''
    lora_config = LoraConfig(
          r=16, #lora_alpha=32, 
          task_type="CAUSAL_LM", target_modules=[
              "q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj",
              "down_proj"])
    model = prepare_model_for_kbit_training(model)

    # Create PeftModel from quantized model and configuration
    model = get_peft_model(model, lora_config)
  
    return model, lora_config
  
  def train_model(self, model, tokenizer, lora_config, training_prompts):
    '''
      Trains the TxGemma 2B predict model.
      Args:
        model: the quantized TxGemma model
        tokenizer: the tokenizer for the chosen TxGemma model
        lora_config: the LoRA configuration for finetuning
        training_prompts: the dataset with the prompt template filled
      Returns:
        model
    '''
    trainer = SFTTrainer(
    model=model,
    train_dataset=training_prompts["train"],
    eval_dataset = training_prompts["test"],
    args=SFTConfig(
        f"{self.model_id}-{self.trained_model_name}",
        push_to_hub=self.push,
        per_device_train_batch_size=self.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=5,
        max_seq_length=512,
        optim="paged_adamw_8bit",
        report_to="none",
        #new
        num_train_epochs=self.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        gradient_checkpointing=True,
        disable_tqdm=False,
        load_best_model_at_end=True),
    peft_config=lora_config,
    formatting_func=self.formatting_func)

    trainer.train() 

    return model