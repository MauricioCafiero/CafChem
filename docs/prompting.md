# Prompting tips for LLM-based Agents 
(based on "Prompt Engineering for LLMs" by Berryman & Ziegler )

## Table of Contents
1. [General Agent tips](#general-agent-tips)
2. [Prompt content](#prompt-content)
3. [Prompt Structure](#prompt-structure)
4. [Agent workflow](#agent-workflow)
5. [Formatting the output of the LLM](#formatting-the-output-of-the-llm)

## General Agent tips
- Use the LLM as little as possible. Augment with code when appropriate.
- This makes the Agent less general, but improves the quality of the results.
- Agent structure: identify goal --> set tasks --> perform tasks --> collect context --> finish.
- Make a test harness. You can get an LLM to evaluate the results of the agent tests. 


## Prompt content
- The LLM can only read from start to finish; it cannot go back and reread something. Thus, information has to flow forward. Have your text refer to information to come rather than refer to information earlier in the prompt.
- Ask for positives rather than negatives.
- Give a reason for a request.
- Avoid absolutes.
- Tool results can be in unstructured string format, markdown, or JSON.
- If you have too much context from web searches, RAG, or tool calls, you can summarize using an LLM call, or even do heirachical summarizing if needed.
- If the LLM is not arriving at the correct answer, try to coax chain-of-thought by use of phrases such as "let's approach this problme step by step."
- Tool names should be meaningful and descriptive. Camelcase works well.
- Limit the number of tools.


## Prompt Structure
- One very good way to arrange your prompt is in markdown format. Using headers (#, ##, ###, etc) and bullet points to structure information.
ChatML is also a good choice. Build a conversation using:
```
<|im_start|>system
*some text describing the work needed, the tools available, and the output format desired* <|im_end|>
<|im_start|>user
*some text decribing the user request and perhaps the tool context* <|im_end|>
<|im_start|>assistant
*some text* <|im_end|>
...
```
- LLMs are very good at continuing patterns, so formatting your prompt in a specific pattern and then starting the LLM's response pattern will likely lead to the LLM continuing the pattern.
- Few shot examples are a good way to giving the LLM a structure to follow and will help it understand your request.
  * In some cases few-shot examples can lead to weird results if the model picks up the wrong pattern.
  * Few-shotting can bias your LLMs answers.
- Information in the middle of a prompt is often lost/ignored by the LLM.
- The end of the prompt should restate the question first posed at the begining (*refocusing*). A sandwich structure of: question --> context --> question works well.
- A report format (in markdown) is an excellent format. It can sections such as:
```
# Table of contents
1. [Introduction](#introduction)
2. [Task](#task)
3. [Tools](#tools)
4. [Response](#response)

## Introduction
system messages to the LLM to describe in general what you want from it

## Task 
The specific task

## Tools

### Tool 1
description of tool 1

### Tool 2
description of tool 2

## Response
how you want the LLM to format your response.
```
You can also include an examples section, *etc*.

## Agent workflow
- The agent app's job is to translate the user's request into an LLM request, then into tool calls, and back. 
- The agent app should prioritize context collection.
- The agent app should maintain state needed to solve the uder's request.
- Chat (instruct) models *or* completion models can be used; the prompt should be adjusted as needed. For example, a ChatML structue works well for completions. 

## Formatting the output of the LLM
- Because of how they are trained, LLMs often produce responses with un-needed preamble and extra detail at the end. The answer toy need is often in the middle.
- Requesting a specific format of answer will help with parsing.
- The probabilities of the tokens in the response (or logprobs) can help you determine certainty and uncertainty in the answer.

