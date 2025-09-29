# Halluciguard: LLM Agents to Fact Check and Guard Against Hallucinations of Other LLMs

A set of engineered prompts to use LLMs to fact check and detect hallucinations in other LLMs.

Agent Prompts folder -- contains usable prompts for checker agents (can be ported to any agentic framework)

Example checker prompts:
 * fact_checker.yaml: An agent that checks facts and claims made by other LLMs and evaluates their truthfulness. Built to detect hallucinations by other agents.
 * general_checker_agent.yaml: An agent that evaluates the work done by an arbitrary agent, given its prompt (including inputs) and produced outputs. Assigns a grade 'A'-'F' to work done and explains the rationale for the    grade.
 * logical_inference.yaml / logical_evaluator.yaml: LLM-based logic inference engine to check whether LLMs claims follow logical rules.
 * hallucinator.yaml: An agent that tells the truth sometimes and hallucinates with certain probability. Used for testing the other agents.
 
Demo: python run.py 
  * Uses an agent to make some true and some false claims
  * Run the checker and report an accuracy
