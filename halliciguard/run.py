import json

from LLMAgents import LLMAgent, LLMConfig
import os

# setup agents
dir = os.path.dirname(os.path.abspath(__file__))
hallucinator = LLMAgent(LLMConfig(), dir + os.sep + 'agent_prompts' + os.sep + 'hallucinator.yaml')
fact_checker = LLMAgent(LLMConfig(), dir + os.sep + 'agent_prompts' + os.sep + 'fact_checker.yaml')

# Step 1: Hallucination
hallucinated_response = hallucinator.ask({
    'num_facts': 10,
    'hallucination_prob': 0.8,
}).strip()
json_resp = json.loads(hallucinated_response)
facts = json_resp['facts']
truth = [not bool(b) for b in json_resp['hallucinations']]
fact_sources = json_resp['sources']
print('facts:', facts, '\ntruth:', truth, '\nsources:', fact_sources)

# Step 2: Fact Checking
fact_checker_response = fact_checker.ask({
    'statements_list': facts,
})
json_resp2 = json.loads(fact_checker_response)
print('fact checker response:', json_resp2)

# Step 3: Accuracy Calculation
acc_count = 0
for i in range(len(json_resp2['results'])):
    print(f"Statement: {json_resp2['results'][i]['statement']}")
    print(f"  Predicted label: {json_resp2['results'][i]['label']}, Truth: {truth[i]}")
    print(f" fact source: {fact_sources[i]}, checker source: {json_resp2['results'][i]['source']}")
    print(f"reason for prediction: {json_resp2['results'][i]['reason']}\n")
    if json_resp2['results'][i]['label'] == str(truth[i]):
        acc_count += 1
acc = acc_count / len(json_resp2['results']) * 100
print('accuracy:', acc)