import json
import random
from arklex.evaluation.get_documents import load_docs
from arklex.evaluation.chatgpt_utils import (chatgpt_chatbot, query_chatbot, filter_convo, adjust_goal,
                                               flip_hist, generate_goals, format_chat_history_str, flip_hist_content_only)


def check_goal_completion(goal, convo):
    convo_str = format_chat_history_str(flip_hist_content_only(convo[2:]))
    prompt = f"Here is a conversation between a user and a customer service chatbot assistant:\n{convo_str}\n\nThe user's goal is the following: {goal}\nOutput False if the user needs to learn more information regarding their goal. Output True otherwise. Only onput True or False and nothing else."
    output = chatgpt_chatbot([{'role': 'user', 'content': prompt}])
    return output == "True"

def conversation(model_api, goal, summary, model_params, synthetic_data_params, env_config):
    history = []
    instructional_prompt = f'Replicate the writing behavior of a human customer. You are interacting with a customer service chatbot for the following company: {summary}\nYou have the following goal when interacting with this chatbot:\n{goal}\n Have a conversation with the chatbot while trying to achieve this goal. Make sure the conversation is natural. For example, if the chatbot asks you a question you should answer it.'
    start_text = "Humans write short questions with typos and a neutral sentiment. Here are some examples of what a human customer would type: [how much is it?, Can you send info to my email, yes I need a job, want to check both proposals to rent and buy, How much does it cost a [PRODUCT_HERE], Im interested in [PRODUCT_HERE], hi i would like to rent out [PRODUCT_HERE] but im wondering which countries are available for rental]. Replicate the writing behavior of a human customer and begin the conversation with a question to achieve your goal."
    history.append({'role': 'system','content': instructional_prompt})
    history.append({'role': 'user', 'content': start_text})
    chatbot_history = []
    
    # First turn
    response = chatgpt_chatbot(history)
    history.append({'role': 'assistant', 'content': response})
    chatbot_history.append({'role': 'user', 'content': response})
    response = query_chatbot(model_api, chatbot_history, model_params, env_config)
    history.append({'role': 'user', 'content': response['answer']})
    chatbot_history.append({'role': 'assistant', 'content': response['answer']})

    # Subsequent turns
    for i in range(synthetic_data_params['max_turns']):
        response = chatgpt_chatbot(history)
        history.append({'role': 'assistant', 'content': response})
        chatbot_history.append({'role': 'user', 'content': response})
        response = query_chatbot(model_api, chatbot_history, model_params, env_config)
        history.append({'role': 'user', 'content': response['answer']})
        chatbot_history.append({'role': 'assistant', 'content': response['answer']})
        if check_goal_completion(goal, history):
            break
    return history

def simulate_conversations(model_api, model_params, synthetic_data_params, config):
    documents = load_docs(config['documents_dir'], config, synthetic_data_params['num_goals'] * 2)
    summary = config['intro']
    env_config = {
        "workers": config['workers'],
        "tools": config.get("tools", [])
    }
    
    final_goals = []
    if synthetic_data_params.get('goals', None):
        raw_goals = []
        cases = synthetic_data_params['goals']
        for stage, categories in cases.items():
            for first_level, second_levels in categories.items():
                for second_level, goals in second_levels.items():
                    raw_goal = goals[0]
                    raw_goals.append(raw_goal)
        
        # goal adaptation
        final_goals = []
        for goal in raw_goals:
            doc = random.choice(documents) if documents else {"content": summary}
            new_goal = adjust_goal(doc["content"], goal)
            final_goals.append(new_goal)
    else:
        # If we have task_docs with test cases, use those instead of generating goals
        if config.get('task_docs') and isinstance(config['task_docs'], list) and len(config['task_docs']) > 0:
            try:
                with open(config['task_docs'][0], 'r') as f:
                    test_cases = json.load(f)
                final_goals = []
                for test_case in test_cases[:synthetic_data_params['num_goals']]:
                    final_goals.append(test_case['description'])
            except Exception as e:
                print(f"Error loading test cases: {e}")
                final_goals = generate_goals(documents if documents else [{"content": summary}], synthetic_data_params)
        else:
            final_goals = generate_goals(documents if documents else [{"content": summary}], synthetic_data_params)
    
    try:
        conversations = []
        for goal in final_goals:
            convo = conversation(
                model_api,
                goal,
                summary,
                model_params,
                synthetic_data_params,
                env_config,
            )
            conversations.append(convo)
    except Exception as e:
        print("Generate conversations failed")
        print("Error: ", e)
        conversations = []
    return conversations, final_goals

if __name__ == "__main__":
    model_api = "http://adaptation.cs.columbia.edu:55231/qa/richtech/v1alpha1"
    synthetic_data_params = {'num_convos': 5, 'num_goals': 3, 'max_turns': 10}
    model_params = {}
    convos  = simulate_conversations(model_api, model_params, synthetic_data_params)
    with open('p1_sample_convos.json', 'w') as f:
        json.dump(convos, f, indent=5)