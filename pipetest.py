import chainlit as cl
from typing import List
import transformers
import requests
import torch

API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "Can you please let us know more details about your ",
})

##model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
model_id = "McGill-NLP/Llama-3-8b-Web"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

messages = [
    {"role": "system", "content": "You are a jedi chatbot who always responds in master yoda speak!"},
    {"role": "user", "content": "Who are you?"},
]
prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    prompt,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
print(outputs[0]["generated_text"][len(prompt):])
##state = template.format(**valid[0])
##pipe = pipeline(model="McGill-NLP/Llama-3-8b-Web", device=0, torch_dtype='auto')
##out = pipe(state, return_full_text=False)[0]
##print("Action:", out['generated_text'])
##prompt1 = "the capital of usa is"
##print(pipe(prompt1))
	  


def get_prompt(instruction: str, history: List[str]) -> str:
    system = "You are an AI assistant that gives helpful answers. You answer the questions in a short and concise way."
    prompt = f"### System:\n{system}\n\n### User:\n"
    if len(history) > 0:
        prompt += f"This is the conversation history: {''.join(history)}. Now answer the question: "
    prompt += f"{instruction}\n\n### Response:\n"
    print(prompt)
    return prompt

@cl.on_message
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history")
    msg = cl.Message(content="")
    await msg.send()

    prompt = get_prompt(message.content, message_history)

    response = ""
    for word in outputs(prompt, stream=True):
        await msg.stream_token(word) 
        response += word
    await msg.update()
    message_history.append(response)

@cl.on_chat_start
def on_chat_start():
    cl.user_session.set("message_history", [])
    



history = []
"""
question = "Which city is the capital of Austria?"

answer = ""

for word in llm(get_prompt(question), stream=True):
    print(word, end="", flush=True)
    answer += word
print()

history.append(answer)

question = "And which is of the United States?"

for word in llm(get_prompt(question, history), stream=True):
    print(word, end="", flush=True)
print()
"""