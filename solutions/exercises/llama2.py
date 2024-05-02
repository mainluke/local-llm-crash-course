from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained(
    "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF", model_file="Meta-Llama-3-8B-Instruct.Q2_K.gguf"
)


def get_prompt(instruction: str) -> str:
    system = "You are an AI assistant that gives helpful answers. You answer the question in a short and concise way."
    prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{instruction} [/INST]"
    print(f"Prompt created: {prompt}")
    return prompt


question = "Which city is the capital of India?"
prompt = get_prompt(question)
for word in llm(prompt, stream=True):
    print(word, end="", flush=True)
print()
