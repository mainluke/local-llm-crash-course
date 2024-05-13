from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
import chainlit as cl
import torch

model_name_or_path = "nvidia/Llama3-ChatQA-1.5-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")

pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
)

llm = HuggingFacePipeline(pipeline=pipe)

template = """
System: {{
Du bist ein sehr hilfreicher KI Assistent.
Du antwortest dem Benutzer und gibst hilfreiche Antworten.
Antworte in Deutsch, und mit einer Antwort.}}
{history}
User: {input}
Assistant: 
"""

prompt = PromptTemplate(input_variables=["history", "input"], template=template)
memory = ConversationBufferWindowMemory(k=3)

@cl.on_chat_start
async def initialize_chat():
    llm_chain = ConversationChain(prompt=prompt, llm=llm, memory=memory)
    cl.user_session.set("llm_chain", llm_chain)

def build_prompt(memory, user_input):
    prompt_text = "\n".join(memory[-3:])  
    prompt_text += f"\nUser: {user_input}\nAssistant:"
    return prompt_text

@cl.on_message
async def handle_message(message: cl.Message):
    llm_chain = cl.user_session.get("llm_chain")
    cb = cl.AsyncLangchainCallbackHandler()
    cb.answer_reached = True
    res = await cl.make_async(llm_chain)(message.content, callbacks=[cb])
    response_text = res['response']
    await cl.Message(content=response_text).send()


