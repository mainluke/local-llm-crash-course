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

llm=HuggingFacePipeline(pipeline=pipe)

template = """
System: {{
You are a helpful assistant that provides information and engages in casual conversation.
Respond naturally to user queries and provide useful information.
Please, write a single reply only!}}
{history}
User: {input}
Assistant: 
"""

@cl.on_chat_start
async def start():
    llm_chain = ConversationChain(prompt=prompt, llm=llm, memory=memory)
    cl.user_session.set("llm_chain", llm_chain)


@cl.on_message
async def main(message:cl.message):
    llm_chain = cl.user_session.get("llm_chain")
    cb = cl.AsyncLangchainCallbackHandler( )
    cb.answer_reached = True
    res = await cl.make_async(llm_chain)(message.content, callbacks=[cb])
    await cl.Message(content=res['response']).send()

prompt = PromptTemplate(input_variables=["history", "input"], template=template)
memory=ConversationBufferWindowMemory(k=3)