from langchain.llms import HuggingFacePipeline
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
import chainlit as cl
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GPTQConfig

model_id = "gaurav021201/Meta-Llama-3-8B-GPTQ"
quantization_config_loading = GPTQConfig(bits=4)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config_loading, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512
    )

llm=HuggingFacePipeline(pipeline=pipe)

template = """
{history}
Question: {input}
"""

prompt = PromptTemplate(input_variables=["history", "input"], template=template)
memory=ConversationBufferWindowMemory(k=3)

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