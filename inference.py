from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "shenzhi-wang/Llama3-8B-Chinese-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto"
)

messages = [
    {"role":"user","content":"你好，我是一个聊天机器人。"},
]

input_ids = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to(model.device)

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.9,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]
print(outputs.shape)
print(tokenizer.decode(response, skip_special_tokens=True))