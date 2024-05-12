from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

model_id = "shenzhi-wang/Llama3-8B-Chinese-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto"
)


@app.route('/')
def home():
    return "Post Chat Json to /completion"


@app.route('/completion', methods=['POST'])
def completion():
    content = request.json
    if 'context' in content.keys():
        temperature = content['temperature'] if 'temperature' in content.keys() else 0.6
        top_p = content['top_p'] if 'top_p' in content.keys() else 0.9
        top_k = content['top_k'] if 'top_k' in content.keys() else 50
        max_new_tokens = content['max_new_tokens'] if 'max_new_tokens' in content.keys() else 256
        
        messages = content['context']
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        response = outputs[0][input_ids.shape[-1]:]
        return jsonify({"response": tokenizer.decode(response, skip_special_tokens=True)})
    else:
        return "Need Context", 400


if __name__ == '__main__':
    app.run(debug=True)