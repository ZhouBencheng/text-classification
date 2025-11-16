from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch
import torch.nn.functional as F
import gradio as gr

model_dir = "bentom/my-bert-classification"

config = AutoConfig.from_pretrained(model_dir, num_labels=3, finetuning_task="text-classification")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir, config=config)

def inference(input_text):
    inputs = tokenizer.batch_encode_plus(
            [input_text],
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
    
    with torch.no_grad():
        logits = model(**inputs).logits
        
    predicted_class_id = logits.argmax().item()
    output = model.config.id2label[predicted_class_id]
    return output

with gr.Blocks(css="""
               .message,svelte-w6rprc,svelte-w6rprc,svelte-w6rprc (font-size:20px; margin-top: 20px}
               #component-21 >div.wrap.svelte-w6rprc (height: 600px;}
               """) as demo:
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(placeholder="Insert your prompt here:", scale=2, container=False)
            answer = gr.Textbox(lines=0, label="Answer")
            generate_bt = gr.Button("Generate", scale=1)
    inputs = [input_text]
    outputs = [answer]
    generate_bt.click(
        fn=inference, inputs=inputs, outputs=outputs, show_progress=True
    )
    examples = [
        ["My last two weather pics from the storm on August 2nd. People packed up realfastafter the temp dropped and winds picked up."],
        ["Lying clinton sinking! Donald Trump singing: Let's Make America Great Again!"],
    ]


demo.queue()
demo.launch()
