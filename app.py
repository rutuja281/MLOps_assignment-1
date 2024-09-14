import gradio as gr
from huggingface_hub import InferenceClient
import torch
from transformers import pipeline

# Inference client setup
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")
pipe = pipeline("text-generation", "microsoft/Phi-3-mini-4k-instruct", torch_dtype=torch.bfloat16, device_map="auto")

# Global flag to handle cancellation
stop_inference = False

def respond(
    message,
    history: list[tuple[str, str]],
    system_message="You are a friendly Chatbot.",
    max_tokens=512,
    temperature=0.7,
    top_p=0.95,
    use_local_model=False,
):
    global stop_inference
    stop_inference = False  # Reset cancellation flag

    if history is None:
        history = []

    if use_local_model:
        messages = [{"role": "system", "content": system_message}]
        for val in history:
            if val[0]:
                messages.append({"role": "user", "content": val[0]})
            if val[1]:
                messages.append({"role": "assistant", "content": val[1]})
        messages.append({"role": "user", "content": message})

        response = ""
        for output in pipe(
            messages,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=top_p,
        ):
            if stop_inference:
                response = "Inference cancelled."
                yield history + [(message, response)]
                return
            token = output['generated_text'][-1]['content']
            response += token
            yield history + [(message, response)]  # Yield history + new response

    else:
        messages = [{"role": "system", "content": system_message}]
        for val in history:
            if val[0]:
                messages.append({"role": "user", "content": val[0]})
            if val[1]:
                messages.append({"role": "assistant", "content": val[1]})
        messages.append({"role": "user", "content": message})

        response = ""
        for message_chunk in client.chat_completion(
            messages,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p=top_p,
        ):
            if stop_inference:
                response = "Inference cancelled."
                yield history + [(message, response)]
                return
            if stop_inference:
                response = "Inference cancelled."
                break
            token = message_chunk.choices[0].delta.content
            response += token
            yield history + [(message, response)]  # Yield history + new response


def cancel_inference():
    global stop_inference
    stop_inference = True

# Enhanced custom CSS for a modern look
custom_css = """
body {
    background-color: #f0f0f5;
    font-family: 'Arial', sans-serif;
}

.gradio-container {
    max-width: 900px;
    margin: 0 auto;
    padding: 20px;
    background: white;
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
    border-radius: 15px;
}

#title {
    text-align: center;
    font-size: 2.5em;
    color: #4CAF50;
    font-weight: bold;
    margin-bottom: 15px;
}

.gr-button {
    background-color: #4CAF50;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 30px;
    font-size: 16px;
    cursor: pointer;
    transition: background-color 0.3s ease;
}

.gr-button:hover {
    background-color: #388E3C;
}

.gr-slider input[type=range] {
    -webkit-appearance: none;
    appearance: none;
    width: 100%;
    height: 8px;
    background: #4CAF50;
    border-radius: 5px;
    outline: none;
    opacity: 0.9;
    transition: opacity .15s ease-in-out;
}

.gr-slider input[type=range]:hover {
    opacity: 1;
}

.gr-textbox input {
    border: 2px solid #4CAF50;
    border-radius: 8px;
    padding: 10px;
    font-size: 16px;
}

.gr-textbox input:focus {
    border-color: #388E3C;
    outline: none;
}

.gr-chatbox {
    background-color: #e8f5e9;
    border-radius: 10px;
    padding: 15px;
    max-height: 300px;
    overflow-y: auto;
    font-size: 16px;
}

.gr-chatbox .message {
    margin: 10px 0;
}

.gr-chatbox .user {
    font-weight: bold;
    color: #00796B;
}

.gr-chatbox .bot {
    font-weight: bold;
    color: #388E3C;
}
"""

# Define the interface with a custom layout
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("<h1 id='title'>🌟 Ask me anything! 🌟</h1>")
    
    with gr.Row():
        system_message = gr.Textbox(value="You are a friendly Chatbot.", label="System message", interactive=True)
        use_local_model = gr.Checkbox(label="Use Local Model", value=False)

    with gr.Row():
        max_tokens = gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens")
        temperature = gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature")
        top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)")

    # Chat history box with styled messages
    chat_history = gr.Chatbot(label="Chat", elem_classes=["gr-chatbox"])

    user_input = gr.Textbox(show_label=False, placeholder="Type your message here...", elem_id="user_input")

    with gr.Row():
        submit_button = gr.Button("Send", elem_classes=["gr-button"])
        cancel_button = gr.Button("Cancel Inference", elem_classes=["gr-button"], variant="danger")

    # Connect the buttons to the appropriate functions
    submit_button.click(respond, [user_input, chat_history, system_message, max_tokens, temperature, top_p, use_local_model], chat_history)
    cancel_button.click(cancel_inference)

    user_input.submit(respond, [user_input, chat_history, system_message, max_tokens, temperature, top_p, use_local_model], chat_history)

if __name__ == "__main__":
    demo.launch(share=False)
