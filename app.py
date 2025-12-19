import gradio as gr
from inference import predict_image
from segmentation import progressive_word_segmentation
from PIL import Image
import matplotlib.pyplot as plt
import io

def recognize_char(image):
    image.save("temp.png")
    preds = predict_image("temp.png")
    labels = [p[1] for p in preds]
    confs = [p[2]*100 for p in preds]

    plt.figure(figsize=(4,2))
    plt.barh(labels[::-1], confs[::-1])
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    return f"Prediction: {labels[0]}", Image.open(buf)

def recognize_word(image):
    image.save("temp_word.png")
    return progressive_word_segmentation("temp_word.png")

with gr.Blocks() as demo:
    gr.Markdown("## AksharaVision — Kannada OCR")

    with gr.Tab("Character OCR"):
        img = gr.Image(type="pil")
        out = gr.Textbox()
        chart = gr.Image()
        gr.Button("Predict").click(recognize_char, img, [out, chart])

    with gr.Tab("Word OCR"):
        wimg = gr.Image(type="pil")
        wout = gr.Textbox(lines=12)
        gr.Button("Run").click(recognize_word, wimg, wout)

demo.launch()
