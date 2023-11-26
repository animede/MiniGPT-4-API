import numpy as np
from PIL import Image
import re
import torch
import html
import gradio as gr
#from client_api  import reset, upload, ask, generate, visualize
from class_client_api import ImageAPI

key=""
url = "http://0.0.0.0:8001"
api_client = ImageAPI(url) #ImageAPIのインスタンス変数として定義

def gradio_reset():
     global key
     responce, message , key = api_client.reset()
     chatbot=[]
     image="white.png"
     text_input=""
     print("key=",key)
     return chatbot,   image, text_input

def gradio_upload(image):
    global key
    #image = {'image': <PIL.Image.Image image mode=RGB size=767x432 at 0x7F66A983F160>, 'mask': <PIL.Image.Image image mode=RGB size=767x432 at 0x7F6692691D60>}
    print("upload=",image)
    pil_image=image["image"]
    mask=image["mask"]
    responce, message, chatbot =  api_client.upload(key, pil_image , mask)
    return 

def gradio_ask(user_message):
    global key
    if len(user_message) == 0:
        text_box_show = 'Input should not be empty!'
    else:
        text_box_show = ''
    task_name=matched = re.findall(r'\[(.*?)\]', user_message)
    if len(task_name)==0:
         task = ""
    else:
         task_name=task_name[0]
         if task_name=="":
              task = ""
         else:
              task = "[" + task_name + "]"
    user_message=user_message.replace(task,"")
    responce, message , chatbot =  api_client.ask(key, task , user_message)
    return text_box_show

def gradio_stream_answer(temperature):
    global key
    print("gradio_stream_answer")
    print(" temperature=", temperature)
    responce, message, clean_text, text, html_txt , chatbot = api_client.generate(key, temperature)
    print(  chatbot )
    genatae = str(chatbot[0][1])
    genatae =  genatae.replace("\\","")
    print(genatae)
    return  genatae 

def gradio_visualize():
    global key
    print("gradio_visualize")
    responce, message , html_colored_list , pil_image , chatbot  = api_client.visualize(key)
    html_txt=html_colored_list[1]
    return pil_image,html_txt

def gradio_taskselect(idx):
    prompt_list = [
        '',
        '[grounding] describe this image in detail',
        '[refer] ',
        '[detection] ',
        '[identify] what is this ',
        '[vqa] '
    ]
    instruct_list = [
        '**Hint:** Type in whatever you want',
        '**Hint:** Send the command to generate a grounded image description',
        '**Hint:** Type in a phrase about an object in the image and send the command',
        '**Hint:** Type in a caption or phrase, and see object locations in the image',
        '**Hint:** Draw a bounding box on the uploaded image then send the command. Click the "clear" botton on the top right of the image before redraw',
        '**Hint:** Send a question to get a short answer',
    ]
    return prompt_list[idx], instruct_list[idx]

def example_trigger(text_input, image):
    # set the upload flag to true when receive a new image.
    # if there is an old image (and old conversation), set the replace flag to true to reset the conv later.
    return 

title = """<h1 align="center">MiniGPT-v2 API Demo by Megu channel</h1>"""
introduction = '''
For Abilities Involving Visual Grounding:
1. Grounding: CLICK **Send** to generate a grounded image description.
2. Refer: Input a referring object and CLICK **Send**.
3. Detection: Write a caption or phrase, and CLICK **Send**.
4. Identify: Draw the bounding box on the uploaded image window and CLICK **Send** to generate the bounding box. (CLICK "clear" button before re-drawing next time).
5. VQA: Input a visual question and CLICK **Send**.
6. No Tag: Input whatever you want and CLICK **Send** without any tagging

You can also simply chat in free form!
'''
text_input = gr.Textbox(placeholder='Upload your image and chat', interactive=True, show_label=False, container=False,  scale=8)
with gr.Blocks() as demo:
    gr.Markdown(title)
    gradio_reset()
    with gr.Row():
        with gr.Column(scale=1):
            if gr.Image!="":
                   image = gr.Image(type="pil", tool='sketch', brush_radius=20)
            temperature = gr.Slider(
                minimum=0.1,
                maximum=1.5,
                value=0.6,
                step=0.1,
                interactive=True,
                label="Temperature",
            )
            clear      = gr.Button("Restart",variant='primary' )
            img_upload = gr.Button("Image Uplad")
            gr.Markdown(introduction)
        with gr.Column():
            text_output = gr.Markdown()
            genatae  = gr.Markdown()
            pil_img = gr.Image(label="画像")
            dataset = gr.Dataset(
                components=[gr.Textbox(visible=False)],
                samples=[['No Tag'], ['Grounding'], ['Refer'], ['Detection'], ['Identify'], ['VQA']],
                type="index",
                label='Task Shortcuts',
            )
            task_inst = gr.Markdown('**Hint:** Upload your image and chat')
            with gr.Row():
                text_input.render()
                send = gr.Button("Send", variant='primary', size='sm', scale=1)
        
    with gr.Row():
        with gr.Column():
            gr.Examples(examples=[
                ["examples_v2/office.jpg", "[grounding] describe this image in detail"],
                ["examples_v2/sofa.jpg", "[detection] sofas"],
                ["examples_v2/2000x1372_wmkn_0012149409555.jpg", "[refer] the world cup"],
                ["examples_v2/KFC-20-for-20-Nuggets.jpg", "[identify] what is this {<4><50><30><65>}"],
            ], inputs=[image, text_input], fn=example_trigger,
                outputs=[])
        with gr.Column():
            gr.Examples(examples=[
                ["examples_v2/glip_test.jpg", "[vqa] where should I hide in this room when playing hide and seek"],
                ["examples_v2/float.png", "Please write a poem about the image"],
                ["examples_v2/thief.png", "Is the weapon fateful"],
                ["examples_v2/cockdial.png", "What might happen in this image in the next second"],
            ], inputs=[image, text_input], fn=example_trigger,
                outputs=[])

    clear.click(gradio_reset,    queue=False)    #reset　リセット　&　GETキー
    img_upload.click(gradio_upload,   [image],[], queue=False) #アップロードイメージ
    
    dataset.click(
        gradio_taskselect,
        inputs=[dataset],
        outputs=[text_input, task_inst],
        show_progress="hidden",
        postprocess=False,
        queue=False,
    )
    #text- boxでenterをした時
    text_input.submit(
        gradio_ask,
        [text_input],
        [text_input], queue=False
    ).success(
        gradio_stream_answer,
        [temperature],
        [genatae ]
    ).success(
        gradio_visualize,
        [],
        [pil_img, text_output],
        queue=False,
    )
    #send ボタンをクリックした時
    send.click(
        gradio_ask,
        [text_input],
        [text_input], queue=False
    ).success(
        gradio_stream_answer,
        [temperature],
        [genatae]
    ).success(
        gradio_visualize,
        [],
        [pil_img,text_output],
        queue=False,
    )

demo.launch(share=False)
