import argparse
import os
import random
from collections import defaultdict

import cv2
import re
import numpy as np
from PIL import Image
import torch
import html
import torchvision.transforms as T
import torch.backends.cudnn as cudnn

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Conversation, SeparatorStyle, Chat
# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

# ==========    args  
def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='eval_configs/minigptv2_eval.yaml', help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--options", nargs="+", help="override some settings in the used config, the key-value pair "
                                                                                                                           "in xxx=yyy format will be merged into config file (deprecate), "
                                                                                                                           "change to --cfg-options instead.", )
    args = parser.parse_args()
    return args

#==========  INIT
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
cudnn.benchmark = False
cudnn.deterministic = True
print('Initializing Chat')
args = parse_args()
cfg = Config(args)
device = 'cuda:{}'.format(args.gpu_id)
model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to(device)
bounding_box_size = 100
vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
model = model.eval()

CONV_VISION = Conversation(
    system="",
    roles=(r"<s>[INST] ", r" [/INST]"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="",
)
#バウンディングboxで使用する色を定義
colors = [(255, 0, 0),    (0, 255, 0),    (0, 0, 255),    (210, 210, 0),    (255, 0, 255),    (0, 255, 255),    (114, 128, 250),    (0, 165, 255),
              (0, 128, 0),    (144, 238, 144),    (238, 238, 175),    (255, 191, 0),    (0, 128, 0),    (226, 43, 138),    (255, 0, 255),    (0, 215, 255),]
color_map = {
    f"{color_id}": f"#{hex(color[2])[2:].zfill(2)}{hex(color[1])[2:].zfill(2)}{hex(color[0])[2:].zfill(2)}" for
    color_id, color in enumerate(colors )
}
used_colors = colors
chat = Chat(model, vis_processor, device=device)

# ===================================     FastAPI  ==============================
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse,StreamingResponse,JSONResponse
from pydantic import BaseModel
from io import BytesIO
import json
import base64

from function import mask2bbox, reverse_escape , escape_markdown, visualize_all_bbox_together

app = FastAPI()

global  chatbot
global  chat_state
global  gr_img
global  img_list
global  upload_flag
global  replace_flag
global  out_image

#　>>>>>>>>>>>>>>>　リセット　<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
@app.post("/reset/")
def reset( ):
        print(">>>>>  reset(")
        global  chatbot, chat_state,  gr_img,  img_list, upload_flag, replace_flag, out_image
        chatbot=[]
        chat_state=None
        gr_img=[]
        img_list=[]
        upload_flag=0
        replace_flag=0
        out_image=None
        return {'message': "complete"}

#　>>>>>>>>>>>>>>>　LLMへ　イメージのアップロード　<<<<<<<<<<<<<<<<<<<<<<<<<<
@app.post("/uploadfile/")
#def uploadfile(file: UploadFile ):
def upload_file(file: UploadFile = File(...), mask:UploadFile=File(...)):
    global   chatbot, chat_state, gr_img, img_list,upload_flag, replace_flag
    print(">>>>>  uploadfile")
    if file:
            image_data = file.file.read()
            pil_img = Image.open(BytesIO(image_data))  # バイナリデータをPIL形式に変換
            mask_img =  mask.file.read()
            pil_mask= Image.open(BytesIO(mask_img))  # バイナリデータをPIL形式に変換
    else:
        return {"message":"Error"}#"Error"
    #元画像と同じサイズのマスクを作成
    #w, h = pil_img.size
    #mask = Image.new('RGB', [w,h], (0,0,0))
    gr_img = {"image" :pil_img ,"mask":pil_mask}
    upload_flag = 1     # set the upload flag to true when receive a new image.
    if img_list:                # if there is an old image (and old conversation), set the replace flag to true to reset the conv later.
        replace_flag = 1
    
    #print("gr_img =",gr_img )
    return {"message": "Uploadded","chatbot": chatbot}

#　>>>>>>>>save_tmp_img
def save_tmp_img(visual_img):
    file_name = "".join([str(random.randint(0, 9)) for _ in range(5)]) + ".jpg"
    file_path = "/tmp/gradio" + file_name
    visual_img.save(file_path)
    return file_path

#　>>>>>>>>>>>>>>>　LLMへ　ask　<<<<<<<>>><<<<<<<>><<<<<<<<<<<<<<<<<<<<<<<<<
class Ask(BaseModel):
     user_message: str  
@app.post("/ask/")
def ask(gen_request:Ask):
    global  chatbot, chat_state,  gr_img,  img_list, upload_flag, replace_flag, out_image
    print(">>>>>  ask")
    user_message = gen_request.user_message
    #if len(user_message) == 0:
    #    text_box_show = 'Input should not be empty!'
    #else:
     #   text_box_show = ''
    if isinstance(gr_img, dict):   #gr_imgが辞書型か？
        gr_img, mask = gr_img['image'], gr_img['mask']  #辞書型ならimageとmaskを抜き出す
    else:
        mask = None  #辞書型でないなら gr_imgはimageであり、maskはなし

    # ユーザーが[identify]でバウンディングboxの位置を指定している時の処理
    if '[identify]' in user_message: # ユーザーがバウンディングboxの位置を指定しているかどうか
        ##-?: ハイフン（-）が0回または1回現れることを示す. \d+: 1つ以上の数字（0から9までのいずれか）が1回以上現れることを示します
        integers = re.findall(r'-?\d+', user_message) #txtから数字列を探し、その数字をintegersにリストで記録
        if len(integers) != 4:  #  ユーザーが4箇所のbboxを指定していない場合 
            bbox = mask2bbox(mask)
            user_message = user_message + bbox
    if chat_state is None:
        chat_state = CONV_VISION.copy()
    if upload_flag:
        if replace_flag:
            chat_state = CONV_VISION.copy()  # new image, reset everything
            replace_flag = 0
            chatbot = []
        img_list = []
        llm_message = chat.upload_img(gr_img, chat_state, img_list)
        upload_flag = 0
    chat.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]
    
    if '[identify]' in user_message:
        visual_img, _ = visualize_all_bbox_together(gr_img, user_message,colors)
        if visual_img is not None:
            file_path = save_tmp_img(visual_img)            
            #chatbot = chatbot + [[(gr_img), None]]
            chatbot = chatbot + [[(file_path,), None]]
            out_image=visual_img #バウンディングbox付きの画像をout_imageに保管
    result="Accepted"
    return {'message':result, "chatbot":chatbot}

#　>>>>>>>>>>>>>>>　LLMからの結果をストリームで取得　　<<<<<<<>>>>>>>>>>
@app.post("/get_stream/")
def get_stream( temperature:float=0.6):
    global  chatbot, chat_state,  gr_img,  img_list, upload_flag, replace_flag, out_image
    print(">>>>>  get_stream")
    generator_obj = stream_answer(chatbot, chat_state, img_list, temperature)
    try:
        for result in generator_obj:
            output= result
            print(".", end="")
    except:
        return {'message':"Generate error, try ask"}
    print(output)
    html_txt=reverse_escape(chatbot[-1][1])
    print("html_txt=", html_txt)
    text=html_txt.replace("<p>","").replace("</p","").replace("{","").replace("}","")
    # 数値と特殊記号を除去する正規表現
    clean_text = re.sub(r"[<>0-9]", "", text)
    print("clean_text=",clean_text)
    result="Generated"
    return {'message':result, "chatbot":chatbot, "html_txt":html_txt,"text":text,"clean_text":clean_text}

#　>>>>>>>>>>>>>>>　LLMからの結果を視覚化　<<<<<<<>>>>>>>>>>>>>>>　
@app.post("/visualize/")
def visualize():
    global  chatbot, chat_state,  gr_img,  img_list, upload_flag, replace_flag, out_image
    print(">>>>>  visualize")
    if isinstance(gr_img, dict):   #  gr_imgはも元の画像, maskはイメージと同じサイズの黒い画像, 画像はPLIオブジェクト
        gr_img, mask = gr_img['image'], gr_img['mask']
    try:
        unescaped = reverse_escape(chatbot[-1][1])
    except:
        return {'message':"visualize error ,try ask"}
    visual_img, generation_color = visualize_all_bbox_together(gr_img, unescaped,colors)
    print("0-unescaped-",unescaped)
    if '[identify]' in chatbot [0][0]:
        visual_img=out_image
    print("generation_color =",generation_color )    
    if visual_img is not None:
        if len(generation_color):
            chatbot[-1][1] = generation_color
        chatbot = chatbot + [[None, "file_path"]]#DUMMY
        try:
            html_colored_list=chatbot[-2]
        except:
            html_colored_list=""
    else:
        visual_img = gr_img
        html_colored_list=["None",unescaped]
    print("html_colored_list",html_colored_list)
    #jesonで返信するためにbase64にエンコード
    try:
        img_byte_array = BytesIO()
        visual_img.save(img_byte_array, format="PNG")
        img_base64 = base64.b64encode(img_byte_array.getvalue()).decode()
        result="Created"
    except:
        img_base64=""
        result="error"
    return {'message':result, "chatbot": chatbot ,"html_colored_list":html_colored_list,"visual_img":img_base64}

#LLMからの結果をストリームで表示
def stream_answer(chatbot, chat_state, img_list, temperature):
    if len(img_list) > 0:
        if not isinstance(img_list[0], torch.Tensor):
            chat.encode_img(img_list)
    streamer = chat.stream_answer(conv=chat_state,
                                  img_list=img_list,
                                  temperature=temperature,
                                  max_new_tokens=500,
                                  max_length=2000)
    output = ''
    for new_output in streamer:
        escapped = escape_markdown(new_output)
        output += escapped
        chatbot[-1][1] = output
        yield chatbot, chat_state
    chat_state.messages[-1][1] = '</s>'
    return chatbot, chat_state

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
