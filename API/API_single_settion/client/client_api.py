import requests
from PIL import Image
from io import BytesIO
import json
import base64


def  reset(url ):
    url = url+"/reset/"      # FastAPIエンドポイントのURL
    response = requests.post(url)       # レスポンス
    if response.status_code == 200:
       result = response.json()
       return response.status_code, result.get("message") , result.get("key")      
    else:
       return response.status_code,"None","None"

def upload(url , pil_image="",mask=""):     #  image_file_path  送信するPIL形式の画像データ
    #バイナリストリームに画像を保存
    img_byte_arr = BytesIO()
    pil_image.save(img_byte_arr, format='PNG')
    img_data = img_byte_arr.getvalue()
    if mask=="":
        width, height = pil_image.size
        mask = Image.new('L', (width, height), 0)
        
    img_byte_arr_m= BytesIO()
    mask.save(img_byte_arr_m, format='PNG')
    mask_img = img_byte_arr_m.getvalue()
    # ファイルをアップロードするためのリクエストを作成
    files = {
            "file": ("img.png", BytesIO(img_data), "image/png"),
            "mask": ("mask.png", BytesIO(mask_img), "image/png"),
     }
    url =url+"/uploadfile/"      # FastAPIエンドポイントのURL
    response = requests.post(url, files=files)    # レスポンス
    if response.status_code == 200:
       result = response.json()
       return response.status_code, result.get("message") ,  result.get("chatbot")
    else:
       return response.status_code,"None","None"

def ask(url ,  task , user_message):
    data = {
       "user_message": task + user_message,
          }
    url =url+"/ask/"     # FastAPIエンドポイントのURL
    response = requests.post(url, json=data)      # レスポンス
    if response.status_code == 200:
       result = response.json()
       return response.status_code, result.get("message"), result.get("chatbot")
    else:
       return response.status_code,"None","None"

def generate(url , key, temperature=0.6):
    data = {
        "temperature": temperature,
        }
    url =url+"/get_stream/"      # FastAPIエンドポイントのURL
    response = requests.post(url, json=data)    # レスポンス
    if response.status_code == 200: 
        result = response.json()
        return response.status_code, result.get("message"), result.get("clean_text"), result.get("text"), result.get("html_txt") , result.get("chatbot")
    else:
        return response.status_code,"None","None","None","None","None"

def visualize(url ):
    url =url+"/visualize/"      # FastAPIエンドポイントのURL
    response = requests.post(url)     # レスポンス
    if response.status_code == 200:
        result = response.json()
        if result.get("message")=="Created":
            decoded_image_data = base64.b64decode(result.get("visual_img"))
            pil_image = Image.open(BytesIO(decoded_image_data ))
            chatbot=result.get("chatbot")
            return response.status_code,  result.get("message"),  result.get("html_colored_list"),   pil_image ,   chatbot
        else:     #通信は成功したが上手く生成出来なかった時
            print(result.get("message"))
            return response.status_code,   result.get("message"),"None","None","None"
    else:
        return response.status_code, "None","None","None","None"

