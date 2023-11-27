import requests
from PIL import Image
from io import BytesIO
import base64

class ImageAPI:
    def __init__(self, url):
        self.url = url

    def reset(self):
        response = requests.post(self.url + "/reset/")     # レスポンス
        if response.status_code == 200:
            result = response.json()
            return response.status_code, result.get("message"), result.get("key")
        else:
            return response.status_code, "None", "None"

    def upload(self, key, pil_image, mask=None):     #  image_file_path  送信するPIL形式の画像データ
        print("api_key=", key)
        #バイナリストリームに画像を保存
        img_byte_arr = BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        img_data = img_byte_arr.getvalue()

        if mask is None:
            width, height = pil_image.size
            mask = Image.new('L', (width, height), 0)

        img_byte_arr_m = BytesIO()
        mask.save(img_byte_arr_m, format='PNG')
        mask_img = img_byte_arr_m.getvalue()
        # ファイルをアップロードするためのリクエストを作成
        files = {
            "file": ("img.png", BytesIO(img_data), "image/png"),
            "mask": ("mask.png", BytesIO(mask_img), "image/png"),
            "key": (None, key)
        }

        response = requests.post(self.url + "/uploadfile/", files=files) # レスポンス
        if response.status_code == 200:
            result = response.json()
            return response.status_code, result.get("message"), result.get("chatbot")
        else:
            return response.status_code, "None", "None"

    def ask(self, key, task, user_message):
        data = {
            "user_message": task + user_message,
            "key": key
        }
        response = requests.post(self.url + "/ask/", json=data)   # レスポンス
        if response.status_code == 200:
            result = response.json()
            return response.status_code, result.get("message"), result.get("chatbot")
        else:
            return response.status_code, "None", "None"

    def generate(self, key, temperature=0.6):
        print(" temperature=", temperature)
        data = {
            "temperature": temperature,
            "key": key
        }
        response = requests.post(self.url + "/get_stream/", json=data)   # レスポンス
        if response.status_code == 200:
            result = response.json()
            return response.status_code, result.get("message"), result.get("clean_text"), result.get("text"), result.get("html_txt"), result.get("chatbot")
        else:
            return response.status_code, "None", "None", "None", "None", "None"

    def visualize(self, key):
        data = {
            "key": key
        }
        response = requests.post(self.url + "/visualize/", json=data)   # レスポンス
        if response.status_code == 200:
            result = response.json()
            if result.get("message") == "Created":
                decoded_image_data = base64.b64decode(result.get("visual_img"))
                pil_image = Image.open(BytesIO(decoded_image_data))
                chatbot = result.get("chatbot")
                return response.status_code, result.get("message"), result.get("html_colored_list"), pil_image, chatbot
            #通信は成功したが上手く生成出来なかった時
            else:  
                print(result.get("message"))
                return response.status_code, result.get("message"), "None", "None", "None"
        else:
            return response.status_code, "None", "None", "None", "None"


