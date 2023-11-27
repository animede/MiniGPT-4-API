import requests
from PIL import Image
from io import BytesIO
import json

# 送信するデータを準備

data = {
   "user_message": "[grounding] describe this image in detail",
   }
#data ={
#    "user_message": "[refer] clock tower",
#   }
#data ={
#    "user_message": "[detection] clock tower",
#   }
#data ={
#    "user_message": "[identify] clock tower",
#   }
#data ={
#    "user_message": "[vqa]Please explaine colore of a clock tower",
#   }

# FastAPIエンドポイントのURL
url = 'http://0.0.0.0:8001/ask/'  # FastAPIサーバーのURLに合わせて変更してください
# POSTリクエストを送信
response = requests.post(url, json=data)
# レスポンスを表示 return {"message": "ask_completed ","chatbot":chatbot}
if response.status_code == 200:
    result = response.json()
    print("サーバーからの応答message:", result.get("message"))
    print("サーバーからの応答chatbot:", result.get("chatbot"))
    print("サーバーからの応答text_box_show:", result.get("text_box_show"))
else:
    print("リクエストが失敗しました。ステータスコード:", response.status_code)

