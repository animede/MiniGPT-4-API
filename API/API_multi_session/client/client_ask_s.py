import requests
from PIL import Image
from io import BytesIO
import json

# 送信するデータを準備
data = {
   "user_message": "[grounding] describe this image in detail",
   "key": "20231124233503IqK3y",
  }
#data ={
#    "user_message": "[refer] color of  clock tower",
#    "key":  "20231120144545NbrnT"
#   }
#data ={
#    "user_message": "[detection] clock tower",
#    "key":  "20231120144545NbrnT",
#  }
#data ={
#    "user_message": "[identify] color of clock tower",
#    "key": "20231120144545NbrnT"
 #  }
#data ={
#    "user_message": "[vqa]Please explaine colore of a clock tower",
#   "key": "20231120144545NbrnT",
#  }

# FastAPIエンドポイントのURL
url = 'http://0.0.0.0:8001/ask/'  # FastAPIサーバーのURLに合わせて変更してください
# POSTリクエストを送信
response = requests.post(url, json=data)
# レスポンスを表示 return {"message": "ask_completed ","chatbot":chatbot}
if response.status_code == 200:
    result = response.json()
    print("サーバーからの応答message:", result.get("message"))
    print("サーバーからの応答chatbot:", result.get("chatbot"))
else:
    print("リクエストが失敗しました。ステータスコード:", response.status_code)

