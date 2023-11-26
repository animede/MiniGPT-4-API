import requests
from PIL import Image
from io import BytesIO
import json
import base64
from PIL import Image

# FastAPIエンドポイントのURL
url = 'http://0.0.0.0:8001/visualize/'  # FastAPIサーバーのURLに合わせて変更してください
# 送信するデータを準備
data = {
   "key": "20231124233503IqK3y",
}
response = requests.post(url, json=data)
# レスポンスを表示
if response.status_code == 200:
    result = response.json()
    if result.get("message")=="Created":
        print("サーバーからの応答message:", result.get("message"))
        print("サーバーからの応答chatbot:", result.get("chatbot"))
        print("サーバーからの応答html_colored_list:", result.get("html_colored_list"))
        decoded_image_data = base64.b64decode(result.get("visual_img"))
        pil_image = Image.open(BytesIO(decoded_image_data ))
        print("サーバーからの応答visual_img:", pil_image)
        pil_image.show()
    else:
        print(result.get("message"))
else:
    print("リクエストが失敗しました。ステータスコード:", response.status_code)

