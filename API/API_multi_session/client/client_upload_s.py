import requests
from PIL import Image
from io import BytesIO
# 送信するPIL形式の画像データ
image_file_path = '00016-331097358.png'
# FastAPIエンドポイントのURL
url = 'http://0.0.0.0:8001/uploadfile/'  # FastAPIサーバーのURLに合わせて変更してください
#Maskを作成。IDENTIFYタスクで使う。その他の場合はイメージと同じ大きさのファイルを作成
pil_image=Image.open(image_file_path )
width, height = pil_image.size
mask = Image.new('L', (width, height), 0)
# ファイルデータをバイナリ形式に変換
img_byte_arr_m = BytesIO()
mask.save(img_byte_arr_m, format='PNG')
mask_img = img_byte_arr_m.getvalue()
# ファイルデータをバイナリ形式で読み込む
file_data = open(image_file_path, "rb").read()
# ファイルをアップロードするためのリクエストを作成
files = {
    "file": ("img.png", BytesIO(file_data), "image/png"),
    "mask": ("mask.png", BytesIO(mask_img), "image/png"),
    "key": (None, "20231124233503IqK3y")
}
# POSTリクエストを送信
response = requests.post(url, files=files)
# レスポンスを表示
if response.status_code == 200:
    result = response.json()
    print("サーバーからの応答message:", result.get("message"))
    print("サーバーからの応答chatbot:", result.get("chatbot"))
else:
    print("リクエストが失敗しました。ステータスコード:", response.status_code)
