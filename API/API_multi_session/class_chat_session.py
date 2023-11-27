import pprint

class ChatSession:
    max_sessions = 10  # 最大セッション数の設定

    def __init__(self):
        self.chat_sessions = {}
        self.session_keys = []

    def create_session(self, key):
        """新しいチャットセッションを作成する"""
        if key in self.chat_sessions:
            raise ValueError("Session key already exists.")

        self.chat_sessions[key] = {
            "chatbot": [],
            "chat_state": None,
            "gr_img": [],
            "img_list": [],
            "upload_flag": 0,
            "replace_flag": 0,
            "out_image": None,
        }

        self.session_keys.append(key)
        if len(self.session_keys) > ChatSession.max_sessions:
            first_key = self.session_keys.pop(0)
            del self.chat_sessions[first_key]

        pprint.pprint(self.chat_sessions)
        #print(self.session_keys)

    def read(self, key, sub_key):
        """指定されたキーのデータを読み取る"""
        return self.chat_sessions[key][sub_key]

    def write(self, key, sub_key, data):
        """指定されたキーのデータを書き込む"""
        self.chat_sessions[key][sub_key] = data
        #pprint.pprint(self.chat_sessions)

    def read_all(self, key):
        """特定のキーのすべてのデータを読み取る"""
        session = self.chat_sessions[key]
        return (session["chatbot"], session["chat_state"], session["gr_img"],
                session["img_list"], session["upload_flag"], 
                session["replace_flag"], session["out_image"])

    def write_all(self, key, data):
        """特定のキーのすべてのデータを書き込む"""
        session = self.chat_sessions[key]
        session["chatbot"] = data["chatbot"]
        session["chat_state"] = data["chat_state"]
        session["gr_img"] = data["gr_img"]
        session["img_list"] = data["img_list"]
        session["upload_flag"] = data["upload_flag"]
        session["replace_flag"] = data["replace_flag"]
        session["out_image"] = data["out_image"]
        #pprint.pprint(self.chat_sessions)
