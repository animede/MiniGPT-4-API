from collections import defaultdict
from PIL import Image
import numpy as np
import cv2
import html
import re
#　＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊　　　内部関数　＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊
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

#2つの矩形（Rectangle） rect1 と rect2 が重なっているかどうかを判定する
def is_overlapping(rect1, rect2):
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2
    return not (x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4)#矩形が重なっている場合、返される値は True であり、重なっていない場合は False

#この関数 computeIoU(bbox1, bbox2) は、2つのバウンディングボックス（Bounding Box） bbox1 と bbox2 の間の交差部分の面積と、
#それらのバウンディングボックスのIoU（Intersection over Union）スコアを計算するための関数です。
#2つのバウンディングボックスの重なり具合を数値化して比較することができます。
def computeIoU(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    #intersection_x1、intersection_y1、intersection_x2、intersection_y2 を計算、2つのバウンディングボックスの交差部分の左上隅 と右下隅
    intersection_x1 = max(x1, x3)
    intersection_y1 = max(y1, y3)
    intersection_x2 = min(x2, x4)
    intersection_y2 = min(y2, y4)
    #交差部分の面積 
    intersection_area = max(0, intersection_x2 - intersection_x1 + 1) * max(0, intersection_y2 - intersection_y1 + 1)
    #各バウンディングボックスの面積
    bbox1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    bbox2_area = (x4 - x3 + 1) * (y4 - y3 + 1)
    #IoUスコアを計算します。これは、交差部分の面積をバウンディングボックスの合計面積から引いた後、合計面積で割った値です。
    #IoUは常に0から1の間の値を取り、0は重なりがないことを、1は完全に重なっていることを示します。
    union_area = bbox1_area + bbox2_area - intersection_area
    iou = intersection_area / union_area
    return iou

#ユーザー指定のバウンディングbox部分のマスクを作成　＊＊＊＊＊　サンプルではここが呼ばれることはない　＊＊＊＊＊
def mask2bbox(mask):
    print("-----   mask2bbox")
    print(mask)
    if mask is None:
        return ''
    mask = mask.resize([100, 100], resample=Image.NEAREST)#mask を100x100に縮小し、リサンプリングには最も近い隣接ピクセルの値を使用する
    #mask多次元配列から、非ゼロ行と、非ゼロ列を抽出。マスク内で実際に物体が存在する領域を特定する
    mask = np.array(mask)[:, :, 0]
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    #非ゼロ行と非ゼロ列から、バウンディングボックスの情報を抽出
    if rows.sum():
        #非ゼロ行の最小インデックス rmin と最大インデックス rmax、および非ゼロ列の最小インデックス cmin と最大インデックス cmax を取得
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        #抽出したバウンディングボックス情報を文字列としてフォーマット
        #フォーマットは {{<left><top><right><bottom>}} の形式で、バウンディングボックスの左上隅と右下隅の座標
        bbox = '{{<{}><{}><{}><{}>}}'.format(cmin, rmin, cmax, rmax)
    else:
        bbox = ''
    return bbox

def escape_markdown(text):    #文字列に"<" → "\<"    に変換 
    md_chars = ['<', '>']
    for char in md_chars:
       text = text.replace(char, '\\' + char)
    return text

def reverse_escape(text):           #　'\<' を　'<'　に変換
    md_chars = ['\\<', '\\>']
    for char in md_chars:
        text = text.replace(char, char[1:])
    return text

def extract_substrings(string):
    import re
    print("-----  extract_substrings")
    # first check if there is no-finished bracket 最後に現れる } のインデックスを検索、 index 変数に格納さ、 } が見つからなければ、indexには -1 が設定される
    index = string.rfind('}')
    if index != -1:
        string = string[:index + 1]
    #<p> 文字列に続き、最短一致（non-greedy）任意文字列（.*?）が続き、その後に }があるパターンでを設定。ただし、 } の後に < が続いていない場合
    pattern = r'<p>(.*?)\}(?!<)'
    #パターンに一致する部分文字列を matches というリストとして抽出す。
    matches = re.findall(pattern, string)
    substrings = [match for match in matches]#matchesと同じLISTが得られるが、COPYではないのでお互いに依存はしない、
    #例：['A young man</p> {<8><0><62><91>', 'a suit</p> {<7><35><56><88>', 'tie</p> {<29><46><41><89>', 'his desk</p> {<0><85><100><100>', 'a laptop</p> {<55><52><85><87>']
    return substrings

# 画像とeverse_escapeで変換された位置情報を含む文字列から、バウンディングbox付き画像とhtmlで記述された色付き文字列を生成する
# generationはuser_message または　reverse_escape(chatbot[-1][1]), reverse_escapeで変換された位置情報を含む文字列
# <p>sofas</p> {<57><49><70><72>}<delim>{<26><48><43><71>}
# OUT:　バウンディングboxと文字が書き込まれた画像、　htmlで記述された色付き文章
def visualize_all_bbox_together(image, generation,colors):
    bounding_box_size = 100
    print("++++++    visualize_all_bbox_together")
    print("generation=",generation)
    if image is None:
        return None, ''
    #HTMLエンコードされた文字列を元のテキストにデコード（逆変換）する
    generation = html.unescape(generation)
    image_width, image_height = image.size
    image = image.resize([500, int(500 / image_width * image_height)])
    image_width, image_height = image.size
    # first check if there is no-finished bracket
    string_list = extract_substrings(generation)
    if string_list:  # grounding 　または　 detection の時
        mode = 'all'
        entities = defaultdict(list)
        i = 0
        j = 0
        for string in string_list:
            try:
                obj, string = string.split('</p>')
            except ValueError:
                print('wrong string: ', string)
                continue
            bbox_list = string.split('<delim>')
            flag = False
            for bbox_string in bbox_list:
                integers = re.findall(r'-?\d+', bbox_string)
                if len(integers) == 4:
                    x0, y0, x1, y1 = int(integers[0]), int(integers[1]), int(integers[2]), int(integers[3])
                    left = x0 / bounding_box_size * image_width
                    bottom = y0 / bounding_box_size * image_height
                    right = x1 / bounding_box_size * image_width
                    top = y1 / bounding_box_size * image_height
                    entities[obj].append([left, bottom, right, top])
                    j += 1
                    flag = True
            if flag:
                i += 1
    else:
        integers = re.findall(r'-?\d+', generation)
        if len(integers) == 4:  #  refer　の時
            mode = 'single'
            entities = list()
            x0, y0, x1, y1 = int(integers[0]), int(integers[1]), int(integers[2]), int(integers[3])
            left = x0 / bounding_box_size * image_width
            bottom = y0 / bounding_box_size * image_height
            right = x1 / bounding_box_size * image_width
            top = y1 / bounding_box_size * image_height
            entities.append([left, bottom, right, top])
        else:
            return None, ''  # don't detect any valid bbox to visualize
    if len(entities) == 0:
        return None, ''
    if isinstance(image, Image.Image):
        image_h = image.height
        image_w = image.width
        image = np.array(image)
    elif isinstance(image, str):
        if os.path.exists(image):
            pil_img = Image.open(image).convert("RGB")
            image = np.array(pil_img)[:, :, [2, 1, 0]]
            image_h = pil_img.height
            image_w = pil_img.width
        else:
            raise ValueError(f"invaild image path, {image}")
    elif isinstance(image, torch.Tensor):
        image_tensor = image.cpu()
        reverse_norm_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:, None, None]
        reverse_norm_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:, None, None]
        image_tensor = image_tensor * reverse_norm_std + reverse_norm_mean
        pil_img = T.ToPILImage()(image_tensor)
        image_h = pil_img.height
        image_w = pil_img.width
        image = np.array(pil_img)[:, :, [2, 1, 0]]
    else:
        raise ValueError(f"invaild image format, {type(image)} for {image}")
    indices = list(range(len(entities)))
    new_image = image.copy()
    previous_bboxes = []
    text_size = 0.5    # size of text
    text_line = 1      # thickness of text     int(max(1 * min(image_h, image_w) / 512, 1))
    box_line = 2
    (c_width, text_height), _ = cv2.getTextSize("F", cv2.FONT_HERSHEY_COMPLEX, text_size, text_line)
    base_height = int(text_height * 0.675)
    text_offset_original = text_height - base_height
    text_spaces = 2
    used_colors = colors  # random.sample(colors, k=num_bboxes)
    color_id = -1
    for entity_idx, entity_name in enumerate(entities):
        # ++++   mode == 'single' or mode == 'identify'
        if mode == 'single' or mode == 'identify':
            bboxes = entity_name
            bboxes = [bboxes]
        else:
            bboxes = entities[entity_name]
        color_id += 1 #色を更新
        for bbox_id, (x1_norm, y1_norm, x2_norm, y2_norm) in enumerate(bboxes):
            skip_flag = False
            orig_x1, orig_y1, orig_x2, orig_y2 = int(x1_norm), int(y1_norm), int(x2_norm), int(y2_norm)
            color = used_colors[entity_idx % len(used_colors)]  # tuple(np.random.randint(0, 255, size=3).tolist())
            new_image = cv2.rectangle(new_image, (orig_x1, orig_y1), (orig_x2, orig_y2), color, box_line)
             # ++++   mode == 'all'
            if mode == 'all':
                l_o, r_o = box_line // 2 + box_line % 2, box_line // 2 + box_line % 2 + 1
                x1 = orig_x1 - l_o
                y1 = orig_y1 - l_o
                if y1 < text_height + text_offset_original + 2 * text_spaces:
                    y1 = orig_y1 + r_o + text_height + text_offset_original + 2 * text_spaces
                    x1 = orig_x1 + r_o
                # add text background
                (text_width, text_height), _ = cv2.getTextSize(f"  {entity_name}", cv2.FONT_HERSHEY_COMPLEX, text_size,  text_line)
                text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2 = x1, y1 - (
                            text_height + text_offset_original + 2 * text_spaces), x1 + text_width, y1
                for prev_bbox in previous_bboxes:
                    if computeIoU((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2), prev_bbox['bbox']) > 0.95 and \
                            prev_bbox['phrase'] == entity_name:
                        skip_flag = True
                        break
                    while is_overlapping((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2), prev_bbox['bbox']):
                        text_bg_y1 += (text_height + text_offset_original + 2 * text_spaces)
                        text_bg_y2 += (text_height + text_offset_original + 2 * text_spaces)
                        y1 += (text_height + text_offset_original + 2 * text_spaces)

                        if text_bg_y2 >= image_h:
                            text_bg_y1 = max(0, image_h - (text_height + text_offset_original + 2 * text_spaces))
                            text_bg_y2 = image_h
                            y1 = image_h
                            break
                if not skip_flag:
                    alpha = 0.5
                    for i in range(text_bg_y1, text_bg_y2):
                        for j in range(text_bg_x1, text_bg_x2):
                            if i < image_h and j < image_w:
                                if j < text_bg_x1 + 1.35 * c_width:
                                    # original color
                                    bg_color = color
                                else:
                                    # white
                                    bg_color = [255, 255, 255]
                                new_image[i, j] = (alpha * new_image[i, j] + (1 - alpha) * np.array(bg_color)).astype(
                                    np.uint8)
                    cv2.putText(
                        new_image, f"  {entity_name}", (x1, y1 - text_offset_original - 1 * text_spaces),
                        cv2.FONT_HERSHEY_COMPLEX, text_size, (0, 0, 0), text_line, cv2.LINE_AA
                    )
                    previous_bboxes.append(
                        {'bbox': (text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2), 'phrase': entity_name})
     # ++++   mode == 'all'
    if mode == 'all':
        def color_iterator(colors):
            while True:
                for color in colors:
                    yield color
        color_gen = color_iterator(colors)
        def colored_phrases(match): # Add colors to phrases and remove <p></p>
            phrase = match.group(1)
            color = next(color_gen)
            return f'<span style="color:rgb{color}">{phrase}</span>'
        generation = re.sub(r'{<\d+><\d+><\d+><\d+>}|<delim>', '', generation)
        generation_colored = re.sub(r'<p>(.*?)</p>', colored_phrases, generation)
    else:
        generation_colored = ''
    pil_image = Image.fromarray(new_image)
    # pil_imageはバウンディングboxと文字が書き込まれた画像、　generation_coloredはhtmlで記述された色付き文章
    return pil_image, generation_colored

def  get_stream_answer(chatbot, chat_state, img_list, temperature):
    generator_obj = stream_answer(chatbot, chat_state, img_list, temperature)
    try:
        for result in generator_obj:
            output= result
            print(".", end="")
    except:
        return {'message':"Generate error, try ask"}
    print(output)
    return output
    
