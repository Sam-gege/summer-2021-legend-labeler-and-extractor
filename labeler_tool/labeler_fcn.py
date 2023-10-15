from fcns import *
from copy import deepcopy
import pandas as pd
def draw_symbol_text(symbol_text,OCR):
    global rgb
    special_char = ' /&$(),-.!:'
    numbers_char = '0123456789'
    lower_char = 'abcdefghijklmnopqrstuvwxyz'
    upper_char = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    french_char = 'é'  # as seen in the word 'café',otherwise it will be recognized as 'cafe' with low confidence
    custom_char = special_char + numbers_char + lower_char + upper_char + french_char
    '''
    alist: the list of symbol text pairs
    '''
    if not symbol_text:
        return
    if len(symbol_text[-1])==1:
        warnings.warn("wrong length, you forgot to press 's' in the end")
        return
    rgb3=rgb.copy()
    i=0
    for [symbolcc,textcc] in symbol_text:
        # draw symbol using red box
        xs,ys,ws,hs=symbolcc.loc
        cv2.rectangle(rgb3, (xs, ys), (xs + ws - 1, ys + hs - 1), (0, 0, 255), 1)
        # draw text using green box
        xt,yt,wt,ht=textcc.loc
        cv2.rectangle(rgb3, (xt, yt), (xt + wt - 1, yt + ht - 1), (0, 255, 0), 1)
        # OCR
        if OCR:
            ocr_area = rgb[(yt-1):(yt+ht+1),(xt-1):(xt+wt+1)]
            results = pytesseract.image_to_data(ocr_area, output_type=pytesseract.Output.DICT,
                                                config=f"-c tessedit_char_whitelist='{custom_char}' --psm 6")
            print(f"'{' '.join([i for i in results['text'] if i])}',")
        # draw blue line
        cv2.line(rgb3, (xs+ws//2, ys+hs//2), (xt+wt//2, yt+ht//2), (255, 0, 0), thickness=1, lineType=8)
        # write numbers in black
        cv2.putText(rgb3,
            f'{i}',
            (xs, ys-2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4, (0, 0, 0), 1)
        if textcc.text == '':
            cv2.putText(rgb3,
                f'{i}',
                (xt, yt-2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (0, 0, 0), 1)
        else:
            cv2.putText(rgb3,
                f'{i}:{textcc.text}',
                (xt, yt-2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (0, 0, 0), 1)
        i+=1
    show_images(rgb3)
def adjust_boxes(sym_txt,pos_string):
    for pos,strList in pos_string.items():
        pos1=int(pos[:-1])
        pos2=pos[-1]
        if pos2=='t':
            sym_txt[pos1][1]+=strList
        if pos2=='s':
            sym_txt[pos1][0]+=strList
def add_text(sym_txt,pos_txt):
    for i in range(len(pos_txt)):
        sym_txt[i][1].addText(pos_txt[i])
    # for pos,txt in pos_txt.items():
    #     sym_txt[pos][1].addText(txt)
def mouse_callback(event,x,y,flags,param):
    global combined_cc,is_symbol,rgb2,mask,CC
    if event == cv2.EVENT_RBUTTONDOWN:
        if is_symbol:
            cv2.circle(rgb2, (x, y), 3, (0, 0, 255), -1)
        else:
            cv2.circle(rgb2, (x, y), 3, (0, 255, 0), -1)
        if mask[y,x]!=0:
            combined_cc.append(CC[mask[y,x]-1])
class ConnectedC:
    def __init__(self, loc: tuple):
        self.loc = loc[0:4]
        self.left = loc[0]
        self.top = loc[1]
        self.right = loc[0] + loc[2] - 1
        self.bottom = loc[1] + loc[3] - 1
        self.text = ''

    def __add__(self, other):
        if isinstance(other, int):
            return self
        elif isinstance(other, ConnectedC):
            left = min(self.left, other.left)
            top = min(self.top, other.top)
            right = max(self.right, other.right)
            bottom = max(self.bottom, other.bottom)
            w = right - left + 1
            h = bottom - top + 1
            return ConnectedC((left, top, w, h))
        elif isinstance(other, list):
            links={'w':self.top,'s':self.bottom,'a':self.left,'d':self.right}
            for i in other:
                if i[:3]=='exp':
                    times=int(i[3:]) if len(i)>3 else 1
                    for j in range(times):
                        links['w']-=1
                        links['a']-=1
                        links['s']+=1
                        links['d']+=1
                else:
                    locator=i[0]
                    operator=i[1]
                    value = int(i[2:])
                    if operator=='+':
                        if locator=='w' or locator=='a':
                            links[locator]-=value
                        elif locator=='s' or locator=='d':
                            links[locator]+=value
                    elif operator=='-':
                        if locator=='w' or locator=='a':
                            links[locator]+=value
                        elif locator=='s' or locator=='d':
                            links[locator]-=value
            top,bottom,left,right=links['w'],links['s'],links['a'],links['d']
            w = right - left + 1
            h = bottom - top + 1
            return ConnectedC((left, top, w, h))
    def __radd__(self, other):
        return self.__add__(other)

    def addText(self, text):
        self.text = text
    def __str__(self):
        return str(self.loc)
    def __iter__(self):
        return iter(self.loc)


def select_sym_txt(input_fn, canny_thres, kernel,size_filter):
    global combined_cc, is_symbol, rgb2, mask, CC, rgb
    '''cell1'''
    rgb, gray,_,_ = read_image(f'images/processed/{input_fn}')
    canny = cv2.Canny(gray, canny_thres[0], canny_thres[0])
    # delete long vertical lines
    mask_y = np.ones((99, 1), dtype=np.uint8)
    y = cv2.morphologyEx(canny, cv2.MORPH_OPEN, mask_y, iterations=1)
    canny -= y
    # delete long horizontal lines
    mask_x = np.ones((1, 99), dtype=np.uint8)
    x = cv2.morphologyEx(canny, cv2.MORPH_OPEN, mask_x, iterations=1)
    canny -= x
    ## morph must use odd number!!
    filled = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, np.ones(kernel, dtype=int))
    ## find connected components
    _, _, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8, ltype=cv2.CV_32S)
    CC = []
    for x, y, w, h, A in stats:
        if not size_filter:
            CC.append(ConnectedC((x, y, w, h)))
        else:
            if ((w>=6 or h>=6) and (w<=120 or h<=120)) or (w >= 120 and h<=80 and w >= 2 * h) or (w <= 6 and h >= 1.5 * w) or (2<h<=6 and w>=1.5*h):
                CC.append(ConnectedC((x, y, w, h)))

    '''cell2'''
    rgb1 = rgb.copy()
    mask = np.zeros(rgb.shape[:2], dtype=int)
    i = 1
    for c in CC:
        x, y, w, h = c
        mask[y:y + h, x:x + w] = i
        cv2.rectangle(rgb1, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 1)
        i += 1

    '''cell3'''
    rgb2 = rgb1.copy()
    is_symbol = True
    combined_cc = []
    cv2.namedWindow('select symbols and texts', cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback('select symbols and texts', mouse_callback)
    symbol_text = []
    while (1):
        cv2.imshow('select symbols and texts', rgb2)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            break
        if k == ord('s'):
            if is_symbol:
                symbol_text.append([sum(combined_cc)])
            else:
                symbol_text[-1].append(sum(combined_cc))
            is_symbol = not is_symbol
            combined_cc = []
    draw_symbol_text(symbol_text,OCR=False)
    return symbol_text,rgb,filled
