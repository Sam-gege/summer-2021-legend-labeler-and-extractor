from cv2 import cv2
import numpy as np
import re
import pytesseract
import statistics
import matplotlib.pyplot as plt
import warnings
import imutils
import math
import json
import pandas as pd
from difflib import SequenceMatcher
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
def set_whitelist(print_out):
    special_char=' /&$(),-.!:'
    numbers_char='0123456789'
    lower_char='abcdefghijklmnopqrstuvwxyz'
    upper_char='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    french_char='é' # as seen in the word 'café',otherwise it will be recognized as 'cafe' with low confidence
    custom_char=special_char+numbers_char+lower_char+upper_char+french_char
    if print_out:
        print(
            f'''\033[1mOCR whitelist:\033[0m
        \033[1mspecial char:\033[0m
        {special_char}
        
        \033[1mnumbers:\033[0m
        {numbers_char}
        
        \033[1mfrench:\033[0m
        {french_char}
        
        \033[1mlower & upper:\033[0m
        {lower_char+upper_char}
        '''
        )
    return custom_char
def read_image(filename,resized_diagonal=None):
    rgb=None
    extension = ['png', 'jpg','jpeg','PNG']
    for ext in extension:
        rgb = cv2.imread(f'{filename}.{ext}')  # read to greyscale
        if rgb is not None:
            break
    if rgb is None:
        cap = cv2.VideoCapture(f'{filename}.gif')
        _, rgb = cap.read()
        cap.release()
    if rgb is None:
        warnings.warn('file type not supported')
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    if resized_diagonal is not None:
        h,w,_=rgb.shape
        resized_width=int(resized_diagonal*w/math.sqrt(w**2+h**2))
        resized_rgb= imutils.resize(rgb, width=resized_width)
        resized_gray= imutils.resize(gray, width=resized_width)
        return resized_rgb,resized_gray,w/resized_width,rgb
    else:
        return rgb, gray,1,rgb

def show_images(*args):
    i=1
    for img in args:
        cv2.namedWindow(f'{i}', cv2.WINDOW_NORMAL)
        cv2.imshow(f'{i}', img)
        i+=1
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def CC_n_OCR(bw,rgb,whitelist,recog_txt_dict,pix,desc,criteria,psm):
    Height, Width = rgb.shape[:2]
    _,_,stats,_ = cv2.connectedComponentsWithStats(bw, connectivity=8, ltype=cv2.CV_32S)
    recognition_results=dict()
    stats_filter=[]
    for x,y,w,h,A in stats:
        if (x,y,w,h) in recog_txt_dict.keys():
            results = recog_txt_dict[(x, y, w, h)]
            loc, conf, text = results['loc'], results['conf'], results['text']
            recognition_results[(x, y, w, h)] = {'loc': loc, 'conf': conf, 'text': text}
        elif criteria(w,h):
            stats_filter.append((x,y,w,h,A))
    for x,y,w,h,A in tqdm(stats_filter, ncols=100, desc=desc):
        if x>=pix and y>=pix and x+w<=Width+1 and y+h<=Height+1:
            x -= pix
            y -= pix
            w += 2 * pix
            h += 2 * pix
        ocr_area = rgb[y:y + h, x:x + w]
        if (x,y,w,h) in recog_txt_dict.keys():
            results=recog_txt_dict[(x,y,w,h)]
            loc,conf,text=results['loc'],results['conf'],results['text']
        else:
            results = pytesseract.image_to_data(ocr_area, output_type=pytesseract.Output.DICT,
                                        config=f"-c tessedit_char_whitelist='{whitelist}' --psm {psm}")
            loc,conf,text=[],[],[]
            for i in range(len(results['conf'])):
                if int(results["conf"][i])>=0:# record results that have none negative confidence
                    loc.append((int(results['left'][i]),int(results['top'][i]),
                               int(results['width'][i]),int(results['height'][i])))
                    conf.append(results['conf'][i])
                    text.append(results['text'][i])
        recognition_results[(x,y,w,h)]={'loc':loc,'conf':conf,'text':text}
    return recognition_results


def initial_result_filter(rgb, recognition_results, is_text_criteria, letter_num_symbols_criteria,
                          text_color, symbol_color,convert_sym_txt):
    '''filter recognition results and show image'''
    rgb1 = rgb.copy()  #
    mask_bw = np.zeros(rgb.shape[:2], dtype=np.uint8)
    symbols = []
    recog_txt_dict = dict()
    # symbols that are consists of numbers and one letter with high confidence,
    # potentially convert to texts in 'symbol_text_convert' function
    letter_number_symbols = dict()
    for (x, y, w, h), results in recognition_results.items():
        is_text = False
        for i in range(len(results['text'])):
            conf = results['conf'][i]
            txt = results['text'][i]
            # is text if confidence is high or the number of recognized letters is >=3
            if is_text_criteria(conf, txt):
                is_text = True
                break
        if is_text:  # if text_locs is not empty, i.e. it's text
            '''text split'''
            xt, wt = x, w
            #         xt,wt,symbols,_=text_split(x,y,w,h,gap_thres,conf_thres,results,symbols,symbol_color,
            #                                    image=rgb1,
            #                                    right_split=False,
            #                                   )
            mask_bw[y:y + h, xt:xt + wt] = 255
            recog_txt_dict[(x, y, w, h)] = results
            cv2.rectangle(rgb1, (xt, y), (xt + wt - 1, y + h - 1), text_color, 1)  # text
        else:
            if len(results['text']) == 1:
                conf = results['conf'][0]
                txt = results['text'][0]
                if letter_num_symbols_criteria(conf, txt):
                    letter_number_symbols[(x, y, w, h)] = results
            symbols.append((x, y, w, h))
            cv2.rectangle(rgb1, (x, y), (x + w - 1, y + h - 1), symbol_color, 1)  # symbol

    '''convert symbols and text'''
    if convert_sym_txt:
        symbols = symbol_text_convert(mask_bw, symbols, rgb1, rgb, text_color, symbol_color, letter_number_symbols)
    '''get text height'''
    text_heights = [k[3] for k in recog_txt_dict.keys()]
    txt_h = int(np.median(text_heights))
    return rgb1, mask_bw, symbols, recog_txt_dict, letter_number_symbols, txt_h


def check_result(loc,pix,OCR_results,rgb,whitelist):
    if pix!=0:
        print('warning: pix is not 0!')
    rgb_copy=rgb.copy()
    x,y=loc
    found=False
    for (x1,y1,w,h),v in OCR_results.items():
        if x1==x and y1==y:
            found=True
            # print('loc:',v['loc'])
            break
    if not found:
        print('not found')
        return
    print(f'AR={w/h}')
    x -= pix
    y -= pix
    w += 2 * pix
    h += 2 * pix
    cv2.rectangle(rgb_copy, (x, y), (x + w - 1, y + h - 1), (255, 255, 0), 1) # blue
    ocr_area=rgb[y:y+h, x:x+w]
    results = pytesseract.image_to_data(ocr_area, output_type=pytesseract.Output.DICT,
                                                config=f"-c tessedit_char_whitelist='{whitelist}' --psm 7")
    recognition_results = dict()
    loc, conf, text = [], [], []
    for i in range(len(results['conf'])):
        if int(results["conf"][i]) >= 0:  # record results that have none negative confidence
            loc.append((int(results['left'][i]), int(results['top'][i]),
                        int(results['width'][i]), int(results['height'][i])))
            conf.append(results['conf'][i])
            text.append(results['text'][i])
    recognition_results[(x, y, w, h)] = {'loc': loc, 'conf': conf, 'text': text}
    print(recognition_results)
    show_images(rgb_copy)

def text_split(x,y,w,h,gap_thres,conf_thres,results,symbols,symbol_color,image,right_split,criteria):
    text_locs=[]
    for i in range(len(results['text'])):
        conf = results['conf'][i]
        txt=results['text'][i]
        # is text if confidence is high or the number of recognized letters is >=3
        if criteria(conf,conf_thres,txt):
            text_locs.append(results['loc'][i])
    ######
    xt,wt=x,w
    left_gap=min(text_locs,key=lambda x:x[0])[0]
    rightmost=max(text_locs,key=lambda x:x[0]+x[2])
    right_gap=w-(rightmost[0]+rightmost[2])
    # return unchanged if both left and right gap > gap thres
    if left_gap>gap_thres and right_gap>gap_thres:
        return x,w,symbols,' '.join(results['text']),results
    if left_gap>gap_thres:
        xt+=left_gap
        wt-=left_gap
        ws=left_gap-1
        symbols.append((x,y,ws,h))
        cv2.rectangle(image, (x, y), (x + ws - 1, y + h - 1), symbol_color, 1) # symbol:red
    elif right_split and right_gap>gap_thres:
        wt-=right_gap
        xs=x+w-right_gap+1
        ws=right_gap-1
        symbols.append((xs,y,ws,h))
        cv2.rectangle(image, (xs, y), (xs + ws - 1, y + h - 1), symbol_color, 1) # symbol:red
    final_loc = []
    final_conf = []
    final_text=[]
    for i in range(len(results['text'])):
        loc=results['loc'][i]
        conf=results['conf'][i]
        txt = results['text'][i]
        if left_gap<=loc[0] and loc[0]+loc[2]<=rightmost[0]+rightmost[2]:
            final_loc.append(loc)
            final_conf.append(conf)
            final_text.append(txt)
    final_results={'loc':final_loc,'conf':final_conf,'text':final_text}
    final_text=' '.join(final_text)
    return xt,wt,symbols,final_text,final_results

def text_narrow(y,h,results,height_diff_thres):
    ## narrow text
    y_min=min(results['loc'],key=lambda x:x[1])[1]
    h_max_box=max(results['loc'],key=lambda x:x[1]+x[3]) # use max text height instead of bounding box height
    h_max=h_max_box[1]+h_max_box[3]
    if y_min>height_diff_thres or h-(y_min+h_max)>height_diff_thres:
        y+=y_min
        h=h_max-y_min
    return y,h

def symbol_text_convert(mask_bw,symbols,image,original_img,text_color,symbol_color,letter_number_symbols=None):
    '''
    convert symbol to text if they overlap with each other
    In the meantime, convert text to symbol if symbol contains text
    '''
    ## merge symbols if they contains each other
    symbols1=[]
    symbols.sort(key=lambda x: x[2]*x[3])
    for i in range(len(symbols)):
        x, y, w, h = symbols[i]
        symbols1.append((x, y, w, h))
        for j in range(i+1,len(symbols)):
            xl,yl,wl,hl=symbols[j]
            if xl<x and yl<y and x+w<xl+wl and y+h<yl+hl:
                image[y:y+h,x:x+w]=original_img[y:y+h,x:x+w]
                symbols1.pop()
                break


    symbols_filtered=[]
    for (x,y,w,h) in symbols1:
        mask_slice=mask_bw[y:y+h,x:x+w]
        mask_slice_small = mask_bw[(y + 1):(y + h - 1), (x + 1):(x + w - 1)]
        pos = cv2.findNonZero(mask_slice)
        if pos is None:
            symbols_filtered.append((x,y,w,h))
            continue
        pos1 = cv2.findNonZero(mask_slice_small)
        if pos1 is None:
            pos1=[]
        num_edge_pix=len(pos)-len(pos1)
        if num_edge_pix==0: # ==> symbol contains text, convert text to symbol
            symbols_filtered.append((x,y,w,h))
            mask_bw[y:y+h,x:x+w]=0 # delete text on mask
            cv2.rectangle(image, (x, y), (x + w - 1, y + h - 1), symbol_color, -1) # filled symbol
        else: # ==> symbol overlap text, convert symbol to text if w/h <=3
            if w/h>3:
                symbols_filtered.append((x,y,w,h))
            else:
                mask_bw[y:y+h,x:x+w]=255
                cv2.rectangle(image, (x, y), (x + w - 1, y + h - 1), text_color, -1) # filled text
    # symbols that are consists of numbers and one letter with high confidence,
    # convert to texts if there is text in its near left
    symbols_final=[]
    for (x, y, w, h) in symbols_filtered:
        if (x,y,w,h) in letter_number_symbols:
            left_most=x-w if x-w>=0 else 0
            mask_slice = mask_bw[y:y + h, left_most:x]
            pos = cv2.findNonZero(mask_slice)
            if pos is not None:
                mask_bw[y:y + h, x:x + w] = 255
                cv2.rectangle(image, (x, y), (x + w - 1, y + h - 1), text_color, -1)  # filled text
                continue
        symbols_final.append((x, y, w, h))
    return symbols_final


def round_odd(num):
    return int(num//2*2+1)

def get_search_box(x,y,w,h,Width,Height,w_max,w_s_h,w_s_v,perc,direction):
    '''
    w_s_h: horizontal search box width
    w_s_v: vertical search box height
    '''
    if not w_max:
        w_max=w
    h_s = int(h * perc)
    if direction == 'left':
        left = x - w_s_h if x - w_s_h > 0 else 0
        right = x
        top = y + (h - h_s) // 2
        bottom = top + h_s
    elif direction == 'right':
        left = x+w
        right = x + w_max + w_s_h if x + w_max + w_s_h < Width else Width
        top = y + (h - h_s) // 2
        bottom = top + h_s
    elif direction == 'up':
        left = x
        right = left + w
        top = y - w_s_v if y - w_s_v > 0 else 0
        bottom = y
    elif direction == 'down':
        left = x + (w - h_s) // 2
        right = left + h_s
        top=y+h
        bottom=y+h+w_s_v if y+h+w_s_v<Height else Height
    return left,right,top,bottom

def connect_bounding_box(mask,image,kernel,perc):
    mask1=mask.copy()
    Height,Width=image.shape[:2]
    '''search box width is in percentage.'''
    morph_h,morph_w=kernel
    _,_,stats,_ = cv2.connectedComponentsWithStats(mask, connectivity=8, ltype=cv2.CV_32S)
    for x,y,w,h,_ in stats:
        '''search right'''
        left,right,top,bottom=get_search_box(x,y,w,h,Width,Height,None,morph_w,morph_h,
                                             perc=perc,direction='right')
        mask_slice=mask[top:bottom, left:right]
        pos=cv2.findNonZero(mask_slice)
        if pos is not None:
            left_most=min(pos,key=lambda i:i[0,0])[0]
            cv2.line(mask1, (x+w,top+left_most[1]), (left+left_most[0],top+left_most[1]), 255, 1)
        '''search left'''
        left,right,top,bottom=get_search_box(x,y,w,h,Width,Height,None,morph_w,morph_h,
                                             perc=perc,direction='left')
        mask_slice=mask[top:bottom, left:right]
        pos=cv2.findNonZero(mask_slice)
        if pos is not None:
            right_most=max(pos,key=lambda i:i[0,0])[0]
            cv2.line(mask1,(x,top+right_most[1]),(left+right_most[0],top+right_most[1]),255,1)
        '''search up'''
        left,right,top,bottom=get_search_box(x,y,w,h,Width,Height,None,morph_w,morph_h,
                                             perc=perc,direction='up')
        mask_slice=mask[top:bottom, left:right]
        pos=cv2.findNonZero(mask_slice)
        if pos is not None:
            bottom_most=max(pos,key=lambda i:i[0,1])[0]
            cv2.line(mask1,(left+bottom_most[0],y),(left+bottom_most[0],top+bottom_most[1]),255,1)
        '''search down'''
        left,right,top,bottom=get_search_box(x,y,w,h,Width,Height,None,morph_w,morph_h,
                                             perc=perc,direction='down')
        mask_slice=mask[top:bottom, left:right]
        pos=cv2.findNonZero(mask_slice)
        if pos is not None:
            top_most=min(pos,key=lambda i:i[0,1])[0]
            cv2.line(mask1,(left+top_most[0],y),(left+top_most[0],top+top_most[1]),255,1)
    return mask1

def symbol_search(rgb,texts,symbols,box_size,direction,search_box_color):
    '''
    box_size: tuple of two elements:
    First element: search box width in pixel.
    Second element: search box height in percentage of text height
    '''
    # text mask
    rgb_copy=rgb.copy()
    Width=rgb.shape[1]
    mask = np.zeros(rgb.shape[:2])
    i=1
    w_max=max(texts,key=lambda x:x[2])[2]
    w_s,perc=box_size
    for x,y,w,h,_ in texts:
        left,right,top,bottom=get_search_box(x, y, w, h, Width, None, w_max, w_s, w_s, perc, direction)
        mask[top:bottom, left:right] = i
        i+=1
        cv2.rectangle(rgb_copy, (left,top), (right - 1, bottom - 1), search_box_color, 1)
    ##
    text_symbol=dict()
    for (x,y,w,h) in symbols:
        ## check if overlaps with search box, then link texts with symbols
        mask_slice=mask[y:y+h,x:x+w]
        pos=cv2.findNonZero(mask_slice) # find non zero positions
        if pos is not None:
            txt=texts[int(statistics.mode([mask_slice[y,x] for [[x,y]] in pos]))-1]
            # merge symbols if their center distance <= 50
            if txt not in text_symbol.keys():
                text_symbol[txt]=(x,y,w,h)
            elif center_dist(text_symbol[txt], (x,y,w,h))<=50:
                text_symbol[txt]=merge_box(text_symbol[txt],(x,y,w,h))
            else:# if there are two symbols and their distances > 50, choose the one closest to text
                x_old,y_old,w_old,h_old=text_symbol[txt]
                if direction == 'left':# if search direction is left, choose the symbol with largest x
                    text_symbol[txt]=max((x_old,y_old,w_old,h_old),(x,y,w,h),key=lambda x: x[0])
                elif direction == 'right':
                    text_symbol[txt] = min((x_old, y_old, w_old, h_old), (x, y, w, h), key=lambda x: x[0])
                elif direction=='up':# if search up, choose the symbol with largest y
                    text_symbol[txt] = max((x_old, y_old, w_old, h_old), (x, y, w, h), key=lambda x: x[1])

    text_symbol=[(k, v) for k, v in text_symbol.items()]
    ## detele overlapped text_symbol pairs
    filtered_text_symbol=[]
    overlapped_indices = set()
    for i in range(len(text_symbol)):
        loc1=merge_box(text_symbol[i][0][:-1],text_symbol[i][1]) # merge text and symbol
        not_overlap=True
        for j in range(i+1,len(text_symbol)):
            loc2=merge_box(text_symbol[j][0][:-1],text_symbol[j][1])
            if jaccard(loc1,loc2)>0.1:
                not_overlap=False
                overlapped_indices.add(i)
                overlapped_indices.add(j)
                break
        if not_overlap and i not in overlapped_indices:
            filtered_text_symbol.append(text_symbol[i])
    filtered_text_symbol=dict(filtered_text_symbol)

    # loop through confirmed symbols again, connect text if multiple texts overlap the same symbol
    symbols_filtered = [v for k, v in filtered_text_symbol.items()]
    for (x, y, w, h) in symbols_filtered:
        mask_slice = mask[y:y + h, x:x + w]
        pos = cv2.findNonZero(mask_slice)  # find non zero positions
        if pos is not None:
            text_indices = {int(mask_slice[y, x]) for [[x, y]] in pos}
            if len(text_indices) > 1:
                txts = [texts[i - 1] for i in text_indices]  # element is in the form of (x,y,w,h,'text')
                old_symbol = None
                for txt in txts:  # delete the key-value pair in text_symbol dict
                    popped = filtered_text_symbol.pop(txt, None)
                    if popped:
                        old_symbol = popped
                txt = merge_text(txts)
                if old_symbol:
                    filtered_text_symbol[txt] = old_symbol
    final_text_symbol=[]
    for text,symbol in filtered_text_symbol.items():
        xt,yt,wt,ht,txt=text
        xs,ys,ws,hs=symbol
        if (direction=='up' and xs<xt+wt//2<xs+ws) or ys<yt+ht//2<ys+hs:
            final_text_symbol.append((text,symbol))
    ## plot
    for text,symbol in final_text_symbol:
        txt=text[-1] # the actual text
        x,y,w,h=symbol
        cv2.putText(rgb_copy,
                f'{txt}',
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (0, 0, 0), 1)
    symbol_centers=[left_right_center(i[1], i[1]) for i in final_text_symbol]
    text_centers=[left_right_center(i[0], i[0]) for i in final_text_symbol]
    return rgb_copy,mask,final_text_symbol,text_centers,symbol_centers

def center_dist(rect1,rect2):
    x1,y1,w1,h1=rect1
    x2,y2,w2,h2=rect2
    dx=x1+w1//2-(x2+w2//2)
    dy=y1+h1//2-(y2+h2//2)
    return np.sqrt(dx*dx+dy*dy)

def merge_box(rect1,rect2):
    x1,y1,w1,h1=rect1
    x2,y2,w2,h2=rect2
    x=min(x1,x2)
    y=min(y1,y2)
    w=max(x1+w1,x2+w2)-x
    h=max(y1+h1,y2+h2)-y
    return (x,y,w,h)
def merge_text(texts):
    '''
    texts: a list of (x,y,w,h,'txt')
    '''
    x=min(texts,key=lambda i:i[0])[0]
    y=min(texts,key=lambda i:i[1])[1]
    right_most=max(texts,key=lambda i:i[0]+i[2])
    w=right_most[0]+right_most[2]-x
    bottom_most=max(texts,key=lambda i:i[1]+i[3])
    h=bottom_most[1]+bottom_most[3]-y
    texts.sort(key=lambda i:i[1]) # sort based on y
    txt=' '.join([i[-1] for i in texts])
    return (x,y,w,h,txt)


def left_right_center(text,symbol):
    xt,yt,wt,ht=text[:4]
    xs,ys,ws,hs=symbol[:4]
    left=min(xt,xs)
    top=min(yt,ys)
    right=max(xt+wt,xs+ws)-1
    bottom=max(yt+ht,ys+hs)-1
    return [left,right,int((left+right)//2),int((top+bottom)//2),top,bottom]

def remove_outliers(n_neighbors,direction,centers,sym_centers,text_symbol,txt_h,ratio):
    ctr=[[i[2],i[3]] for i in centers]
    if len(ctr)<n_neighbors:
        return dict()
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree',n_jobs=-1,p=1).fit(ctr)
    distances, indices = nbrs.kneighbors(ctr)
    output=dict()
    count=0
    max_dist_thres=max(txt_h*15,2.5*min(distances,key=lambda x: x[1])[1])
    for i in range(len(indices)):
        index=[]
        for j in range(len(indices[i])):
            if distances[i][j]<=max_dist_thres:
                index.append(indices[i][j])
        if all(distances[i][1:]>max_dist_thres):
            continue
        indx0=index[0]
        '''diffs for text'''
        left0,right0,top0,bottom0=centers[indx0][0],centers[indx0][1],centers[indx0][4],centers[indx0][5]
        xc0,yc0=centers[indx0][2],centers[indx0][3]
        left_diff=[abs(centers[k][0]-left0) for k in index[1:]]
        right_diff=[abs(centers[k][1]-right0) for k in index[1:]]
        top_diff=[abs(centers[k][4]-top0) for k in index[1:]]
        bottom_diff=[abs(centers[k][5]-bottom0) for k in index[1:]]
        xc_diff = [abs(centers[k][2] - xc0) for k in index[1:]]
        yc_diff=[abs(centers[k][3]-yc0) for k in index[1:]]
        '''diffs for symbols'''
        left0_sym, right0_sym, top0_sym, bottom0_sym=\
            sym_centers[indx0][0],sym_centers[indx0][1],sym_centers[indx0][4],sym_centers[indx0][5]
        xc0_sym,yc0_sym = sym_centers[indx0][2],sym_centers[indx0][3]
        left_diff_sym = [abs(sym_centers[k][0] - left0_sym) for k in index[1:]]
        top_diff_sym=[abs(sym_centers[k][4]-top0_sym) for k in index[1:]]
        bottom_diff_sym=[abs(sym_centers[k][5]-bottom0_sym) for k in index[1:]]
        xc_diff_sym = [abs(sym_centers[k][2] - xc0_sym) for k in index[1:]]
        yc_diff_sym=[abs(sym_centers[k][3]-yc0_sym) for k in index[1:]]
        '''diffs for center x of symbol to left of text'''
        xc_sym_left_diff=[abs((sym_centers[k][2]-centers[k][0])-(xc0_sym-left0)) for k in index[1:]]

        if direction=='left':
            diffs=left_diff+top_diff+xc_diff+yc_diff
            diffs_sym=left_diff_sym+top_diff_sym+xc_diff_sym+yc_diff_sym
        elif direction=='right':
            diffs=left_diff+top_diff+xc_diff+yc_diff
            diffs_sym = left_diff_sym + top_diff_sym+xc_diff_sym+yc_diff_sym
        elif direction=='up':
            diffs=bottom_diff+left_diff+xc_diff+yc_diff
            diffs_sym=bottom_diff_sym+left_diff_sym+xc_diff_sym+yc_diff_sym
        ################### test
        # if i == 0:
        #     print('stairs index', index)
        #     print('stairs dist', distances[i])
        #     print('stairs====xc0_sym', xc0_sym)
        #     print('stairs====left0', left0)
        #     print('exit====xc_sym', sym_centers[1][2])
        #     print('exit====left', centers[1][0])
        #     print('xc_sym_left_diff',xc_sym_left_diff)
        #     print('diffs',diffs)
        # else:
        #     break
            ################
        for k in range(len(diffs)):
            if diffs[k]*ratio<12 and diffs_sym[k] * ratio < 12: # if text diff<10
                # if search left or right, diffs for center x of symbol to left of text should also <15
                # else, continue
                if (direction=='left' or direction=='right') and xc_sym_left_diff[k%(len(index)-1)]*ratio>=15:
                    continue
                (xt,yt,wt,ht,txt),(xs,ys,ws,hs)=text_symbol[i]
                output[count]={
                    'symbol_loc':{'x':int(round((xs)*ratio)),'y':int(round((ys)*ratio)),
                                  'w':int(round((ws)*ratio)),'h':int(round((hs)*ratio))},
                    'text_loc':{'x':int(round((xt)*ratio)),'y':int(round((yt)*ratio)),
                                'w':int(round((wt)*ratio)),'h':int(round((ht)*ratio))},
                    'text':txt
                }
                count+=1
                break
    return output

def plot_legends(truth,pred,rgb_original,text_color,symbol_color):
    ###
    rgb4=rgb_original.copy()//2
    for _,legend in pred.items():
        symbol_loc,text_loc,txt=legend['symbol_loc'],legend['text_loc'],legend['text']
        xs,ys,ws,hs=symbol_loc['x'],symbol_loc['y'],symbol_loc['w'],symbol_loc['h']
        xt,yt,wt,ht=text_loc['x'],text_loc['y'],text_loc['w'],text_loc['h']
        rgb4[yt:yt+ht,xt:xt+wt]=rgb_original[yt:yt+ht,xt:xt+wt]
        rgb4[ys:ys+hs,xs:xs+ws]=rgb_original[ys:ys+hs,xs:xs+ws]
        cv2.rectangle(rgb4, (xt, yt), (xt + wt-1, yt + ht-1), text_color, 1) # green
        cv2.rectangle(rgb4, (xs, ys), (xs + ws-1, ys + hs-1), symbol_color, 1) # red
        cv2.putText(rgb4,f'{txt}',(xs, ys),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1) # black
    ###
    for _,legend in truth.items():
        symbol_loc,text_loc,txt=legend['symbol_loc'],legend['text_loc'],legend['text']
        xs,ys,ws,hs=symbol_loc['x'],symbol_loc['y'],symbol_loc['w'],symbol_loc['h']
        xt,yt,wt,ht=text_loc['x'],text_loc['y'],text_loc['w'],text_loc['h']
        cv2.rectangle(rgb4, (xt, yt), (xt + wt-1, yt + ht-1), symbol_color, 1) # reverse color
        cv2.rectangle(rgb4, (xs, ys), (xs + ws-1, ys + hs-1), text_color, 1) # reverse color
        cv2.putText(rgb4,f'{txt}',(xs, ys-10),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1) # black
    show_images(rgb4)

def jaccard(loc1:tuple,loc2:tuple):
    x1,y1,w1,h1=loc1
    x2,y2,w2,h2=loc2
    rec1=(x1,y1,x1+w1-1,y1+h1-1)
    rec2=(x2,y2,x2+w2-1,y2+h2-1)
    if rec2[0]>rec1[2] or rec2[2]<rec1[0] or rec2[1]>rec1[3] or rec2[3]<rec1[1]:
        return 0
    dx=min(rec1[2],rec2[2])-max(rec1[0],rec2[0])
    dy=min(rec1[3],rec2[3])-max(rec1[1],rec2[1])
    intersect_area=(dx+1)*(dy+1)
    union_area=w1*h1+w2*h2-intersect_area
    return intersect_area/union_area

def check_contains(loc_true,loc_pred):
    '''check if predicted bounding box contains true bounding box'''
    xt,yt,wt,ht=loc_true
    xp,yp,wp,hp=loc_pred
    if xt>=xp and yt>=yp and xt+wt<=xp+wp and yt+ht<=yp+hp:
        return True
def calc_confusion_matrix(df,truth,pred,thres,key):
    truth_match, pred_match, text_truth_match, text_pred_match = set(),set(),0,0
    partially_matched_pred=set()
    for i in truth:
        for j in pred:
            '''check bounding box Jaccard Index'''
            JI=jaccard(i[key],j[key])
            if JI==0:
                continue
            if JI>=thres or check_contains(i[key],j[key]):
                # df.loc[key,'TP']+=1
                pred_match.add(j[key])
            elif 0<JI<thres: # partially match, False positive
                if j[key] not in partially_matched_pred:
                    df.loc[key,'FP']+=1
                    pred_match.add(j[key])
                partially_matched_pred.add(j[key])
            truth_match.add(i[key])
            '''also do text similarity check if key is text_loc'''
            if key=='text_loc':
                sim=SequenceMatcher(None, i['text'].lower(), j['text'].lower()).ratio()
                if sim==0:
                    continue
                if 0<sim<thres:
                    df.loc['text','FP']+=1
                elif sim>=thres:
                    df.loc['text','TP']+=1
                text_truth_match+=1
                text_pred_match+=1
    df.loc[key,'TP']+=len(pred_match)
    df.loc[key,'FP']+=(len(pred)-len(pred_match))
    df.loc[key,'FN']+=(len(truth)-len(truth_match))
    if key=='text_loc':
        df.loc['text','FP']+=(len(pred)-text_pred_match)
        df.loc['text','FN']+=(len(truth)-text_truth_match)


def main(filename):
    text_color = (0, 255, 0)  # text
    symbol_color = (0, 0, 255)  # symbol
    search_box_color = (255, 0, 0)  # search box
    '''read files'''
    rgb, gray, ratio, rgb_original = read_image(f'processed/{filename}', resized_diagonal=2500)
    Height, Width = rgb.shape[:2]
    canny = cv2.Canny(gray, 50, 100)  # detect edge

    '''set OCR whitelist'''
    whitelist = set_whitelist(print_out=False)  # set OCR white list

    '''morph must use odd number!!'''
    filled = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, np.ones((1, 5), dtype=int))
    '''Initial OCR'''
    # lower limit of connected component bounding box is 2.5/ratio, so that the bounding box in the original image
    # is >2
    recognition_results = CC_n_OCR(bw=filled,
                                   rgb=rgb,
                                   whitelist=whitelist,
                                   recog_txt_dict=dict(),
                                   pix=1,
                                   desc='Initial OCR',
                                   criteria=lambda w, h: (max(2.5 / ratio, 6) <= w <= 120 and max(2.5 / ratio,
                                                                                                  6) <= h <= 120) or \
                                                         (w / h >= 2 and max(2.5 / ratio, 3) < h <= 80),
                                   psm=7
                                   )
    if ratio < 0.4:
        conf_thres = 50
    else:
        conf_thres = 80
    gap_thres = 15
    is_text_criteria0 = lambda conf, txt: (conf > conf_thres and sum(c.isalpha() for c in txt) > 1) or \
                                          sum(c.isalpha() for c in txt) >= 4 or \
                                          (conf > conf_thres and len(txt) >= 4) or \
                                          (conf > conf_thres and txt in {'A', 'a'}) and \
                                          not re.search(r'(([a-km-z])\2{2,})', txt.lower()) and \
                                          not re.search(r'(([l])\2{3,})', txt.lower())
    # symbols that are consists of numbers and one letter with high confidence,
    # potentially convert to texts in 'symbol_text_convert' function
    letter_num_symbols_criteria = lambda conf, txt: conf > conf_thres and (txt.isalnum() or txt in {'&', '/'})
    '''filter recognition results and show image'''
    rgb1, mask_bw, symbols, recog_txt_dict, letter_number_symbols, txt_h = \
        initial_result_filter(rgb, recognition_results, is_text_criteria0, letter_num_symbols_criteria,
                              text_color, symbol_color, True)

    '''re-OCR with larger morph length if txt_h is too high'''
    if txt_h > 30:
        filled = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, np.ones((1, round_odd(txt_h)), dtype=int))
        recognition_results_new = CC_n_OCR(bw=filled,
                                           rgb=rgb,
                                           whitelist=whitelist,
                                           recog_txt_dict=dict(),
                                           pix=2,
                                           desc=f'txt_h>30, OCR again',
                                           criteria=lambda w, h: (max(2.5 / ratio, 6) <= w <= 120 and max(2.5 / ratio,
                                                                                                          6) <= h <= 120) or \
                                                                 (w / h >= 2 and max(2.5 / ratio, 3) < h <= 80),
                                           psm=7
                                           )
    '''filter re-OCR recognition results and show image'''
    rgb1, mask_bw, symbols, recog_txt_dict, letter_number_symbols, txt_h = \
        initial_result_filter(rgb, recognition_results, is_text_criteria0, letter_num_symbols_criteria,
                              text_color, symbol_color, True)

    '''connect texts'''
    # if txt_h<=15:
    #     morph_h=3
    # elif 15<txt_h<=22:
    #     morph_h=5
    # else:
    #     morph_h=7
    morph_h = 1
    kernel = (morph_h, int(round(1.7 * txt_h)))
    print(f'text_h:{txt_h}, morph_h:{morph_h},morph_v:{kernel[1]}')
    mask_bw1 = connect_bounding_box(mask_bw, rgb, kernel=kernel, perc=0.5)

    '''OCR again'''
    recognition_results1 = CC_n_OCR(bw=mask_bw1,
                                    rgb=rgb,
                                    whitelist=whitelist,
                                    recog_txt_dict=recog_txt_dict,
                                    pix=0,
                                    desc='Final OCR',
                                    criteria=lambda w, h: Width > w and Height > h,
                                    psm=6
                                    )
    '''filter recognition results and show image'''
    rgb2 = rgb1.copy()
    texts = []
    mask_bw2 = np.zeros(rgb.shape[:2], dtype=np.uint8)
    # no 3+(incl.) chars except l
    # no 4+ (inclusive) l's
    is_text_criteria = lambda conf, conf_thres, txt: (conf > conf_thres / 2 and sum(c.isalpha() for c in txt) > 1) or \
                                                     (sum(c.isalpha() for c in txt) >= 3) or \
                                                     (conf > conf_thres and len(txt) >= 4) or \
                                                     (conf > conf_thres and txt in {'A', 'a'}) and \
                                                     not re.search(r'(([a-km-z])\2{2,})', txt.lower()) and \
                                                     not re.search(r'(([l])\2{3,})', txt.lower())
    for (x, y, w, h), results in recognition_results1.items():
        is_text = False
        for i in range(len(results['text'])):
            conf = int(results['conf'][i])
            txt = results['text'][i]
            # check if it's text
            if is_text_criteria0(conf, txt):
                is_text = True
                break

        if is_text:
            '''text split'''
            x, w, symbols, recognized_text, results = text_split(x, y, w, h, gap_thres, conf_thres, results, symbols,
                                                                 symbol_color,
                                                                 image=rgb2,
                                                                 right_split=False,
                                                                 criteria=is_text_criteria
                                                                 )
            '''text narrow'''
            y, h = text_narrow(y, h, results, height_diff_thres=3)
            mask_bw2[y:y + h, x:x + w] = 255
            texts.append((x, y, w, h, recognized_text))
            cv2.rectangle(rgb2, (x, y), (x + w - 1, y + h - 1), (0, 0, 0), 1)  # black
    #         cv2.putText(rgb2,
    #             f'{recognized_text}',
    #             (x, y),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             0.4, (0, 0, 0), 1)
    '''convert symbols and text'''
    symbols = symbol_text_convert(mask_bw2, symbols, rgb2, rgb, text_color, symbol_color, dict())

    '''search and link'''
    '''serach left'''
    direction = 'left'
    rgb3_left, search_mask, text_symbol_left, text_centers, symbol_centers = symbol_search(rgb2, texts, symbols,
                                                                                           box_size=(7 * txt_h, 1),
                                                                                           direction=direction,
                                                                                           search_box_color=search_box_color)

    output_left = remove_outliers(4, direction, text_centers, symbol_centers, text_symbol_left, txt_h, ratio)

    '''search right'''
    direction = 'right'
    rgb3_right, search_mask, text_symbol_right, text_centers, symbol_centers = symbol_search(rgb2, texts, symbols,
                                                                                             box_size=(5 * txt_h, 1),
                                                                                             direction=direction,
                                                                                             search_box_color=search_box_color)
    output_right = remove_outliers(4, direction, text_centers, symbol_centers, text_symbol_right, txt_h, ratio)
    '''search up'''
    direction = 'up'
    rgb3_up, search_mask, text_symbol_up, text_centers, symbol_centers = symbol_search(rgb2, texts, symbols,
                                                                                       box_size=(4 * txt_h, 1),
                                                                                       direction=direction,
                                                                                       search_box_color=search_box_color)

    ##
    # text_symbol_up=[text_symbol_up[5],text_symbol_up[4],text_symbol_up[3],text_symbol_up[2]]
    # text_centers=[text_centers[5],text_centers[4],text_centers[3],text_centers[2]]
    # symbol_centers=[symbol_centers[5],symbol_centers[4],symbol_centers[3],symbol_centers[2]]
    ##
    output_up = remove_outliers(4, direction, text_centers, symbol_centers, text_symbol_up, txt_h, ratio)
    output, rgb3 = max((output_left, rgb3_left), (output_right, rgb3_right), (output_up, rgb3_up),
                       key=lambda x: len(x[0].keys()))
    # plot on original image
    rgb4 = rgb_original.copy() // 2
    for _, legend in output.items():
        symbol_loc, text_loc, txt = legend['symbol_loc'], legend['text_loc'], legend['text']
        xs, ys, ws, hs = symbol_loc['x'], symbol_loc['y'], symbol_loc['w'], symbol_loc['h']
        xt, yt, wt, ht = text_loc['x'], text_loc['y'], text_loc['w'], text_loc['h']
        rgb4[yt:yt + ht, xt:xt + wt] = rgb_original[yt:yt + ht, xt:xt + wt]
        rgb4[ys:ys + hs, xs:xs + ws] = rgb_original[ys:ys + hs, xs:xs + ws]
        cv2.rectangle(rgb4, (xt, yt), (xt + wt - 1, yt + ht - 1), text_color, 1)  # text
        cv2.rectangle(rgb4, (xs, ys), (xs + ws - 1, ys + hs - 1), symbol_color, 1)  # symbol
        cv2.putText(rgb4, f'{txt}', (xs, ys), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)  # black
    '''save output'''
    with open(f'label_pred/{filename}.json', 'w') as fi:
        json.dump(output, fi)
    '''evaluate performance'''
    ## read truth and prediction
    truth_fn = f'label_truth/{filename}.json'
    pred_fn = f'label_pred/{filename}.json'
    with open(truth_fn) as f:
        truth = json.load(f)
    with open(pred_fn) as f:
        pred = json.load(f)
    ## plot
    # plot_legends(truth,pred,rgb_original,text_color,symbol_color)

    truth = [{'symbol_loc': (v['symbol_loc']['x'], v['symbol_loc']['y'], v['symbol_loc']['w'], v['symbol_loc']['h']),
              'text_loc': (v['text_loc']['x'], v['text_loc']['y'], v['text_loc']['w'], v['text_loc']['h']),
              'text': v['text']} for k, v in truth.items()]

    pred = [{'symbol_loc': (v['symbol_loc']['x'], v['symbol_loc']['y'], v['symbol_loc']['w'], v['symbol_loc']['h']),
             'text_loc': (v['text_loc']['x'], v['text_loc']['y'], v['text_loc']['w'], v['text_loc']['h']),
             'text': v['text']} for k, v in pred.items()]
    thres = 0.7
    df = pd.DataFrame(np.zeros((3, 3), dtype=int), index=['symbol_loc', 'text_loc', 'text'])
    df.columns = ['TP', 'FP', 'FN']
    '''check if symbol_loc, text_loc and text match'''
    calc_confusion_matrix(df, truth, pred, thres, 'symbol_loc')
    calc_confusion_matrix(df, truth, pred, thres, 'text_loc')
    return df




if __name__ == '__main__':
    pass


