import json
import math
import re
import statistics
import sys
import warnings
from difflib import SequenceMatcher

import imutils
import numpy as np
import pandas as pd
import pytesseract
from cv2 import cv2
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

def set_whitelist(print_out):
    special_char = ' /&$(),-.!:>*'
    numbers_char = '0123456789'
    lower_char = 'abcdefghijklmnopqrstuvwxyz'
    upper_char = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    french_char = 'é'  # as seen in the word 'café',otherwise it will be recognized as 'cafe' with low confidence
    custom_char = special_char + numbers_char + lower_char + upper_char + french_char
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
        {lower_char + upper_char}
        '''
        )
    return custom_char


def read_image(filename, resized_diagonal=None):
    rgb = None
    extension = {'png', 'jpg', 'jpeg'}
    if len(filename.split('.')) == 2:
        filename, ext = filename.split('.')
    if ext.lower() in extension:
        rgb = cv2.imread(f'{filename}.{ext}')  # read to greyscale
    elif ext == 'gif':
        cap = cv2.VideoCapture(f'{filename}.gif')
        _, rgb = cap.read()
        cap.release()
    if rgb is None:
        warnings.warn('file type not supported')
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    if resized_diagonal is not None:
        h, w, _ = rgb.shape
        resized_width = int(resized_diagonal * w / math.sqrt(w ** 2 + h ** 2))
        resized_rgb = imutils.resize(rgb, width=resized_width)
        resized_gray = imutils.resize(gray, width=resized_width)
        return resized_rgb, resized_gray, w / resized_width, rgb, filename.split('/')[-1]
    else:
        return rgb, gray, 1, rgb, filename.split('/')[-1]


def show_images(filename, **args):
    try:
        for i in args.keys():
            cv2.namedWindow(f'{filename} {i}', cv2.WINDOW_NORMAL)
            cv2.imshow(f'{filename} {i}', args[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except TypeError:
        print('show_images: wrong input type!')
        cv2.destroyAllWindows()


def canny_filled(gray):
    Height, Width = gray.shape[:2]
    canny = cv2.Canny(gray, 50, 100)  # detect edge
    '''morph must use odd number!!'''
    filled = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, np.ones((1, 5), dtype=int))
    '''extract CC whose area < 16 and calculate'''
    _, _, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8, ltype=cv2.CV_32S)
    counter = 0
    filtered_stats_x = []
    filtered_stats_y = []
    for x, y, w, h, A in stats:
        if A < 16:
            filtered_stats_x.append(x / Width)
            filtered_stats_y.append(y / Height)
            counter += 1
    sd_x = np.std(filtered_stats_x) * 100
    sd_y = np.std(filtered_stats_y) * 100
    if counter > 800 and min(sd_x, sd_y) >= 20:
        canny = cv2.Canny(gray, 100, 200)
    return canny


def CC_n_OCR(bw, rgb, whitelist, recog_txt_dict, pix, desc, criteria, psm):
    Height, Width = rgb.shape[:2]
    _, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8, ltype=cv2.CV_32S)
    recognition_results = dict()
    stats_filter = []
    for x, y, w, h, A in stats:
        if (x, y, w, h) in recog_txt_dict.keys():
            results = recog_txt_dict[(x, y, w, h)]
            loc, conf, text = results['loc'], results['conf'], results['text']
            recognition_results[(x, y, w, h)] = {'loc': loc, 'conf': conf, 'text': text}
        elif criteria(w, h):
            stats_filter.append((x, y, w, h, A))
    for x, y, w, h, A in tqdm(stats_filter, ncols=100, desc=desc):
        if x >= pix and y >= pix and x + w <= Width + 1 and y + h <= Height + 1:
            x -= pix
            y -= pix
            w += 2 * pix
            h += 2 * pix
        ocr_area = rgb[y:y + h, x:x + w]
        if (x, y, w, h) in recog_txt_dict.keys():
            results = recog_txt_dict[(x, y, w, h)]
            loc, conf, text = results['loc'], results['conf'], results['text']
        else:
            results = pytesseract.image_to_data(ocr_area, output_type=pytesseract.Output.DICT,
                                                config=f"-c tessedit_char_whitelist='{whitelist}' --psm {psm}")
            loc, conf, text = [], [], []
            for i in range(len(results['conf'])):
                if int(results["conf"][i]) >= 0:  # record results that have none negative confidence
                    loc.append((int(results['left'][i]), int(results['top'][i]),
                                int(results['width'][i]), int(results['height'][i])))
                    conf.append(results['conf'][i])
                    text.append(results['text'][i])
        recognition_results[(x, y, w, h)] = {'loc': loc, 'conf': conf, 'text': text}
    return recognition_results


def initial_result_filter(rgb, recognition_results, is_text_criteria, letter_num_symbols_criteria,
                          text_color, symbol_color, convert_sym_txt, txt_h=None):
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
            if is_text_criteria(conf, txt, h, txt_h):
                is_text = True
                break
        if is_text:  # if text_locs is not empty, i.e. it's text
            # narrow the text if it's just adjacent to any box in its near top or bottom. see 66
            x, y, w, h = format_text_box((x, y, w, h), mask_bw)
            mask_bw[y:y + h, x:x + w] = 255
            recog_txt_dict[(x, y, w, h)] = results
            cv2.rectangle(rgb1, (x, y), (x + w - 1, y + h - 1), text_color, 1)  # text
        else:
            if len(results['text']) == 1:
                conf = results['conf'][0]
                txt = results['text'][0]
                if letter_num_symbols_criteria(conf, txt):
                    letter_number_symbols[(x, y, w, h)] = results
            symbols.append((x, y, w, h))
            cv2.rectangle(rgb1, (x, y), (x + w - 1, y + h - 1), symbol_color, 1)  # symbol

    '''get text height'''
    text_heights = [k[3] for k in recog_txt_dict.keys()]
    txt_h = int(np.median(text_heights))

    '''convert symbols and text'''
    if convert_sym_txt:
        symbols = symbol_text_convert(mask_bw, symbols, rgb1, rgb, text_color, symbol_color, txt_h,
                                      letter_number_symbols)

    return rgb1, mask_bw, symbols, recog_txt_dict, letter_number_symbols, txt_h


def format_text_box(text_loc, mask):
    Height = mask.shape[0]
    x, y, w, h = text_loc
    mask_slice = mask[y:y + h, x:x + w]
    pos = cv2.findNonZero(mask_slice)
    if pos is None:
        if y == 0:
            return x, y, w, h
        slice_top = mask[y - 1:y, x:x + w]
        pos = cv2.findNonZero(slice_top)
        if pos is not None:
            y += 1
            h -= 1

        if y + h - 1 == Height:
            return x, y, w, h
        slice_bottom = mask[y + h:y + h + 1, x:x + w]
        pos = cv2.findNonZero(slice_bottom)
        if pos is not None:
            h -= 1
    return x, y, w, h


def check_result(loc, pix, OCR_results, rgb, whitelist, filename):
    if pix != 0:
        print('warning: pix is not 0!')
    rgb_copy = rgb.copy()
    x, y = loc
    found = False
    for (x1, y1, w, h), v in OCR_results.items():
        if x1 == x and y1 == y:
            found = True
            # print('loc:',v['loc'])
            break
    if not found:
        print('not found')
        return
    print(f'AR={w / h}')
    x -= pix
    y -= pix
    w += 2 * pix
    h += 2 * pix
    cv2.rectangle(rgb_copy, (x, y), (x + w - 1, y + h - 1), (255, 255, 0), 1)  # blue
    ocr_area = rgb[y:y + h, x:x + w]
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
    show_images(filename, check_result=rgb_copy)


def text_symbol_refit(symbol_loc, bw):
    xs, ys, ws, hs = symbol_loc
    _, _, stats, _ = cv2.connectedComponentsWithStats(bw[ys:ys + hs, xs:xs + ws], connectivity=8, ltype=cv2.CV_32S)
    filtered_stats = []
    for x, y, w, h, _ in stats:
        if x < 0 or x == 0 and y == 0 and w == ws and h == hs:
            continue
        filtered_stats.append((x, y, w, h))
    if len(filtered_stats) == 0:
        return symbol_loc
    final_symbol = filtered_stats[0]
    for x, y, w, h in filtered_stats:
        final_symbol = merge_box(final_symbol, (x, y, w, h))
    x, y, w, h = final_symbol
    return (x + xs, y + ys, w, h)


def text_split(x, y, w, h, gap_thres, conf_thres, results, symbols, symbol_color, image, bw, right_split, criteria,
               txt_h=None):
    text_locs = []
    for i in range(len(results['text'])):
        conf = results['conf'][i]
        txt = results['text'][i]
        # is text if confidence is high or the number of recognized letters is >=3
        if criteria(conf, txt, h, txt_h):
            text_locs.append(results['loc'][i])
    ######
    xt, wt = x, w
    left_gap = min(text_locs, key=lambda x: x[0])[0]
    rightmost = max(text_locs, key=lambda x: x[0] + x[2])
    right_gap = w - (rightmost[0] + rightmost[2])
    # return unchanged if both left and right gap > gap thres
    if left_gap > gap_thres and right_gap > gap_thres:
        return x, w, symbols, ' '.join(results['text']), results
    if left_gap > gap_thres:
        xt += left_gap
        wt -= left_gap
        ws = left_gap - 1
        x, y, ws, h = text_symbol_refit((x, y, ws, h), bw)
        symbols.append((x, y, ws, h))
        cv2.rectangle(image, (x, y), (x + ws - 1, y + h - 1), symbol_color, 1)  # symbol:red
    elif right_split and right_gap > gap_thres:
        wt -= right_gap
        xs = x + w - right_gap + 1
        ws = right_gap - 1
        xs, y, ws, h = text_symbol_refit((xs, y, ws, h), bw)
        symbols.append((xs, y, ws, h))
        cv2.rectangle(image, (xs, y), (xs + ws - 1, y + h - 1), symbol_color, 1)  # symbol:red
    final_loc = []
    final_conf = []
    final_text = []
    for i in range(len(results['text'])):
        loc = results['loc'][i]
        conf = results['conf'][i]
        txt = results['text'][i]
        if left_gap <= loc[0] and loc[0] + loc[2] <= rightmost[0] + rightmost[2]:
            final_loc.append(loc)
            final_conf.append(conf)
            final_text.append(txt)
    final_results = {'loc': final_loc, 'conf': final_conf, 'text': final_text}
    final_text = ' '.join(final_text)
    return xt, wt, symbols, final_text, final_results


def text_narrow(x, y, w, h, bw, results, height_diff_thres):
    ## narrow text
    y_min = min(results['loc'], key=lambda x: x[1])[1]
    y_max_box = max(results['loc'], key=lambda x: x[1] + x[3])  # use max text height instead of bounding box height
    y_max = y_max_box[1] + y_max_box[3]
    if y_min > height_diff_thres or h - (y_min + y_max) > height_diff_thres:
        y += y_min
        h = y_max - y_min
    ## get filled cc bounding box
    _, y1, _, h1 = text_symbol_refit((x, y, w, h), bw)
    return min((y, h), (y1, h1), key=lambda x: x[1])


def symbol_text_convert(mask_bw, symbols, image, original_img, text_color, symbol_color, txt_h,
                        letter_number_symbols=None):
    '''
    convert symbol to text if they overlap with each other
    In the meantime, convert text to symbol if symbol contains text
    '''
    ## merge symbols if they contains each other
    symbols1 = []
    symbols.sort(key=lambda x: x[2] * x[3])
    for i in range(len(symbols)):
        x, y, w, h = symbols[i]
        symbols1.append((x, y, w, h))
        for j in range(i + 1, len(symbols)):
            xl, yl, wl, hl = symbols[j]
            if xl < x and yl < y and x + w < xl + wl and y + h < yl + hl:
                image[y:y + h, x:x + w] = original_img[y:y + h, x:x + w]
                symbols1.pop()
                break

    symbols_filtered = []
    for (x, y, w, h) in symbols1:
        mask_slice = mask_bw[y:y + h, x:x + w]
        mask_slice_small = mask_bw[(y + 1):(y + h - 1), (x + 1):(x + w - 1)]
        pos = cv2.findNonZero(mask_slice)
        if pos is None:
            symbols_filtered.append((x, y, w, h))
            continue
        ## if height of overlapped area is 1. Ignore. see 298
        top_most = min(pos, key=lambda i: i[0, 1])[0]
        bottom_most = max(pos, key=lambda i: i[0, 1])[0]
        if bottom_most[1] == top_most[1]:
            symbols_filtered.append((x, y, w, h))
            continue
        pos1 = cv2.findNonZero(mask_slice_small)
        if pos1 is None:
            pos1 = []
        num_edge_pix = len(pos) - len(pos1)
        if num_edge_pix == 0:  # ==> symbol contains text, convert text to symbol
            symbols_filtered.append((x, y, w, h))
            mask_bw[y:y + h, x:x + w] = 0  # delete text on mask
            cv2.rectangle(image, (x, y), (x + w - 1, y + h - 1), symbol_color, -1)  # filled symbol
        else:  # ==> symbol overlap text, convert symbol to text if w/h <=3 or h<=3*text height
            if w / h > 3 or h > 3 * txt_h:
                symbols_filtered.append((x, y, w, h))
            else:
                mask_bw[y:y + h, x:x + w] = 255
                cv2.rectangle(image, (x, y), (x + w - 1, y + h - 1), text_color, -1)  # filled text

    # symbols that are consists of numbers and one letter with high confidence,
    # convert to texts if there is text in its near left
    symbols_final = []
    for (x, y, w, h) in symbols_filtered:
        if (x, y, w, h) in letter_number_symbols:
            left_most = x - w if x - w >= 0 else 0
            mask_slice = mask_bw[y:y + h, left_most:x]
            pos = cv2.findNonZero(mask_slice)
            if pos is not None:
                mask_bw[y:y + h, x:x + w] = 255
                cv2.rectangle(image, (x, y), (x + w - 1, y + h - 1), text_color, -1)  # filled text
                continue
        symbols_final.append((x, y, w, h))
    return symbols_final


def round_odd(num):
    return int(num // 2 * 2 + 1)


def get_search_box(x, y, w, h, Width, Height, w_max, w_s_h, w_s_v, perc, direction):
    '''
    w_s_h: horizontal search box width
    w_s_v: vertical search box height
    '''
    if not w_max:
        w_max = w
    h_s = int(h * perc)
    if direction == 'left':
        left = x - w_s_h if x - w_s_h > 0 else 0
        right = x
        top = y + (h - h_s) // 2
        bottom = top + h_s
    elif direction == 'right':
        left = x + w
        right = x + w_max + w_s_h if x + w_max + w_s_h < Width else Width
        top = y + (h - h_s) // 2
        bottom = top + h_s
    elif direction == 'up':
        left = x + (w - h_s) // 2
        right = left + h_s
        top = y - w_s_v if y - w_s_v > 0 else 0
        bottom = y
    elif direction == 'down':
        left = x + (w - h_s) // 2
        right = left + h_s
        top = y + h
        bottom = y + h + w_s_v if y + h + w_s_v < Height else Height
    return left, right, top, bottom


def connect_bounding_box(mask, kernel, perc, only_search_up=False):
    mask1 = mask.copy()
    Height, Width = mask.shape[:2]
    '''search box width is in percentage.'''
    morph_h, morph_w = kernel
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8, ltype=cv2.CV_32S)
    for x, y, w, h, _ in stats:
        '''search up'''
        left, right, top, bottom = get_search_box(x, y, w, h, Width, Height, None, morph_w, morph_h,
                                                  perc=perc, direction='up')
        mask_slice = mask[top:bottom, left:right]
        pos = cv2.findNonZero(mask_slice)
        if pos is not None:
            bottom_most = max(pos, key=lambda i: i[0, 1])[0]
            cv2.line(mask1, (left + bottom_most[0], y), (left + bottom_most[0], top + bottom_most[1]), 255, 1)
        if not only_search_up:
            '''search down'''
            left, right, top, bottom = get_search_box(x, y, w, h, Width, Height, None, morph_w, morph_h,
                                                      perc=perc, direction='down')
            mask_slice = mask[top:bottom, left:right]
            pos = cv2.findNonZero(mask_slice)
            if pos is not None:
                top_most = min(pos, key=lambda i: i[0, 1])[0]
                cv2.line(mask1, (left + top_most[0], y), (left + top_most[0], top + top_most[1]), 255, 1)
            '''search right'''
            left, right, top, bottom = get_search_box(x, y, w, h, Width, Height, None, morph_w, morph_h,
                                                      perc=perc, direction='right')
            mask_slice = mask[top:bottom, left:right]
            pos = cv2.findNonZero(mask_slice)
            if pos is not None:
                left_most = min(pos, key=lambda i: i[0, 0])[0]
                cv2.line(mask1, (x + w, top + left_most[1]), (left + left_most[0], top + left_most[1]), 255, 1)
            '''search left'''
            left, right, top, bottom = get_search_box(x, y, w, h, Width, Height, None, morph_w, morph_h,
                                                      perc=perc, direction='left')
            mask_slice = mask[top:bottom, left:right]
            pos = cv2.findNonZero(mask_slice)
            if pos is not None:
                right_most = max(pos, key=lambda i: i[0, 0])[0]
                cv2.line(mask1, (x, top + right_most[1]), (left + right_most[0], top + right_most[1]), 255, 1)
    return mask1


def symbol_search(rgb, texts, symbols, box_size, direction, search_box_color):
    '''
    box_size: tuple of two elements:
    First element: search box width in pixel.
    Second element: search box height in percentage of text height
    '''
    '''text mask'''
    rgb_copy = rgb.copy()
    Width = rgb.shape[1]
    mask = np.zeros(rgb.shape[:2])
    i = 1
    w_max = max(texts, key=lambda x: x[2])[2]
    w_s, perc = box_size
    for x, y, w, h, _ in texts:
        left, right, top, bottom = get_search_box(x, y, w, h, Width, None, w_max, w_s, w_s, perc, direction)
        mask[top:bottom, left:right] = i
        i += 1
        cv2.rectangle(rgb_copy, (left, top), (right - 1, bottom - 1), search_box_color, 1)
    '''link symbol with text'''
    text_symbol = dict()
    for (x, y, w, h) in symbols:
        ## check if overlaps with search box, then link texts with symbols
        mask_slice = mask[y:y + h, x:x + w]
        pos = cv2.findNonZero(mask_slice)  # find non zero positions
        if pos is not None:
            txt = texts[int(statistics.mode([mask_slice[y, x] for [[x, y]] in pos])) - 1]
            '''check relative positions of symbol and text and AR of symbol and text'''
            '''if text is a single A or a, ignore. (because before we treat a single A or a as text)'''
            result = check_sym_txt((x, y, w, h), txt, direction, strict=False)
            if result is False:
                continue
            '''merge symbols if their center distance <= 50'''
            if txt not in text_symbol.keys():
                text_symbol[txt] = (x, y, w, h)
            elif center_dist(text_symbol[txt], (x, y, w, h), p=1) <= 50:
                text_symbol[txt] = merge_box(text_symbol[txt], (x, y, w, h))
            else:  # if there are two symbols and their distances > 50, choose the one closest to text
                x_old, y_old, w_old, h_old = text_symbol[txt]
                if direction == 'left':  # if search direction is left, choose the symbol with largest x
                    text_symbol[txt] = max((x_old, y_old, w_old, h_old), (x, y, w, h), key=lambda x: x[0])
                elif direction == 'right':
                    text_symbol[txt] = min((x_old, y_old, w_old, h_old), (x, y, w, h), key=lambda x: x[0])
                elif direction == 'up':  # if search up, choose the symbol with largest y
                    text_symbol[txt] = max((x_old, y_old, w_old, h_old), (x, y, w, h), key=lambda x: x[1])

    text_symbol = [(k, v) for k, v in text_symbol.items()]

    '''detele overlapped text_symbol pairs'''
    filtered_text_symbol = []
    overlapped_indices = set()
    for i in range(len(text_symbol)):
        loc1 = merge_box(text_symbol[i][0][:-1], text_symbol[i][1])  # merge text and symbol
        not_overlap = True
        for j in range(i + 1, len(text_symbol)):
            loc2 = merge_box(text_symbol[j][0][:-1], text_symbol[j][1])
            if jaccard(loc1, loc2) > 0.1:
                not_overlap = False
                overlapped_indices.add(i)
                overlapped_indices.add(j)
                break
        if not_overlap and i not in overlapped_indices:
            filtered_text_symbol.append(text_symbol[i])
    filtered_text_symbol = dict(filtered_text_symbol)

    '''loop through confirmed symbols again, connect text if multiple texts overlap the same symbol'''
    '''and if texts max left difference or max center x difference is within threshold'''
    symbols_filtered = [v for k, v in filtered_text_symbol.items()]
    for (x, y, w, h) in symbols_filtered:
        mask_slice = mask[y:y + h, x:x + w]
        pos = cv2.findNonZero(mask_slice)  # find non zero positions
        if pos is not None:
            text_indices = {int(mask_slice[y, x]) - 1 for [[x, y]] in pos}
            if len(text_indices) > 1:
                txts = [texts[i] for i in text_indices]  # element in the form of (x,y,w,h,'text')
                max_left_diff = max(txts, key=lambda x: x[0])[0] - min(txts, key=lambda x: x[0])[0]
                max_center = max(txts, key=lambda x: x[0] + x[2] // 2)
                min_center = min(txts, key=lambda x: x[0] + x[2] // 2)
                max_center_diff = max_center[0] + max_center[2] // 2 - (min_center[0] + min_center[2] // 2)
                if max_left_diff <= 15 or max_center_diff <= 15:
                    old_symbol = None
                    for txt in txts:  # delete the key-value pair in text_symbol dict
                        popped = filtered_text_symbol.pop(txt, None)
                        if popped:
                            old_symbol = popped
                    txt = merge_text(txts)
                    if old_symbol:
                        filtered_text_symbol[txt] = old_symbol

    # '''loop through all symbols again, delete out of the axis symbol-text pairs'''
    # final_text_symbol = []
    # for text, symbol in filtered_text_symbol.items():
    #     xt, yt, wt, ht, txt = text
    #     xs, ys, ws, hs = symbol
    #     if (direction == 'up' and (xs < xt + wt // 2 < xs + ws) or (
    #             xt < xs + ws // 2 < xt + wt)) or ys < yt + ht // 2 < ys + hs:
    #         final_text_symbol.append((text, symbol))
    final_text_symbol = [(k, v) for k, v in filtered_text_symbol.items()]

    '''plot'''
    for text, symbol in final_text_symbol:
        txt = text[-1]  # the actual text
        x, y, w, h = symbol
        cv2.putText(rgb_copy,
                    f'{txt}',
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 0), 1)
    symbol_centers = [left_right_center(i[1], i[1]) for i in final_text_symbol]
    text_centers = [left_right_center(i[0], i[0]) for i in final_text_symbol]
    return rgb_copy, mask, final_text_symbol, text_centers, symbol_centers


def check_sym_txt(symbol_loc, text_loc, direction, strict=True):
    xs, ys, ws, hs = symbol_loc
    xt, yt, wt, ht, txt = text_loc
    AR_s = ws / hs  # width/height
    inv_AR_s = hs / ws
    inv_AR_t = ht / wt  # height/width
    # return false if aspect ratio of symbol or text is wrong
    if AR_s >= 10 or inv_AR_s >= 6 or inv_AR_t >= 6:
        return False
    if txt in {'A', 'a', '/', '&', '(', ')'}:
        return False
    if direction == 'up':
        if ys + hs - 1 < yt and ((xs < xt + wt // 2 < xs + ws) or (xt < xs + ws // 2 < xt + wt)):
            return True
    if direction == 'left':
        if xs + ws - 1 < xt:
            if strict:
                if (ys < yt + ht // 4 < ys + hs) or (ys < yt + 3 * ht // 4 < ys + hs):
                    return True
            else:
                if (ys < yt + ht // 4 < ys + hs) or (ys < yt + 3 * ht // 4 < ys + hs) or \
                        (yt < ys + hs // 4 < yt + ht) or (yt < ys + 3 * hs // 4 < yt + ht):
                    return True
    if direction == 'right':
        if xt + wt - 1 < xs:
            if strict:
                if (ys < yt + ht // 4 < ys + hs) or (ys < yt + 3 * ht // 4 < ys + hs):
                    return True
            else:
                if (ys < yt + ht // 4 < ys + hs) or (ys < yt + 3 * ht // 4 < ys + hs) or \
                        (yt < ys + hs // 4 < yt + ht) or (yt < ys + 3 * hs // 4 < yt + ht):
                    return True
    return False


def center_dist(rect1, rect2, p):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    if p == 2:
        dx = x1 + w1 // 2 - (x2 + w2 // 2)
        dy = y1 + h1 // 2 - (y2 + h2 // 2)
        return np.sqrt(dx * dx + dy * dy)
    elif p == 1:
        return abs(x1 - x2) + abs(y1 - y2)


def merge_box(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    x = min(x1, x2)
    y = min(y1, y2)
    w = max(x1 + w1, x2 + w2) - x
    h = max(y1 + h1, y2 + h2) - y
    return (x, y, w, h)


def merge_text(texts):
    '''
    texts: a list of (x,y,w,h,'txt')
    '''
    x = min(texts, key=lambda i: i[0])[0]
    y = min(texts, key=lambda i: i[1])[1]
    right_most = max(texts, key=lambda i: i[0] + i[2])
    w = right_most[0] + right_most[2] - x
    bottom_most = max(texts, key=lambda i: i[1] + i[3])
    h = bottom_most[1] + bottom_most[3] - y
    texts.sort(key=lambda i: i[1])  # sort based on y
    txt = ' '.join([i[-1] for i in texts])
    return (x, y, w, h, txt)


def left_right_center(text, symbol):
    xt, yt, wt, ht = text[:4]
    xs, ys, ws, hs = symbol[:4]
    left = min(xt, xs)
    top = min(yt, ys)
    right = max(xt + wt, xs + ws) - 1
    bottom = max(yt + ht, ys + hs) - 1
    return [left, right, int((left + right) // 2), int((top + bottom) // 2), top, bottom]


def remove_outliers(n_neighbors, direction, centers, sym_centers, text_symbol, txt_h, ratio):
    '''if search left or right: compute text top-left distances'''
    '''if search up: compute text center distances'''
    '''centers: [left,right,center_x,center_y,top,bottom]'''
    if direction == 'up':
        ctr = [[i[2], i[3]] for i in centers]
    elif direction == 'left' or direction == 'right':
        ctr = [[i[0], i[4]] for i in centers]
    output = dict()
    if len(ctr) < n_neighbors:
        return output
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree', n_jobs=-1, p=1).fit(ctr)
    distances, indices = nbrs.kneighbors(ctr)
    count = 0
    max_dist_thres = max(txt_h * 15, 2.5 * min(distances, key=lambda x: x[1])[1])
    for i in range(len(indices)):
        index = []
        for j in range(len(indices[i])):
            if distances[i][j] <= max_dist_thres:
                index.append(indices[i][j])
        if all(distances[i][1:] > max_dist_thres):
            continue
        indx0 = index[0]
        '''diffs for text'''
        left0, right0, top0, bottom0 = centers[indx0][0], centers[indx0][1], centers[indx0][4], centers[indx0][5]
        xc0, yc0 = centers[indx0][2], centers[indx0][3]
        left_diff = [abs(centers[k][0] - left0) for k in index[1:]]
        right_diff = [abs(centers[k][1] - right0) for k in index[1:]]
        top_diff = [abs(centers[k][4] - top0) for k in index[1:]]
        bottom_diff = [abs(centers[k][5] - bottom0) for k in index[1:]]
        xc_diff = [abs(centers[k][2] - xc0) for k in index[1:]]
        yc_diff = [abs(centers[k][3] - yc0) for k in index[1:]]
        '''diffs for symbols'''
        left0_sym, right0_sym, top0_sym, bottom0_sym = \
            sym_centers[indx0][0], sym_centers[indx0][1], sym_centers[indx0][4], sym_centers[indx0][5]
        xc0_sym, yc0_sym = sym_centers[indx0][2], sym_centers[indx0][3]
        left_diff_sym = [abs(sym_centers[k][0] - left0_sym) for k in index[1:]]
        right_diff_sym = [abs(sym_centers[k][1] - right0_sym) for k in index[1:]]
        top_diff_sym = [abs(sym_centers[k][4] - top0_sym) for k in index[1:]]
        bottom_diff_sym = [abs(sym_centers[k][5] - bottom0_sym) for k in index[1:]]
        xc_diff_sym = [abs(sym_centers[k][2] - xc0_sym) for k in index[1:]]
        yc_diff_sym = [abs(sym_centers[k][3] - yc0_sym) for k in index[1:]]
        '''diffs for center x of symbol to left of text'''
        xc_sym_left_diff = [abs((sym_centers[k][2] - centers[k][0]) - (xc0_sym - left0)) for k in index[1:]]

        if direction == 'left':
            diffs = left_diff + top_diff + xc_diff + yc_diff + left_diff
            diffs_sym = left_diff_sym + top_diff_sym + xc_diff_sym + yc_diff_sym + xc_diff_sym

        # note the last of diffs is 'left_diff' and the last of diffs_sym is 'right_diff_sym'
        # which means if text is left aligned and symbol is right aligned, see test10
        # or text is left aligned and symbol is middle aligned, see 66
        elif direction == 'right':
            diffs = left_diff + top_diff + xc_diff + yc_diff + left_diff + left_diff
            diffs_sym = left_diff_sym + top_diff_sym + xc_diff_sym + yc_diff_sym + right_diff_sym + xc_diff_sym
        elif direction == 'up':
            diffs = bottom_diff + left_diff + xc_diff + yc_diff
            diffs_sym = bottom_diff_sym + left_diff_sym + xc_diff_sym + yc_diff_sym
        ################### test
        # if i == 0:
        #     insp=text_symbol[i][0][-1]
        #     print('text centers or top-left',ctr)
        #     print(f'{insp} index: {index}')
        #     print(f'{insp} dist: {distances[i]}' )
        #
        #     print(f'{insp} diffs: {diffs}')
        #     print(f'{insp} diffs_syms: {diffs_sym}')

        ################
        for k in range(len(diffs)):
            if diffs[k] * ratio < 12 and diffs_sym[k] * ratio < 12:  # if text diff<10
                # if search left or right, diffs for center x of symbol to left of text should also <15
                # else, continue
                if (direction == 'left' or direction == 'right') and xc_sym_left_diff[
                    k % (len(index) - 1)] * ratio >= 15:
                    continue
                (xt, yt, wt, ht, txt), (xs, ys, ws, hs) = text_symbol[i]
                output[count] = {
                    'symbol_loc': (xs, ys, ws, hs),
                    'text_loc': (xt, yt, wt, ht),
                    'text': txt
                }
                count += 1
                break
    return output


def connect_lone_texts(output, texts, rgb, ratio, txt_h, direction):
    Height, Width = rgb.shape[:2]
    output = [(v['symbol_loc'], v['text_loc'], v['text']) for k, v in output.items()]
    '''text set'''
    text_set = [i[1] for i in output]

    '''legend text mask---bw'''
    legend_text_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    for x, y, w, h in text_set:
        legend_text_mask[y:y + h, x:x + w] = 255

    '''text mask---bw'''
    text_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    for x, y, w, h, txt in texts:
        mask_slice = legend_text_mask[y:y + h, x:x + w]
        pos = cv2.findNonZero(mask_slice)
        if pos is None:
            text_mask[y:y + h, x:x + w] = 255

    '''link lone texts horizontally'''
    kernel = (int(round(0.6 * txt_h)), 1)
    text_mask = connect_bounding_box(text_mask, kernel=kernel, perc=0.5, only_search_up=True)
    _, _, stats, _ = cv2.connectedComponentsWithStats(text_mask, connectivity=8, ltype=cv2.CV_32S)
    filtered_stats = []

    for x, y, w, h, _ in stats:
        if x == 0 and y == 0 and w == Width and h == Height:
            continue
        filtered_stats.append((x, y, w, h))
    #     print(filtered_stats)

    ###
    text_mask = np.zeros(rgb.shape[:2])
    i = 1
    for x, y, w, h, txt in texts:
        mask_slice = legend_text_mask[y:y + h, x:x + w]
        pos = cv2.findNonZero(mask_slice)
        if pos is None:
            text_mask[y:y + h, x:x + w] = i
        i += 1

    filtered_texts = []
    for (x, y, w, h) in filtered_stats:
        mask_slice = text_mask[y:y + h, x:x + w]
        pos = cv2.findNonZero(mask_slice)
        if pos is not None:
            text_indices = {int(mask_slice[y, x]) for [[x, y]] in pos}
            txts = [texts[i - 1] for i in text_indices]
            if len(text_indices) > 1:
                txt = merge_text(txts)
            else:
                txt = txts[0]
            filtered_texts.append(txt)
    '''now we have connected lone texts'''
    '''create legend text mask--numbers'''
    legend_text_mask = np.zeros(rgb.shape[:2])
    i = 1
    for x, y, w, h in text_set:
        legend_text_mask[y:y + h, x:x + w] = i
        i += 1
    '''connected lone texts search up to find its legend text'''
    for x, y, w, h, txt in filtered_texts:
        left = x
        right = x + w // 2
        top = max(0, y - int(round(0.8 * txt_h)))
        bottom = y
        mask_slice = legend_text_mask[top:bottom, left:right]
        pos = cv2.findNonZero(mask_slice)
        if pos is not None:
            bottom_most = max(pos, key=lambda i: i[0, 1])[0]  # find the bottom most point
            x_bm, y_bm = bottom_most
            text_index = int(mask_slice[y_bm, x_bm]) - 1
            symbol, (xt, yt, wt, ht), lone_txt = output[text_index]
            merged = merge_text([(xt, yt, wt, ht, lone_txt), (x, y, w, h, txt)])
            '''only merge if relative position of merged text and symbol is good'''
            if check_sym_txt(symbol, merged, direction, strict=False):
                output[text_index] = (symbol, merged[:-1], merged[-1])

    '''save output as dict'''
    count = 0
    output_dict = dict()
    for (xs, ys, ws, hs), (xt, yt, wt, ht), txt in output:
        output_dict[count] = {
            'symbol_loc': {'x': int(round(xs * ratio)), 'y': int(round(ys * ratio)),
                           'w': int(round(ws * ratio)), 'h': int(round(hs * ratio))},
            'text_loc': {'x': int(round(xt * ratio)), 'y': int(round(yt * ratio)),
                         'w': int(round(wt * ratio)), 'h': int(round(ht * ratio))},
            'text': txt
        }
        count += 1
    return output_dict


def jaccard(loc1: tuple, loc2: tuple):
    x1, y1, w1, h1 = loc1
    x2, y2, w2, h2 = loc2
    rec1 = (x1, y1, x1 + w1 - 1, y1 + h1 - 1)
    rec2 = (x2, y2, x2 + w2 - 1, y2 + h2 - 1)
    if rec2[0] > rec1[2] or rec2[2] < rec1[0] or rec2[1] > rec1[3] or rec2[3] < rec1[1]:
        return 0
    dx = min(rec1[2], rec2[2]) - max(rec1[0], rec2[0])
    dy = min(rec1[3], rec2[3]) - max(rec1[1], rec2[1])
    intersect_area = (dx + 1) * (dy + 1)
    union_area = w1 * h1 + w2 * h2 - intersect_area
    return intersect_area / union_area


def calc_confusion_matrix(df, truth, pred, thres, rgb_original, text_color, symbol_color, plot):
    if plot:
        rgb_TP_FP = rgb_original.copy() // 2
        rgb_FN = rgb_original.copy() // 2
    truth_matched = set()
    pred_matched = set()
    partially_matched = set()
    for i in truth:
        for j in pred:

            JI_symbol = jaccard(i['symbol_loc'], j['symbol_loc'])
            JI_text = jaccard(i['text_loc'], j['text_loc'])
            if JI_symbol > 0 and JI_text > 0:
                partially_matched.add(i['symbol_loc'])

            if JI_symbol >= thres or check_contains(i['symbol_loc'], j['symbol_loc']) or \
                    check_contains(j['symbol_loc'], i['symbol_loc']):

                if JI_text >= thres or check_contains(i['text_loc'], j['text_loc']):
                    txt1 = re.sub(' ', '', i['text'].lower())  # lower then remove space
                    txt2 = re.sub(' ', '', j['text'].lower())
                    sim = SequenceMatcher(None, txt1, txt2).ratio()
                    if sim >= thres:
                        df['TP'] += 1
                        truth_matched.add(i['symbol_loc'])
                        pred_matched.add(j['symbol_loc'])
                        if plot:
                            xs, ys, ws, hs = j['symbol_loc']
                            xt, yt, wt, ht = j['text_loc']
                            rgb_TP_FP[yt:yt + ht, xt:xt + wt] = rgb_original[yt:yt + ht, xt:xt + wt]
                            rgb_TP_FP[ys:ys + hs, xs:xs + ws] = rgb_original[ys:ys + hs, xs:xs + ws]
                            cv2.rectangle(rgb_TP_FP, (xt, yt), (xt + wt - 1, yt + ht - 1), text_color, 1)  # text
                            cv2.rectangle(rgb_TP_FP, (xs, ys), (xs + ws - 1, ys + hs - 1), symbol_color, 1)  # symbol
                        break
    df['FP'] = len(pred) - len(pred_matched)
    df['FN'] = len(truth) - len(truth_matched | partially_matched)
    if plot:
        for i in pred:
            if i['symbol_loc'] not in pred_matched:
                xs, ys, ws, hs = i['symbol_loc']
                xt, yt, wt, ht = i['text_loc']
                rgb_TP_FP[yt:yt + ht, xt:xt + wt] = rgb_original[yt:yt + ht, xt:xt + wt]
                rgb_TP_FP[ys:ys + hs, xs:xs + ws] = rgb_original[ys:ys + hs, xs:xs + ws]
                cv2.rectangle(rgb_TP_FP, (xt, yt), (xt + wt - 1, yt + ht - 1), text_color, 1)  # text
                cv2.rectangle(rgb_TP_FP, (xs, ys), (xs + ws - 1, ys + hs - 1), symbol_color, 1)  # symbol
                cv2.line(rgb_TP_FP, (xt, yt), (xt + wt - 1, yt + ht - 1), symbol_color, 1)
                cv2.line(rgb_TP_FP, (xt + wt - 1, yt), (xt, yt + ht - 1), symbol_color, 1)
                cv2.line(rgb_TP_FP, (xs, ys), (xs + ws - 1, ys + hs - 1), text_color, 1)
                cv2.line(rgb_TP_FP, (xs + ws - 1, ys), (xs, ys + hs - 1), text_color, 1)
        for i in truth:
            if i['symbol_loc'] not in truth_matched and i['symbol_loc'] not in partially_matched:
                xs, ys, ws, hs = i['symbol_loc']
                xt, yt, wt, ht = i['text_loc']
                rgb_FN[yt:yt + ht, xt:xt + wt] = rgb_original[yt:yt + ht, xt:xt + wt]
                rgb_FN[ys:ys + hs, xs:xs + ws] = rgb_original[ys:ys + hs, xs:xs + ws]
                cv2.rectangle(rgb_FN, (xt, yt), (xt + wt - 1, yt + ht - 1), text_color, 1)  # text
                cv2.rectangle(rgb_FN, (xs, ys), (xs + ws - 1, ys + hs - 1), symbol_color, 1)  # symbol
        return rgb_TP_FP, rgb_FN


def check_contains(loc_true, loc_pred):
    '''check if predicted bounding box contains true bounding box'''
    xt, yt, wt, ht = loc_true
    xp, yp, wp, hp = loc_pred
    if xt >= xp and yt >= yp and xt + wt <= xp + wp and yt + ht <= yp + hp:
        if xp * yp <= 4 * xt * yt:  # if area of predicted box is no more than 4 times of true box
            return True


def main(filename):
    text_color = (0, 255, 0)  # text
    symbol_color = (0, 0, 255)  # symbol
    search_box_color = (255, 0, 0)  # search box
    '''read files'''
    rgb, gray, ratio, rgb_original, filename = read_image(f'corpus/{filename}', resized_diagonal=2500)
    # rgb,gray,ratio,rgb_original = read_image(f'others/test_img/{filename}',resized_diagonal=2500)
    Height, Width = rgb.shape[:2]

    # get canny edge and filled
    canny = canny_filled(gray)

    # delete long vertical lines
    mask_y = np.ones((91, 1), dtype=np.uint8)
    y = cv2.morphologyEx(canny, cv2.MORPH_OPEN, mask_y, iterations=1)
    canny -= y
    # delete long horizontal lines
    mask_x = np.ones((1, 99), dtype=np.uint8)
    x = cv2.morphologyEx(canny, cv2.MORPH_OPEN, mask_x, iterations=1)
    canny -= x

    '''morph must use odd number!!'''
    filled = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, np.ones((1, 5), dtype=int))

    '''set OCR whitelist'''
    whitelist = set_whitelist(print_out=False)  # set OCR white list
    '''Initial OCR'''
    # lower limit of connected component bounding box is 2.5/ratio, so that the bounding box in the original image
    # is >2
    OCR_criteria = lambda w, h: (max(2.5 / ratio, 6) <= w <= 120 and max(2.5 / ratio, 6) <= h <= 120) or \
                                (2 <= w / h <= 30 and max(2.5 / ratio, 3) < h <= 80)
    recognition_results = CC_n_OCR(bw=filled,
                                   rgb=rgb,
                                   whitelist=whitelist,
                                   recog_txt_dict=dict(),
                                   pix=1,
                                   desc='Initial OCR',
                                   criteria=OCR_criteria,
                                   psm=7
                                   )

    if ratio < 0.4:
        conf_thres = 50
    elif ratio < 0.6:
        conf_thres = 60
    else:
        conf_thres = 80
    gap_thres = 15
    is_text_criteria0 = lambda conf, txt, h, txt_h: ((conf > conf_thres and sum(c.isalpha() for c in txt) > 2) or \
                                                     (conf > 80 and sum(c.isalpha() for c in txt) > 1) or \
                                                     sum(c.isalpha() for c in txt) >= 4 or \
                                                     (conf > conf_thres and len(txt) >= 4) or \
                                                     (conf > conf_thres and txt in {'A', 'a', '(', ')', '>', '*'})) and \
                                                    (not re.search(r'(([a-km-z])\2{2,})', txt.lower())) and \
                                                    (not re.search(r'(([l])\2{3,})', txt.lower())) and \
                                                    (not re.search(r'\(\d{,3}\)',
                                                                   txt))  # remove text like '(13)' see 88

    # symbols that are consists of numbers and one letter with high confidence,
    # or 2+ digits with '-' then 2+ digits, like '467-512'. see 192_1
    # potentially convert to texts in 'symbol_text_convert' function
    letter_num_symbols_criteria = lambda conf, txt: (conf >= conf_thres and (txt.isalnum() or txt in {'&', '/'})) or \
                                                    re.search(r'\w{2,}-\w{2,}', txt) or \
                                                    (conf >= conf_thres and sum(c.isalpha() for c in txt) == 1) or \
                                                    (conf >= conf_thres / 2 and sum(c.isalnum() for c in txt) >= 3)

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
                                           criteria=OCR_criteria,
                                           psm=7
                                           )

    '''filter re-OCR recognition results and show image'''
    '''also filter out texts whose height>3*txt_h at this step'''
    is_text_criteria_new = lambda conf, txt, h, txt_h: ((conf > conf_thres and sum(c.isalpha() for c in txt) > 2) or \
                                                        (conf > 80 and sum(c.isalpha() for c in txt) > 1) or \
                                                        sum(c.isalpha() for c in txt) >= 4 or \
                                                        (conf > conf_thres and len(txt) >= 4) or \
                                                        (conf > conf_thres and txt in {'A', 'a', '(', ')', '>',
                                                                                       '*'})) and \
                                                       (not re.search(r'(([a-km-z])\2{2,})', txt.lower())) and \
                                                       (not re.search(r'(([l])\2{3,})', txt.lower())) and \
                                                       (not re.search(r'\(\d{,3}\)', txt)) and \
                                                       h <= 3 * txt_h

    rgb1, mask_bw, symbols, recog_txt_dict, letter_number_symbols, txt_h = \
        initial_result_filter(rgb, recognition_results, is_text_criteria_new, letter_num_symbols_criteria,
                              text_color, symbol_color, True, txt_h)
    morph_h = 1
    kernel = (morph_h, int(round(1.7 * txt_h)))
    mask_bw1 = connect_bounding_box(mask_bw, kernel=kernel, perc=0.5)
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
    is_text_criteria_loose = lambda conf, txt, h, txt_h: ((conf > conf_thres * 0.8 and sum(
        c.isalpha() for c in txt) > 1) or \
                                                          (sum(c.isalnum() for c in txt) >= 3) or \
                                                          (conf > conf_thres and len(txt) >= 4) or \
                                                          (conf > conf_thres and txt in {'A', 'a', '(', ')', '>',
                                                                                         '*'})) and \
                                                         (not re.search(r'(([a-km-z])\2{2,})', txt.lower())) and \
                                                         (not re.search(r'(([l])\2{3,})', txt.lower())) and \
                                                         (not re.search(r'\(\d{,3}\)', txt))

    for (x, y, w, h), results in recognition_results1.items():
        is_text = False
        for i in range(len(results['text'])):
            conf = int(results['conf'][i])
            txt = results['text'][i]
            # check if it's text
            if is_text_criteria0(conf, txt, h, txt_h):
                is_text = True
                break

        if is_text:
            '''text split'''
            x, w, symbols, recognized_text, results = text_split(x, y, w, h, gap_thres, conf_thres, results, symbols,
                                                                 symbol_color,
                                                                 image=rgb2,
                                                                 bw=canny,
                                                                 right_split=False,
                                                                 criteria=is_text_criteria_loose
                                                                 )
            '''text narrow'''
            y, h = text_narrow(x, y, w, h, filled, results, height_diff_thres=2)
            mask_bw2[y:y + h, x:x + w] = 255
            texts.append((x, y, w, h, recognized_text))
            cv2.rectangle(rgb2, (x, y), (x + w - 1, y + h - 1), (0, 0, 0), 1)  # black
    #         cv2.putText(rgb2,
    #             f'{recognized_text}',
    #             (x, y),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             0.4, (0, 0, 0), 1)
    '''convert symbols and text'''
    symbols = symbol_text_convert(mask_bw2, symbols, rgb2, rgb, text_color, symbol_color, txt_h, dict())
    '''search and link'''
    '''serach left'''
    direction = 'left'
    rgb3_left, search_mask, text_symbol_left, text_centers, symbol_centers = symbol_search(rgb2, texts, symbols,
                                                                                           box_size=(6 * txt_h, 1),
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
                                                                                       box_size=(3 * txt_h, 1),
                                                                                       direction=direction,
                                                                                       search_box_color=search_box_color)

    ##
    # text_symbol_up=[text_symbol_up[5],text_symbol_up[4],text_symbol_up[3],text_symbol_up[2]]
    # text_centers=[text_centers[5],text_centers[4],text_centers[3],text_centers[2]]
    # symbol_centers=[symbol_centers[5],symbol_centers[4],symbol_centers[3],symbol_centers[2]]
    ##
    output_up = remove_outliers(4, direction, text_centers, symbol_centers, text_symbol_up, txt_h, ratio)
    output, rgb3, direction = max((output_left, rgb3_left, 'left'), (output_right, rgb3_right, 'right'),
                                  (output_up, rgb3_up, 'up'), key=lambda x: len(x[0].keys()))
    if not output:
        print('not enough elements in all search directions')
    else:
        output = connect_lone_texts(output, texts, rgb, ratio, txt_h, direction)
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
    thres = 0.3
    df = pd.DataFrame(np.zeros((1, 3), dtype=int), index=[f'file: {filename}'])
    df.columns = ['TP', 'FP', 'FN']
    rgb_TP_FP, rgb_FN = calc_confusion_matrix(df, truth, pred, thres, rgb_original, text_color, symbol_color, plot=True)
    cv2.imwrite(f'results/{filename}_TP_FP.png', rgb_TP_FP)
    cv2.imwrite(f'results/{filename}_FN.png', rgb_FN)
    return df


if __name__ == '__main__':
    '''
    code is far away from bug with the animal protecting
      ┏┓　　　┏┓
    ┏┛┻━━━┛┻┓
    ┃　　　　　　　┃ 　
    ┃　　　━　　　┃
    ┃　┳┛　┗┳　┃
    ┃　　　　　　　┃
    ┃　　　┻　　　┃
    ┃　　　　　　　┃
    ┗━┓　　　┏━┛
    　　┃　　　┃
    　　┃　　　┃
    　　┃　　　┗━━━┓
    　　┃　　　　　　　┣┓
    　　┃　　　　　　　┏┛
    　　┗┓┓┏━┳┓┏┛
    　　　┃┫┫　┃┫┫
    　　　┗┻┛　┗┻┛ 
    '''
    filename = sys.argv[1]
    df = main(filename)
    print(df)
