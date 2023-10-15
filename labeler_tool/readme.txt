step0: run 'labeler.ipynb'. This labeler will free you from selecting bounding boxes manually and type in all the texts.



step1: press 's' key to swap. red dot is symbol, green dot is text,
select the bounding boxes of symbol and texts pairs using red and green dots respectively, those boxes will later be merged 
and symbol-text pairs linked.




step2: If the bounding boxes in the previous step is not perfect, you can adjust the bounding box sizes manually, using:
    '13s':['a+4','s-1','exp3']:

13: the 13th symbol-text pair (IDs of symbol-text pairs are shown in the previous step)
s: means symbol. use 't' for text
a+4: a is left; s is bottom; d is right; w is top. +n/-n is to expand or shrink the sides by certain pixels
exp3: to expand the box by 3 pixels in all four directions.




step3:set OCR=True to recognize the texts, then after OCR, copy the result into the cell, make some corrections if needed, 
set OCR=False, then run this cell again. You will now see the texts are added onto the top of the bounding boxes.

step4: save to csv if all good (save to csv because it's easier to access and make corrections if needed,
this csv will later be converted to json in csv2json.ipynb in 'images' folder)

