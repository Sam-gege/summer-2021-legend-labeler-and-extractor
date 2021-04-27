### Legend corpus, labeler and extractor

### label_truth:
contains true labels, in JSON format

### label_pred: 
contains predicted labels, in JSON format

### corpus: 
contains original floor plans.

### results: 
contains outputs from the tool. FN highlights all false negatives; TP_FP highlights all true positives and false positives, whereas false positives are crossed out.

### How to run:
    1. Create 4 folders: label_truth, label_pred, corpus and results in the same directory as the detector. 
    2. Save images and its true label in corpus and label_truth folder respectively. 
    3. Run the command: python3 legend_detector_v2.py filename


### Change log:

### legend_detector_v1:  
    1. 83.9% precision and 87.9% recall

### legend_detector_v2: 
    1. used a tighter text similarity threshold, so that detector_v1 only achieved 81.1% precision and 87.8% recall. 
    2. Add a new feature to reduce FP: search box stops at horizontal or vertical walls. detector_v2 now achieves 82.8% precision and 88.2% recall.
