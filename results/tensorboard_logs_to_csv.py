import tensorflow as tf
import glob
import os
import pandas as pd

logging_dir = './logs'
event_paths = glob.glob(os.path.join(logging_dir, "*","event*"))

def sum_log(path):
    runlog = pd.DataFrame(columns=['metric', 'value'])
    try:
        for e in tf.train.summary_iterator(path):
            for v in e.summary.value:
                if 'val' in v.tag:
                        category_fold = path.split("/")[2]
                        category = category_fold.split("_")[0]
                        fold = category_fold.split("=")[1]

                        r = {'metric': v.tag, 'value':v.simple_value, 'category': category, 'fold': fold}
                        runlog = runlog.append(r, ignore_index=True)
    except:
        print('Event file possibly corrupt: {}'.format(path))
        return None
    runlog['epoch'] = [item for sublist in [[i]*5 for i in range(0, len(runlog)//5)] for item in sublist]
    return runlog

all_log = pd.DataFrame()
for path in event_paths:
    log = sum_log(path)
    if log is not None:
        if all_log.shape[0] == 0:
            all_log = log
        else:
            all_log = all_log.append(log)

