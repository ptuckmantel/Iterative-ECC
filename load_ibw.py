import numpy as np
import matplotlib.pyplot as plt
from igor import binarywave

def correct_label(label):
    label = [x for x in label if x]  # Remove the empty lists
    label = label[0]  # Remove the unnecessary inception

    corrected_label = []

    for i in label:
        i = i.decode('UTF-8')
        if len(i) == 0:  # Skip empty channel names
            pass
        else:  # Correct the duplicate letters
            if 'Trace' in i:
                i = i.split('Trace')[0]
                corrected_label.append(i + 'Trace')
            elif 'Retrace' in i:
                i = i.split('Retrace')[0]
                corrected_label.append(i + 'Retrace')
            else:
                corrected_label.append(i)

    return corrected_label

def parse_meta(metadata, params):
    # Separate metadata in items for each params
    sep = str(metadata).split('\\r')
    values = []
    for p in params:
        #Define default value of the param as None in case the param is not found
        val=None
        # Find all entries in sep that contain the parameter name
        ind = [i for i, s in enumerate(sep) if p in s]
        for i in ind:
            # ent is string containinĝ the parameter in it
            ent = sep[i]
            # ceck that parameter is only the param and it doesn't have characters before and/or after
            sp = ent.split(': ')
            label = sp[0]
            if label == p:
                val = float(sp[1])
        if val == None:
            print('Parameter ' + p + ' not found in metadata')
            
        values.append(val)
    return values
    

def load_file(path, include_meta = True, meta_params = None):
    
    #Extract data from binarywave
    tmpdata = binarywave.load(path)['wave']
    #Get metadata
    note = tmpdata['note']
    data = tmpdata['wData']
    # Clean up channel list
    label_list = correct_label(tmpdata['labels'])

    # Remove FrequencyRetrace to save memory    

    try: 
        ind = label_list.index('FrequencyRetrace')
        del label_list[ind]
        del data[ind]
    except ValueError:
        pass

    ind = []

    # Append scan data to dat dict
    dat  = {}
    for i in range(len(label_list)):
        dat[label_list[i]] = data[:,:, i]

    if include_meta:
        if not meta_params:
            meta_params = ['FastScanSize', 
                       'SlowScanSize', 
                       'ScanRate', 
                       'ScanPoints', 
                       'ScanLines', 
                       'InvOLS', 
                       'SpringConstant', 
                       'TipVoltage', 
                       'DriveAmplitude',
                       'DeflectionSetpointVolts']

    
        # Append metadata info
        meta_values = parse_meta(note, meta_params)

        for v, m in zip(meta_values, meta_params):
            dat[m] = v
    return dat
