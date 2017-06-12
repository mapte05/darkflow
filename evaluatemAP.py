    # --------------------------------------------------------
    # Simar Mangat, Maneesh Apte, Priy Sekhar
    # Heavily modified from https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py
    #

    import xml.etree.ElementTree as ET
    import os
    import pickle
    import numpy as np
    import json
    import collections

    detpath = './VOCout/'
    annopath = './VOCdevkit/VOC2012/Annotations/'
    imagesetfile = './VOCdevkit/VOC2012/ImageSets/Main/train.txt'
    detpath = 'VOCdevkit/VOC2012/JPEGImages/out/'
    classname = 'person'
    cachedir = 'cache'
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    # Parses the VOC XML files
    def parse_rec(filename):
        tree = ET.parse(filename)
        objects = []
        for obj in tree.findall('object'):
            obj_struct = {}
            obj_struct['name'] = obj.find('name').text
            obj_struct['pose'] = obj.find('pose').text
            obj_struct['truncated'] = int(obj.find('truncated').text)
            obj_struct['difficult'] = int(obj.find('difficult').text)
            bbox = obj.find('bndbox')
            obj_struct['bbox'] = [int(bbox.find('xmin').text),
                                  int(bbox.find('ymin').text),
                                  int(bbox.find('xmax').text),
                                  int(bbox.find('ymax').text)]
            objects.append(obj_struct)
        return objects

    # Parses the predictions made by darkflow
    def parse_pred(imagesetfile, predpath):
        unaccounted_files = 0
        json_store = collections.defaultdict(list)
        with open(imagesetfile, 'r') as f:
            lines = f.readlines()
        imagenames = [x.strip() for x in lines]
        for image in imagenames:
            image_json_path = predpath + image + '.json'
            # try catch statements needed for files that didn't fully go through prediction
            try:
                with open(image_json_path) as json_data:
                    try:
                        predictions = json.load(json_data)
                        for p in predictions:
                                bbox = []
                                bbox.append(p['topleft']['x'])
                                bbox.append(p['topleft']['y'])
                                bbox.append(p['bottomright']['x'])
                                bbox.append(p['bottomright']['y'])
                                store = [image, p['confidence'], bbox]
                                json_store[p['label']].append(store)
                    except ValueError:
                        unaccounted_files += 1
                        continue
            except IOError:
                continue
        return json_store

    # Compute VOC AP given preicision and recall
    def voc_ap(rec, prec, 07_metric=False):
        if 07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap
    # Top level function for evaluating VOC
    # Params
    #   detpath: path to directory containing predictions
    #   annopath: path to directory containing original VOC annotations
    #   imagesetfile: path to file containing a list of all images used for training
    #   classname: the particular class being evaluated (dog, person, etc. )
    #   ovthresh: the threshold for bounding box overlap 
    def voc_eval(detpath,
                 annopath,
                 imagesetfile,
                 classname,
                 cachedir,
                 ovthresh=0.5,
                 use_07_metric=False):
        # first load gt
        if not os.path.isdir(cachedir):
            os.mkdir(cachedir)
        cachefile = os.path.join(cachedir, 'annots.pkl')
        # read list of images
        with open(imagesetfile, 'r') as f:
            lines = f.readlines()
        imagenames = [x.strip() for x in lines]

        if not os.path.isfile(cachefile):
            # load annots
            recs = {}
            for i, imagename in enumerate(imagenames):
                recs[imagename] = parse_rec(annopath + imagename + '.xml')
                if i % 100 == 0:
                    print('Reading annotation for {}/{}'.format(i + 1, len(imagenames)))
            # save
            print('Saving cached annotations to {}'.format(cachefile))
            with open(cachefile, 'wb') as f:
                pickle.dump(recs, f)
        else:
            # load
            with open(cachefile, 'rb') as f:
                recs = pickle.load(f)

        # extract gt objects for this class
        class_recs = {}
        npos = 0
        for imagename in imagenames:
            R = [obj for obj in recs[imagename] if obj['name'] == classname]
            bbox = np.array([x['bbox'] for x in R])
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
            det = [False] * len(R)
            npos = npos + sum(~difficult)
            class_recs[imagename] = {'bbox': bbox,
                                     'difficult': difficult,
                                     'det': det}

        class_images = parse_pred(imagesetfile, detpath)[classname]
        nd = len(class_images)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[class_images[d][0]]
            bb = class_images[d][2]
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)

            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                       (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
        return rec, prec, ap


    # Main methods to evaluate VOC
    map_measure = []
    for classname in classes:
        rec, prec, ap = voc_eval(detpath, annopath, imagesetfile, classname, cachedir)
        print(classname, ap)
        map_measure.append(ap)

    print("mAP is", np.mean(map_measure))
