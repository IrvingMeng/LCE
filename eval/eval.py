import argparse
import numpy as np
import os


def parse_option():
    parser = argparse.ArgumentParser(description='Train image-based re-id model')
    parser.add_argument('--query_path', type=str, required=True, help='path to old file')
    parser.add_argument('--gallery_path', type=str, required=True, help='path to old file')    
    args = parser.parse_args()
    return args


def compute_ap_cmc(index, good_index, junk_index):
    ap = 0
    cmc = np.zeros(len(index)) 
    
    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1.0
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        # ap = ap + d_recall*(old_precision + precision)/2
        ap = ap + d_recall*precision

    return ap, cmc


def evaluate(distmat, q_pids, g_pids, q_camids, g_camids):
    num_q, num_g = distmat.shape
    index = np.argsort(distmat, axis=1) # from small to large

    num_no_gt = 0 # num of query imgs without groundtruth
    num_r1 = 0
    CMC = np.zeros(len(g_pids))
    AP = 0

    for i in range(num_q):
        # groundtruth index
        query_index = np.argwhere(g_pids==q_pids[i])
        camera_index = np.argwhere(g_camids==q_camids[i])
        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        if good_index.size == 0:
            num_no_gt += 1
            continue
        # remove gallery samples that have the same pid and camid with query
        junk_index = np.intersect1d(query_index, camera_index)

        ap_tmp, CMC_tmp = compute_ap_cmc(index[i], good_index, junk_index)
        if CMC_tmp[0]==1:
            num_r1 += 1
        CMC = CMC + CMC_tmp
        AP += ap_tmp

    if num_no_gt > 0:
        print("{} query imgs do not have groundtruth.".format(num_no_gt))

    CMC = CMC / (num_q - num_no_gt)
    mAP = AP / (num_q - num_no_gt)

    return CMC, mAP


def main(args):
    assert os.path.isfile('{}/query.npz'.format(args.query_path))
    assert os.path.isfile('{}/gallery.npz'.format(args.gallery_path))
    query = np.load('{}/query.npz'.format(args.query_path))
    gallery = np.load('{}/gallery.npz'.format(args.gallery_path))

    qf, q_pids, q_camids = query['arr_0'], query['arr_1'], query['arr_2']
    gf, g_pids, g_camids = gallery['arr_0'], gallery['arr_1'], gallery['arr_2']

    m, n = qf.shape[0], gf.shape[0]
    distmat = np.zeros((m,n))
    qf = qf/ np.linalg.norm(qf, axis=1, keepdims=True)
    gf = gf/ np.linalg.norm(gf, axis=1, keepdims=True)
    distmat = - np.dot(qf, gf.T)

    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    print("Results ----------------------------------------")
    print('top1:{:.1%} top5:{:.1%} top10:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], mAP))
    print("------------------------------------------------")


if __name__ == '__main__':
    args = parse_option()
    main(args)
