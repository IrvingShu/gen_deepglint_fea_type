import os
import sys
import numpy as np
import os.path as osp
import matio
import struct
import argparse

data_size = 1862120


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-dir', type=str, help='feature dir')
    parser.add_argument('--feature-dims', type=int, help='feature dims', default=512)
    parser.add_argument('--output', type=str, help='where to save the feature after pca')
    return parser.parse_args(argv)


def write_bin(path, m):
    rows, cols = m.shape
    with open(path, 'wb') as f:
        f.write(struct.pack('4i', rows,cols,cols*4,5))
        f.write(m.data)

def main(args):
    print('===> args:\n', args)
    fea_root = args.feature_dir
    fea_len = args.feature_dims    
    out_put = args.output
    
    with open('./testlist/testdata_lmk.txt') as f:
        lines = f.readlines()
        features_all = np.zeros( (data_size, fea_len), dtype=np.float32 )
        i = 0

        for line in lines:
            fea_path = fea_root + line.split(' ')[0] + '_feat.bin'
            if not osp.exists(fea_path):
                print('Not existed: %s' %fea_path)
            else:
                feat = matio.load_mat(fea_path)
                x_vec = feat[:fea_len].reshape((1,-1))
                features_all[i,:] = x_vec   
                if i % 10000 == 0:
                    print('processing', i)
                i = i + 1
        write_bin(out_put, features_all)
        
        print('Finished')

if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))                  
