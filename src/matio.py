"""
We use the same format as Megaface(http://megaface.cs.washington.edu) 
except that we merge all files into a single binary file.
"""
import struct
import numpy as np
import sys,os

cv_type_to_dtype = {
    5 : np.dtype('float32')
}

dtype_to_cv_type = {v : k for k,v in cv_type_to_dtype.items()}

def write_mat(f, m):
    """Write mat m to file f"""
    if len(m.shape) == 1:
        rows = m.shape[0]
        cols = 1
    else:
        rows, cols = m.shape
    header = struct.pack('iiii', rows, cols, cols * 4, dtype_to_cv_type[m.dtype])
    f.write(header)
    f.write(m.data)


def read_mat(f):
    """
    Reads an OpenCV mat from the given file opened in binary mode
    """
    rows, cols, stride, type_ = struct.unpack('iiii', f.read(4*4))
    mat = np.fromstring(f.read(rows*stride),dtype=cv_type_to_dtype[type_])
    return mat.reshape(rows,cols)


def load_mat(filename):
    """
    Reads a OpenCV Mat from the given filename
    """
    return read_mat(open(filename,'rb'))

def save_mat(filename, m):
    """Saves mat m to the given filename"""
    return write_mat(open(filename,'wb'), m)


def main():
    """ Demo """
    fn="insightface.bin"
    mat=load_mat(fn)
    print mat.shape,mat.dtype
    #save_mat("insightface.bin",mat)
    print(mat[0])



if __name__ == '__main__':
    main()
