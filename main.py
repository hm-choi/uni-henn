import os, sys
from uni_henn.utils.context import Context

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Please input argv.")
    elif len(sys.argv) > 2:
        raise ValueError("Please input only one argv.")

    file_type = int(sys.argv[1])
    if file_type < 1 or file_type > 7:
        raise ValueError("Incorrect file type. Please input between 1 and 7.")

    """
    When creating a context, you can adjust several parameters.
    @param N        Number of slots                         (default: 8192)
    @param depth    Depth of ciphertext                     (default: 11)
    @param LogQ     Fraction log scale                      (default: 32)
    @param LogP     Fraction log scale + Integer log scale  (default: 40)
    @param scale    Scale value used in ecoding             (default: 2**LogQ)
    """
    # context = Context(N = 2**14, depth = 8, LogQ = 40, LogP = 60)
    context = Context()
    
    files = ['example/M1_test.py',
             'example/M2_test.py',
             'example/M3_test.py',
             'example/M4_test.py',
             'example/M5_test.py',
             'example/M6_test.py',
             'example/M7_test.py']
    
    test_filename = files[file_type - 1]
    with open(test_filename) as test_file:
        test_code = test_file.read()

    sys.argv = [test_filename, context]
    exec(test_code)
