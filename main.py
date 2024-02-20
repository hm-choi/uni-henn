import os, sys
from uni_henn.utils.context import Context

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please input argv")
        exit(0)

    """
    When creating a context, you can adjust several parameters.
    @param N        Number of slots                         (default: 8192)
    @param depth    Depth of ciphertext                     (default: 11)
    @param LogQ     Fraction log scale                      (default: 32)
    @param LogP     Fraction log scale + Integer log scale  (default: 40)
    @param scale    Scale value used in ecoding             (default: 2**LogQ)
    """
    context = Context(N = 2**14, depth = 8, LogQ = 40, LogP = 60)
    # context = Context()
    
    files = {'1': 'example/M1_test.py',
            '2': 'example/M2_test.py',
            '3': 'example/M3_test.py',
            '4': 'example/M4_test.py',
            '5': 'example/M5_test.py',
            '6': 'example/M6_test.py',}
    
    TestFile = open(files[sys.argv[1]])
    TestCode = TestFile.read()

    sys.argv = [files[sys.argv[1]], context]
    exec(TestCode)
    TestFile.close()