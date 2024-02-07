import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please input argv")
        exit(0)

    files = {'1': 'example/M1_test.py',
            '2': 'example/M2_test.py',
            '3': 'example/M3_test.py',
            '4': 'example/M4_test.py',
            '5': 'example/M5_test.py',
            '6': 'example/M6_test.py',
            '7': 'example/M7_test.py',}

    exec(open(files[sys.argv[1]]).read())