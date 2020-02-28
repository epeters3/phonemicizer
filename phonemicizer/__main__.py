"""
The CLI tool. Pipe all stdin text through the trained
phonemicizer, writing the output to stdout.
"""

import sys

from phonemicizer.predict import TrainedPhonemicizer

if __name__ == "__main__":
    model = TrainedPhonemicizer()
    for line in sys.stdin.readlines():
        sys.stdout.write(model.predict(line) + "\n")
    sys.stdout.flush()
