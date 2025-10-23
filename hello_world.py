#!/usr/bin/env python3

import sys, time


def fprint(text):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\n")


def main():
    fprint("Whaddup!")


if __name__ == "__main__":
    main()
