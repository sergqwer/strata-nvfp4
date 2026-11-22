import sys
# eol.py [--fix] FILE... : count CRLF / bare LF; --fix turns bare LF into CRLF in files that are mostly CRLF
fix = sys.argv[1] == "--fix"
for p in sys.argv[2 if fix else 1:]:
    b = open(p, "rb").read()
    crlf = b.count(b"\r\n"); lf = b.count(b"\n") - crlf
    print(f"{p}: CRLF {crlf}, bare LF {lf}")
    if fix and lf and crlf > lf:
        b = b.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        open(p, "wb").write(b)
        print("  fixed")
