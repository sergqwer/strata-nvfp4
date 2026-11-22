R = r"D:\Projects\strata-dec3"
def sub(path, old, new, n=1):
    p = R + "\\" + path
    b = open(p, "rb").read().decode("utf-8")
    old = old.replace("\n", "\r\n"); new = new.replace("\n", "\r\n")
    c = b.count(old)
    assert c == n, (path, c, old[:80])
    b = b.replace(old, new)
    open(p, "wb").write(b.encode("utf-8"))

sub(r"src\core\nv_defaults.cpp",
"""            {"STRATA_EARLY_READ", "1"},
""",
"""            {"STRATA_EARLY_READ", "1"},
            // decode round 3 D6: PDL edges in the MTP draft graphs (their kernels read only weights before the wait): the same drafts
            // (dumps identical), CLI fixed struct steps 0.788 -> 0.776 ms a round, round graph 0.310 -> 0.305 (bench-d3 NOTES-d3.md D6).
            {"STRATA_MTP_PDL", "1"},
""")
sub(r"src\core\nv_defaults_test.cpp",
"""    CHECK(nv_default_env(12, 0, false).size() == 29);""",
"""    CHECK(nv_default_env(12, 0, false).size() == 30);""", n=2)
print("ok")
