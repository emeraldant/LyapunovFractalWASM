"""Build a self-hosted sim for a problem without prepare() (attention): the submission, then
the sim template, then a module-level call wrapped so it can never break the submission's own
result. A check run of the output returns the judge's result plus the sim's report.

  python3 okc/tools/make_selfsim.py <submission template> <sim template> <output> "<call>"
"""
import os
import subprocess
import sys

OKC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def main(sub, sim, out, call):
    src = open(os.path.join(OKC, sub)).read()
    sim_src = open(os.path.join(OKC, sim)).read()
    tail = ("\n\n# " + "-" * 87 + "\n# Sim (okc/" + sim + "), run once at import\n\n" + sim_src +
            "\n\ntry:\n    " + call + "\nexcept Exception:  # the sim must never break the submission\n"
            "    import traceback\n    print('sim failed:', traceback.format_exc()[-1500:])\n"
            "torch.cuda.empty_cache()\n")
    # the sim's imports go to the top with the submission's (bundle resolves its @bundle lines)
    tmp = os.path.join(OKC, out + ".src")
    open(tmp, "w").write(src + tail)
    subprocess.check_call([sys.executable, os.path.join(OKC, "tools", "bundle.py"), tmp, os.path.join(OKC, out)])
    os.remove(tmp)


if __name__ == "__main__":
    main(*sys.argv[1:5])
