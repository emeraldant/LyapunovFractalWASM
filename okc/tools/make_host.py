"""Build a sim host: a real submission whose prepare() first runs a sim (untimed, on the
judge's GPU), then does its own work, so a check run returns both its normal result and the
sim's printed report.

  python3 okc/tools/make_host.py <host submission> <sim template> <output> "<call>"

e.g. make_host.py muse_moe_w4a16/submission.py sim/attention_sim_src.py probes/attn_host.py \
     "run_attention_sim({'current': attention_kernel_fn})"
The sim template's `# @bundle` lines are resolved by tools/bundle.py.
"""
import os
import subprocess
import sys

OKC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def main(host, sim, out, call, extra="", after=False):
    # Missing bundled names fail here, not on the judge's GPU.
    subprocess.check_call([sys.executable, os.path.join(OKC, "tools", "check_bundle.py"), os.path.join(OKC, sim)])
    src = open(os.path.join(OKC, host)).read()
    sim_src = open(os.path.join(OKC, sim)).read()
    hook = ("def prepare(weights: Weights):\n"
            "    try:\n"
            f"        {call}\n"
            "    except Exception as e:  # the sim must never break the host's own result\n"
            "        import traceback\n"
            "        print('sim failed:', ''.join(traceback.format_exc())[-1500:])\n"
            "    torch.cuda.empty_cache()\n")
    if src.count("def prepare(weights: Weights):\n") == 0 and src.count("def prepare(weights):\n") == 1:
        src = src.replace("def prepare(weights):\n", "def prepare(weights: Weights):\n")   # untyped hosts
        if "Weights" not in src.split("def prepare")[0]:
            src = src.replace("def prepare(weights: Weights):\n", "Weights = dict\n\n\ndef prepare(weights: Weights):\n")
    assert src.count("def prepare(weights: Weights):\n") == 1, "host prepare() signature not found"
    if after:
        # Run the sim at the end of prepare(), with the host's `state` and `weights`.
        start = src.index("def prepare(weights: Weights):\n")
        end = src.index("\n    return state\n", start)
        tail = ("\n    try:\n"
                f"        {call}\n"
                "    except Exception as e:  # the sim must never break the host's own result\n"
                "        import traceback\n"
                "        print('sim failed:', ''.join(traceback.format_exc())[-1500:])\n"
                "    torch.cuda.empty_cache()")
        src = src[:end] + tail + src[end:]
    else:
        src = src.replace("def prepare(weights: Weights):\n", hook, 1)
    marker = src.index("def prepare(weights: Weights):")
    src = src[:marker] + "# " + "-" * 87 + "\n# Sim (okc/" + sim + ")\n\n" + sim_src + "\n" + extra + "\n\n" + src[marker:]
    tmp = os.path.join(OKC, out + ".src")
    open(tmp, "w").write(src)
    subprocess.check_call([sys.executable, os.path.join(OKC, "tools", "bundle.py"), tmp, os.path.join(OKC, out)])
    os.remove(tmp)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--after"]
    main(*args[:4], *(args[4:5] or [""]), after="--after" in sys.argv)
