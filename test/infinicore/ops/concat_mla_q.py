import argparse

import torch

import infinicore


def ic(tensor):
    return infinicore.from_torch(tensor)


def run_case(dtype):
    ql_nope = torch.randn((2, 8, 512), device="cuda", dtype=dtype)
    q_pe = torch.randn((2, 8, 64), device="cuda", dtype=dtype)
    expected = torch.cat((ql_nope, q_pe), dim=-1)
    actual = torch.empty_like(expected)
    infinicore.concat_mla_q(ic(ql_nope), ic(q_pe), out=ic(actual))
    infinicore.sync_device()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def run_graph_replay_case():
    device = infinicore.device("cuda", 0)
    ql_nope = torch.zeros((1, 8, 512), device="cuda", dtype=torch.bfloat16)
    q_pe = torch.zeros((1, 8, 64), device="cuda", dtype=torch.bfloat16)
    actual = torch.empty((1, 8, 576), device="cuda", dtype=torch.bfloat16)

    infinicore.start_graph_recording(device)
    infinicore.concat_mla_q(ic(ql_nope), ic(q_pe), out=ic(actual))
    graph = infinicore.stop_graph_recording()

    ql_replacement = torch.randn_like(ql_nope)
    qpe_replacement = torch.randn_like(q_pe)
    ic(ql_nope).copy_(ic(ql_replacement))
    ic(q_pe).copy_(ic(qpe_replacement))
    infinicore.sync_stream()
    graph.run()
    infinicore.sync_stream()
    torch.testing.assert_close(
        actual,
        torch.cat((ql_replacement, qpe_replacement), dim=-1),
        rtol=0,
        atol=0,
    )


def test():
    torch.manual_seed(0)
    for dtype in (torch.float16, torch.bfloat16):
        run_case(dtype)
    run_graph_replay_case()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.parse_args()
    test()
    print("concat_mla_q ok")
