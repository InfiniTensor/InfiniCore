import argparse

import torch

import infinicore


def ic(tensor):
    return infinicore.from_torch(tensor)


def pack_int4(weight):
    values = weight.to(torch.int16) & 0xF
    blocks = values.reshape(*values.shape[:-1], -1, 32)
    packed = blocks[..., :16] | (blocks[..., 16:] << 4)
    return packed.reshape(*values.shape[:-1], -1).to(torch.int8).contiguous()


def run_chain(hidden, scores, bias, w1, s1, w2, s2):
    tokens, hidden_size = hidden.shape
    experts = scores.shape[1]
    topk = 2
    total = tokens * topk

    router_weights = torch.empty((tokens, topk), device="cuda", dtype=torch.float32)
    router_ids = torch.empty((tokens, topk), device="cuda", dtype=torch.int32)
    infinicore.grouped_topk_vendor(
        ic(scores),
        1,
        1,
        topk,
        True,
        1.0,
        ic(bias),
        "sigmoid",
        out=(ic(router_weights), ic(router_ids)),
    )

    counts = torch.empty((experts,), device="cuda", dtype=torch.int32)
    sorted_ids = torch.empty((total,), device="cuda", dtype=torch.int32)
    inverse = torch.empty_like(sorted_ids)
    infinicore.moe_argsort_bincount_with_inv_pos_(
        ic(counts), ic(sorted_ids), ic(inverse), ic(router_ids), experts
    )

    a1 = torch.empty((total, hidden_size), device="cuda", dtype=torch.int8)
    a1_scale = torch.empty((total, 1), device="cuda", dtype=torch.float32)
    infinicore.moe_expand_input_with_inv_pos_(
        ic(a1), ic(a1_scale), ic(hidden), ic(inverse), topk, 128, 1
    )

    a2_width = w1.shape[1]
    a2 = torch.empty((total, a2_width), device="cuda", dtype=torch.bfloat16)
    infinicore.w4a8_group_gemm_(
        ic(a2),
        ic(a1),
        ic(w1),
        ic(a1_scale),
        ic(s1),
        ic(counts),
        None,
        None,
        True,
        True,
    )

    a2_quant = torch.empty((total, a2_width // 2), device="cuda", dtype=torch.int8)
    a2_scale = torch.empty((total, 1), device="cuda", dtype=torch.float32)
    infinicore.moe_silu_and_mul_quant_(ic(a2_quant), ic(a2_scale), ic(a2), 1)

    a3 = torch.empty((total, hidden_size), device="cuda", dtype=torch.bfloat16)
    infinicore.w4a8_group_gemm_(
        ic(a3),
        ic(a2_quant),
        ic(w2),
        ic(a2_scale),
        ic(s2),
        ic(counts),
        ic(sorted_ids),
        None,
        True,
        True,
    )

    output = torch.empty((tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    infinicore.moe_sum_vendor_(
        ic(output),
        ic(a3.view(tokens, topk, hidden_size)),
        ic(router_weights),
        None,
        1.0,
        1.0,
    )
    return {
        "router_weights": router_weights,
        "router_ids": router_ids,
        "counts": counts,
        "sorted_ids": sorted_ids,
        "inverse": inverse,
        "a1": a1,
        "a1_scale": a1_scale,
        "a2": a2,
        "a2_quant": a2_quant,
        "a2_scale": a2_scale,
        "a3": a3,
        "output": output,
    }


def test():
    torch.manual_seed(0)
    device = infinicore.device("cuda", 0)
    tokens, experts, hidden_size, a2_width = 1, 4, 128, 128
    hidden = torch.zeros((tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    scores = torch.zeros((tokens, experts), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((experts,), device="cuda", dtype=torch.float32) * 0.1

    w1_raw = torch.randint(
        -8, 8, (experts, a2_width, hidden_size), device="cuda", dtype=torch.int8
    )
    w2_raw = torch.randint(
        -8,
        8,
        (experts, hidden_size, a2_width // 2),
        device="cuda",
        dtype=torch.int8,
    )
    w1 = pack_int4(w1_raw)
    w2 = pack_int4(w2_raw)
    s1 = torch.rand((experts, a2_width, 1), device="cuda", dtype=torch.float32) * 0.02
    s2 = (
        torch.rand((experts, hidden_size, 1), device="cuda", dtype=torch.float32) * 0.02
    )

    infinicore.start_graph_recording(device)
    graph_outputs = run_chain(hidden, scores, bias, w1, s1, w2, s2)
    graph = infinicore.stop_graph_recording()

    hidden_replacement = torch.randn_like(hidden)
    scores_replacement = torch.randn_like(scores)
    ic(hidden).copy_(ic(hidden_replacement))
    ic(scores).copy_(ic(scores_replacement))
    infinicore.sync_stream()
    graph.run()
    infinicore.sync_stream()
    actual = {name: tensor.clone() for name, tensor in graph_outputs.items()}

    expected = run_chain(hidden_replacement, scores_replacement, bias, w1, s1, w2, s2)
    infinicore.sync_stream()
    for name in actual:
        if actual[name].dtype in (torch.int8, torch.int32, torch.int64):
            torch.testing.assert_close(
                actual[name],
                expected[name],
                rtol=0,
                atol=0,
                msg=lambda message, name=name: f"{name}: {message}",
            )
        else:
            torch.testing.assert_close(
                actual[name],
                expected[name],
                rtol=2e-2,
                atol=3e-2,
                msg=lambda message, name=name: f"{name}: {message}",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.parse_args()
    test()
    print("glm_moe_graph ok")
