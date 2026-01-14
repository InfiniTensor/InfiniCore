import infinicore as ic

device = ic.device("cuda:0")

q = ic.empty((1, 1, 4), dtype=ic.float16, device=device)

print(q.info)

q = ic.softmax(q, dim=-1)

print(q)
