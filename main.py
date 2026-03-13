import infinicore
a = infinicore.eye(3,device=infinicore.device("cuda", 0))
print(a)