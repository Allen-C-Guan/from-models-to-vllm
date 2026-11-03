"""CPU 数学练习：递推、三角求解、梯度和分段续算。"""
import torch
from gdn_math import toy,recurrent,chunkwise

def main():
    torch.set_num_threads(2)
    args=toy()
    for method in (recurrent,chunkwise):
        o,s,*_=method(*args)
        print(method.__name__, '\n逐 token 输出:\n',o,'\n最终状态:\n',s)
    ro,rs,*_=recurrent(*args)
    for c in (1,2,3,8):
        co,cs,*_=chunkwise(*args,chunk=c)
        torch.testing.assert_close(co,ro,rtol=1e-10,atol=1e-10)
        torch.testing.assert_close(cs,rs,rtol=1e-10,atol=1e-10)
    # 两条计算路径的全部输入都参与梯度比较，包括非零入口状态。
    grads=[]
    for method in (recurrent,chunkwise):
        aa=[x.clone().requires_grad_() for x in args]
        o,s,*_=method(*aa)
        grads.append(torch.autograd.grad(o.square().sum()+s.square().sum(),aa))
    for gr,gc in zip(*grads):torch.testing.assert_close(gr,gc,rtol=1e-10,atol=1e-10)
    # 在块中间暂停，传入状态后续算，应与完整序列相同。
    left=[x[:1] for x in args[:5]]+[args[5]]
    o1,s1,*_=chunkwise(*left,chunk=2)
    right=[x[1:] for x in args[:5]]+[s1]
    o2,s2,*_=chunkwise(*right,chunk=2)
    torch.testing.assert_close(torch.cat((o1,o2)),ro)
    torch.testing.assert_close(s2,rs)
    print('递推、不同块长、梯度和分段续算一致。')

if __name__=='__main__':main()
