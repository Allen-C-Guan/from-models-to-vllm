"""第13章：以整数和消息队列逐轮核验通信算法；CPU，无通信库或硬件测量。

source ./env.sh
python code/Book6-推理系统导论/ch13/collective_walkthrough.py
输出默认位于 log/book6-ch13/collective-algorithms/walkthrough.json。
"""
import argparse
import json
from pathlib import Path


def ring(p=4):
    """输入每rank (p,)；RS输出 (1,)；AG恢复 (p,)。每轮读旧状态再写新状态。"""
    original = [[10**r * (j + 1) for j in range(p)] for r in range(p)]
    expected = [sum(row[j] for row in original) for j in range(p)]
    active = [(r - 1) % p for r in range(p)]
    values = [original[r][active[r]] for r in range(p)]
    rs = []
    for step in range(p - 1):
        messages = [{'src': r, 'dst': (r + 1) % p, 'chunk': active[r],
                     'value': values[r]} for r in range(p)]
        new_active, new_values = [None] * p, [None] * p
        for msg in messages:
            dst, j = msg['dst'], msg['chunk']
            new_active[dst] = j
            new_values[dst] = msg['value'] + original[dst][j]
        active, values = new_active, new_values
        rs.append({'round': step + 1, 'messages': messages,
                   'chunks_after': active[:], 'values_after': values[:]})
    assert active == list(range(p)) and values == expected
    held = [{r: values[r]} for r in range(p)]
    active = list(range(p))
    ag = []
    for step in range(p - 1):
        messages = [{'src': r, 'dst': (r + 1) % p, 'chunk': active[r],
                     'value': held[r][active[r]]} for r in range(p)]
        new_active = [None] * p
        for msg in messages:
            dst, j = msg['dst'], msg['chunk']
            assert j not in held[dst]
            held[dst][j] = msg['value']
            new_active[dst] = j
        active = new_active
        ag.append({'round': step + 1, 'messages': messages,
                   'chunks_after': [sorted(x) for x in held]})
    assert all([h[j] for j in range(p)] == expected for h in held)
    return {'input': original, 'expected': expected, 'reduce_scatter': rs,
            'all_gather': ag, 'send_chunks_per_rank_per_phase': p - 1}


def recursive(p=4):
    """递归折半RS、反向倍增AG；另核验完整张量递归倍增All-Reduce。"""
    assert p > 1 and p & (p - 1) == 0
    original = [[10**r * (j + 1) for j in range(p)] for r in range(p)]
    expected = [sum(row[j] for row in original) for j in range(p)]
    states = [dict(enumerate(row)) for row in original]
    halving = []
    distance = p // 2
    while distance:
        new = []
        for r in range(p):
            peer = r ^ distance
            keys = sorted(states[r])
            keep = keys[:len(keys)//2] if r < peer else keys[len(keys)//2:]
            new.append({j: states[r][j] + states[peer][j] for j in keep})
        states = new
        halving.append({'distance': distance, 'states': states})
        distance //= 2
    assert all(s == {r: expected[r]} for r, s in enumerate(states))
    doubling_gather = []
    distance = 1
    while distance < p:
        new = []
        for r in range(p):
            peer = r ^ distance
            assert not states[r].keys() & states[peer].keys()
            new.append(states[r] | states[peer])
        states = new
        doubling_gather.append({'distance': distance, 'states': states})
        distance *= 2
    assert all([s[j] for j in range(p)] == expected for s in states)
    states = original
    doubling_reduce = []
    distance = 1
    while distance < p:
        states = [[a + b for a, b in zip(states[r], states[r ^ distance])]
                  for r in range(p)]
        doubling_reduce.append({'distance': distance, 'states': states})
        distance *= 2
    assert all(s == expected for s in states)
    return {'halving': halving, 'doubling_gather': doubling_gather,
            'doubling_reduce': doubling_reduce, 'expected': expected}


def alltoall(p=4):
    """用(src,dst)标记 (m,) 分块；核验直接轮转与按距离二进制分阶段转发。"""
    assert p > 1 and p & (p - 1) == 0
    direct = [[(r, r)] for r in range(p)]
    rounds = []
    for k in range(1, p):
        messages = [{'src': r, 'dst': (r + k) % p, 'packet': (r, (r + k) % p)}
                    for r in range(p)]
        for msg in messages:
            direct[msg['dst']].append(msg['packet'])
        rounds.append(messages)
    assert all(sorted(x) == [(r, dst) for r in range(p)] for dst, x in enumerate(direct))
    states = [[(r, dst) for dst in range(p)] for r in range(p)]
    staged = []
    distance = 1
    count = 0
    while distance < p:
        new = [[] for _ in range(p)]
        messages = []
        for r, packets in enumerate(states):
            send = [packet for packet in packets if ((packet[1] - r) % p) & distance]
            keep = [packet for packet in packets if not ((packet[1] - r) % p) & distance]
            new[r].extend(keep)
            new[(r + distance) % p].extend(send)
            count += len(send)
            messages.append({'src': r, 'dst': (r + distance) % p, 'packets': send})
        states = new
        staged.append({'distance': distance, 'messages': messages, 'states': states})
        distance *= 2
    assert all(sorted(x) == [(r, dst) for r in range(p)] for dst, x in enumerate(states))
    assert count == p * (p // 2) * (p.bit_length() - 1)
    return {'pairwise': rounds, 'binary_distance_forwarding': staged,
            'direct_network_chunks_total': p * (p - 1),
            'forwarded_network_chunks_total': count}


def dissemination(p=4):
    """集合仅表示通知隐含的到达知识，不是实际屏障消息载荷。"""
    known = [{r} for r in range(p)]
    rounds = []
    distance = 1
    while distance < p:
        known = [known[r] | known[(r - distance) % p] for r in range(p)]
        rounds.append({'distance': distance, 'known': [sorted(x) for x in known]})
        distance *= 2
    assert all(x == set(range(p)) for x in known)
    return rounds


def trees():
    """核验正文四卡树形走查，含中间打包和Reduce数值。"""
    broadcast = [True, False, False, False]
    for edges in [[(0, 2)], [(0, 1), (2, 3)]]:
        before = broadcast[:]
        for src, dst in edges:
            assert before[src]
            broadcast[dst] = True
    assert all(broadcast)
    scatter = [['A', 'B', 'C', 'D'], [], [], []]
    for edges in [[(0, 2, ['C', 'D'])], [(0, 1, ['B']), (2, 3, ['D'])]]:
        for src, dst, payload in edges:
            assert not scatter[dst]
            scatter[dst] = payload[:]
            scatter[src] = [x for x in scatter[src] if x not in payload]
    assert scatter == [['A'], ['B'], ['C'], ['D']]
    gather = [x[:] for x in scatter]
    for edges in [[(1, 0), (3, 2)], [(2, 0)]]:
        for src, dst in edges:
            gather[dst] += gather[src]
    assert gather[0] == ['A', 'B', 'C', 'D']
    reduce = [[10**r, 2*10**r] for r in range(4)]
    for edges in [[(1, 0), (3, 2)], [(2, 0)]]:
        for src, dst in edges:
            reduce[dst] = [a+b for a, b in zip(reduce[dst], reduce[src])]
    assert reduce[0] == [1111, 2222]
    return {'broadcast': broadcast, 'scatter': scatter, 'gather_root': gather[0],
            'reduce_root': reduce[0]}


def build_examples():
    return {'evidence': '整数构造例；算法状态核验，不是NPU通信实测',
            'trees_4': trees(),
            'ring_4': ring(4), 'ring_8': ring(8),
            'recursive_4': recursive(4), 'recursive_8': recursive(8),
            'alltoall_4': alltoall(4), 'alltoall_8': alltoall(8),
            'barrier': {p: dissemination(p) for p in [3, 4, 5, 8]}}


def main():
    root = Path(__file__).resolve().parents[3]
    if not (root / 'env.sh').exists():
        root = Path.cwd()
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=root/'log/book6-ch13/collective-algorithms/walkthrough.json')
    args = ap.parse_args()
    result = build_examples()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print('PASS: trees; ring/halving/doubling/all-to-all (4,8 ranks); barrier (3,4,5,8 ranks)')
    print(args.out)


if __name__ == '__main__':
    main()
