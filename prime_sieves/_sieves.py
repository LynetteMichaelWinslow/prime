import os as o1
o1.environ["NUMBA_THREADING_LAYER"] = "omp"
import time as t1
import numpy as np
import math as m1
from numba import njit as nj1, prange as pg1, get_num_threads as gth
from numba.typed import List as lt1

a1 = np.array([1,7,11,13,17,19,23,29], dtype=np.int64)
a2 = np.array([2,3,5,7,11,13,17,19,23,29], dtype=np.int64)
a3 = np.array([
    1,11,13,17,19,23,29,31,37,41,43,47,
    53,59,61,67,71,73,79,83,89,97,101,103,
    107,109,113,121,127,131,137,139,143,149,
    151,157,163,167,169,173,179,181,187,191,
    193,197,199,209
], dtype=np.int64)
a4 = np.array([2,3,5,7,11,13,17,19,23,29], dtype=np.int64)
a5 = np.array([1,7,11,13,17,19,23,29], dtype=np.int64)
a6 = np.array([
    1,11,13,17,19,23,29,31,37,41,43,47,
    53,59,61,67,71,73,79,83,89,97,101,103,
    107,109,113,121,127,131,137,139,143,149,
    151,157,163,167,169,173,179,181,187,191,
    193,197,199,209
], dtype=np.int64)
a7 = np.array([1,7,11,13,17,19,23,29], dtype=np.int64)
a8 = np.array([2,3,5,7,11,13,17,19,23,29], dtype=np.int64)
a9 = np.array([
    1,11,13,17,19,23,29,31,37,41,43,47,
    53,59,61,67,71,73,79,83,89,97,101,103,
    107,109,113,121,127,131,137,139,143,149,
    151,157,163,167,169,173,179,181,187,191,
    193,197,199,209
], dtype=np.int64)

@nj1(fastmath=True)
def f1(n1):
    if n1 < 2:
        return np.empty(0, dtype=np.int64)
    x1 = m1.log(n1)
    if n1 < 60184:
        y1 = int(1.25506 * n1 / x1)
    else:
        y1 = int(n1 / (x1 - 1.1))
    z1 = (n1 + 1) >> 1
    u1 = (z1 + 7) >> 3
    v1 = np.full(u1, 0xFF, dtype=np.uint8)
    v1[0] &= 0xFE
    w1 = ((int(n1**0.5)) + 1) >> 1
    for i1 in range(1, w1):
        if v1[i1 >> 3] & (1 << (i1 & 7)):
            j1 = (i1 << 1) + 1
            k1 = (j1 * j1) >> 1
            while k1 < z1:
                v1[k1 >> 3] &= ~(1 << (k1 & 7))
                k1 += j1
    r1 = np.empty(y1, dtype=np.int64)
    s1 = 0
    r1[s1] = 2
    s1 += 1
    for t1 in range(1, z1):
        if v1[t1 >> 3] & (1 << (t1 & 7)):
            m2 = (t1 << 1) + 1
            if m2 <= n1:
                r1[s1] = m2
                s1 += 1
            else:
                break
    return r1[:s1]

@nj1(fastmath=True)
def f2(n1):
    if n1 < 2:
        return np.empty(0, dtype=np.int64)
    x1 = m1.log(n1)
    if n1 < 60184:
        y1 = int(1.25506 * n1 / x1)
    else:
        y1 = int(n1 / (x1 - 1.1))
    z1 = np.zeros(n1 + 1, dtype=np.int64)
    u1 = np.empty(y1, dtype=np.int64)
    v1 = 0
    for i1 in range(2, n1 + 1):
        if z1[i1] == 0:
            u1[v1] = i1
            v1 += 1
            z1[i1] = i1
        for j1 in range(v1):
            k1 = u1[j1]
            if k1 > z1[i1] or i1 * k1 > n1:
                break
            z1[i1 * k1] = k1
            if i1 % k1 == 0:
                break
    return u1[:v1]

@nj1(fastmath=True)
def f3(arr1, b1, c1, arr2):
    x1 = c1 - b1 + 1
    y1 = (x1 + 1) >> 1
    for z1 in arr2:
        if z1 < 7:
            continue
        u1 = z1 * z1
        if u1 > c1:
            break
        v1 = ((b1 + z1 - 1) // z1) * z1
        if v1 < u1:
            v1 = u1
        v1 += z1 * (1 - (v1 & 1))
        w1 = (v1 - b1) >> 1
        while w1 + 8 * z1 < y1:
            for _ in range(8):
                arr1[w1 >> 3] &= ~(1 << (w1 & 7))
                w1 += z1
        while w1 < y1:
            arr1[w1 >> 3] &= ~(1 << (w1 & 7))
            w1 += z1

@nj1(parallel=True, fastmath=True, nogil=True)
def f4(arr1, b1, c1, arr2):
    x1 = c1 - b1 + 1
    y1 = (x1 + 1) >> 1
    z1 = (y1 + 7) >> 3
    u1 = gth()
    v1 = np.full((u1, z1), 0xFF, dtype=np.uint8)
    w1 = (len(arr2) + u1 - 1) // u1
    for i1 in pg1(u1):
        r1 = v1[i1]
        s1 = i1 * w1
        t1 = s1 + w1
        if t1 > len(arr2):
            t1 = len(arr2)
        for j1 in range(s1, t1):
            k1 = arr2[j1]
            if k1 < 7:
                continue
            m2 = k1 * k1
            if m2 > c1:
                continue
            n1 = ((b1 + k1 - 1) // k1) * k1
            if n1 < m2:
                n1 = m2
            n1 += k1 * (1 - (n1 & 1))
            o1 = (n1 - b1) >> 1
            while o1 + 8 * k1 < y1:
                for _ in range(8):
                    r1[o1 >> 3] &= ~(1 << (o1 & 7))
                    o1 += k1
            while o1 < y1:
                r1[o1 >> 3] &= ~(1 << (o1 & 7))
                o1 += k1
    for p1 in range(z1):
        arr1[p1] = 0xFF
    for q1 in range(u1):
        r1 = v1[q1]
        for s2 in range(z1):
            arr1[s2] &= r1[s2]

@nj1(fastmath=True)
def f5(arr1, b1, c1, arr2):
    x1 = c1 - b1 + 1
    y1 = (x1 + 1) >> 1
    for z1 in arr2:
        if z1 < 7:
            continue
        u1 = z1 * z1
        if u1 > c1:
            break
        v1 = ((b1 + z1 - 1) // z1) * z1
        if v1 < u1:
            v1 = u1
        v1 += z1 * (1 - (v1 & 1))
        w1 = (v1 - b1) >> 1
        while w1 + 8 * z1 < y1:
            for _ in range(8):
                arr1[w1 >> 3] &= ~(1 << (w1 & 7))
                w1 += z1
        while w1 < y1:
            arr1[w1 >> 3] &= ~(1 << (w1 & 7))
            w1 += z1

@nj1(fastmath=True)
def f6(n1, z1=10000000):
    z1 = 50000000
    if n1 < 2:
        return np.empty(0, dtype=np.int64)
    x2 = int(n1**0.5)
    y2 = f1(x2)
    u2 = m1.log(n1)
    if n1 < 60184:
        v2 = int(1.25506 * n1 / u2)
    else:
        v2 = int(n1 / (u2 - 1.1))
    w2 = np.empty(v2, dtype=np.int64)
    r2 = 0
    for s1 in a2:
        if s1 <= n1:
            w2[r2] = s1
            r2 += 1
        else:
            break
    t2 = 31
    if t2 > n1:
        return w2[:r2]
    if n1 >= 3e10:
        base1 = 210
        wh1 = a3
    else:
        base1 = 30
        wh1 = a1
    tt1 = t2
    while True:
        if tt1 > n1:
            break
        p1 = tt1 + z1 - 1
        if p1 > n1:
            p1 = n1
        if p1 < 7:
            break
        q1 = p1 - tt1 + 1
        o1 = (q1 + 1) >> 1
        h1 = (o1 + 7) >> 3
        g1 = np.full(h1, 0xFF, dtype=np.uint8)
        f3(g1, tt1, p1, y2)
        dd1 = (tt1 // base1) * base1
        while dd1 <= p1:
            for mm1 in wh1:
                nn1 = dd1 + mm1
                if nn1 < tt1:
                    continue
                if nn1 > p1:
                    break
                if nn1 <= n1:
                    kk1 = (nn1 - tt1) >> 1
                    if g1[kk1 >> 3] & (1 << (kk1 & 7)):
                        w2[r2] = nn1
                        r2 += 1
            dd1 += base1
        tt1 = p1 + 2
    return w2[:r2]

@nj1(fastmath=True)
def f7(n1, z1=10000000):
    z1 = 50000000
    if n1 < 2:
        return np.empty(0, dtype=np.int64)
    x2 = int(n1**0.5)
    y2 = f1(x2)
    u2 = m1.log(n1)
    if n1 < 60184:
        v2 = int(1.25506 * n1 / u2)
    else:
        v2 = int(n1 / (u2 - 1.1))
    w2 = np.empty(v2, dtype=np.int64)
    r2 = 0
    for s1 in a2:
        if s1 <= n1:
            w2[r2] = s1
            r2 += 1
        else:
            break
    t2 = 31
    if t2 > n1:
        return w2[:r2]
    if n1 >= 3e10:
        base1 = 210
        wh1 = a3
    else:
        base1 = 30
        wh1 = a1
    tt1 = t2
    while True:
        if tt1 > n1:
            break
        p1 = tt1 + z1 - 1
        if p1 > n1:
            p1 = n1
        if p1 < 7:
            break
        q1 = p1 - tt1 + 1
        o1 = (q1 + 1) >> 1
        h1 = (o1 + 7) >> 3
        g1 = np.full(h1, 0xFF, dtype=np.uint8)
        f4(g1, tt1, p1, y2)
        dd1 = (tt1 // base1) * base1
        while dd1 <= p1:
            for mm1 in wh1:
                nn1 = dd1 + mm1
                if nn1 < tt1:
                    continue
                if nn1 > p1:
                    break
                if nn1 <= n1:
                    kk1 = (nn1 - tt1) >> 1
                    if g1[kk1 >> 3] & (1 << (kk1 & 7)):
                        w2[r2] = nn1
                        r2 += 1
            dd1 += base1
        tt1 = p1 + 2
    return w2[:r2]

@nj1(fastmath=True)
def f8(n1, z1=10000000):
    z1 = 50000000
    if n1 < 2:
        return np.empty(0, dtype=np.int64)
    x2 = int(n1**0.5)
    y2 = f2(x2)
    u2 = m1.log(n1)
    if n1 < 60184:
        v2 = int(1.25506 * n1 / u2)
    else:
        v2 = int(n1 / (u2 - 1.1))
    w2 = np.empty(v2, dtype=np.int64)
    r2 = 0
    for s1 in a2:
        if s1 <= n1:
            w2[r2] = s1
            r2 += 1
        else:
            break
    t2 = 31
    if t2 > n1:
        return w2[:r2]
    if n1 >= 3e10:
        base1 = 210
        wh1 = a3
    else:
        base1 = 30
        wh1 = a1
    tt1 = t2
    while True:
        if tt1 > n1:
            break
        p1 = tt1 + z1 - 1
        if p1 > n1:
            p1 = n1
        if p1 < 7:
            break
        q1 = p1 - tt1 + 1
        o1 = (q1 + 1) >> 1
        h1 = (o1 + 7) >> 3
        g1 = np.full(h1, 0xFF, dtype=np.uint8)
        f5(g1, tt1, p1, y2)
        dd1 = (tt1 // base1) * base1
        while dd1 <= p1:
            for mm1 in wh1:
                nn1 = dd1 + mm1
                if nn1 < tt1:
                    continue
                if nn1 > p1:
                    break
                if nn1 <= n1:
                    kk1 = (nn1 - tt1) >> 1
                    if g1[kk1 >> 3] & (1 << (kk1 & 7)):
                        w2[r2] = nn1
                        r2 += 1
            dd1 += base1
        tt1 = p1 + 2
    return w2[:r2]

@nj1(fastmath=True)
def f9(n1, z1=10000000):
    z1 = 50000000
    if n1 < 2:
        return np.empty(0, dtype=np.int64)
    x2 = int(n1**0.5)
    y2 = f2(x2)
    u2 = m1.log(n1)
    if n1 < 60184:
        v2 = int(1.25506 * n1 / u2)
    else:
        v2 = int(n1 / (u2 - 1.1))
    w2 = np.empty(v2, dtype=np.int64)
    r2 = 0
    for s1 in a2:
        if s1 <= n1:
            w2[r2] = s1
            r2 += 1
        else:
            break
    t2 = 31
    if t2 > n1:
        return w2[:r2]
    if n1 >= 3e10:
        base1 = 210
        wh1 = a3
    else:
        base1 = 30
        wh1 = a1
    tt1 = t2
    while True:
        if tt1 > n1:
            break
        p1 = tt1 + z1 - 1
        if p1 > n1:
            p1 = n1
        if p1 < 7:
            break
        q1 = p1 - tt1 + 1
        o1 = (q1 + 1) >> 1
        h1 = (o1 + 7) >> 3
        g1 = np.full(h1, 0xFF, dtype=np.uint8)
        f4(g1, tt1, p1, y2)
        dd1 = (tt1 // base1) * base1
        while dd1 <= p1:
            for mm1 in wh1:
                nn1 = dd1 + mm1
                if nn1 < tt1:
                    continue
                if nn1 > p1:
                    break
                if nn1 <= n1:
                    kk1 = (nn1 - tt1) >> 1
                    if g1[kk1 >> 3] & (1 << (kk1 & 7)):
                        w2[r2] = nn1
                        r2 += 1
            dd1 += base1
        tt1 = p1 + 2
    return w2[:r2]

def f10(x1):
    x1 = x1.strip().lower()
    if x1.endswith('m'):
        return int(float(x1[:-1]) * 1e6)
    return int(float(x1))

def f11(a1n):
    if a1n > 2e10:
        print("Recommended parallel for N > 2e10")
    print("Choose the sieve method:")
    print("1) Eratosthenes single-thread")
    print("2) Eratosthenes parallel")
    print("3) Euler single-thread")
    print("4) Euler parallel")
    b1 = input().strip()
    c1 = t1.time()
    if a1n <= 10000000:
        if b1 == '3' or b1 == '4':
            d1 = f2(a1n)
        else:
            d1 = f1(a1n)
    else:
        if a1n > 20000000000:
            if b1 == '1':
                b1 = '2'
            elif b1 == '3':
                b1 = '4'
        if b1 == '1':
            d1 = f6(a1n)
        elif b1 == '2':
            d1 = f7(a1n)
        elif b1 == '3':
            d1 = f8(a1n)
        else:
            d1 = f9(a1n)
    e1 = t1.time()
    print(f"sieve up to {a1n} completed, {len(d1)} primes found.")
    print(f"Largest prime: {d1[-1]}")
    print(f"Time taken: {e1 - c1:.4f} seconds")

@nj1(fastmath=True)
def f12(n1):
    if n1 < 2:
        return np.empty(0, dtype=np.int64)
    x1 = m1.log(n1)
    if n1 < 60184:
        y1 = int(1.25506 * n1 / x1)
    else:
        y1 = int(n1 / (x1 - 1.1))
    z1 = (n1 + 1) >> 1
    u1 = (z1 + 7) >> 3
    v1 = np.full(u1, 0xFF, dtype=np.uint8)
    v1[0] &= 0xFE
    w1 = ((int(n1**0.5)) + 1) >> 1
    for i1 in range(1, w1):
        if v1[i1 >> 3] & (1 << (i1 & 7)):
            j1 = (i1 << 1) + 1
            k1 = (j1 * j1) >> 1
            while k1 < z1:
                v1[k1 >> 3] &= ~(1 << (k1 & 7))
                k1 += j1
    r1 = np.empty(y1, dtype=np.int64)
    s1 = 0
    r1[s1] = 2
    s1 += 1
    for t1 in range(1, z1):
        if v1[t1 >> 3] & (1 << (t1 & 7)):
            m2 = (t1 << 1) + 1
            if m2 <= n1:
                r1[s1] = m2
                s1 += 1
            else:
                break
    return r1[:s1]

@nj1(fastmath=True)
def f13(n1):
    if n1 < 2:
        return np.empty(0, dtype=np.int64)
    x1 = m1.log(n1)
    if n1 < 60184:
        y1 = int(1.25506 * n1 / x1)
    else:
        y1 = int(n1 / (x1 - 1.1))
    z1 = np.zeros(n1 + 1, dtype=np.int64)
    u1 = np.empty(y1, dtype=np.int64)
    v1 = 0
    for i1 in range(2, n1 + 1):
        if z1[i1] == 0:
            u1[v1] = i1
            v1 += 1
            z1[i1] = i1
        for j1 in range(v1):
            k1 = u1[j1]
            if k1 > z1[i1] or i1 * k1 > n1:
                break
            z1[i1 * k1] = k1
            if i1 % k1 == 0:
                break
    return u1[:v1]

@nj1(fastmath=True)
def f14(arr1, b1, c1, arr2):
    x1 = c1 - b1 + 1
    y1 = (x1 + 1) >> 1
    for z1 in arr2:
        if z1 < 7:
            continue
        u1 = z1 * z1
        if u1 > c1:
            break
        v1 = ((b1 + z1 - 1) // z1) * z1
        if v1 < u1:
            v1 = u1
        v1 += z1 * (1 - (v1 & 1))
        w1 = (v1 - b1) >> 1
        while w1 + 8 * z1 < y1:
            for _ in range(8):
                arr1[w1 >> 3] &= ~(1 << (w1 & 7))
                w1 += z1
        while w1 < y1:
            arr1[w1 >> 3] &= ~(1 << (w1 & 7))
            w1 += z1

@nj1(parallel=True, fastmath=True, nogil=True)
def f15(arr1, b1, c1, arr2):
    x1 = c1 - b1 + 1
    y1 = (x1 + 1) >> 1
    z1 = (y1 + 7) >> 3
    u1 = gth()
    v1 = np.full((u1, z1), 0xFF, dtype=np.uint8)
    w1 = (len(arr2) + u1 - 1) // u1
    for i1 in pg1(u1):
        r1 = v1[i1]
        s1 = i1 * w1
        t1 = s1 + w1
        if t1 > len(arr2):
            t1 = len(arr2)
        for j1 in range(s1, t1):
            k1 = arr2[j1]
            if k1 < 7:
                continue
            m2 = k1 * k1
            if m2 > c1:
                continue
            n2 = ((b1 + k1 - 1) // k1) * k1
            if n2 < m2:
                n2 = m2
            n2 += k1 * (1 - (n2 & 1))
            o2 = (n2 - b1) >> 1
            while o2 + 8 * k1 < y1:
                for _ in range(8):
                    r1[o2 >> 3] &= ~(1 << (o2 & 7))
                    o2 += k1
            while o2 < y1:
                r1[o2 >> 3] &= ~(1 << (o2 & 7))
                o2 += k1
    for p1 in range(z1):
        arr1[p1] = 0xFF
    for q1 in range(u1):
        r1 = v1[q1]
        for s2 in range(z1):
            arr1[s2] &= r1[s2]

@nj1(fastmath=True)
def f16(arr1, b1, c1, arr2):
    x1 = c1 - b1 + 1
    y1 = (x1 + 1) >> 1
    for z1 in arr2:
        if z1 < 7:
            continue
        u1 = z1 * z1
        if u1 > c1:
            break
        v1 = ((b1 + z1 - 1) // z1) * z1
        if v1 < u1:
            v1 = u1
        v1 += z1 * (1 - (v1 & 1))
        w1 = (v1 - b1) >> 1
        while w1 + 8 * z1 < y1:
            for _ in range(8):
                arr1[w1 >> 3] &= ~(1 << (w1 & 7))
                w1 += z1
        while w1 < y1:
            arr1[w1 >> 3] &= ~(1 << (w1 & 7))
            w1 += z1

@nj1(fastmath=True)
def f17(n1, y1=10000000):
    y1 = 50000000
    if n1 < 2:
        return np.empty(0, dtype=np.int64)
    x2 = int(n1**0.5)
    u2 = f12(x2)
    v2 = m1.log(n1)
    if n1 < 60184:
        w2 = int(1.25506 * n1 / v2)
    else:
        w2 = int(n1 / (v2 - 1.1))
    r2 = np.empty(w2, dtype=np.int64)
    s2 = 0
    for i1 in a8:
        if i1 <= n1:
            r2[s2] = i1
            s2 += 1
        else:
            break
    j1 = 31
    if j1 > n1:
        return r2[:s2]
    if n1 >= 3e10:
        base1 = 210
        wh1 = a9
    else:
        base1 = 30
        wh1 = a7
    while True:
        if j1 > n1:
            break
        k1 = j1 + y1 - 1
        if k1 > n1:
            k1 = n1
        if k1 < 7:
            break
        l1 = k1 - j1 + 1
        m1x = (l1 + 1) >> 1
        o1 = (m1x + 7) >> 3
        p1 = np.full(o1, 0xFF, dtype=np.uint8)
        f14(p1, j1, k1, u2)
        q1 = (j1 // base1) * base1
        while q1 <= k1:
            for h1 in wh1:
                a1x = q1 + h1
                if a1x < j1:
                    continue
                if a1x > k1:
                    break
                if a1x <= n1:
                    b1x = (a1x - j1) >> 1
                    if p1[b1x >> 3] & (1 << (b1x & 7)):
                        r2[s2] = a1x
                        s2 += 1
            q1 += base1
        j1 = k1 + 2
    return r2[:s2]

@nj1(fastmath=True)
def f18(n1, y1=10000000):
    y1 = 50000000
    if n1 < 2:
        return np.empty(0, dtype=np.int64)
    x2 = int(n1**0.5)
    u2 = f12(x2)
    v2 = m1.log(n1)
    if n1 < 60184:
        w2 = int(1.25506 * n1 / v2)
    else:
        w2 = int(n1 / (v2 - 1))
    r2 = np.empty(w2, dtype=np.int64)
    s2 = 0
    for i1 in a8:
        if i1 <= n1:
            r2[s2] = i1
            s2 += 1
        else:
            break
    j1 = 31
    if j1 > n1:
        return r2[:s2]
    if n1 >= 3e10:
        base1 = 210
        wh1 = a9
    else:
        base1 = 30
        wh1 = a7
    while True:
        if j1 > n1:
            break
        k1 = j1 + y1 - 1
        if k1 > n1:
            k1 = n1
        if k1 < 7:
            break
        l1 = k1 - j1 + 1
        m1x = (l1 + 1) >> 1
        o1 = (m1x + 7) >> 3
        p1 = np.full(o1, 0xFF, dtype=np.uint8)
        f15(p1, j1, k1, u2)
        q1 = (j1 // base1) * base1
        while q1 <= k1:
            for h1 in wh1:
                a1x = q1 + h1
                if a1x < j1:
                    continue
                if a1x > k1:
                    break
                if a1x <= n1:
                    b1x = (a1x - j1) >> 1
                    if p1[b1x >> 3] & (1 << (b1x & 7)):
                        r2[s2] = a1x
                        s2 += 1
            q1 += base1
        j1 = k1 + 2
    return r2[:s2]

@nj1(fastmath=True)
def f19(n1, y1=10000000):
    y1 = 50000000
    if n1 < 2:
        return np.empty(0, dtype=np.int64)
    x2 = int(n1**0.5)
    u2 = f13(x2)
    v2 = m1.log(n1)
    if n1 < 60184:
        w2 = int(1.25506 * n1 / v2)
    else:
        w2 = int(n1 / (v2 - 1.1))
    r2 = np.empty(w2, dtype=np.int64)
    s2 = 0
    for i1 in a8:
        if i1 <= n1:
            r2[s2] = i1
            s2 += 1
        else:
            break
    j1 = 31
    if j1 > n1:
        return r2[:s2]
    if n1 >= 3e10:
        base1 = 210
        wh1 = a9
    else:
        base1 = 30
        wh1 = a7
    while True:
        if j1 > n1:
            break
        k1 = j1 + y1 - 1
        if k1 > n1:
            k1 = n1
        if k1 < 7:
            break
        l1 = k1 - j1 + 1
        m1x = (l1 + 1) >> 1
        o1 = (m1x + 7) >> 3
        p1 = np.full(o1, 0xFF, dtype=np.uint8)
        f16(p1, j1, k1, u2)
        q1 = (j1 // base1) * base1
        while q1 <= k1:
            for h1 in wh1:
                a1x = q1 + h1
                if a1x < j1:
                    continue
                if a1x > k1:
                    break
                if a1x <= n1:
                    b1x = (a1x - j1) >> 1
                    if p1[b1x >> 3] & (1 << (b1x & 7)):
                        r2[s2] = a1x
                        r2 += 1
            q1 += base1
        j1 = k1 + 2
    return r2[:s2]

@nj1(fastmath=True)
def f20(n1, y1=10000000):
    y1 = 50000000
    if n1 < 2:
        return np.empty(0, dtype=np.int64)
    x2 = int(n1**0.5)
    u2 = f13(x2)
    v2 = m1.log(n1)
    if n1 < 60184:
        w2 = int(1.25506 * n1 / v2)
    else:
        w2 = int(n1 / (v2 - 1.1))
    r2 = np.empty(w2, dtype=np.int64)
    s2 = 0
    for i1 in a8:
        if i1 <= n1:
            r2[s2] = i1
            s2 += 1
        else:
            break
    j1 = 31
    if j1 > n1:
        return r2[:s2]
    if n1 >= 3e10:
        base1 = 210
        wh1 = a9
    else:
        base1 = 30
        wh1 = a7
    while True:
        if j1 > n1:
            break
        k1 = j1 + y1 - 1
        if k1 > n1:
            k1 = n1
        if k1 < 7:
            break
        l1 = k1 - j1 + 1
        m1x = (l1 + 1) >> 1
        o1 = (m1x + 7) >> 3
        p1 = np.full(o1, 0xFF, dtype=np.uint8)
        f15(p1, j1, k1, u2)
        q1 = (j1 // base1) * base1
        while q1 <= k1:
            for h1 in wh1:
                a1x = q1 + h1
                if a1x < j1:
                    continue
                if a1x > k1:
                    break
                if a1x <= n1:
                    b1x = (a1x - j1) >> 1
                    if p1[b1x >> 3] & (1 << (b1x & 7)):
                        r2[s2] = a1x
                        s2 += 1
            q1 += base1
        j1 = k1 + 2
    return r2[:s2]

def f21(x1):
    x1 = x1.strip().lower()
    if x1.endswith('m'):
        return int(float(x1[:-1]) * 1e6)
    return int(float(x1))

def f22(a1n):
    if a1n > 2e10:
        print("Recommended parallel for N > 2e10")
    print("Choose the sieve method:")
    print("1) Eratosthenes single-thread")
    print("2) Eratosthenes parallel")
    print("3) Euler single-thread")
    print("4) Euler parallel")
    c1 = input().strip()
    d1 = t1.time()
    if a1n <= 10000000:
        if c1 == '3' or c1 == '4':
            e1 = f13(a1n)
        else:
            e1 = f12(a1n)
    else:
        if a1n > 20000000000:
            if c1 == '1':
                c1 = '2'
            elif c1 == '3':
                c1 = '4'
        if c1 == '1':
            e1 = f17(a1n)
        elif c1 == '2':
            e1 = f23(a1n)
        elif c1 == '3':
            e1 = f19(a1n)
        else:
            e1 = f24(a1n)
    f1x = t1.time()
    print(f"sieve up to {a1n} completed, {len(e1)} primes found.")
    print(f"Largest prime: {e1[-1]}")
    print(f"Time taken: {f1x - d1:.4f} seconds")

@nj1(parallel=True, fastmath=True)
def f23(a1n, y1=50000000):
    if a1n < 2:
        return np.empty(0, dtype=np.int64)
    x1 = int(a1n**0.5)
    u1 = f12(x1)
    u1 = u1[u1 >= 7]
    v1 = m1.log(a1n)
    if a1n < 60184:
        w1 = int(1.25506 * a1n / v1)
    else:
        w1 = int(a1n / (v1 - 1.1))
    r1 = []
    for i1 in a8:
        if i1 <= a1n:
            r1.append(i1)
        else:
            break
    j1 = 31
    if j1 > a1n:
        return np.array(r1, dtype=np.int64)
    if a1n >= 3e10:
        base1 = 210
        wh1 = a9
    else:
        base1 = 30
        wh1 = a7
    k1 = (a1n - j1 + y1) // y1
    l1 = lt1()
    for _ in range(k1):
        l1.append(np.empty(0, dtype=np.int64))
    for i2 in pg1(k1):
        n2 = j1 + i2 * y1
        o2 = n2 + y1 - 1
        if o2 > a1n:
            o2 = a1n
        if o2 < 7:
            l1[i2] = np.empty(0, dtype=np.int64)
            continue
        p2 = o2 - n2 + 1
        q2 = (p2 + 1) >> 1
        r2 = (q2 + 7) >> 3
        s2 = np.full(r2, 0xFF, dtype=np.uint8)
        f15(s2, n2, o2, u1)
        t2 = np.empty(q2, dtype=np.int64)
        u2 = 0
        v2 = (n2 // base1) * base1
        while v2 <= o2:
            for h2 in wh1:
                a2x = v2 + h2
                if a2x < n2:
                    continue
                if a2x > o2:
                    break
                if a2x <= a1n:
                    b2x = (a2x - n2) >> 1
                    if s2[b2x >> 3] & (1 << (b2x & 7)):
                        t2[u2] = a2x
                        u2 += 1
            v2 += base1
        l1[i2] = t2[:u2]
    aa1 = len(r1)
    for i2 in range(k1):
        aa1 += len(l1[i2])
    bb1 = np.empty(aa1, dtype=np.int64)
    cc1 = 0
    for dd1 in r1:
        bb1[cc1] = dd1
        cc1 += 1
    for i2 in range(k1):
        ee1 = l1[i2]
        ff1 = len(ee1)
        bb1[cc1 : cc1 + ff1] = ee1
        cc1 += ff1
    return bb1

@nj1(parallel=True, fastmath=True)
def f24(a1n, y1=50000000):
    if a1n < 2:
        return np.empty(0, dtype=np.int64)
    x1 = int(a1n**0.5)
    u1 = f13(x1)
    u1 = u1[u1 >= 7]
    v1 = m1.log(a1n)
    if a1n < 60184:
        w1 = int(1.25506 * a1n / v1)
    else:
        w1 = int(a1n / (v1 - 1.1))
    r1 = []
    for i1 in a8:
        if i1 <= a1n:
            r1.append(i1)
        else:
            break
    j1 = 31
    if j1 > a1n:
        return np.array(r1, dtype=np.int64)
    if a1n >= 3e10:
        base1 = 210
        wh1 = a9
    else:
        base1 = 30
        wh1 = a7
    k1 = (a1n - j1 + y1) // y1
    l1 = lt1()
    for _ in range(k1):
        l1.append(np.empty(0, dtype=np.int64))
    for i2 in pg1(k1):
        n2 = j1 + i2 * y1
        o2 = n2 + y1 - 1
        if o2 > a1n:
            o2 = a1n
        if o2 < 7:
            l1[i2] = np.empty(0, dtype=np.int64)
            continue
        p2 = o2 - n2 + 1
        q2 = (p2 + 1) >> 1
        r2 = (q2 + 7) >> 3
        s2 = np.full(r2, 0xFF, dtype=np.uint8)
        f15(s2, n2, o2, u1)
        t2 = np.empty(q2, dtype=np.int64)
        u2 = 0
        v2 = (n2 // base1) * base1
        while v2 <= o2:
            for h2 in wh1:
                a2x = v2 + h2
                if a2x < n2:
                    continue
                if a2x > o2:
                    break
                if a2x <= a1n:
                    b2x = (a2x - n2) >> 1
                    if s2[b2x >> 3] & (1 << (b2x & 7)):
                        t2[u2] = a2x
                        u2 += 1
            v2 += base1
        l1[i2] = t2[:u2]
    aa1 = len(r1)
    for i2 in range(k1):
        aa1 += len(l1[i2])
    bb1 = np.empty(aa1, dtype=np.int64)
    cc1 = 0
    for dd1 in r1:
        bb1[cc1] = dd1
        cc1 += 1
    for i2 in range(k1):
        ee1 = l1[i2]
        ff1 = len(ee1)
        bb1[cc1 : cc1 + ff1] = ee1
        cc1 += ff1
    return bb1

def f25():
    f1(10)
    f2(10)
    x1 = np.full(1, 0xFF, dtype=np.uint8)
    y1 = np.array([2,3,5,7], dtype=np.int64)
    f3(x1, 10, 20, y1)
    f4(x1, 10, 20, y1)
    f5(x1, 10, 20, y1)
    f6(10)
    f7(10)
    f8(10)
    f9(10)
    f12(10)
    f13(10)
    f14(x1, 10, 20, y1)
    f15(x1, 10, 20, y1)
    f16(x1, 10, 20, y1)
    f17(10)
    f18(10)
    f19(10)
    f20(10)
    f21("10")
    f23(10)
    f24(10)

if __name__ == "__main__":
    f25()
    u1 = input("Please enter the range N (e.g., 3e10, 50000000, 50m): ").strip()
    n1 = f10(u1) if u1 else 30000000000
    if n1 <= 6e10:
        f22(n1)
    else:
        f11(n1)